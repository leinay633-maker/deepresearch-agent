"""SSRF-resistant URL validation and bounded text fetching.

DNS is resolved fail-closed before every request and redirect hop.  The standard
library transport resolves the hostname again when it opens the socket, so this
module cannot completely pin DNS against a rebinding race.  Production
deployments should additionally enforce egress filtering or use a transport
that connects to the validated address while preserving TLS SNI/Host headers.
"""

from __future__ import annotations

import ipaddress
import gzip
import io
import re
import socket
from collections.abc import Callable, Iterable
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 3
MAX_URL_LENGTH = 8192

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_BLOCKED_HOSTS = frozenset({"localhost", "metadata.google.internal"})
_CROSS_ORIGIN_SENSITIVE_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "cookie", "cookie2"}
)
_STRUCTURED_TEXT_MIME_TYPES = frozenset(
    {
        "application/atom+xml",
        "application/javascript",
        "application/json",
        "application/ld+json",
        "application/problem+json",
        "application/rss+xml",
        "application/xhtml+xml",
        "application/xml",
    }
)
_CHARSET_RE = re.compile(r"(?:^|;)\s*charset\s*=\s*[\"']?([^;\"']+)", re.IGNORECASE)


class URLPolicyError(ValueError):
    """Raised when a URL or one of its resolved addresses is unsafe."""


class SafeHTTPError(RuntimeError):
    """Raised when a bounded HTTP fetch cannot be completed safely."""


class ResponseTooLargeError(SafeHTTPError):
    """Raised when a response exceeds the configured byte budget."""


class UnsupportedContentTypeError(SafeHTTPError):
    """Raised when a response is not a supported textual representation."""


class FetchedText(str):
    """Text plus the canonical URL reached after validated redirects.

    It remains a string for backwards compatibility with existing crawler
    callers, while allowing provenance-aware callers to use ``final_url``.
    """

    final_url: str
    redirect_chain: tuple[str, ...]

    def __new__(
        cls,
        text: str,
        *,
        final_url: str,
        redirect_chain: tuple[str, ...],
    ) -> "FetchedText":
        result = super().__new__(cls, text)
        result.final_url = final_url
        result.redirect_chain = redirect_chain
        return result


# Less acronym-heavy aliases are convenient for callers and backwards-compatible
# with either common spelling in downstream code.
UrlPolicyError = URLPolicyError
SafeHttpError = SafeHTTPError


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def no_redirect_urlopen(request: Request, timeout: float | None = None) -> Any:
    """Open one HTTP request without following redirects automatically."""

    opener = build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def validate_url(
    url: str,
    *,
    resolver: Callable[..., Iterable[tuple[Any, ...]]] | None = None,
) -> str:
    """Validate an HTTP(S) URL and every A/AAAA address it currently resolves to.

    Resolution is deliberately fail-closed: a lookup error, no usable address,
    or any non-global address rejects the entire hostname.
    """

    if not isinstance(url, str) or not url:
        raise URLPolicyError("URL must be a non-empty string")
    if len(url) > MAX_URL_LENGTH:
        raise URLPolicyError(f"URL exceeds {MAX_URL_LENGTH} characters")
    if url != url.strip() or any(character.isspace() for character in url):
        raise URLPolicyError("URL must not contain whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        raise URLPolicyError("URL must not contain control characters")

    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise URLPolicyError(f"invalid URL: {exc}") from exc

    if scheme not in _ALLOWED_SCHEMES:
        raise URLPolicyError("only http and https URLs are allowed")
    if not parsed.netloc or not hostname:
        raise URLPolicyError("URL must include a hostname")
    if "@" in parsed.netloc or parsed.username is not None or parsed.password is not None:
        raise URLPolicyError("URL userinfo is not allowed")
    if "\\" in parsed.netloc:
        raise URLPolicyError("backslashes are not allowed in URL authority")
    if port is not None and not 1 <= port <= 65535:
        raise URLPolicyError("URL port must be between 1 and 65535")

    normalized_host = _normalize_hostname(hostname)
    if normalized_host in _BLOCKED_HOSTS or normalized_host.endswith(".localhost"):
        raise URLPolicyError(f"blocked hostname: {normalized_host}")
    if normalized_host.endswith(".metadata.google.internal"):
        raise URLPolicyError(f"blocked metadata hostname: {normalized_host}")

    literal = _ip_literal(normalized_host)
    if literal is not None:
        _require_global_unicast(literal, normalized_host)
        return url

    resolve = resolver or socket.getaddrinfo
    service_port = port or (443 if scheme == "https" else 80)
    try:
        answers = resolve(
            normalized_host,
            service_port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except (OSError, socket.gaierror) as exc:
        raise URLPolicyError(f"DNS resolution failed for {normalized_host}: {exc}") from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for answer in answers:
        try:
            family = answer[0]
            sockaddr = answer[4]
            raw_address = str(sockaddr[0])
        except (IndexError, TypeError):
            continue
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        address = _ip_literal(raw_address.split("%", 1)[0])
        if address is not None:
            addresses.append(address)

    if not addresses:
        raise URLPolicyError(f"DNS resolution returned no A/AAAA addresses for {normalized_host}")
    for address in addresses:
        _require_global_unicast(address, normalized_host)
    return url


def fetch_text_url(
    url: str,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    resolver: Callable[..., Iterable[tuple[Any, ...]]] | None = None,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Fetch a textual HTTP(S) response with SSRF and resource protections."""

    if max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be positive")
    if max_redirects < 0:
        raise ValueError("max_redirects must be non-negative")

    request_headers = {
        "Accept": "text/*, application/json, application/*+json, application/xml, "
        "application/*+xml;q=0.9",
        "Accept-Encoding": "identity",
        **(headers or {}),
    }
    open_request = opener or no_redirect_urlopen
    current_url = url
    current_headers = request_headers
    redirect_chain = [url]
    redirect_count = 0

    while True:
        validate_url(current_url, resolver=resolver)
        request = Request(current_url, headers=current_headers, method="GET")
        response = _open_once(open_request, request, timeout)
        try:
            status = _response_status(response)
            if 300 <= status < 400:
                location = _header_value(getattr(response, "headers", None), "Location")
                if not location:
                    raise SafeHTTPError(f"HTTP {status} redirect is missing Location")
                if redirect_count >= max_redirects:
                    raise SafeHTTPError(f"redirect limit exceeded ({max_redirects})")
                next_url = urljoin(current_url, location)
                # Validate before opening the next hop.  This is what prevents a
                # public endpoint from redirecting the crawler into a private net.
                validate_url(next_url, resolver=resolver)
                current_headers = _redirect_headers(
                    current_headers,
                    from_url=current_url,
                    to_url=next_url,
                )
                current_url = next_url
                redirect_chain.append(next_url)
                redirect_count += 1
                continue
            if not 200 <= status < 300:
                raise SafeHTTPError(f"HTTP request failed with status {status}")

            response_headers = getattr(response, "headers", None)
            content_type_header = _header_value(response_headers, "Content-Type")
            media_type = _media_type(content_type_header)
            if not _is_textual_media_type(media_type):
                shown_type = media_type or "missing"
                raise UnsupportedContentTypeError(
                    f"unsupported response Content-Type: {shown_type}"
                )

            content_encoding = _header_value(response_headers, "Content-Encoding")
            normalized_encoding = (content_encoding or "identity").strip().lower()
            if normalized_encoding not in {"identity", "none", "gzip", "x-gzip"}:
                raise SafeHTTPError(
                    f"unsupported response Content-Encoding: {content_encoding.strip()}"
                )

            declared_length = _content_length(response_headers)
            if declared_length is not None and declared_length > max_response_bytes:
                raise ResponseTooLargeError(
                    f"response exceeds {max_response_bytes} byte limit"
                )
            raw = _read_bounded(response, max_response_bytes)
            if normalized_encoding in {"gzip", "x-gzip"}:
                raw = _decompress_gzip_bounded(raw, max_response_bytes)
            charset = _response_charset(response_headers, content_type_header)
            try:
                text = raw.decode(charset, errors="replace")
            except LookupError:
                text = raw.decode("utf-8", errors="replace")
            return FetchedText(
                text,
                final_url=current_url,
                redirect_chain=tuple(redirect_chain),
            )
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()


safe_fetch_text = fetch_text_url


def _decompress_gzip_bounded(raw: bytes, max_bytes: int) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
            return _read_bounded(compressed, max_bytes)
    except (OSError, EOFError) as exc:
        raise SafeHTTPError(f"invalid gzip response: {exc}") from exc


def _normalize_hostname(hostname: str) -> str:
    if "%" in hostname:
        raise URLPolicyError("IPv6 zone identifiers are not allowed")
    host = hostname.rstrip(".").lower()
    if not host:
        raise URLPolicyError("URL hostname is empty")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise URLPolicyError("URL hostname is not valid IDNA") from exc


def _redirect_headers(
    headers: dict[str, str],
    *,
    from_url: str,
    to_url: str,
) -> dict[str, str]:
    """Preserve normal request headers but never forward credentials cross-origin."""

    if _same_origin(from_url, to_url):
        return headers
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _CROSS_ORIGIN_SENSITIVE_HEADERS
    }


def _same_origin(left: str, right: str) -> bool:
    """Compare normalized HTTP origins, including each scheme's default port."""

    left_parts = urlsplit(left)
    right_parts = urlsplit(right)
    try:
        left_host = _normalize_hostname(left_parts.hostname or "")
        right_host = _normalize_hostname(right_parts.hostname or "")
        left_port = left_parts.port or (443 if left_parts.scheme.lower() == "https" else 80)
        right_port = right_parts.port or (443 if right_parts.scheme.lower() == "https" else 80)
    except ValueError:
        # Both URLs are validated immediately before this function is called,
        # but treating malformed input as cross-origin is the safe fallback.
        return False
    return (
        left_parts.scheme.lower(),
        left_host,
        left_port,
    ) == (
        right_parts.scheme.lower(),
        right_host,
        right_port,
    )


def _ip_literal(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _require_global_unicast(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    hostname: str,
) -> None:
    candidates = [address]
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        candidates.append(address.ipv4_mapped)

    for candidate in candidates:
        site_local = bool(getattr(candidate, "is_site_local", False))
        if (
            not candidate.is_global
            or candidate.is_private
            or candidate.is_loopback
            or candidate.is_link_local
            or candidate.is_multicast
            or candidate.is_reserved
            or candidate.is_unspecified
            or site_local
        ):
            raise URLPolicyError(
                f"hostname {hostname} resolves to non-global address {candidate}"
            )


def _open_once(opener: Callable[..., Any], request: Request, timeout: float) -> Any:
    try:
        return opener(request, timeout=timeout)
    except HTTPError as exc:
        if 300 <= exc.code < 400:
            return exc
        raise SafeHTTPError(f"HTTP request failed with status {exc.code}") from exc
    except URLPolicyError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize transport failures for callers.
        raise SafeHTTPError(str(exc)) from exc


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = getattr(response, "code", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        status = getcode() if callable(getcode) else 200
    try:
        return int(status)
    except (TypeError, ValueError) as exc:
        raise SafeHTTPError(f"invalid HTTP response status: {status}") from exc


def _header_value(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is not None:
            return str(value)
    items = getattr(headers, "items", None)
    if callable(items):
        for key, value in items():
            if str(key).lower() == name.lower():
                return str(value)
    return None


def _media_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def _is_textual_media_type(media_type: str) -> bool:
    return bool(
        media_type.startswith("text/")
        or media_type in _STRUCTURED_TEXT_MIME_TYPES
        or media_type.endswith("+json")
        or media_type.endswith("+xml")
    )


def _content_length(headers: Any) -> int | None:
    value = _header_value(headers, "Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError:
        return None
    return max(length, 0)


def _response_charset(headers: Any, content_type: str | None) -> str:
    get_content_charset = getattr(headers, "get_content_charset", None)
    if callable(get_content_charset):
        charset = get_content_charset()
        if charset:
            return str(charset).strip()
    if content_type:
        match = _CHARSET_RE.search(content_type)
        if match:
            return match.group(1).strip()
    return "utf-8"


def _read_bounded(response: Any, max_response_bytes: int) -> bytes:
    payload = bytearray()
    while True:
        remaining_with_sentinel = max_response_bytes + 1 - len(payload)
        chunk = response.read(min(64 * 1024, remaining_with_sentinel))
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise SafeHTTPError("HTTP response returned non-byte content")
        payload.extend(chunk)
        if len(payload) > max_response_bytes:
            raise ResponseTooLargeError(
                f"response exceeds {max_response_bytes} byte limit"
            )
    return bytes(payload)
