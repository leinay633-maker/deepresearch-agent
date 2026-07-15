from __future__ import annotations

import socket
import gzip
from collections.abc import Callable
from typing import Any

import pytest

from deepresearch_agent.url_policy import (
    ResponseTooLargeError,
    SafeHTTPError,
    UnsupportedContentTypeError,
    URLPolicyError,
    fetch_text_url,
    validate_url,
)


PUBLIC_V4 = "93.184.216.34"
PUBLIC_V6 = "2606:4700:4700::1111"


def resolver_with(*addresses: str) -> Callable[..., list[tuple[Any, ...]]]:
    def resolve(host: str, port: int, **kwargs: Any) -> list[tuple[Any, ...]]:
        del host, kwargs
        answers: list[tuple[Any, ...]] = []
        for address in addresses:
            if ":" in address:
                answers.append(
                    (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, port, 0, 0))
                )
            else:
                answers.append(
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
                )
        return answers

    return resolve


class FakeResponse:
    def __init__(
        self,
        body: bytes = b"ok",
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = headers or {"Content-Type": "text/plain; charset=utf-8"}
        self.offset = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "https://user:password@example.com/",
        "http://localhost/",
        "http://api.localhost/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://0.0.0.0/",
        "http://224.0.0.1/",
        "http://240.0.0.1/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://[::ffff:10.0.0.1]/",
    ],
)
def test_validate_url_rejects_unsafe_targets(url: str) -> None:
    with pytest.raises(URLPolicyError):
        validate_url(url, resolver=resolver_with(PUBLIC_V4))


def test_validate_url_allows_global_ipv4_and_ipv6() -> None:
    assert validate_url(f"https://{PUBLIC_V4}/") == f"https://{PUBLIC_V4}/"
    assert (
        validate_url(f"https://[{PUBLIC_V6}]/")
        == f"https://[{PUBLIC_V6}]/"
    )


def test_dns_resolution_rejects_if_any_a_or_aaaa_address_is_non_global() -> None:
    mixed_resolver = resolver_with(PUBLIC_V4, "10.20.30.40", PUBLIC_V6)

    with pytest.raises(URLPolicyError, match="non-global"):
        validate_url("https://public.example/page", resolver=mixed_resolver)


def test_dns_resolution_rejects_mapped_private_ipv4() -> None:
    with pytest.raises(URLPolicyError, match="non-global"):
        validate_url(
            "https://public.example/page",
            resolver=resolver_with("::ffff:192.168.1.10"),
        )


def test_dns_resolution_is_fail_closed() -> None:
    def failing_resolver(*args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        del args, kwargs
        raise socket.gaierror("lookup failed")

    with pytest.raises(URLPolicyError, match="DNS resolution failed"):
        validate_url("https://unresolved.example/", resolver=failing_resolver)
    with pytest.raises(URLPolicyError, match="no A/AAAA"):
        validate_url("https://empty.example/", resolver=resolver_with())


def test_redirect_target_is_revalidated_before_it_is_opened() -> None:
    requested_urls: list[str] = []

    def opener(request: Any, timeout: float) -> FakeResponse:
        del timeout
        requested_urls.append(request.full_url)
        return FakeResponse(
            status=302,
            headers={"Location": "http://169.254.169.254/latest/meta-data/"},
        )

    with pytest.raises(URLPolicyError):
        fetch_text_url(
            "https://public.example/start",
            timeout=1.0,
            resolver=resolver_with(PUBLIC_V4),
            opener=opener,
        )

    assert requested_urls == ["https://public.example/start"]


def test_fetch_allows_at_most_three_redirect_hops() -> None:
    redirects = {
        "https://one.example/": "https://two.example/",
        "https://two.example/": "https://three.example/",
        "https://three.example/": "https://four.example/",
    }
    requested_urls: list[str] = []

    def opener(request: Any, timeout: float) -> FakeResponse:
        del timeout
        requested_urls.append(request.full_url)
        location = redirects.get(request.full_url)
        if location:
            return FakeResponse(status=302, headers={"Location": location})
        return FakeResponse(b"final text")

    text = fetch_text_url(
        "https://one.example/",
        timeout=1.0,
        resolver=resolver_with(PUBLIC_V4),
        opener=opener,
    )

    assert text == "final text"
    assert text.final_url == "https://four.example/"
    assert text.redirect_chain[-1] == "https://four.example/"
    assert requested_urls == [
        "https://one.example/",
        "https://two.example/",
        "https://three.example/",
        "https://four.example/",
    ]

    redirects["https://four.example/"] = "https://five.example/"
    with pytest.raises(SafeHTTPError, match="redirect limit"):
        fetch_text_url(
            "https://one.example/",
            timeout=1.0,
            resolver=resolver_with(PUBLIC_V4),
            opener=opener,
        )


def test_fetch_redirects_share_one_absolute_timeout_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    seen_timeouts: list[float] = []

    monkeypatch.setattr(
        "deepresearch_agent.url_policy.time.monotonic", lambda: now[0]
    )

    def opener(request: Any, timeout: float) -> FakeResponse:
        seen_timeouts.append(timeout)
        if request.full_url == "https://one.example/":
            now[0] += 0.6
            return FakeResponse(
                status=302,
                headers={"Location": "https://two.example/"},
            )
        return FakeResponse(b"final")

    text = fetch_text_url(
        "https://one.example/",
        timeout=1.0,
        resolver=resolver_with(PUBLIC_V4),
        opener=opener,
    )

    assert text == "final"
    assert seen_timeouts == pytest.approx([1.0, 0.4])


def test_fetch_removes_jina_bearer_and_other_sensitive_headers_on_cross_origin_redirect() -> None:
    requested_headers: list[dict[str, str]] = []

    def opener(request: Any, timeout: float) -> FakeResponse:
        del timeout
        requested_headers.append(
            {name.lower(): value for name, value in request.header_items()}
        )
        if request.full_url == "https://r.jina.ai/https://public.example/page":
            return FakeResponse(
                status=302,
                headers={"Location": "https://redirected.example/reader-result"},
            )
        return FakeResponse(b"reader content")

    text = fetch_text_url(
        "https://r.jina.ai/https://public.example/page",
        timeout=1.0,
        headers={
            "Authorization": "Bearer jina-secret",
            "Cookie": "session=secret",
            "X-Request-Id": "safe-to-forward",
        },
        resolver=resolver_with(PUBLIC_V4),
        opener=opener,
    )

    assert text == "reader content"
    assert requested_headers[0]["authorization"] == "Bearer jina-secret"
    assert requested_headers[0]["cookie"] == "session=secret"
    assert "authorization" not in requested_headers[1]
    assert "cookie" not in requested_headers[1]
    assert requested_headers[1]["x-request-id"] == "safe-to-forward"


def test_fetch_preserves_authorization_for_same_origin_redirect() -> None:
    requested_headers: list[dict[str, str]] = []

    def opener(request: Any, timeout: float) -> FakeResponse:
        del timeout
        requested_headers.append(
            {name.lower(): value for name, value in request.header_items()}
        )
        if request.full_url == "https://r.jina.ai/first":
            return FakeResponse(status=302, headers={"Location": "/second"})
        return FakeResponse(b"same-origin reader content")

    fetch_text_url(
        "https://r.jina.ai/first",
        timeout=1.0,
        headers={"Authorization": "Bearer jina-secret"},
        resolver=resolver_with(PUBLIC_V4),
        opener=opener,
    )

    assert requested_headers[1]["authorization"] == "Bearer jina-secret"


def test_fetch_streams_with_a_hard_response_byte_limit() -> None:
    response = FakeResponse(b"01234567890")

    with pytest.raises(ResponseTooLargeError, match="10 byte limit"):
        fetch_text_url(
            "https://public.example/large",
            timeout=1.0,
            max_response_bytes=10,
            resolver=resolver_with(PUBLIC_V4),
            opener=lambda request, timeout: response,
        )

    assert response.closed is True


def test_fetch_rejects_declared_oversize_and_non_text_mime() -> None:
    oversize = FakeResponse(
        b"small",
        headers={"Content-Type": "text/plain", "Content-Length": "100"},
    )
    with pytest.raises(ResponseTooLargeError):
        fetch_text_url(
            "https://public.example/large",
            timeout=1.0,
            max_response_bytes=10,
            resolver=resolver_with(PUBLIC_V4),
            opener=lambda request, timeout: oversize,
        )

    image = FakeResponse(b"PNG", headers={"Content-Type": "image/png"})
    with pytest.raises(UnsupportedContentTypeError, match="image/png"):
        fetch_text_url(
            "https://public.example/image",
            timeout=1.0,
            resolver=resolver_with(PUBLIC_V4),
            opener=lambda request, timeout: image,
        )


def test_fetch_decompresses_gzip_with_a_decoded_size_limit() -> None:
    response = FakeResponse(
        gzip.compress("可验证正文".encode()),
        headers={"Content-Type": "text/html; charset=utf-8", "Content-Encoding": "gzip"},
    )
    text = fetch_text_url(
        "https://public.example/page",
        timeout=1.0,
        max_response_bytes=100,
        resolver=resolver_with(PUBLIC_V4),
        opener=lambda request, timeout: response,
    )
    assert text == "可验证正文"

    bomb = FakeResponse(
        gzip.compress(b"x" * 200),
        headers={"Content-Type": "text/plain", "Content-Encoding": "gzip"},
    )
    with pytest.raises(ResponseTooLargeError):
        fetch_text_url(
            "https://public.example/large-gzip",
            timeout=1.0,
            max_response_bytes=100,
            resolver=resolver_with(PUBLIC_V4),
            opener=lambda request, timeout: bomb,
        )
