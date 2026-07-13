from __future__ import annotations

import re


def tokenize(text: str) -> set[str]:
    """Return comparable English/number tokens and CJK unigrams/bigrams."""

    tokens = {token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", text)}
    for sequence in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text):
        tokens.update(sequence)
        tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def split_sentences(text: str) -> list[str]:
    # Chinese sentence punctuation does not need trailing whitespace. A period
    # does: splitting on every dot corrupts versions (3.14.0) and hostnames.
    parts = re.split(r"(?<=[!?。！？])\s*|(?<=\.)\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]
