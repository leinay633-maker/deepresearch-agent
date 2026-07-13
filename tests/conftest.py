from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _stable_local_retrieval_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests offline unless a test run explicitly opts into hybrid retrieval."""

    if "LOCAL_RETRIEVAL_MODE" not in os.environ:
        monkeypatch.setenv("LOCAL_RETRIEVAL_MODE", "keyword")
