"""Shared test helpers: a fake HTTP session that serves saved fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import requests

from tracker import settings


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int = 200, text: str = "", content: bytes | None = None,
                 headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.headers = headers or {}

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error for url: https://example.test/?token=secret", response=self)


def fixture_text(*parts: str) -> str:
    """Read a file from tests/fixtures/."""
    return Path(settings.FIXTURES_DIR, *parts).read_text(encoding="utf-8")


class RoutedSession:
    """Fake requests.Session: `handler(url, params)` returns a FakeResponse.

    Every call is recorded in `calls` as (url, params, headers).
    """

    def __init__(self, handler: Callable[[str, dict[str, Any]], FakeResponse]) -> None:
        self.handler = handler
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, Any], dict[str, str] | None]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, headers=None, timeout=None, **kwargs: Any) -> FakeResponse:
        params = dict(params or {})
        self.calls.append((url, params, headers))
        return self.handler(url, params)


def no_wait(fetcher) -> None:
    """Remove rate-limit sleeps and retry backoff for fast tests."""
    fetcher.min_interval_s = 0
    fetcher.backoff_min_s = fetcher.backoff_max_s = 0
