"""Shared fetcher interface: HTTP with retries and rate limiting, raw copies,
availability stamping and conversion to the store schema.

A fetcher subclasses `BaseFetcher`, sets `source`, and implements
`_fetch(start, end)` returning rows in the store schema (usually built with
`self.frame(...)`). Callers use `fetch(start, end)`, which also validates the
result and logs row counts.
"""

from __future__ import annotations

import gzip
import logging
import time
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from tracker import settings, store
from tracker.registry import Registry, load_registry


class FetchError(RuntimeError):
    """A fetch failed in a way that retrying won't fix (bad key, bad request...)."""


class RetryableHTTPError(RuntimeError):
    """HTTP 429 or 5xx: worth retrying with backoff."""


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (RetryableHTTPError, requests.ConnectionError, requests.Timeout))


def period_end(dates: pd.Series, frequency: str) -> pd.Series:
    """First instant after the period that starts at each date (naive dates)."""
    dates = pd.to_datetime(dates)
    if frequency == "D":
        return dates + pd.Timedelta(days=1)
    if frequency == "W":
        return dates + pd.Timedelta(days=7)
    if frequency == "M":
        return dates + pd.DateOffset(months=1)
    if frequency == "Q":
        return dates + pd.DateOffset(months=3)
    raise ValueError(f"unknown frequency {frequency!r}")


def estimate_availability(
    dates: pd.Series, frequency: str, lag_days: float | pd.Series, fetched_at: pd.Timestamp
) -> tuple[pd.Series, pd.Series]:
    """Estimate when each value became public: period end + publication lag.

    `lag_days` is a number or a per-row Series. Returns (available_at,
    vintage_type). Where the estimate is not earlier than `fetched_at` we are
    seeing the value "live", so available_at is the fetch time and the
    vintage type is `live`; otherwise `estimated`.
    """
    lag = pd.to_timedelta(lag_days, unit="D") if isinstance(lag_days, pd.Series) else pd.Timedelta(days=lag_days)
    est = (period_end(dates, frequency) + lag).dt.tz_localize("UTC")
    live = est >= fetched_at
    available = est.where(~live, fetched_at)
    vintage = pd.Series(np.where(live, "live", "estimated"), index=dates.index)
    return available, vintage


class BaseFetcher(ABC):
    """Base class for one data source."""

    source: str = ""
    min_interval_s: float = 0.5  # minimum seconds between requests (rate limit)
    timeout_s: float = 60.0
    max_attempts: int = 5
    backoff_min_s: float = 2.0
    backoff_max_s: float = 60.0

    def __init__(
        self,
        registry: Registry | None = None,
        config: dict[str, Any] | None = None,
        raw_dir: Path | None = None,
        session: requests.Session | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.registry = registry or load_registry()
        self.data_dir = data_dir  # store location, for fetchers that check what's already stored
        self.config = config if config is not None else settings.load_sources().get(self.source, {})
        self.raw_root = raw_dir or (settings.RAW_DIR / self.source)
        self.session = session or requests.Session()
        self.log = logging.getLogger(f"tracker.fetchers.{self.source}")
        self.fetched_at = pd.Timestamp.now(tz="UTC")
        self.requests_made = 0
        self.issues: list[dict[str, Any]] = []
        # Set True when a call budget stopped the fetch early (e.g. a long
        # backfill spread over several runs); the run is logged as "partial".
        self.partial = False
        self._last_request = 0.0

    # ------------------------------------------------------------------ public

    def fetch(self, start: date | str, end: date | str | None = None) -> pd.DataFrame:
        """Download [start, end] and return rows in the store schema."""
        self.fetched_at = pd.Timestamp.now(tz="UTC").floor("s")
        self.issues = []
        self.partial = False
        start_d = pd.Timestamp(start).date()
        end_d = pd.Timestamp(end).date() if end is not None else self.fetched_at.date()
        self.log.info("%s: fetching %s to %s", self.source, start_d, end_d)
        df = self._fetch(start_d, end_d)
        df = self._finalise(df)
        self.log.info("%s: %d rows fetched in %d requests", self.source, len(df), self.requests_made)
        return df

    @abstractmethod
    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        """Source-specific download and parsing; returns store-schema rows."""

    # ----------------------------------------------------------------- helpers

    def api_key(self) -> str:
        """The source's secret from .env (named by `key_env` in sources.yaml)."""
        name = self.config.get("key_env")
        if not name:
            raise FetchError(f"{self.source}: no key_env configured in sources.yaml")
        value = settings.get_secret(name)
        if not value:
            raise FetchError(f"{self.source}: {name} is not set in .env")
        return value

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        raw_name: str | None = None,
        headers: dict[str, str] | None = None,
        raw_ext: str = "json",
    ) -> requests.Response:
        """GET with rate limiting and retries; saves the raw body if `raw_name`."""
        retrying = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=self.backoff_min_s, max=self.backoff_max_s),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        response = retrying(self._get_once, url, params, headers)
        if raw_name:
            self.save_raw(raw_name, response.content, raw_ext)
        return response

    def wait_turn(self) -> None:
        """Sleep until `min_interval_s` has passed since the previous request,
        then count this one. Call before every request (the rate limit)."""
        wait = self.min_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        self.requests_made += 1

    def _get_once(self, url: str, params: dict[str, Any] | None, headers: dict[str, str] | None) -> requests.Response:
        self.wait_turn()
        response = self.session.get(url, params=params, headers=headers, timeout=self.timeout_s)
        status = response.status_code
        if status == 429 or status >= 500:
            self.log.warning("%s: HTTP %s from %s, will retry", self.source, status, url)
            raise RetryableHTTPError(f"HTTP {status} from {url}")
        if status >= 400:
            body = settings.redact(response.text[:300])
            raise FetchError(f"{self.source}: HTTP {status} from {url}: {body}")
        return response

    def report_issue(self, indicator: str, country: str, check: str, severity: str, detail: str) -> None:
        """Record a data-quality problem found while fetching (the runner
        writes these to quality_issues)."""
        self.log.warning("%s: %s %s %s: %s", self.source, indicator, country, check, detail)
        self.issues.append({
            "source": self.source, "indicator": indicator, "country": country,
            "check": check, "severity": severity, "detail": detail,
        })

    def save_raw(self, name: str, content: bytes | str, ext: str = "json", compress: bool = False) -> Path:
        """Save an untouched response under data/raw/<source>/<YYYY-MM-DD>/.

        `compress=True` gzips it (the file gets a `.gz` suffix); the content is
        unchanged, for sources whose responses are large and compress well.
        """
        folder = self.raw_root / self.fetched_at.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        data = content.encode("utf-8") if isinstance(content, str) else content
        if compress:
            data, ext = gzip.compress(data), f"{ext}.gz"
        path = folder / f"{self.fetched_at.strftime('%H%M%S')}_{safe}.{ext}"
        path.write_bytes(data)
        return path

    def frame(
        self,
        indicator_id: str,
        data: pd.DataFrame,
        available_at: pd.Series | None = None,
        vintage_type: pd.Series | str | None = None,
        lag_by_country: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Turn `data` (columns date, country, value) into store-schema rows.

        Unit and frequency come from the registry. Without `available_at`,
        availability is estimated from the indicator's publication lag,
        overridden per country by `lag_by_country` where given.
        """
        ind = self.registry.indicator(indicator_id)
        if data.empty:
            return store.empty_frame()
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(data["date"]).dt.normalize(),
                "country": data["country"].astype(str).to_numpy(),
                "value": pd.to_numeric(data["value"], errors="coerce").to_numpy(),
            }
        )
        out["indicator"] = ind.id
        out["unit"] = ind.unit
        out["frequency"] = ind.frequency
        out["source"] = self.source
        out["fetched_at"] = self.fetched_at
        if available_at is None:
            lag: float | pd.Series = ind.publication_lag_days
            if lag_by_country:
                lag = out["country"].map(lag_by_country).fillna(ind.publication_lag_days).astype(float)
            avail, vintage = estimate_availability(out["date"], ind.frequency, lag, self.fetched_at)
            out["available_at"] = avail
            out["vintage_type"] = vintage
        else:
            out["available_at"] = pd.to_datetime(pd.Series(available_at).to_numpy(), utc=True)
            out["vintage_type"] = vintage_type if vintage_type is not None else "estimated"
        out = out[np.isfinite(out["value"])]
        return out[store.COLUMNS].reset_index(drop=True)

    def _finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return store.empty_frame()
        unknown = sorted(set(df["indicator"]) - set(self.registry.indicators))
        if unknown:
            raise FetchError(f"{self.source}: indicators {unknown} are not defined in config/indicators.yaml")
        wrong = sorted(set(df["source"]) - {self.source})
        if wrong:
            raise FetchError(f"{self.source}: rows tagged with other sources {wrong}")
        return store.validate(df)
