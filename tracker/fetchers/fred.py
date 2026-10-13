"""FRED / ALFRED fetcher (https://fred.stlouisfed.org/docs/api/fred/).

Series are listed in `config/sources.yaml` under `fred.series`. For official
series (`vintages: true`) we request every real-time period since the start
date (ALFRED), so each value carries the date it was actually published and
`vintage_type='source_vintage'`. Market series use current values with
availability estimated from the registry's publication lag. Series whose
indicator is `retired` in the registry are skipped (PLAN D56).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from tracker import store
from tracker.fetchers.base import BaseFetcher, FetchError

OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
PAGE_LIMIT = 100_000  # FRED's maximum `limit`
MISSING = "."  # FRED's marker for a missing observation


def parse_observations(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse a FRED observations JSON payload into date, value, realtime_start.

    Missing values ('.') are dropped.
    """
    rows = payload.get("observations", [])
    if not rows:
        return pd.DataFrame(columns=["date", "value", "realtime_start"])
    df = pd.DataFrame(rows)[["date", "value", "realtime_start"]]
    df = df[df["value"] != MISSING].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    return df.dropna(subset=["value"]).reset_index(drop=True)


def vintage_available_at(realtime_start: pd.Series) -> pd.Series:
    """ALFRED gives the publication *date*; we treat a vintage as known by the
    end of that day (next midnight UTC), which avoids any intraday look-ahead."""
    return (pd.to_datetime(realtime_start) + pd.Timedelta(days=1)).dt.tz_localize("UTC")


class FredFetcher(BaseFetcher):
    """Fetch the FRED series configured in sources.yaml."""

    source = "fred"
    min_interval_s = 0.6  # FRED allows 120 requests/minute; stay near 100

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        series = self.config.get("series", [])
        if not series:
            raise FetchError("fred: no series configured in sources.yaml")
        retired = [s["series_id"] for s in series if self.registry.indicator(s["indicator"]).retired]
        if retired:
            self.log.info("fred: skipping retired series %s", ", ".join(retired))
        series = [s for s in series if s["series_id"] not in retired]
        key = self.api_key()
        frames = [self._fetch_series(spec, key, start, end) for spec in series]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    def _fetch_series(self, spec: dict[str, Any], key: str, start: date, end: date) -> pd.DataFrame:
        series_id = spec["series_id"]
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "limit": PAGE_LIMIT,
        }
        vintages = bool(spec.get("vintages"))
        if vintages:
            # Every real-time period since `start`. Observations dated on or after
            # `start` were first published after it, so no vintage is clipped.
            params.update(realtime_start=start.isoformat(), realtime_end="9999-12-31", output_type=1)

        parsed = []
        offset = 0
        while True:
            params["offset"] = offset
            payload = self.get(OBSERVATIONS_URL, params, raw_name=f"{series_id}_{offset}").json()
            parsed.append(parse_observations(payload))
            count = int(payload.get("count", 0))
            offset += PAGE_LIMIT
            if offset >= count:
                break
        obs = pd.concat(parsed, ignore_index=True)
        self.log.info("fred: %s: %d observations%s", series_id, len(obs), " (all vintages)" if vintages else "")
        if obs.empty:
            return store.empty_frame()

        data = pd.DataFrame({"date": obs["date"], "country": spec["country"], "value": obs["value"]})
        if vintages:
            return self.frame(
                spec["indicator"], data,
                available_at=vintage_available_at(obs["realtime_start"]),
                vintage_type="source_vintage",
            )
        return self.frame(spec["indicator"], data)
