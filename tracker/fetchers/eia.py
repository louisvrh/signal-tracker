"""EIA API v2 fetcher (https://www.eia.gov/opendata/documentation.php).

* Hourly Lower-48 demand (EIA-930, route electricity/rto/region-data) is
  aggregated to (a) daily total load and (b) weekday overnight load, the mean
  of the hours 00:00-05:00 local time. Day boundaries use US Central time for
  the whole Lower 48 (a simplification; see NOTES).
* Monthly retail electricity sales to industry (route electricity/retail-sales).

EIA-930 timestamps are UTC and label the hour by its *end*, so the hour
labelled 06 covers 05:00-06:00 UTC.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from tracker import store
from tracker.fetchers.base import BaseFetcher, FetchError
from tracker.fetchers.load import daily_load

PAGE_LENGTH = 5000  # EIA's maximum rows per JSON response


def parse_rows(payload: dict[str, Any], value_field: str) -> pd.DataFrame:
    """EIA v2 JSON -> DataFrame(period, value); raises on API errors."""
    if "error" in payload:
        raise FetchError(f"eia: {payload['error']}")
    resp = payload.get("response")
    if resp is None:
        raise FetchError(f"eia: unexpected response: {str(payload)[:200]}")
    rows = resp.get("data", [])
    if not rows:
        return pd.DataFrame(columns=["period", "value"])
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "period": df["period"].astype(str),
        "value": pd.to_numeric(df[value_field], errors="coerce"),
    }).dropna(subset=["value"])


def hourly_to_daily(hourly: pd.DataFrame, tz: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate hour-ending UTC demand (MW, EIA `period` strings) to local days.

    Returns (daily_total, weekday_overnight, incomplete) as
    `tracker.fetchers.load.daily_load` does: MWh over complete local days, the
    weekday mean MW over local 00:00-05:00, and days with missing hours.
    """
    if hourly.empty:
        return daily_load(pd.DatetimeIndex([], tz="UTC"), [], tz)
    end_utc = pd.to_datetime(hourly["period"], format="%Y-%m-%dT%H").dt.tz_localize("UTC")
    return daily_load(pd.DatetimeIndex(end_utc - pd.Timedelta(hours=1)), hourly["value"].to_numpy(), tz)


class EIAFetcher(BaseFetcher):
    """US Lower-48 load (daily and overnight) and monthly industrial sales."""

    source = "eia"
    min_interval_s = 0.5

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        key = self.api_key()
        frames = []
        if self.config.get("hourly_demand"):
            frames.append(self._fetch_load(key, start, end))
        for spec in self.config.get("monthly_sales", []):
            frames.append(self._fetch_sales(key, spec, start, end))
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    def _paged(self, route: str, params: dict[str, Any], value_field: str, label: str) -> pd.DataFrame:
        url = f"{self.config['base_url']}/{route}/data/"
        frames, offset = [], 0
        while True:
            page = dict(params, offset=offset, length=PAGE_LENGTH)
            payload = self.get(url, page, raw_name=f"{label}_{offset}").json()
            frames.append(parse_rows(payload, value_field))
            total = int(payload["response"].get("total", 0))
            offset += PAGE_LENGTH
            if offset >= total:
                break
        return pd.concat(frames, ignore_index=True)

    def _fetch_load(self, key: str, start: date, end: date) -> pd.DataFrame:
        cfg = self.config["hourly_demand"]
        tz = cfg["timezone"]
        # Pad by a day on each side so every local day is complete.
        params = {
            "api_key": key, "frequency": "hourly", "data[0]": "value",
            "facets[respondent][]": cfg["respondent"], "facets[type][]": "D",
            "start": f"{(start - timedelta(days=1)).isoformat()}T00",
            "end": f"{(end + timedelta(days=1)).isoformat()}T23",
            "sort[0][column]": "period", "sort[0][direction]": "asc",
        }
        hourly = self._paged("electricity/rto/region-data", params, "value", f"hourly_{cfg['respondent']}")
        daily, overnight, incomplete = hourly_to_daily(hourly, tz)
        in_range = lambda d: d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))]  # noqa: E731
        daily, overnight, incomplete = in_range(daily), in_range(overnight), in_range(incomplete)
        # Today and yesterday are normally still filling in; only older gaps are issues.
        cutoff = pd.Timestamp(self.fetched_at.tz_convert(tz).date()) - pd.Timedelta(days=2)
        for row in incomplete[incomplete["date"] < cutoff].itertuples():
            self.report_issue(cfg["daily_indicator"], cfg["country"], "missing_hours", "warning",
                              f"{row.date.date()}: {row.hours:g} of {row.expected:g} hours; day not stored")
        country = cfg["country"]
        return pd.concat([
            self.frame(cfg["daily_indicator"], daily.assign(country=country)),
            self.frame(cfg["overnight_indicator"], overnight.assign(country=country)),
        ], ignore_index=True)

    def _fetch_sales(self, key: str, spec: dict[str, Any], start: date, end: date) -> pd.DataFrame:
        params = {
            "api_key": key, "frequency": "monthly", "data[0]": "sales",
            "facets[stateid][]": spec["state"], "facets[sectorid][]": spec["sector"],
            "start": f"{start.year:04d}-{start.month:02d}", "end": f"{end.year:04d}-{end.month:02d}",
            "sort[0][column]": "period", "sort[0][direction]": "asc",
        }
        rows = self._paged("electricity/retail-sales", params, "sales", f"sales_{spec['sector']}")
        data = pd.DataFrame({"date": pd.to_datetime(rows["period"], format="%Y-%m"),
                             "country": spec["country"], "value": rows["value"].to_numpy()})
        return self.frame(spec["indicator"], data)
