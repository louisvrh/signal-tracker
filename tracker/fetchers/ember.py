"""Ember monthly electricity fetcher (https://api.ember-energy.org/v1/docs).

Monthly generation (total and by fuel) and demand, in TWh, for the countries
listed on each indicator. Data is CC-BY-4.0 ("Monthly electricity data,
Ember"). Countries publish at different speeds, so availability is estimated
with per-country publication lags from sources.yaml.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from tracker import store
from tracker.fetchers.base import BaseFetcher, FetchError


def parse_records(payload: dict[str, Any], value_field: str) -> pd.DataFrame:
    """Ember JSON -> DataFrame(date, country, series, value)."""
    data = payload.get("data")
    if data is None:
        raise FetchError(f"ember: unexpected response: {str(payload)[:200]}")
    if not data:
        return pd.DataFrame(columns=["date", "country", "series", "value"])
    df = pd.DataFrame(data)
    if value_field not in df.columns:
        raise FetchError(f"ember: field '{value_field}' missing from response")
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "country": df["entity_code"].astype(str),
        "series": df["series"].astype(str) if "series" in df.columns else "Demand",
        "value": pd.to_numeric(df[value_field], errors="coerce"),
    })
    return out.dropna(subset=["value"]).reset_index(drop=True)


class EmberFetcher(BaseFetcher):
    """Monthly generation and demand from Ember's API."""

    source = "ember"
    min_interval_s = 0.3  # Ember documents no rate limit; be polite anyway

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        key = self.api_key()
        base = self.config["base_url"]
        month = lambda d: f"{d.year:04d}-{d.month:02d}"  # noqa: E731
        # Ember's end_date is exclusive, so ask for the month after `end`.
        after_end = (pd.Timestamp(end).replace(day=1) + pd.DateOffset(months=1)).date()
        common = {"start_date": month(start), "end_date": month(after_end), "api_key": key}
        frames = []

        demand = self.config.get("demand")
        if demand:
            for country in self.registry.countries_for(demand["indicator"]):
                payload = self.get(f"{base}/electricity-demand/monthly", dict(common, entity_code=country),
                                   raw_name=f"demand_{country}").json()
                recs = parse_records(payload, "demand_twh")
                frames.append(self._frame(demand["indicator"], recs))

        gen = self.config.get("generation", [])
        if gen:
            by_series = {g["series"]: g["indicator"] for g in gen}
            countries = sorted(set().union(*(self.registry.countries_for(g["indicator"]) for g in gen)))
            for country in countries:
                payload = self.get(
                    f"{base}/electricity-generation/monthly",
                    dict(common, entity_code=country, series=",".join(by_series)),
                    raw_name=f"generation_{country}",
                ).json()
                recs = parse_records(payload, "generation_twh")
                for series, part in recs.groupby("series"):
                    if series in by_series:
                        frames.append(self._frame(by_series[series], part))
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    def _frame(self, indicator_id: str, recs: pd.DataFrame) -> pd.DataFrame:
        allowed = set(self.registry.countries_for(indicator_id))
        recs = recs[recs["country"].isin(allowed)]
        return self.frame(indicator_id, recs, lag_by_country=self.config.get("publication_lag_days_by_country"))
