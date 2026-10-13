"""Open-Meteo historical weather fetcher (https://open-meteo.com/en/docs/historical-weather-api).

For every city in countries.yaml we download the daily mean 2 m temperature
(local-time days, the country's time zone) and combine the cities with their
population weights into:

* `openmeteo_temp_mean`: weighted mean temperature (°C);
* `openmeteo_hdd18`: weighted heating degree days, sum of w * max(0, 18 - T);
* `openmeteo_cdd22`: weighted cooling degree days, sum of w * max(0, T - 22).

Degree days are computed per city before weighting, because they are
non-linear in temperature.

The free tier (non-commercial, CC-BY 4.0) allows 600 calls/minute,
5,000/hour and 10,000/day, where a request counts as one call per location
per 2 weeks of data. A full 2015 backfill is ~34,000 such calls, so the
fetcher works in (country, year) chunks, newest first, skips chunks already
complete in the store, and stops at a per-run budget (the run is then logged
as `partial` and continues next time).
"""

from __future__ import annotations

import math
import time
from datetime import date, timedelta
from typing import Any

import pandas as pd

from tracker import store
from tracker.fetchers.base import BaseFetcher, FetchError
from tracker.registry import City


def call_weight(n_locations: int, days: int) -> int:
    """Open-Meteo's cost of a request: one call per location per 14 days."""
    return n_locations * max(1, math.ceil(days / 14))


def parse_locations(payload: Any, cities: list[City]) -> pd.DataFrame:
    """Archive JSON (a dict for one location, a list for several, in request
    order) -> DataFrame(date, city, temp)."""
    if isinstance(payload, dict) and payload.get("error"):
        raise FetchError(f"openmeteo: {payload.get('reason')}")
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(cities):
        raise FetchError(f"openmeteo: expected {len(cities)} locations, got {len(locations)}")
    frames = []
    for city, loc in zip(cities, locations):
        daily = loc.get("daily") or {}
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(daily.get("time", [])),
            "city": city.name,
            "temp": pd.to_numeric(pd.Series(daily.get("temperature_2m_mean", []), dtype="object"), errors="coerce"),
        }))
    return pd.concat(frames, ignore_index=True)


def country_weather(city_temps: pd.DataFrame, weights: dict[str, float], hdd_base: float, cdd_base: float) -> pd.DataFrame:
    """Population-weighted temperature, HDD and CDD per day.

    A day is kept only when every city has a value (weights are not
    renormalised over a partial set of cities).
    """
    df = city_temps.dropna(subset=["temp"]).copy()
    df["w"] = df["city"].map(weights)
    df["hdd"] = (hdd_base - df["temp"]).clip(lower=0)
    df["cdd"] = (df["temp"] - cdd_base).clip(lower=0)
    for col in ("temp", "hdd", "cdd"):
        df[col] = df[col] * df["w"]
    grouped = df.groupby("date").agg(temp=("temp", "sum"), hdd=("hdd", "sum"), cdd=("cdd", "sum"), n=("city", "nunique"))
    complete = grouped[grouped["n"] == len(weights)]
    return complete.drop(columns="n").reset_index()


class OpenMeteoFetcher(BaseFetcher):
    """Population-weighted country temperature and degree days."""

    source = "openmeteo"
    min_interval_s = 1.0

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        cfg = self.config
        ind = cfg["indicators"]
        today = self.fetched_at.date()
        end = min(end, today - timedelta(days=1))  # today is incomplete; future dates are rejected
        countries = self.registry.countries_for(ind["temp_indicator"])
        refetch_from = today - timedelta(days=int(cfg.get("refetch_window_days", 30)))
        chunks = self._plan_chunks(countries, start, end, refetch_from)

        budget = float(cfg.get("max_weighted_calls_per_run", 4000))
        spent, frames = 0.0, []
        for country, s, e in chunks:
            cities = self.registry.countries[country].weather_cities
            weight = call_weight(len(cities), (e - s).days + 1)
            if spent + weight > budget:
                self.partial = True
                self.log.warning("openmeteo: call budget %d reached; %d chunk(s) left for the next run",
                                 budget, len(chunks) - chunks.index((country, s, e)))
                break
            frames.append(self._fetch_chunk(country, s, e))
            spent += weight
            self._pace(weight)
        self.log.info("openmeteo: %d chunk(s) fetched, ~%d weighted calls", len(frames), spent)
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    def _plan_chunks(self, countries: list[str], start: date, end: date, refetch_from: date) -> list[tuple[str, date, date]]:
        """(country, start, end) per calendar year, newest year first, skipping
        years already complete in the store (unless inside the refetch window)."""
        stored = store.latest(indicators=[self.config["indicators"]["temp_indicator"]], countries=countries,
                              start=start, end=end, data_dir=self.data_dir)
        counts = stored.groupby([stored["country"], stored["date"].dt.year]).size().to_dict() if not stored.empty else {}
        chunks = []
        for year in range(end.year, start.year - 1, -1):
            s, e = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
            if s > e:
                continue
            for country in countries:
                complete = counts.get((country, year), 0) >= (e - s).days + 1
                if complete and e < refetch_from:
                    continue
                # Only the refetch window needs downloading again for a complete
                # year (a per-country start: `s` must not carry over to the next country).
                chunks.append((country, max(s, refetch_from) if complete else s, e))
        return chunks

    def _fetch_chunk(self, country: str, start: date, end: date) -> pd.DataFrame:
        ref = self.registry.countries[country]
        cities = ref.weather_cities
        params = {
            "latitude": ",".join(f"{c.lat:.4f}" for c in cities),
            "longitude": ",".join(f"{c.lon:.4f}" for c in cities),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "temperature_2m_mean",
            "timezone": ref.timezone,
        }
        payload = self.get(self.config["base_url"], params, raw_name=f"{country}_{start}_{end}").json()
        city_temps = parse_locations(payload, cities)
        weights = {c.name: c.weight for c in cities}
        weather = country_weather(city_temps, weights, self.config.get("hdd_base", 18.0), self.config.get("cdd_base", 22.0))
        weather["country"] = country
        ind = self.config["indicators"]
        return pd.concat([
            self.frame(ind["temp_indicator"], weather.rename(columns={"temp": "value"})),
            self.frame(ind["hdd_indicator"], weather.rename(columns={"hdd": "value"})),
            self.frame(ind["cdd_indicator"], weather.rename(columns={"cdd": "value"})),
        ], ignore_index=True)

    def _pace(self, weight: float) -> None:
        """Sleep so weighted calls stay under the per-minute allowance."""
        per_minute = float(self.config.get("max_weighted_calls_per_minute", 400))
        delay = weight / per_minute * 60 - self.min_interval_s
        if delay > 0:
            time.sleep(delay)
