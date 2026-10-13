"""NESO fetcher: Great Britain's electricity demand from the NESO data portal (no key).

NESO (the GB system operator) publishes half-hourly demand on a CKAN portal,
https://api.neso.energy, in two datasets with the same columns:

* **Historic Demand Data** (`historic-demand-data`): one CSV per calendar
  year, settled data, updated daily about 21 days in arrears;
* **Demand Data Update** (`daily-demand-update`): one CSV with the last two
  months of actuals up to the current half hour, plus a week of forecasts.

The yearly files give the history; the update file adds the days after the
yearly file's last day (actuals only, `FORECAST_ACTUAL_INDICATOR = A`). Where
both cover a day their values have been identical (checked 2026-09-30); the
yearly file wins, and later corrections arrive as revisions through the
refetch window.

Demand measured: national demand (`ND`, transmission-connected generation
serving GB demand) plus NESO's estimates of embedded (distribution-connected)
wind and solar generation, i.e. demand including embedded generation, the
closest match to ENTSO-E's Actual Total Load (all generation minus exports
plus imports minus storage). It also keeps the growth of rooftop solar and
small wind from reading as falling demand. Each half hour becomes:

* `neso_load`: total over each local (Europe/London) calendar day, MWh;
* `neso_overnight_load`: mean over 00:00-05:00 local time, Monday to
  Friday, MW (as for ENTSO-E, via `tracker.fetchers.load`).

Settlement periods count half hours from local midnight (46 on the spring
clock-change day, 50 in autumn), so each is converted to its UTC start first.
Limits: one CKAN API request per second, and at most two datastore requests
per minute; the CSV downloads are spaced like the latter to be safe.
"""

from __future__ import annotations

import io
import re
import time
from datetime import date

import pandas as pd

from tracker import store
from tracker.fetchers.base import BaseFetcher, FetchError
from tracker.fetchers.load import daily_load

REQUIRED = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND", "EMBEDDED_WIND_GENERATION", "EMBEDDED_SOLAR_GENERATION"]
YEAR_RE = re.compile(r"^Historic Demand Data (\d{4})$")
HALF_HOUR = pd.Timedelta(minutes=30)


DATE_FORMATS = {  # pattern -> format; the yearly files differ (checked 2026-09-30)
    r"\d{4}-\d{2}-\d{2}": "%Y-%m-%d",        # 2025 onwards: 2025-01-01
    r"\d{1,2}-[A-Za-z]{3}-\d{4}": "%d-%b-%Y",  # 2009-2022 and 2024: 01-JAN-2015
    r"\d{1,2}-[A-Za-z]{3}-\d{2}": "%d-%b-%y",  # 2023: 01-Jan-23
}


def parse_settlement_dates(values: pd.Series) -> pd.Series:
    """Settlement dates as dates, in any of the formats the files use (`DATE_FORMATS`)."""
    text = values.astype(str).str.strip().str.title()
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[us]")
    for pattern, fmt in DATE_FORMATS.items():
        match = text.str.fullmatch(pattern)
        if match.any():
            out[match] = pd.to_datetime(text[match], format=fmt, errors="coerce")
    bad = out.isna() | (out.dt.year < 2000) | (out.dt.year > 2100)
    if bad.any():
        raise FetchError(f"neso: unreadable settlement dates, e.g. {values[bad].iloc[0]!r}")
    return out


def parse_demand_csv(text: str) -> pd.DataFrame:
    """A demand CSV -> settlement_date, period, value (MW: ND + embedded wind + embedded solar).

    Forecast rows (`FORECAST_ACTUAL_INDICATOR` F, in the update file) are dropped.
    """
    try:
        df = pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise FetchError(f"neso: CSV could not be parsed: {exc}") from exc
    df.columns = [str(c).strip().upper() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise FetchError(f"neso: CSV lacks columns {missing}: {text[:200]!r}")
    if "FORECAST_ACTUAL_INDICATOR" in df.columns:
        df = df[df["FORECAST_ACTUAL_INDICATOR"].astype(str).str.strip().str.upper() == "A"]
    parts = [pd.to_numeric(df[c], errors="coerce") for c in REQUIRED[2:]]
    return pd.DataFrame({
        "settlement_date": parse_settlement_dates(df["SETTLEMENT_DATE"]).to_numpy(),
        "period": pd.to_numeric(df["SETTLEMENT_PERIOD"], errors="coerce").to_numpy(),
        "value": (parts[0] + parts[1] + parts[2]).to_numpy(),
    }).dropna().reset_index(drop=True)


def interval_starts(settlement_dates: pd.Series, periods: pd.Series, tz: str) -> pd.DatetimeIndex:
    """UTC start of each settlement period: local midnight of its date + 30 minutes per earlier period."""
    midnights = pd.DatetimeIndex(settlement_dates).tz_localize(tz).tz_convert("UTC")
    return midnights + (pd.Index(periods).astype(int) - 1) * HALF_HOUR


class NesoFetcher(BaseFetcher):
    """Daily and weekday-overnight demand for Great Britain."""

    source = "neso"
    min_interval_s = 1.0         # CKAN API: at most one request per second
    download_interval_s = 31.0   # CSV downloads: no more than two a minute
    timeout_s = 120.0
    _last_download = float("-inf")

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        country = self.config["country"]
        tz = self.registry.countries[country].timezone
        years = self._yearly_files()
        pieces = []
        for year in range(start.year, end.year + 1):
            if year not in years:
                self.log.warning("neso: no Historic Demand Data file for %d yet", year)
                continue
            pieces.append(self._download(years[year], f"historic_{year}"))
        halfhours = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["settlement_date", "period", "value"])
        last_settled = halfhours["settlement_date"].max() if not halfhours.empty else None
        if last_settled is None or last_settled < pd.Timestamp(end):  # the yearly files stop short of `end`
            update = self._download(self._update_file(), "demand_data_update")
            if last_settled is not None:
                update = update[update["settlement_date"] > last_settled]
            halfhours = pd.concat([halfhours, update], ignore_index=True)
        if halfhours.empty:
            return store.empty_frame()
        return self._days(halfhours, country, tz, start, end)

    def _days(self, halfhours: pd.DataFrame, country: str, tz: str, start: date, end: date) -> pd.DataFrame:
        halfhours = halfhours.drop_duplicates(["settlement_date", "period"], keep="first")
        starts = interval_starts(halfhours["settlement_date"], halfhours["period"], tz)
        # A period number past the end of its day (e.g. 49 on a 48-period day) would land in the next day.
        own_day = starts.tz_convert(tz).tz_localize(None).normalize() == pd.DatetimeIndex(halfhours["settlement_date"])
        if not own_day.all():
            bad = halfhours[~own_day].iloc[0]
            self.report_issue(self.config["daily_indicator"], country, "bad_period", "warning",
                              f"{int((~own_day).sum())} settlement periods outside their day, e.g. "
                              f"{bad['settlement_date']:%Y-%m-%d} period {int(bad['period'])}; dropped")
        daily, overnight, incomplete = daily_load(starts[own_day], halfhours["value"].to_numpy()[own_day], tz)
        in_range = lambda d: d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))]  # noqa: E731
        daily, overnight, incomplete = in_range(daily), in_range(overnight), in_range(incomplete)
        cutoff = pd.Timestamp(self.fetched_at.tz_convert(tz).date()) - pd.Timedelta(days=2)  # today and yesterday fill in
        for row in incomplete[incomplete["date"] < cutoff].itertuples():
            self.report_issue(self.config["daily_indicator"], country, "missing_hours", "warning",
                              f"{row.date.date()}: {row.hours:g} of {row.expected:g} hours; day not stored")
        self.log.info("neso: %s: %d days, %d weekday nights, %d incomplete days",
                      country, len(daily), len(overnight), len(incomplete))
        return pd.concat([
            self.frame(self.config["daily_indicator"], daily.assign(country=country)),
            self.frame(self.config["overnight_indicator"], overnight.assign(country=country)),
        ], ignore_index=True)

    # ------------------------------------------------------------- portal

    def _package(self, dataset: str) -> dict:
        url = f"{self.config['base_url']}/api/3/action/package_show"
        body = self.get(url, {"id": dataset}, raw_name=f"package_{dataset}").json()
        if not body.get("success"):
            raise FetchError(f"neso: package_show {dataset} failed: {str(body.get('error'))[:200]}")
        return body["result"]

    def _yearly_files(self) -> dict[int, str]:
        """Download URL of each year's Historic Demand Data CSV."""
        found = {}
        for res in self._package(self.config["historic_dataset"])["resources"]:
            match = YEAR_RE.match(str(res.get("name", "")).strip())
            if match and str(res.get("format", "")).upper() == "CSV":
                found[int(match.group(1))] = res["url"]
        if not found:
            raise FetchError("neso: no 'Historic Demand Data <year>' CSV resources found; has the dataset changed?")
        return found

    def _update_file(self) -> str:
        csvs = [r for r in self._package(self.config["update_dataset"])["resources"]
                if str(r.get("format", "")).upper() == "CSV"]
        if len(csvs) != 1:
            raise FetchError(f"neso: expected one CSV in {self.config['update_dataset']}, found {len(csvs)}")
        return csvs[0]["url"]

    def _download(self, url: str, label: str) -> pd.DataFrame:
        wait = self.download_interval_s - (time.monotonic() - self._last_download)
        if wait > 0:
            time.sleep(wait)
        self._last_download = time.monotonic()
        response = self.get(url)
        self.save_raw(label, response.content, "csv", compress=True)
        data = parse_demand_csv(response.content.decode("utf-8-sig"))
        self.log.info("neso: %s: %d half hours", label, len(data))
        return data

