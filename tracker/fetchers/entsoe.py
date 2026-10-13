"""ENTSO-E Transparency Platform fetcher: actual total load per country.

Uses entsoe-py's raw client for Actual Total Load [6.1.A] (documentType A65,
processType A16) and its XML parser. Each country's load becomes:

* `entsoe_load`: total load over each local calendar day (MWh);
* `entsoe_overnight_load`: mean load over 00:00-05:00 local time (MW),
  Monday to Friday.

The API returns UTC timestamps; they are converted to the country's time zone
(countries.yaml) before days are built (`tracker.fetchers.load`). Resolutions
differ by country and year (60, 30 or 15 minutes; most TSOs moved to 15
minutes in 2024-25), so every value is weighted by the time it covers. Days
with missing intervals are not stored and are reported as quality issues.

Areas: a country's area is its countries.yaml `entsoe_code`, unless
sources.yaml `areas` overrides it, either with another code or with a list of
`{area, from, to}` pieces for a country whose published area changed (bidding
zones split or merged). Country codes are preferred to bidding zones: `DE`
(Germany) runs from 2015, while the zones `DE_AT_LU` (to 2018-09, includes
Austria) and `DE_LU` (from 2018-10, includes Luxembourg) are not Germany alone.

Limits: at most one year per request, and 400 requests per minute per IP and
token (beyond that, a 10-minute ban). Requests are split at calendar years
and spaced `min_interval_s` apart. A year of 15-minute data is ~4 MB of XML
(~0.2 MB gzipped), so raw responses are saved gzipped.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests
from entsoe import EntsoeRawClient
from entsoe.exceptions import NoMatchingDataError
from entsoe.parsers import parse_loads
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from tracker import settings, store
from tracker.fetchers.base import BaseFetcher, FetchError, RetryableHTTPError, _is_retryable
from tracker.fetchers.load import daily_load


@dataclass(frozen=True)
class AreaPeriod:
    """An ENTSO-E area code used for a country from `start` (inclusive) to `end` (exclusive)."""

    area: str
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


def area_periods(spec: str | list[dict[str, Any]]) -> list[AreaPeriod]:
    """Parse an `areas` entry: one code, or a list of {area, from, to} (dates, `to` exclusive)."""
    if isinstance(spec, str):
        return [AreaPeriod(spec)]
    periods = []
    for item in spec:
        start = pd.Timestamp(item["from"], tz="UTC") if item.get("from") else None
        end = pd.Timestamp(item["to"], tz="UTC") if item.get("to") else None
        if start is not None and end is not None and end <= start:
            raise FetchError(f"entsoe: area {item['area']}: 'to' ({item['to']}) must be after 'from' ({item['from']})")
        periods.append(AreaPeriod(str(item["area"]), start, end))
    return periods


def request_windows(
    periods: list[AreaPeriod], start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """(area, start, end) requests covering [start, end): each area's share of
    the range, split at calendar-year boundaries (UTC) so no request exceeds a year."""
    windows = []
    for p in periods:
        s = max(start, p.start) if p.start is not None else start
        e = min(end, p.end) if p.end is not None else end
        while s < e:
            year_end = pd.Timestamp(year=s.year + 1, month=1, day=1, tz="UTC")
            windows.append((p.area, s, min(e, year_end)))
            s = min(e, year_end)
    return windows


def parse_load_xml(text: str) -> pd.Series:
    """Actual Total Load XML -> Series of MW indexed by interval start (UTC)."""
    with warnings.catch_warnings():
        # entsoe-py parses the XML with BeautifulSoup's HTML parser, which warns every time.
        warnings.filterwarnings("ignore", message="It looks like you're using an HTML parser")
        df = parse_loads(text, process_type="A16")
    if df.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    s = pd.to_numeric(df["Actual Load"], errors="coerce")
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    return s


class EntsoeFetcher(BaseFetcher):
    """Daily and weekday-overnight load for European countries."""

    source = "entsoe"
    min_interval_s = 1.0  # at most 60 requests/minute, far below the 400 that trigger a ban
    timeout_s = 180.0     # a year of 15-minute data can take 10-15 s to arrive

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        # entsoe-py's own retry is switched off (one attempt, no delay): `_query` retries.
        client = EntsoeRawClient(api_key=self.api_key(), session=self.session, retry_count=1, retry_delay=0,
                                 timeout=int(self.timeout_s))
        frames = []
        for country in self.registry.countries_for(self.config["daily_indicator"]):
            frames.append(self._fetch_country(client, country, start, end))
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    def areas_for(self, country: str) -> list[AreaPeriod]:
        """The country's area periods: sources.yaml `areas`, else countries.yaml `entsoe_code`."""
        spec = (self.config.get("areas") or {}).get(country) or self.registry.countries[country].entsoe_code
        if not spec:
            raise FetchError(f"entsoe: no area for {country} (set entsoe_code in countries.yaml or areas in sources.yaml)")
        return area_periods(spec)

    def _fetch_country(self, client: EntsoeRawClient, country: str, start: date, end: date) -> pd.DataFrame:
        tz = self.registry.countries[country].timezone
        # Pad by a day before and after so the first and last local days are complete.
        req_start = pd.Timestamp(start - timedelta(days=1), tz="UTC")
        req_end = min(pd.Timestamp(end + timedelta(days=2), tz="UTC"), self.fetched_at.ceil("h"))
        pieces = []
        for area, s, e in request_windows(self.areas_for(country), req_start, req_end):
            text = self._query(client, area, s, e)
            if text is not None:
                pieces.append(parse_load_xml(text))
        if not pieces:
            self.log.warning("entsoe: no load data for %s between %s and %s", country, start, end)
            return store.empty_frame()
        series = pd.concat(pieces)
        daily, overnight, incomplete = daily_load(series.index, series.to_numpy(), tz)
        in_range = lambda d: d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))]  # noqa: E731
        daily, overnight, incomplete = in_range(daily), in_range(overnight), in_range(incomplete)
        # Today and yesterday may still be filling in; only older gaps are issues.
        cutoff = pd.Timestamp(self.fetched_at.tz_convert(tz).date()) - pd.Timedelta(days=2)
        for row in incomplete[incomplete["date"] < cutoff].itertuples():
            self.report_issue(self.config["daily_indicator"], country, "missing_hours", "warning",
                              f"{row.date.date()}: {row.hours:g} of {row.expected:g} hours; day not stored")
        self.log.info("entsoe: %s: %d days, %d weekday nights, %d incomplete days",
                      country, len(daily), len(overnight), len(incomplete))
        return pd.concat([
            self.frame(self.config["daily_indicator"], daily.assign(country=country)),
            self.frame(self.config["overnight_indicator"], overnight.assign(country=country)),
        ], ignore_index=True)

    def _query(self, client: EntsoeRawClient, area: str, start: pd.Timestamp, end: pd.Timestamp) -> str | None:
        """One load request with retries on 429/5xx and connection errors;
        saves the raw XML. Returns None when the API has no data for the window."""
        retrying = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=self.backoff_min_s, max=self.backoff_max_s),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        return retrying(self._query_once, client, area, start, end)

    def _query_once(self, client: EntsoeRawClient, area: str, start: pd.Timestamp, end: pd.Timestamp) -> str | None:
        self.wait_turn()
        label = f"load_{area}_{start:%Y%m%d}_{end:%Y%m%d}"
        try:
            text = client.query_load(area, start=start, end=end)
        except NoMatchingDataError:
            self.log.info("entsoe: no data for %s %s to %s", area, start.date(), end.date())
            return None
        except (requests.ConnectionError, requests.Timeout) as exc:
            # requests' messages hold the request URL, security token included:
            # new messages are built from the status and body instead, unchained.
            raise RetryableHTTPError(f"entsoe: {type(exc).__name__} for {label}") from None
        except requests.HTTPError as exc:
            response = exc.response
            status = response.status_code if response is not None else None
            if status is not None and (status == 429 or status >= 500):
                self.log.warning("entsoe: HTTP %s for %s, will retry", status, label)
                raise RetryableHTTPError(f"entsoe: HTTP {status} for {label}") from None
            body = settings.redact(response.text[:300]) if response is not None else ""
            raise FetchError(f"entsoe: HTTP {status} for {label}: {body}") from None
        self.save_raw(label, text, "xml", compress=True)
        return text
