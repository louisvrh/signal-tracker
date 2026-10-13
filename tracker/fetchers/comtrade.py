"""UN Comtrade fetcher, using the official `comtradeapicall` package.

* Monthly total goods exports and imports (HS TOTAL, partner = World) for the
  watchlist reporters, 12 months and all reporters per call.
* Taiwan does not report to Comtrade. Its figures are built as **mirror data**:
  partners' imports from / exports to "Other Asia, nes" (code 490), summed
  over partners, and stored under separate `comtrade_mirror_*` indicators so
  they are never mistaken for reported data.
* China publishes January and February together in some years. If a year
  has February but no January, the combined value is split evenly across the
  two months (the Jan+Feb sum stays right) and a quality issue is logged.

`comtradeapicall` prints errors and returns None instead of raising, and
doesn't expose the raw response. We capture its output to detect errors
(retrying rate limits) and save the returned records as the raw copy.
"""

from __future__ import annotations

import contextlib
import io
import time
from datetime import date
from typing import Any

import pandas as pd
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from tracker import settings, store
from tracker.fetchers.base import BaseFetcher, FetchError, RetryableHTTPError

MAX_RECORDS = 100_000  # free-tier maximum per call
RETRYABLE_MARKERS = ("429", "rate limit", "request error", "500", "502", "503", "504", "timeout")
DATA_COLUMNS = ["period", "reporterCode", "flowCode", "partnerCode", "primaryValue"]


def month_periods(start: date, end: date) -> list[str]:
    """YYYYMM strings for every month from start to end (inclusive)."""
    months = pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M")
    return [p.strftime("%Y%m") for p in months]


def parse_totals(df: pd.DataFrame, code_to_iso: dict[int, str]) -> pd.DataFrame:
    """Reporter totals -> DataFrame(date, country, flow, value) for X and M."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "country", "flow", "value"])
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing:
        raise FetchError(f"comtrade: response lacks columns {missing}")
    df = df[(df["partnerCode"] == 0) & df["flowCode"].isin(["X", "M"])]
    df = df[df["reporterCode"].isin(code_to_iso)]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["period"].astype(str), format="%Y%m"),
        "country": df["reporterCode"].map(code_to_iso),
        "flow": df["flowCode"],
        "value": pd.to_numeric(df["primaryValue"], errors="coerce"),
    }).dropna(subset=["value"])
    return out.drop_duplicates(["date", "country", "flow"], keep="first").reset_index(drop=True)


def mirror_totals(df: pd.DataFrame, exclude_reporters: set[int]) -> pd.DataFrame:
    """Partners' trade with code 490 -> Taiwan's mirror exports and imports.

    Partners' imports from 490 are Taiwan's exports (valued CIF, so a little
    higher than Taiwan's own FOB figure); partners' exports to 490 are its
    imports. Returns DataFrame(date, flow, value, reporters) with flow X/M
    from Taiwan's point of view.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "flow", "value", "reporters"])
    df = df[df["flowCode"].isin(["X", "M"]) & ~df["reporterCode"].isin(exclude_reporters)]
    df = df.drop_duplicates(["period", "reporterCode", "flowCode"], keep="first")
    grouped = df.groupby(["period", "flowCode"]).agg(
        value=("primaryValue", "sum"), reporters=("reporterCode", "nunique")
    ).reset_index()
    return pd.DataFrame({
        "date": pd.to_datetime(grouped["period"].astype(str), format="%Y%m"),
        "flow": grouped["flowCode"].map({"M": "X", "X": "M"}),  # partner imports = Taiwan exports
        "value": grouped["value"].astype(float),
        "reporters": grouped["reporters"].astype(float),
    })


def split_combined_jan_feb(
    totals: pd.DataFrame, countries: list[str], requested: set[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """For the given countries, where a year's February exists but January is
    missing (and January was requested), treat February as the Jan+Feb total
    and split it evenly. Returns (adjusted totals, notes)."""
    notes: list[dict[str, Any]] = []
    extra = []
    for (country, flow), part in totals[totals["country"].isin(countries)].groupby(["country", "flow"]):
        months = set(part["date"])
        for feb in [d for d in months if d.month == 2]:
            jan = feb.replace(month=1)
            if jan in months or jan.strftime("%Y%m") not in requested:
                continue
            idx = part.index[part["date"] == feb][0]
            half = totals.at[idx, "value"] / 2
            totals.at[idx, "value"] = half
            extra.append({"date": jan, "country": country, "flow": flow, "value": half})
            notes.append({"country": country, "flow": flow, "year": feb.year})
    if extra:
        totals = pd.concat([totals, pd.DataFrame(extra)], ignore_index=True)
    return totals, notes


class ComtradeFetcher(BaseFetcher):
    """Monthly goods trade totals, plus Taiwan mirror data."""

    source = "comtrade"
    min_interval_s = 2.0  # stay well under the per-second limit

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        key = self.api_key()
        periods = month_periods(start, end)
        size = int(self.config.get("periods_per_call", 12))
        # Newest chunks first, so a call budget never starves recent data.
        chunks = [periods[i:i + size] for i in range(0, len(periods), size)][::-1]
        budget = int(self.config.get("max_calls_per_run", 60))
        calls_needed = len(chunks) * (1 + bool(self.config.get("mirror")))
        if calls_needed > budget:
            self.partial = True
            self.log.warning("comtrade: %d calls needed, budget %d; fetching newest months first", calls_needed, budget)

        totals_cfg = self.config["totals"]
        country_list = self.registry.countries_for(totals_cfg["exports_indicator"])
        code_to_iso = {self.registry.countries[c].comtrade_reporter: c for c in country_list}
        mirror_cfg = self.config.get("mirror")
        exclude = self._group_reporters() if mirror_cfg else set()

        frames, calls = [], 0
        for chunk in chunks:
            if calls >= budget:
                break
            raw = self._call(key, f"totals_{chunk[0]}_{chunk[-1]}", period=",".join(chunk),
                             reporterCode=",".join(str(c) for c in sorted(code_to_iso)),
                             partnerCode="0", flowCode="X,M")
            calls += 1
            totals = parse_totals(raw, code_to_iso)
            totals, notes = split_combined_jan_feb(totals, self.config.get("combined_jan_feb", []), set(chunk))
            for n in notes:
                self.report_issue(totals_cfg["exports_indicator" if n["flow"] == "X" else "imports_indicator"],
                                  n["country"], "combined_jan_feb", "info",
                                  f"{n['year']}: January missing; February treated as Jan+Feb and split evenly")
            frames.append(self._frames(totals, totals_cfg))

            if mirror_cfg and calls < budget:
                raw = self._call(key, f"mirror_{chunk[0]}_{chunk[-1]}", period=",".join(chunk),
                                 reporterCode=None, partnerCode=str(mirror_cfg["partner_code"]), flowCode="X,M")
                calls += 1
                frames.append(self._mirror_frames(mirror_totals(raw, exclude), mirror_cfg))
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    # ----------------------------------------------------------------- calls

    def _call(self, key: str, label: str, **params: Any) -> pd.DataFrame:
        """One getFinalData call with rate limiting, retries and a raw copy."""
        import comtradeapicall

        def attempt() -> pd.DataFrame:
            wait = self.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            self.requests_made += 1
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                df = comtradeapicall.getFinalData(
                    key, typeCode="C", freqCode="M", clCode="HS", cmdCode="TOTAL",
                    partner2Code="0", customsCode="C00", motCode="0", maxRecords=MAX_RECORDS, **params,
                )
            message = settings.redact(printed.getvalue().strip())
            if df is None:
                if not message or any(m in message.lower() for m in RETRYABLE_MARKERS):
                    raise RetryableHTTPError(f"comtrade: {message or 'no response'}")
                raise FetchError(f"comtrade: {message[:300]}")
            return df

        retrying = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=self.backoff_min_s, max=self.backoff_max_s),
            retry=retry_if_exception_type(RetryableHTTPError),
            reraise=True,
        )
        df = retrying(attempt)
        self.save_raw(label, df.to_json(orient="records"), "json")
        if len(df) >= MAX_RECORDS:
            raise FetchError(f"comtrade: {label} hit the {MAX_RECORDS}-record limit; use fewer periods per call")
        self.log.info("comtrade: %s: %d records", label, len(df))
        return df

    def _group_reporters(self) -> set[int]:
        """Reporter codes that are country groups (e.g. EU), so the mirror sum
        never double counts. Falls back to a fixed list if the reference fails."""
        import comtradeapicall

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ref = comtradeapicall.getReference("reporter")
            groups = set(ref.loc[ref["isGroup"].astype(bool), "id"].astype(int))
        except Exception as exc:  # reference is a convenience; don't fail the fetch
            self.log.warning("comtrade: reporter reference unavailable (%s); using fallback", exc)
            groups = {97}
        return groups | {0, int(self.config.get("mirror", {}).get("partner_code", 490))}

    # --------------------------------------------------------------- framing

    def _frames(self, totals: pd.DataFrame, cfg: dict[str, str]) -> pd.DataFrame:
        lags = self.config.get("publication_lag_days_by_country")
        parts = [
            self.frame(cfg["exports_indicator"], totals[totals["flow"] == "X"], lag_by_country=lags),
            self.frame(cfg["imports_indicator"], totals[totals["flow"] == "M"], lag_by_country=lags),
        ]
        return pd.concat(parts, ignore_index=True)

    def _mirror_frames(self, mirror: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
        if mirror.empty:
            return store.empty_frame()
        mirror = mirror.assign(country=cfg["country"])
        exports = mirror[mirror["flow"] == "X"]
        parts = [
            self.frame(cfg["exports_indicator"], exports),
            self.frame(cfg["imports_indicator"], mirror[mirror["flow"] == "M"]),
            self.frame(cfg["reporters_indicator"], exports.assign(value=exports["reporters"])),
        ]
        return pd.concat(parts, ignore_index=True)
