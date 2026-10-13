"""The dashboard's one shared, cached data loader.

Every page reads data through this module. Observations come from
`store.latest()` (DuckDB queries over the lake's Parquet files, newest vintage,
`as_of=None` as PLAN §4.1 says for the dashboard; the Country page's replay
reads with `as_of`, D43); supporting tables from `store.read_meta()`. Nothing
here writes. Results are cached with `st.cache_data` for an hour; the
sidebar's "Reload data" button clears them.

The cached functions take the data folder as an argument, so a cache entry
never outlives a change of `settings.DATA_DIR` (tests use a temporary one).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import charts
from tracker import precompute, quality, settings, status, store
from tracker.registry import Registry, load_registry
from tracker.signals import chain as chains
from tracker.signals.moving_holidays import holiday_windows as _holiday_windows

TTL_SECONDS = 3600
OBSERVATION_COLUMNS = ["date", "country", "indicator", "value"]


def _data_dir() -> str:
    return str(settings.DATA_DIR)


@st.cache_resource
def registry() -> Registry:
    """The validated indicator and country registry (config files)."""
    return load_registry()


@st.cache_data(ttl=TTL_SECONDS, show_spinner="Loading data...")
def _observations(indicators: tuple[str, ...] | None, countries: tuple[str, ...] | None, data_dir: str) -> pd.DataFrame:
    df = store.latest(indicators=indicators, countries=countries, data_dir=Path(data_dir))
    return df[OBSERVATION_COLUMNS]


def observations(indicators: list[str] | None = None, countries: list[str] | None = None) -> pd.DataFrame:
    """Latest vintage of every stored value: date, country, indicator, value.

    Filter by indicator ids and/or country codes (None means all).
    """
    key_i = tuple(sorted(set(indicators))) if indicators is not None else None
    key_c = tuple(sorted(set(countries))) if countries is not None else None
    return _observations(key_i, key_c, _data_dir())


def end_of_day(day) -> pd.Timestamp:
    """The last instant of a UTC calendar day: "as known on <day>" includes everything published that day."""
    return pd.Timestamp(day).normalize().tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


@st.cache_data(ttl=TTL_SECONDS, max_entries=64, show_spinner="Loading the data as known then...")
def _observations_as_of(countries: tuple[str, ...], as_of: pd.Timestamp, data_dir: str) -> pd.DataFrame:
    df = store.latest(countries=countries, as_of=as_of, data_dir=Path(data_dir))
    return df[OBSERVATION_COLUMNS]


def observations_as_of(countries: list[str], day) -> pd.DataFrame:
    """The data exactly as known at the end of `day` (UTC): `store.latest(as_of=...)`, so
    values published later, and later revisions of earlier values, are left out (D43)."""
    return _observations_as_of(tuple(sorted(set(countries))), end_of_day(day), _data_dir())


@st.cache_data(ttl=TTL_SECONDS)
def _latest_yoy(indicator_id: str, data_dir: str) -> pd.DataFrame:
    ind = registry().indicator(indicator_id)
    breaks = {c: [pd.Timestamp(b.date) for b in bs] for c, bs in ind.breaks.items()}
    return charts.latest_yoy(_observations((indicator_id,), None, data_dir), ind.frequency, ind.unit, breaks)


def latest_yoy(indicator_id: str) -> pd.DataFrame:
    """Latest level and YoY per country of one indicator (see `charts.latest_yoy`), highest YoY first."""
    return _latest_yoy(indicator_id, _data_dir())


@st.cache_data(ttl=TTL_SECONDS)
def _fetch_log(data_dir: str) -> pd.DataFrame:
    return store.read_fetch_log(Path(data_dir))


def fetch_log() -> pd.DataFrame:
    """Every fetch run (one row per source per run), oldest first."""
    return _fetch_log(_data_dir())


@st.cache_data(ttl=TTL_SECONDS)
def _source_status(data_dir: str) -> pd.DataFrame:
    return status.source_status(registry(), data_dir=Path(data_dir))


def source_status() -> pd.DataFrame:
    """One row per configured source: last run, last success, rows, freshness."""
    return _source_status(_data_dir())


@st.cache_data(ttl=TTL_SECONDS)
def _indicator_freshness(data_dir: str) -> pd.DataFrame:
    return status.indicator_freshness(registry(), data_dir=Path(data_dir))


def indicator_freshness() -> pd.DataFrame:
    """One row per registered indicator: latest date, when the next is due, stale flag."""
    return _indicator_freshness(_data_dir())


@st.cache_data(ttl=TTL_SECONDS)
def _quality(data_dir: str) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    table = store.read_quality_issues(Path(data_dir))
    return quality.open_issues(table), quality.last_run_time(table)


def open_quality_issues() -> pd.DataFrame:
    """Issues found by the latest quality run plus recent fetcher-reported ones."""
    return _quality(_data_dir())[0]


def last_quality_run() -> pd.Timestamp | None:
    """When `python -m tracker quality` last ran (UTC), or None."""
    return _quality(_data_dir())[1]


def indicators_with_data() -> list[str]:
    """Ids of registered indicators with at least one stored value, in registry order."""
    fresh = indicator_freshness()
    return fresh.loc[fresh["last_date"].notna(), "indicator"].tolist()


def breaks(indicator_id: str, country: str) -> list[tuple[pd.Timestamp, str]]:
    """Known structural breaks of one series as (date, reason), from indicators.yaml (PLAN D41)."""
    return [(pd.Timestamp(b.date), b.reason) for b in registry().breaks_for(indicator_id, country)]


def break_dates(indicator_id: str, country: str) -> list[pd.Timestamp]:
    """Dates of the known structural breaks of one series."""
    return [date for date, _ in breaks(indicator_id, country)]


def place_name(code: str) -> str:
    """Display name of a country, chokepoint or `WLD`."""
    reg = registry()
    if code in reg.countries:
        return reg.countries[code].name
    if code in reg.chokepoints:
        return reg.chokepoints[code].name
    return "World" if code == "WLD" else code


def last_backup() -> str | None:
    """The time written by scripts/backup.ps1 after its last backup, or None."""
    marker = settings.LOGS_DIR / "last_backup.txt"
    return marker.read_text(encoding="utf-8-sig").strip() if marker.exists() else None


@st.cache_data(ttl=TTL_SECONDS)
def holiday_windows(iso3: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Moving-holiday windows (holiday, start, end) of a watchlist country."""
    ref = registry().countries.get(iso3)
    if ref is None:
        return _holiday_windows("US", [], start_year, end_year)
    return _holiday_windows(ref.holiday_calendar, ref.moving_holidays, start_year, end_year)


@st.cache_resource(ttl=TTL_SECONDS, show_spinner=False)
def _stl_frames(data_dir: str) -> dict[tuple[str, str], list[pd.DataFrame]]:
    # A shared resource, not cache_data: ~850k rows would be copied on every read.
    return precompute.stl_frames(store.read_signal_table(precompute.STL_TABLE, Path(data_dir)))


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def _signal_table(name: str, data_dir: str) -> pd.DataFrame:
    return store.read_signal_table(name, Path(data_dir))


def signals() -> pd.DataFrame:
    """The latest `python -m tracker signals` output (data/signals/signals.parquet; empty before its first run)."""
    return _signal_table("signals", _data_dir())


def signal_flags() -> pd.DataFrame:
    """The divergence flags written with it (data/signals/flags.parquet)."""
    return _signal_table("flags", _data_dir())


DEGREE_DAYS = ("openmeteo_hdd18", "openmeteo_cdd22")


@st.cache_data(ttl=TTL_SECONDS, max_entries=3000, show_spinner=False)
def _chain_result(chain_text: str, indicator: str, country: str, values: pd.Series,
                  data_dir: str, as_of: pd.Timestamp | None = None) -> chains.ChainResult | str:
    chain = chains.parse(chain_text)
    keys = {step.key for step in chain.steps}
    extra: dict[str, object] = {}
    if "stl" in keys:  # the daily run's precomputed STL (tracker.precompute), used when it fits
        extra["stl"] = _stl_frames(data_dir).get((indicator, country), [])
    if "wx" in keys:  # degree days for S18's weather model, as known on the replay day if there is one
        dd = _observations(DEGREE_DAYS, (country,), data_dir) if as_of is None else             _observations_as_of((country,), as_of, data_dir)
        dd = dd[dd["indicator"].isin(DEGREE_DAYS)]
        for dd_id, part in dd.groupby("indicator"):
            extra[str(dd_id)] = pd.Series(part["value"].to_numpy(), index=pd.DatetimeIndex(part["date"])).sort_index()
    ctx = chains.series_context(registry(), indicator, country, values, extra)
    try:
        return chains.apply(chain, values, ctx, convert=True)  # 28d runs as 1m on monthly data (PLAN D37)
    except chains.ChainError as exc:
        return str(exc)


def chain_result(chain_text: str, indicator: str, country: str, values: pd.Series,
                 replay_day=None) -> chains.ChainResult | str:
    """A transform chain applied to one series, or the reason it can't be (a message).

    Lengths are converted to the series' own unit first (28d -> 1m on monthly
    data; the result's `conversions` say which). Cached per chain string and series. A chain with `stl` gets the daily
    run's precomputed decomposition (data/signals/stl.parquet) when it was
    fitted on the same data; otherwise STL is fitted here (about a second for
    a daily series). Weather adjustment (`wx`) gets the country's degree days,
    as known at the end of `replay_day` when the Country page replays a past day (D43).
    """
    as_of = end_of_day(replay_day) if replay_day is not None else None
    return _chain_result(chain_text, indicator, country, values, _data_dir(), as_of)


@st.cache_data(ttl=TTL_SECONDS)
def _hypotheses_tested(data_dir: str) -> pd.DataFrame:
    return store.read_hypotheses_tested(Path(data_dir))


def hypotheses_tested() -> pd.DataFrame:
    """Every logged backtest run (data/meta/hypotheses_tested), failed and exploratory ones included."""
    return _hypotheses_tested(_data_dir())


@st.cache_data(ttl=TTL_SECONDS)
def _reports(reports_dir: str) -> dict[str, str]:
    folder = Path(reports_dir)
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))} if folder.is_dir() else {}


def reports() -> dict[str, str]:
    """The text of every report in reports/ (not diagnostics/), by file name without .md."""
    return _reports(str(settings.REPORTS_DIR))


def clear() -> None:
    """Drop every cached result, so the next page run re-reads the store."""
    st.cache_data.clear()
    _stl_frames.clear()
