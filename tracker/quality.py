"""Data quality checks: freshness, missing countries, gaps inside series,
implausible jumps, duplicates and unit changes.

The checks are pure functions over store data (DataFrames in, issue
DataFrames out). `run_checks()` reads the store and runs them all, and
`record()` appends the findings to `data/meta/quality_issues`. Values are
only flagged; nothing is ever dropped or changed.

Two kinds of finding:

* **State checks** (freshness, missing_country, unexpected_data, gap,
  duplicate, unit_change) look at the whole store on every run. An issue is
  open while the latest run still reports it.
* **Jumps** are reported once, by the first run that sees the new or revised
  value. A store that has never been checked gets its full history checked.

Every run also writes one `quality_run` row, so the time of the last check is
known even when nothing is wrong. Fetchers report their own issues (e.g.
incomplete days) straight to the same table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tracker import settings, store
from tracker.registry import Indicator, Registry, load_registry
from tracker.status import expected_next_by

ISSUE_COLUMNS = ["source", "indicator", "country", "check", "severity", "detail"]
STATE_CHECKS = ("freshness", "missing_country", "unexpected_data", "gap", "duplicate", "unit_change")
EVENT_CHECKS = ("jump",)
QUALITY_CHECKS = STATE_CHECKS + EVENT_CHECKS
RUN_CHECK = "quality_run"
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

# Jumps: robust z of the period-on-period change (log change for series that
# are always positive), computed per series over its whole history.
JUMP_Z = 10.0
JUMP_MIN_RATIO = 1.5    # positive series must also rise by 50% or fall by a third
JUMP_MIN_POINTS = 30    # shorter series are not judged
JUMP_RANGE_FLOOR = 0.1  # scale >= 10% of the series' 5-95% range (zero-heavy series)
INTERVAL_UNITS = {"degC"}  # no true zero, so ratios mean nothing: use differences
FETCHER_ISSUE_DAYS = 7  # before any quality run, show fetcher issues this recent


def _issue(source: str, indicator: str | None, country: str | None, check: str, severity: str, detail: str) -> dict[str, Any]:
    return {"source": source, "indicator": indicator, "country": country,
            "check": check, "severity": severity, "detail": detail}


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS)


def enabled_sources(sources_config: dict[str, dict[str, Any]]) -> set[str]:
    """Names of sources with `enabled: true`."""
    return {name for name, cfg in sources_config.items() if cfg.get("enabled", False)}


def publication_lag(indicator: Indicator, country: str, sources_config: dict[str, dict[str, Any]]) -> int:
    """Publication lag in days for one series: the source's
    `publication_lag_days_by_country` where given, else the indicator's."""
    by_country = (sources_config.get(indicator.source) or {}).get("publication_lag_days_by_country") or {}
    return int(by_country.get(country, indicator.publication_lag_days))


def series_last_dates(latest: pd.DataFrame) -> pd.DataFrame:
    """(source, indicator, country, last_date) for every stored series."""
    if latest.empty:
        return pd.DataFrame(columns=["source", "indicator", "country", "last_date"])
    out = latest.groupby(["source", "indicator", "country"], as_index=False)["date"].max()
    return out.rename(columns={"date": "last_date"})


# ------------------------------------------------------------------ checks


def check_freshness(
    last_dates: pd.DataFrame, registry: Registry, sources_config: dict[str, dict[str, Any]], now: pd.Timestamp
) -> pd.DataFrame:
    """Series whose next period should have been published by `now`
    (next period end + publication lag + grace, as in `status`). Retired
    indicators are skipped (PLAN D56)."""
    enabled = enabled_sources(sources_config)
    rows = []
    for r in last_dates.itertuples(index=False):
        ind = registry.indicators.get(r.indicator)
        if ind is None or ind.retired or ind.source not in enabled or r.country not in registry.countries_for(ind):
            continue
        last = pd.Timestamp(r.last_date)
        due = expected_next_by(last, ind.frequency, publication_lag(ind, r.country, sources_config))
        if now > due:
            rows.append(_issue(ind.source, ind.id, r.country, "freshness", "warning",
                               f"latest {last:%Y-%m-%d}; next period was due by {due:%Y-%m-%d} "
                               f"({(now - due).days} days late)"))
    return _frame(rows)


def check_missing_countries(
    last_dates: pd.DataFrame, registry: Registry, sources_config: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """Countries an indicator should cover (registry) with no stored data.

    An indicator with no data at all gets one issue instead of one per
    country. Countries in the indicator's `no_data_expected` are skipped; if
    data does appear for one of them, an `unexpected_data` info issue says
    the config list can be trimmed. Retired indicators are skipped (PLAN D56)."""
    enabled = enabled_sources(sources_config)
    have = set(zip(last_dates["indicator"], last_dates["country"]))
    rows = []
    for ind in registry.indicators.values():
        if ind.source not in enabled or ind.retired:
            continue
        expected = [c for c in registry.countries_for(ind) if c not in ind.no_data_expected]
        missing = [c for c in expected if (ind.id, c) not in have]
        if missing and len(missing) == len(expected):
            rows.append(_issue(ind.source, ind.id, None, "missing_country", "warning",
                               f"no data stored for any of its {len(expected)} countries"))
        else:
            rows.extend(_issue(ind.source, ind.id, c, "missing_country", "warning", "no data stored")
                        for c in missing)
        rows.extend(_issue(ind.source, ind.id, c, "unexpected_data", "info",
                           "data is stored although the country is in no_data_expected; remove it there")
                    for c in ind.no_data_expected if (ind.id, c) in have)
    return _frame(rows)


def _missing_between(prev: pd.Series, cur: pd.Series, frequency: pd.Series, calendar: pd.Series) -> pd.Series:
    """Number of expected periods strictly between consecutive dates."""
    out = pd.Series(np.nan, index=cur.index)
    daily = frequency == "D"
    weekdays = daily & (calendar == "weekdays") & prev.notna()  # a series' first row has no previous date
    every = daily & ~weekdays
    out[every] = (cur[every] - prev[every]).dt.days - 1
    if weekdays.any():
        begin = (prev[weekdays] + pd.Timedelta(days=1)).to_numpy().astype("datetime64[D]")
        out[weekdays] = np.busday_count(begin, cur[weekdays].to_numpy().astype("datetime64[D]"))
    weekly = frequency == "W"
    out[weekly] = (cur[weekly] - prev[weekly]).dt.days // 7 - 1
    months = (cur.dt.year - prev.dt.year) * 12 + (cur.dt.month - prev.dt.month)
    out[frequency == "M"] = months[frequency == "M"] - 1
    out[frequency == "Q"] = months[frequency == "Q"] // 3 - 1
    return out


def _gap_span(prev: pd.Timestamp, cur: pd.Timestamp, frequency: str, calendar: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last missing period of the gap between two consecutive dates."""
    if frequency == "D":
        first, last = prev + pd.Timedelta(days=1), cur - pd.Timedelta(days=1)
        if calendar == "weekdays":
            first, last = pd.offsets.BDay().rollforward(first), pd.offsets.BDay().rollback(last)
    elif frequency == "W":
        first, last = prev + pd.Timedelta(days=7), cur - pd.Timedelta(days=7)
    else:
        step = pd.DateOffset(months=1 if frequency == "M" else 3)
        first, last = prev + step, cur - step
    return first, last


def _span_text(first: pd.Timestamp, last: pd.Timestamp) -> str:
    return f"{first:%Y-%m-%d}" if first == last else f"{first:%Y-%m-%d} to {last:%Y-%m-%d}"


def check_gaps(latest: pd.DataFrame, registry: Registry, max_listed: int = 3) -> pd.DataFrame:
    """Missing periods inside each series, between its first and last date.

    Expected periods follow the indicator's frequency and `calendar` (all
    days, or weekdays only for trading-day series). Runs of up to its
    `gap_tolerance` missing periods (e.g. exchange holidays) are normal, and
    gaps inside its `known_gaps` (the source's own, never to be filled) are
    not reported. One issue per series, listing the largest gaps."""
    if latest.empty:
        return _frame([])
    known = latest[latest["indicator"].isin(registry.indicators)]
    df = known.sort_values(["indicator", "country", "date"])[["source", "indicator", "country", "date"]]
    df = df.reset_index(drop=True)
    ind = {i.id: i for i in registry.indicators.values()}
    frequency = df["indicator"].map(lambda i: ind[i].frequency)
    calendar = df["indicator"].map(lambda i: ind[i].calendar)
    tolerance = df["indicator"].map(lambda i: ind[i].gap_tolerance)
    prev = df.groupby(["indicator", "country"], sort=False)["date"].shift(1)
    missing = _missing_between(prev, df["date"], frequency, calendar)
    gaps = df.assign(prev=prev, missing=missing, frequency=frequency, calendar=calendar)[missing > tolerance]
    unit = {"D": "days", "W": "weeks", "M": "months", "Q": "quarters"}
    rows = []
    for (source, indicator, country), part in gaps.groupby(["source", "indicator", "country"], sort=True):
        known = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in ind[indicator].known_gaps.get(country, [])]
        spans_all = [_gap_span(r.prev, r.date, r.frequency, r.calendar) for r in part.itertuples()]
        keep = [not any(a <= first and last <= b for a, b in known) for first, last in spans_all]
        part = part.assign(first=[s[0] for s in spans_all], last=[s[1] for s in spans_all])[keep]
        if part.empty:
            continue
        part = part.sort_values("missing", ascending=False, kind="mergesort")
        head = part.iloc[0]
        label = "weekdays" if head["calendar"] == "weekdays" else unit[head["frequency"]]
        spans = [f"{_span_text(r.first, r.last)} ({int(r.missing)})" for r in part.head(max_listed).itertuples()]
        more = f", and {len(part) - max_listed} more" if len(part) > max_listed else ""
        rows.append(_issue(source, indicator, country, "gap", "warning",
                           f"{int(part['missing'].sum())} missing {label} in {len(part)} gap(s): "
                           f"{', '.join(spans)}{more}"))
    return _frame(rows)


def check_jumps(
    latest: pd.DataFrame,
    since: pd.Timestamp | None = None,
    interval_indicators: set[str] | frozenset[str] = frozenset(),
    z_threshold: float = JUMP_Z,
    min_ratio: float = JUMP_MIN_RATIO,
    min_points: int = JUMP_MIN_POINTS,
) -> pd.DataFrame:
    """Implausibly large period-on-period changes within each series.

    The change is the log change for series that are always positive, and
    the plain difference for other series and for `interval_indicators`
    (e.g. temperatures, where ratios mean nothing). It is flagged when its
    robust z exceeds `z_threshold` and, for log changes, the value also moved
    by at least `min_ratio` either way. The z scale is 1.4826 x the MAD of
    the series' non-zero changes, floored at `JUMP_RANGE_FLOOR` x its 5-95%
    range, so series that are often flat or zero (degree days in summer)
    aren't flagged for ordinary moves. With `since`, only
    changes involving a value fetched after `since` are reported. The data
    itself is never changed.
    """
    if latest.empty:
        return _frame([])
    keys = ["indicator", "country"]
    df = latest.sort_values(keys + ["date"])[["source", "indicator", "country", "date", "value", "fetched_at"]]
    df = df.reset_index(drop=True)
    grouped = df.groupby(keys, sort=False)
    positive = (grouped["value"].transform("min") > 0) & ~df["indicator"].isin(interval_indicators)
    x = np.log(df["value"].where(positive)).where(positive, df["value"])
    by_level = x.groupby([df["indicator"], df["country"]], sort=False)
    spread = by_level.transform(lambda s: s.quantile(0.95) - s.quantile(0.05))
    change = by_level.diff()
    by_series = change.groupby([df["indicator"], df["country"]], sort=False)
    median = by_series.transform("median")
    moving = change.where(change != 0)  # flat spells (e.g. zero degree days) would shrink the MAD to 0
    mad = (moving - median).abs().groupby([df["indicator"], df["country"]], sort=False).transform("median")
    points = by_series.transform("count")
    scale = np.maximum(1.4826 * mad, JUMP_RANGE_FLOOR * spread)
    z = (change - median) / scale.where(scale > 0)

    flag = (points >= min_points) & (z.abs() > z_threshold)
    flag &= ~positive | (change.abs() >= np.log(min_ratio))
    prev_value = grouped["value"].shift(1)
    prev_date = grouped["date"].shift(1)
    if since is not None:
        prev_fetched = grouped["fetched_at"].shift(1)
        flag &= (df["fetched_at"] > since) | (prev_fetched > since)

    rows = []
    for i in df.index[flag.fillna(False).astype(bool)]:
        r = df.loc[i]
        move = f"x{r['value'] / prev_value[i]:.3g}" if positive[i] else f"change {change[i]:+.4g}"
        rows.append(_issue(r["source"], r["indicator"], r["country"], "jump", "warning",
                           f"{prev_value[i]:.6g} on {prev_date[i]:%Y-%m-%d} -> {r['value']:.6g} on "
                           f"{r['date']:%Y-%m-%d} ({move}, robust z {z[i]:+.1f})"))
    return _frame(rows)


def check_duplicates(history: pd.DataFrame) -> pd.DataFrame:
    """Stored rows sharing (date, country, indicator, available_at).

    The write rule never creates these, so they point to files copied into
    the lake by hand (e.g. a restore) or two runs writing at once. Identical
    duplicates are a warning; the same key with different values is an error,
    because it's then unclear which value was known."""
    keys = ["indicator", "country", "date", "available_at"]
    dup = history[history.duplicated(keys, keep=False)] if not history.empty else history
    if dup.empty:
        return _frame([])
    per_key = dup.groupby(keys).agg(source=("source", "first"), n=("value", "size"), values=("value", "nunique"))
    per_series = per_key.groupby(["source", "indicator", "country"]).agg(
        keys=("n", "size"), extra=("n", lambda s: int((s - 1).sum())), conflicting=("values", lambda s: int((s > 1).sum()))
    )
    rows = []
    for (source, indicator, country), r in per_series.iterrows():
        severity = "error" if r["conflicting"] else "warning"
        rows.append(_issue(source, indicator, country, "duplicate", severity,
                           f"{r['extra']} duplicate rows over {r['keys']} keys, {r['conflicting']} with conflicting values"))
    return _frame(rows)


def check_units(history: pd.DataFrame, registry: Registry) -> pd.DataFrame:
    """Indicators stored with more than one unit, or with a unit that differs
    from the registry (e.g. an index rebased from 2015=100 to 2021=100)."""
    if history.empty:
        return _frame([])
    first_seen = history.groupby(["source", "indicator", "unit"])["fetched_at"].min().reset_index()
    rows = []
    for (source, ind_id), part in first_seen.groupby(["source", "indicator"]):
        ind = registry.indicators.get(ind_id)
        expected = ind.unit if ind is not None else None
        part = part.sort_values("fetched_at")
        units = part["unit"].tolist()
        if len(units) > 1:
            seen = ", ".join(f"'{u}' from {t:%Y-%m-%d}" for u, t in zip(part["unit"], part["fetched_at"]))
            rows.append(_issue(source, ind_id, None, "unit_change", "error",
                               f"stored with {len(units)} units ({seen}); registry says '{expected}'"))
        elif expected is not None and units[0] != expected:
            rows.append(_issue(source, ind_id, None, "unit_change", "error",
                               f"stored unit '{units[0]}' differs from the registry's '{expected}'"))
    return _frame(rows)


# --------------------------------------------------------------- run + read


def last_run_time(issues_table: pd.DataFrame) -> pd.Timestamp | None:
    """When the latest quality run happened (None if never)."""
    if issues_table.empty:
        return None
    runs = issues_table.loc[issues_table["check"] == RUN_CHECK, "checked_at"]
    return pd.Timestamp(runs.max()).tz_convert("UTC") if not runs.empty else None


def run_checks(
    registry: Registry | None = None,
    sources_config: dict[str, dict[str, Any]] | None = None,
    data_dir: Path | None = None,
    now: pd.Timestamp | None = None,
    jumps_since: pd.Timestamp | None | str = "last_run",
) -> pd.DataFrame:
    """Run every check over the store's enabled sources; returns the issues.

    `jumps_since="last_run"` (the default) reports only jumps involving values
    fetched since the previous quality run (the full history if there was
    none); pass a timestamp, or None for the full history.
    """
    registry = registry or load_registry()
    sources_config = sources_config if sources_config is not None else settings.load_sources()
    now = now or pd.Timestamp.now(tz="UTC")
    if isinstance(jumps_since, str):
        jumps_since = last_run_time(store.read_quality_issues(data_dir))
    enabled = enabled_sources(sources_config)
    latest = store.latest(data_dir=data_dir)
    latest = latest[latest["source"].isin(enabled)]
    history = store.history(data_dir=data_dir)
    history = history[history["source"].isin(enabled)]
    last_dates = series_last_dates(latest)
    interval = {i.id for i in registry.indicators.values() if i.unit in INTERVAL_UNITS}
    parts = [
        check_freshness(last_dates, registry, sources_config, now),
        check_missing_countries(last_dates, registry, sources_config),
        check_jumps(latest, since=jumps_since, interval_indicators=interval),
        check_gaps(latest, registry),
        check_duplicates(history),
        check_units(history, registry),
    ]
    parts = [p for p in parts if not p.empty]
    return sort_issues(pd.concat(parts, ignore_index=True)) if parts else _frame([])


def summary_text(issues: pd.DataFrame) -> str:
    """e.g. '2 errors, 14 warnings, 0 info'."""
    counts = issues["severity"].value_counts() if not issues.empty else pd.Series(dtype=int)
    return (f"{int(counts.get('error', 0))} errors, {int(counts.get('warning', 0))} warnings, "
            f"{int(counts.get('info', 0))} info")


def record(issues: pd.DataFrame, data_dir: Path | None = None, now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Append the issues plus one `quality_run` summary row; returns the check time."""
    now = now or pd.Timestamp.now(tz="UTC")
    marker = _frame([_issue("quality", None, None, RUN_CHECK, "info", summary_text(issues))])
    rows = pd.concat([issues[ISSUE_COLUMNS], marker], ignore_index=True) if not issues.empty else marker
    store.append_quality_issues(rows.assign(checked_at=now), data_dir)
    return now


def open_issues(issues_table: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """The issues to show now: everything the latest quality run found, plus
    fetcher-reported issues since the run before it (or from the last
    `FETCHER_ISSUE_DAYS` days if there is no such run)."""
    if issues_table.empty:
        return _frame([])
    table = issues_table.copy()
    table["checked_at"] = pd.to_datetime(table["checked_at"], utc=True)
    runs = table.loc[table["check"] == RUN_CHECK, "checked_at"]
    now = now or pd.Timestamp.now(tz="UTC")
    if runs.empty:
        latest_run = pd.DataFrame(columns=table.columns)
        since = now - pd.Timedelta(days=FETCHER_ISSUE_DAYS)
    else:
        last = runs.max()
        earlier = runs[runs < last]
        since = earlier.max() if not earlier.empty else last - pd.Timedelta(days=FETCHER_ISSUE_DAYS)
        latest_run = table[(table["checked_at"] == last) & (table["check"] != RUN_CHECK)]
    from_fetchers = table[(table["checked_at"] > since) & ~table["check"].isin(QUALITY_CHECKS + (RUN_CHECK,))]
    out = pd.concat([latest_run, from_fetchers], ignore_index=True)
    return sort_issues(out.drop_duplicates(ISSUE_COLUMNS)[ISSUE_COLUMNS])


def sort_issues(issues: pd.DataFrame) -> pd.DataFrame:
    """Errors first, then warnings, then info; then by check, indicator, country."""
    if issues.empty:
        return issues.reset_index(drop=True)
    rank = issues["severity"].map(SEVERITY_ORDER).fillna(len(SEVERITY_ORDER))
    order = issues.assign(_rank=rank).sort_values(["_rank", "check", "indicator", "country"], na_position="first")
    return order.drop(columns="_rank").reset_index(drop=True)


def format_issues(issues: pd.DataFrame, limit: int | None = None, indent: str = "  ") -> list[str]:
    """One line per issue (severity, check, indicator, country, detail)."""
    shown = issues if limit is None else issues.head(limit)
    lines = []
    for r in shown.itertuples(index=False):
        country = r.country if isinstance(r.country, str) and r.country else "-"
        indicator = r.indicator if isinstance(r.indicator, str) and r.indicator else "-"
        lines.append(f"{indent}{r.severity:<7}  {r.check:<15}  {indicator:<26}  {country:<4}  {r.detail}")
    if limit is not None and len(issues) > limit:
        lines.append(f"{indent}... {len(issues) - limit} more (full list: python -m tracker quality --list)")
    return lines
