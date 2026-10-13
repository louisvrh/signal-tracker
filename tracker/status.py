"""Source status and data freshness for `python -m tracker status`.

An indicator is **stale** when the period after its latest stored period
should have been published by now (its period end + publication lag + a
grace allowance) but isn't in the store. A retired indicator (no longer
fetched, PLAN D56) is never stale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tracker import settings, store
from tracker.fetchers.base import period_end
from tracker.registry import Registry
from tracker.runner import last_success

GRACE_DAYS = {"D": 3, "W": 3, "M": 7, "Q": 14}


def expected_next_by(last_date: pd.Timestamp, frequency: str, lag_days: int) -> pd.Timestamp:
    """When the period after `last_date` should be available (UTC)."""
    next_end = period_end(period_end(pd.Series([pd.Timestamp(last_date)]), frequency), frequency).iloc[0]
    return (next_end + pd.Timedelta(days=lag_days + GRACE_DAYS[frequency])).tz_localize("UTC")


def indicator_freshness(
    registry: Registry,
    data_dir: Path | None = None,
    now: pd.Timestamp | None = None,
    summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per registered indicator: latest date, when the next is due, stale flag
    (never for a retired indicator, which has no next period due)."""
    now = now or pd.Timestamp.now(tz="UTC")
    if summary is None:
        summary = store.source_summary(data_dir)
    summary = summary.set_index("indicator")
    rows = []
    for ind in registry.indicators.values():
        retired = ind.retired is not None
        if ind.id in summary.index:
            last = pd.Timestamp(summary.loc[ind.id, "last_date"])
            due = None if retired else expected_next_by(last, ind.frequency, ind.publication_lag_days)
            rows.append({
                "source": ind.source, "indicator": ind.id, "frequency": ind.frequency,
                "last_date": last.date(), "next_due": due, "stale": due is not None and bool(now > due),
                "days_late": None if due is None else max(0, (now - due).days), "retired": retired,
            })
        else:
            rows.append({
                "source": ind.source, "indicator": ind.id, "frequency": ind.frequency,
                "last_date": None, "next_due": None, "stale": not retired, "days_late": None, "retired": retired,
            })
    return pd.DataFrame(rows)


def source_status(
    registry: Registry,
    sources_config: dict[str, dict[str, Any]] | None = None,
    data_dir: Path | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per configured source: last run, last success, rows, freshness."""
    sources_config = sources_config if sources_config is not None else settings.load_sources()
    fetch_log = store.read_fetch_log(data_dir)
    summary = store.source_summary(data_dir)
    fresh = indicator_freshness(registry, data_dir, now, summary)
    rows = []
    for name, cfg in sources_config.items():
        log_rows = fetch_log[fetch_log["source"] == name] if not fetch_log.empty else fetch_log
        last_run = log_rows.iloc[-1] if not log_rows.empty else None
        ok_time = last_success(fetch_log, name)
        ok_rows = log_rows[log_rows["status"] == "success"] if not log_rows.empty else log_rows
        ind = fresh[fresh["source"] == name]
        src_summary = summary[summary["source"] == name]
        n_stale = int(ind["stale"].sum())
        if src_summary.empty:
            state = "never run" if last_run is None else "no data"
        elif n_stale:
            state = f"STALE {n_stale}/{len(ind)}"
        else:
            state = "fresh"
        rows.append({
            "source": name,
            "enabled": bool(cfg.get("enabled", False)),
            "last_run": None if last_run is None else str(last_run["status"]),
            "last_success": ok_time,
            "rows_written_last": int(ok_rows.iloc[-1]["rows_written"]) if not ok_rows.empty else None,
            "rows_in_lake": int(src_summary["rows"].sum()) if not src_summary.empty else 0,
            "latest_data": pd.Timestamp(src_summary["last_date"].max()).date() if not src_summary.empty else None,
            "freshness": state,
            "last_error": None if last_run is None or last_run["status"] == "success" else last_run["error"],
        })
    return pd.DataFrame(rows)
