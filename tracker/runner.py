"""Runs fetchers, writes their rows to the store and records fetch_log.

Each source runs in isolation: an exception in one source is caught, logged
(with secrets redacted) and recorded as a failed run, and the next source
still runs. Each source resumes from its last successful fetch minus its
`refetch_window_days`, so revisions are re-downloaded and missed days are
caught up automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from tracker import settings, store
from tracker.fetchers import get_fetcher_class
from tracker.registry import Registry, load_registry

log = logging.getLogger(__name__)

DEFAULT_BACKFILL_START = "2015-01-01"
DEFAULT_REFETCH_WINDOW_DAYS = 60


@dataclass
class RunResult:
    """Outcome of fetching one source."""

    source: str
    status: str  # "success", "partial" or "failed"
    rows_fetched: int
    rows_written: int
    start: date | None
    seconds: float = 0.0
    error: str | None = None


def enabled_sources(sources_config: dict[str, dict[str, Any]]) -> list[str]:
    """Names of sources with `enabled: true`, in config order."""
    return [name for name, cfg in sources_config.items() if cfg.get("enabled", False)]


def refetch_window_days(source: str, cfg: dict[str, Any], registry: Registry) -> int:
    """Source-level window from sources.yaml, else the largest of its indicators."""
    if cfg.get("refetch_window_days") is not None:
        return int(cfg["refetch_window_days"])
    windows = [i.refetch_window_days for i in registry.by_source(source)]
    return max(windows) if windows else DEFAULT_REFETCH_WINDOW_DAYS


def last_success(fetch_log: pd.DataFrame, source: str) -> pd.Timestamp | None:
    """Start time of the source's most recent successful run, if any."""
    if fetch_log.empty:
        return None
    ok = fetch_log[(fetch_log["source"] == source) & (fetch_log["status"] == "success")]
    if ok.empty:
        return None
    return pd.Timestamp(ok["started_at"].max())


def resume_start(
    source: str,
    cfg: dict[str, Any],
    registry: Registry,
    fetch_log: pd.DataFrame,
    default_start: date,
    today: date,
) -> date:
    """Where a normal (non-backfill) run should start for this source.

    Sources with `resume_from: backfill_start` always start at their backfill
    start; their fetcher skips what is already stored (used to spread a long
    backfill over many runs).
    """
    backfill_start = pd.Timestamp(cfg.get("backfill_start", default_start)).date()
    if cfg.get("resume_from") == "backfill_start":
        return backfill_start
    previous = last_success(fetch_log, source)
    if previous is None:
        return backfill_start
    start = previous.date() - timedelta(days=refetch_window_days(source, cfg, registry))
    return max(min(start, today), backfill_start)


def run_source(
    source: str,
    start: date,
    end: date | None = None,
    run_id: str | None = None,
    data_dir: Path | None = None,
    fetcher_kwargs: dict | None = None,
    change_tolerance: float | None = None,
) -> RunResult:
    """Fetch one source from `start`, append changes and log the run.

    Never raises for fetch errors: failures are logged, recorded in fetch_log
    (with secrets redacted) and returned as a failed RunResult.
    `change_tolerance` (from sources.yaml) is passed to the store's write rule.
    """
    run_id = run_id or store.new_run_id()
    started = pd.Timestamp.now(tz="UTC")
    fetched = written = 0
    error = None
    try:
        fetcher = get_fetcher_class(source)(**{"data_dir": data_dir, **(fetcher_kwargs or {})})
        df = fetcher.fetch(start, end)
        fetched = len(df)
        written = store.append_changes(df, run_id, data_dir, rel_tol=change_tolerance)
        if fetcher.issues:
            issues = pd.DataFrame(fetcher.issues).assign(checked_at=pd.Timestamp.now(tz="UTC"))
            store.append_quality_issues(issues, data_dir)
        # A partial run (call budget reached) is not a success, so the next
        # run resumes from the previous success / backfill start again.
        status = "partial" if fetcher.partial else "success"
        log.info("%s: %s, %d rows fetched, %d written", source, status, fetched, written)
    except Exception as exc:  # isolate each source
        status = "failed"
        error = settings.redact(f"{type(exc).__name__}: {exc}")[:2000]
        log.error("%s: fetch failed: %s", source, error)
    finished = pd.Timestamp.now(tz="UTC")
    store.append_fetch_log(
        pd.DataFrame([{
            "run_id": run_id, "source": source, "started_at": started, "finished_at": finished,
            "status": status, "rows_fetched": fetched, "rows_written": written, "error": error,
        }]),
        data_dir,
    )
    seconds = (finished - started).total_seconds()
    return RunResult(source, status, fetched, written, start, seconds, error)


def run_all(
    only: str | None = None,
    since: date | None = None,
    backfill: bool = False,
    data_dir: Path | None = None,
    sources_config: dict[str, dict[str, Any]] | None = None,
    registry: Registry | None = None,
    today: date | None = None,
) -> list[RunResult]:
    """Run every enabled source (or just `only`), each in isolation.

    Without `backfill`/`since`, each source resumes from its last successful
    run minus its refetch window. `since` (or `backfill` alone, which uses the
    configured backfill start) re-downloads everything from that date.
    """
    if sources_config is None:
        sources_config = settings.load_sources()
    registry = registry or load_registry()
    today = today or pd.Timestamp.now(tz="UTC").date()
    default_start = pd.Timestamp(
        settings.load_config("sources").get("defaults", {}).get("backfill_start", DEFAULT_BACKFILL_START)
    ).date()

    if only:
        if only not in sources_config:
            raise KeyError(f"source '{only}' is not in config/sources.yaml")
        names = [only]
    else:
        names = enabled_sources(sources_config)

    fetch_log = store.read_fetch_log(data_dir)
    run_id = store.new_run_id()
    results = []
    for name in names:
        cfg = sources_config[name]
        if since is not None:
            start = since
        elif backfill:
            start = pd.Timestamp(cfg.get("backfill_start", default_start)).date()
        else:
            start = resume_start(name, cfg, registry, fetch_log, default_start, today)
        log.info("%s: starting from %s", name, start)
        results.append(run_source(name, start, run_id=run_id, data_dir=data_dir,
                                  change_tolerance=cfg.get("change_tolerance")))
    return results


def summary_table(results: list[RunResult]) -> str:
    """Plain-text table of run results for the console."""
    rows = [("source", "status", "from", "fetched", "written", "secs", "error")]
    for r in results:
        err = (r.error or "")
        rows.append((
            r.source, r.status, str(r.start or ""), str(r.rows_fetched), str(r.rows_written),
            f"{r.seconds:.1f}", err if len(err) <= 70 else err[:67] + "...",
        ))
    return format_table(rows)


def format_table(rows: list[tuple[str, ...]]) -> str:
    """Left-aligned fixed-width table; the first row is the header."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
