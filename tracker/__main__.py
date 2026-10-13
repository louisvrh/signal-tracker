"""Command-line interface: `python -m tracker <command>`."""

from __future__ import annotations

import argparse
import logging
import sys

from tracker import settings

log = logging.getLogger("tracker")


def cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch new data from all enabled sources (or one with --source)."""
    import pandas as pd

    from tracker.runner import run_all, summary_table

    since = pd.Timestamp(args.since).date() if args.since else None
    results = run_all(only=args.source, since=since, backfill=args.backfill)
    print(summary_table(results))
    failed = [r.source for r in results if r.status == "failed"]
    partial = [r.source for r in results if r.status == "partial"]
    if partial:
        print(f"\nPartial (call budget reached, will continue next run): {', '.join(partial)}")
    if failed:
        print(f"\n{len(failed)} source(s) failed: {', '.join(failed)} (details in logs/tracker.log)")
    return 1 if failed else 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show per-source freshness and registry counts."""
    import pandas as pd

    from tracker.registry import load_registry, registry_counts
    from tracker.runner import format_table
    from tracker.status import indicator_freshness, source_status

    def show(value, fmt=str) -> str:
        return "-" if value is None or pd.isna(value) else fmt(value)

    registry = load_registry()
    table = source_status(registry)
    rows = [("source", "enabled", "last run", "last success (UTC)", "rows last", "rows total", "latest data", "freshness")]
    for r in table.itertuples():
        rows.append((
            r.source, "yes" if r.enabled else "no", show(r.last_run),
            show(r.last_success, lambda v: pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")),
            show(r.rows_written_last, lambda v: str(int(v))),
            str(r.rows_in_lake), show(r.latest_data), r.freshness,
        ))
    print(format_table(rows))
    for r in table.itertuples():
        if show(r.last_error) != "-":
            print(f"  {r.source} last error: {str(r.last_error)[:200]}")

    stale = indicator_freshness(registry)
    stale = stale[stale["stale"] & stale["source"].isin(table.loc[table["rows_in_lake"] > 0, "source"])]
    if not stale.empty:
        print("\nStale indicators:")
        for r in stale.itertuples():
            due = "no data" if r.next_due is None else f"next period due {pd.Timestamp(r.next_due).date()}"
            print(f"  {r.indicator}: latest {r.last_date or '-'}, {due}")

    print_quality_summary()
    marker = settings.LOGS_DIR / "last_backup.txt"  # written by scripts/backup.ps1
    backup = marker.read_text(encoding="utf-8-sig").strip() if marker.exists() else r"none recorded (scripts\backup.ps1)"
    print(f"\nLast backup: {backup}")

    counts = registry_counts(registry)
    print(
        f"\nRegistry: {counts['indicators']} indicators, {counts['countries']} countries, "
        f"{counts['chokepoints']} chokepoints"
    )
    print("  by source: " + ", ".join(f"{k}={v}" for k, v in counts["by_source"].items()))
    print("  by role:   " + ", ".join(f"{k}={v}" for k, v in counts["by_role"].items()))
    return 0


def print_quality_summary(limit: int = 15) -> None:
    """Status section: open quality issues from the latest check."""
    from tracker import quality, store

    table = store.read_quality_issues()
    last = quality.last_run_time(table)
    issues = quality.open_issues(table)
    when = "never run (python -m tracker quality)" if last is None else f"last check {last:%Y-%m-%d %H:%M} UTC"
    print(f"\nQuality ({when}): {quality.summary_text(issues)}")
    for line in quality.format_issues(issues, limit=limit):
        print(line)


def cmd_quality(args: argparse.Namespace) -> int:
    """Run the data quality checks and record the findings."""
    from tracker import quality, store

    if args.list:
        issues = quality.open_issues(store.read_quality_issues())
        print(f"Open quality issues: {quality.summary_text(issues)}")
    else:
        issues = quality.run_checks()
        if args.dry_run:
            print(f"Quality checks (dry run, nothing recorded): {quality.summary_text(issues)}")
        else:
            quality.record(issues)
            print(f"Quality checks recorded: {quality.summary_text(issues)}")
    for line in quality.format_issues(issues, limit=None if args.list else 40):
        print(line)
    return 0


def cmd_decompose(args: argparse.Namespace) -> int:
    """Precompute STL seasonal components for the dashboard (data/signals/stl.parquet)."""
    from tracker.precompute import precompute_stl

    result = precompute_stl()
    print(f"STL precomputed for {result['fitted']} series ({result['holiday_adjusted']} also holiday-adjusted); "
          f"{result['too_short']} too short (under 3 years); {result['rows']} rows in {result['seconds']:.0f} s "
          "-> data/signals/stl.parquet")
    return 0


def cmd_weather(args: argparse.Namespace) -> int:
    """Run the weather-normalised demand model; write its outputs and the fit report."""
    from tracker.diagnostics import WEATHER_TABLE, run_weather

    result = run_weather(report=not args.no_report)
    print(f"Weather model: {result['series']} daily load series, {result['rows']} daily rows in "
          f"{result['seconds']:.0f} s -> data/signals/{WEATHER_TABLE}.parquet")
    summary = result["summary"]
    if len(summary):
        for r in summary.itertuples():
            print(f"  {r.indicator:<24} {r.country}  from {r.first}  MAPE ex-2020 {r.mape_ex2020:4.1f}%  "
                  f"weather in YoY {r.weather_in_raw_yoy:4.0%} raw -> {r.weather_in_normalised_yoy:3.0%} normalised  "
                  f"{r.verdict}")
    for path in result["paths"]:
        print(f"  wrote {path.relative_to(settings.ROOT)}")
    return 0


def cmd_signals(args: argparse.Namespace) -> int:
    """Compute composites, divergences and flags as of the latest data (data/signals/)."""
    import pandas as pd

    from tracker.daily_signals import FLAGS_TABLE, SIGNALS_TABLE, describe, latest_rows, run_signals

    result = run_signals()
    table, flags = result["signals"], result["flags"]
    last = latest_rows(table)
    scored = last[(last["signal"] == "activity")]
    active = flags[(flags["date"] == flags["date"].max()) & flags["flag"]] if not flags.empty else flags
    print(f"Signals: {len(table)} rows ({table['signal'].nunique()} signals, {scored['country'].nunique()} countries "
          f"with an activity score) in {result['seconds']:.0f} s -> data/signals/{SIGNALS_TABLE}.parquet, "
          f"{FLAGS_TABLE}.parquet")
    if not table.empty:
        print(f"Latest week: {pd.Timestamp(table['date'].max()):%Y-%m-%d}; active divergence flags: {len(active)}")
    print("\nTop divergences (most negative):")
    for _, row in result["top"].iterrows():
        print("  " + describe(row))
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    """Build and send the morning digest (always a message: all quiet, news, or a failure)."""
    from tracker.alerts.run import find_chat, run_digest
    from tracker.alerts.telegram import TelegramError

    if args.find_chat:
        try:
            print(find_chat())
        except TelegramError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    failed = [s.strip() for s in (args.failed_steps or "").split(",") if s.strip()]
    try:
        result = run_digest(dry_run=args.dry_run, failed_steps=failed, dashboard=args.dashboard,
                            daily_run=args.daily_run)
    except TelegramError as exc:
        log.error("digest not sent: %s", exc)
        print(f"error: digest not sent: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"[dry run: {result.kind} digest, {len(result.text)} characters, not sent]")
        print(result.text)
    elif not result.sent:  # Telegram not set up yet: the message goes to the log instead
        print(f"[{result.kind} digest NOT SENT: Telegram isn't set up (TELEGRAM_CHAT_ID missing from .env; "
              "run python -m tracker digest --find-chat after messaging the bot)]")
        print(result.text)
    else:
        print(f"Sent the {result.kind} digest ({len(result.text)} characters).")
    return 0


def cmd_nowcast(args: argparse.Namespace) -> int:
    """Run nowcast hypotheses (S23) out of sample, log them and write reports/<id>.md."""
    from tracker.research import run_nowcasts

    run_nowcasts(args.hypotheses)
    return 0


def cmd_trading(args: argparse.Namespace) -> int:
    """Run trading and event hypotheses (S24) with costs and benchmarks; write reports/<id>.md."""
    from tracker.research_trading import run_trading

    run_trading(args.hypotheses)
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    """Run event test E2 (S25) on the stored weekly D_off values; write reports/E2.md."""
    from tracker.research_events import run_events

    run_events()
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Exploratory diagnostics (vault closed), logged as exploratory; reports in reports/diagnostics/."""
    if args.what == "controls":
        from tracker.research_controls import run_controls

        run_controls()
    elif args.what == "why":
        from tracker.research_why import run_why

        run_why()
    return 0


def cmd_vault(args: argparse.Namespace) -> int:
    """A hypothesis's one look at the vault, exactly as pre-registered (S26; SIGNALS.md "H5-vault")."""
    from tracker.research_vault import VaultAlreadyOpened, run_vault

    try:
        run_vault(rehearse=args.rehearse)
    except VaultAlreadyOpened as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_journal_score(args: argparse.Namespace) -> int:
    """Score the resolved predictions in docs/JOURNAL.md by Brier score (S25, PLAN §9)."""
    from pathlib import Path

    import pandas as pd

    from tracker import journal
    from tracker.runner import format_table

    path = Path(args.file) if args.file else journal.JOURNAL_FILE
    entries, problems = journal.parse(path.read_text(encoding="utf-8"))
    table = journal.score(entries)
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    if table.empty:
        print(f"No predictions in {path} yet.")
        return 0
    rows = [("made", "resolves", "probability", "outcome", "Brier", "status", "claim")]
    for r in table.itertuples():
        rows.append((str(r.made), str(r.resolves), "-" if pd.isna(r.probability) else f"{r.probability:.0%}",
                     "-" if pd.isna(r.outcome) else ("yes" if r.outcome else "no"),
                     "-" if pd.isna(r.brier) else f"{r.brier:.3f}", r.status, str(r.claim)[:60]))
    print(format_table(rows))
    s = journal.summary(table)
    if s["scored"]:
        print(f"\n{s['scored']} scored: mean Brier {s['brier']:.3f} (always 50%: 0.250; always the base rate of "
              f"{s['base_rate']:.0%}: {s['brier_base_rate']:.3f}). Lower is better.")
    waiting = int((table["status"] == "awaiting outcome").sum())
    if waiting:
        print(f"{waiting} past their resolve date without an outcome: fill in 'Outcome: yes' or 'no'.")
    if args.write:
        print(f"Filled {journal.write_scores(path, entries)} Brier field(s) in {path}.")
    return 0


def cmd_compact(args: argparse.Namespace) -> int:
    """Merge small lake files per source, keeping every row.

    Compares per-indicator row counts and date ranges before and after, and
    exits with 1 if anything differs. Don't run it while a fetch is running.
    """
    import pandas as pd

    from tracker import store
    from tracker.runner import format_table

    before = store.source_summary()
    report = store.compact(include_meta=not args.no_meta)
    after = store.source_summary()
    rows = [("folder", "files merged", "rows")]
    rows += [(r.folder, str(r.files_merged), str(r.rows)) for r in report.itertuples()]
    print(format_table(rows))
    try:
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
    except AssertionError as exc:
        print(f"\nWARNING: store summary changed during compaction: {exc}", file=sys.stderr)
        return 1
    print(f"\nChecked: rows and date ranges of all {len(after)} indicators are unchanged.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with one sub-command per pipeline step."""
    parser = argparse.ArgumentParser(
        prog="python -m tracker",
        description="Signal tracker: fetch, store and analyse economic indicators.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show INFO logs on the console")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("fetch", help="fetch new data from enabled sources")
    p.add_argument("--source", help="fetch only this source")
    p.add_argument("--since", help="start date YYYY-MM-DD (used with --backfill)")
    p.add_argument("--backfill", action="store_true", help="re-download everything from --since")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("status", help="show source freshness, quality issues and registry counts")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("quality", help="run data quality checks and record the issues")
    p.add_argument("--dry-run", action="store_true", help="run the checks without recording them")
    p.add_argument("--list", action="store_true", help="list the open issues without running the checks")
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("decompose", help="precompute STL seasonal components for the dashboard")
    p.set_defaults(func=cmd_decompose)

    p = sub.add_parser("weather", help="run the weather-normalised demand model and write its fit report")
    p.add_argument("--no-report", action="store_true", help="only write data/signals/weather.parquet")
    p.set_defaults(func=cmd_weather)

    p = sub.add_parser("signals", help="compute signals as of the latest data")
    p.set_defaults(func=cmd_signals)

    p = sub.add_parser("digest", help="build and send the morning digest")
    p.add_argument("--dry-run", action="store_true", help="print the message instead of sending it")
    p.add_argument("--failed-steps", help="comma-separated run_daily.ps1 steps that failed (set by the daily run)")
    p.add_argument("--dashboard", help="the daily run's dashboard check: up, restarted, off or DOWN")
    p.add_argument("--daily-run", action="store_true", help="built by run_daily.ps1 (says 'Run OK')")
    p.add_argument("--find-chat", action="store_true",
                   help="one-time setup: save the chat that messaged the bot as TELEGRAM_CHAT_ID, send a test")
    p.set_defaults(func=cmd_digest)

    p = sub.add_parser("nowcast", help="run nowcast hypotheses out of sample (vault closed) and write reports")
    p.add_argument("hypotheses", nargs="+", choices=["H1", "H1b", "H2", "H2b"], help="hypothesis ids (SIGNALS.md)")
    p.set_defaults(func=cmd_nowcast)

    p = sub.add_parser("trading", help="run trading and event hypotheses (vault closed) and write reports")
    p.add_argument("hypotheses", nargs="+", choices=["H3", "H4", "H5", "H6"], help="hypothesis ids (SIGNALS.md)")
    p.set_defaults(func=cmd_trading)

    p = sub.add_parser("events", help="run event test E2 (vault closed) and write reports/E2.md")
    p.set_defaults(func=cmd_events)

    p = sub.add_parser("diagnose", help="exploratory diagnostics (vault closed), reports in reports/diagnostics/")
    p.add_argument("what", choices=["controls", "why"],
                   help="controls: positive and placebo checks of the backtest engine; why: why the nowcasts failed")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("vault", help="a hypothesis's one look at the vault, as pre-registered (SIGNALS.md)")
    p.add_argument("hypothesis", choices=["H5"], help="hypothesis id with a pre-registered vault test")
    p.add_argument("--rehearse", action="store_true",
                   help="run the same code on pre-vault data only (exploratory DX5; the vault stays closed)")
    p.set_defaults(func=cmd_vault)

    p = sub.add_parser("journal-score", help="score resolved predictions in docs/JOURNAL.md (Brier)")
    p.add_argument("--write", action="store_true", help="fill in 'Brier: ___' for entries with an outcome")
    p.add_argument("--file", help="another journal file (default docs/JOURNAL.md)")
    p.set_defaults(func=cmd_journal_score)

    p = sub.add_parser("compact", help="merge small lake files, keeping every row")
    p.add_argument("--no-meta", action="store_true", help="leave data/meta (fetch_log, quality_issues) alone")
    p.set_defaults(func=cmd_compact)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, set up logging and run the chosen command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    settings.setup_logging(verbose=args.verbose)
    try:
        return args.func(args)
    except Exception as exc:  # last-resort guard so errors are logged, redacted
        log.exception("command %s failed", args.command)
        print(f"error: {settings.redact(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
