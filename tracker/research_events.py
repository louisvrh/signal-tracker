"""Event test E2 (S25): probability forecasts from D_off, Brier-scored; logged and reported.

`python -m tracker events` reads the weekly real-time D_off values that the H3
run stored (`data/signals/pit_divergences.parquet`, from
`python -m tracker trading H3`), runs `tracker/backtest/events.py` on monthly
test dates with the vault closed, logs the result to `hypotheses_tested` and
writes `reports/E2.md`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tracker import settings, store
from tracker.backtest import events as ev
from tracker.backtest import pit
from tracker.registry import load_registry
from tracker.research import REPORTS_DIR, TEST_DAY, VAULT_FROM, _fmt
from tracker.research_trading import PIT_TABLE
from tracker.signals import divergence as dv

RUN_VERSION = "v1"


def test_dates(first: pd.Timestamp) -> pd.DatetimeIndex:
    """Monthly test dates on the 15th from the month of `first` to the last before the vault."""
    start = first.tz_convert("UTC").normalize().replace(day=1)
    dates = pd.date_range(start, VAULT_FROM, freq="MS") + pd.Timedelta(days=TEST_DAY - 1)
    return pd.DatetimeIndex([f.test for f in pit.walk_forward(dates, VAULT_FROM, min_train=0)])


def write_report(f: pd.DataFrame, s: dict[str, Any], cal: pd.DataFrame, cal_base: pd.DataFrame,
                 out_dir: Path = REPORTS_DIR) -> Path:
    lines = ["# E2: D_off forecasts whether the next official print is weaker than the last", "",
             f"*Run {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `python -m tracker events` (S25, {RUN_VERSION}). "
             "Definition: SIGNALS.md, operational definitions of 2026-10-07. Monthly test dates on the 15th; D_off as "
             f"known each week (the H3 run's weekly snapshots); vault closed (outcomes known before {VAULT_FROM:%Y-%m-%d}). "
             "Research only, not financial advice.*", "",
             "## Headline (out of sample)", ""]
    if not s.get("n"):
        lines += ["No forecast could be scored."]
    else:
        lines += ["| | Value |", "|---|---|",
                  f"| Forecasts scored (countries, test dates) | {s['n']} ({s['countries']}, {s['test_dates']}) |",
                  f"| Test dates | {s['first_test']} to {s['last_test']} |",
                  f"| Share of prints weaker than the last | {_fmt(s['event_rate'])} |",
                  f"| Brier score: D_off model / base rate / always 50% | {_fmt(s['brier_model'], 3)} / {_fmt(s['brier_base'], 3)} / 0.250 |",
                  f"| **Brier skill score against the base rate** (above 0 = model better) | **{_fmt(s['skill'], 3)}** |",
                  f"| Skill, 1st / 2nd half of test dates | {_fmt(s['skill_first_half'], 3)} / {_fmt(s['skill_second_half'], 3)} |",
                  f"| Paired t-test on squared errors: t (p) | {_fmt(s['t'])} ({_fmt(s['p'], 3)}) |",
                  "", f"**Verdict.** {verdict(s)}"]
    lines += ["", "## Calibration (D_off model)", "", "| Forecast probability | Forecasts | Mean forecast | Happened |",
              "|---|---|---|---|"]
    lines += [f"| {r.bin} | {r.n} | {_fmt(r.mean_p)} | {_fmt(r.observed)} |" for r in cal.itertuples()]
    lines += ["", "Base rate forecasts, same bins:", "", "| Forecast probability | Forecasts | Mean forecast | Happened |",
              "|---|---|---|---|"]
    lines += [f"| {r.bin} | {r.n} | {_fmt(r.mean_p)} | {_fmt(r.observed)} |" for r in cal_base.itertuples()]
    used = f["official"].value_counts()
    lines += ["", "## Method", "",
              f"- D_off end values from weekly snapshots (Mondays), the latest one in the {ev.STALE_DAYS} days before "
              "each monthly test date. The official series is the one that D_off value used: "
              + ", ".join(f"`{k}` ({v})" for k, v in used.items()) + ".",
              "- Event: the YoY of the first month not yet published is lower than the YoY of the last published month. "
              "Each (country, month) is forecast once, on the first test date it is the next print.",
              f"- Logistic regression of the event on D_off, pooled over countries, refitted on each date on events "
              f"already resolved then (labels as known then), needing {ev.MIN_EVENTS}; the base rate is the share of "
              "those events that happened. Scored on outcomes as known when the vault starts.",
              "- Variants tried: this one only.", "",
              "## Caveats", "",
              "- D_off exists only from January 2024 (its activity z-score needs two years of composite scores), so the "
              "sample is about a year and a half and the model's first fits use few events.",
              "- Most history has estimated publication dates (PLAN §5.1); whether a print was out on a test date is "
              "only approximately right for backfilled data."]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "E2.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verdict(s: dict[str, Any]) -> str:
    skill = s.get("skill", float("nan"))
    if skill > 0.05 and s.get("beats_base_both_halves") and s.get("p", 1) < 0.05:
        return "Promising: D_off's probabilities beat the base rate in both halves, by a clear margin. Tentative (short sample)."
    if skill > 0:
        return (f"Weak: a Brier skill of {skill:.3f} over the base rate is small"
                + ("" if s.get("beats_base_both_halves") else " and not in both halves")
                + f" (p {s.get('p', float('nan')):.2f}); not evidence yet.")
    return f"Not supported: D_off's probabilities scored worse than the base rate (skill {skill:.3f})."


def run_events(data_dir: Path | None = None, progress: Callable[[str], None] = print) -> dict[str, Any]:
    started = time.time()
    pit_rows = store.read_signal_table(PIT_TABLE, data_dir)
    if pit_rows.empty:
        raise RuntimeError("no weekly D_off values: run `python -m tracker trading H3` first (it stores them)")
    registry = load_registry()
    indicators = sorted({*dv.OFFICIAL_IP, *dv.OFFICIAL_EXPORTS, dv.MIRROR_REPORTERS})
    vintages = store.Vintages.load(indicators=indicators, data_dir=data_dir)
    first = pd.to_datetime(pit_rows.loc[(pit_rows["signal"] == "d_off") & pit_rows["value"].notna(), "as_of"],
                           utc=True).min()
    dates = test_dates(first)
    f = ev.e2_forecasts(pit_rows, vintages, registry, dates, VAULT_FROM)
    s = ev.brier_scores(f)
    path = write_report(f, s, ev.calibration(f), ev.calibration(f, "p_base"))
    tests = list(f["test_date"]) if not f.empty else None
    store.append_hypotheses_tested(pit.hypothesis_record(
        "E2", "logistic on D_off vs base rate", {"min_events": ev.MIN_EVENTS, "vault_from": str(VAULT_FROM),
                                                 "test_day": TEST_DAY, "stale_days": ev.STALE_DAYS},
        tests, tests, s, status="ok" if s.get("n") else "failed", uses_vault=False,
        notes=f"S25 {RUN_VERSION}; scored on outcomes known before the vault"))
    progress(f"E2: {s.get('n', 0)} forecasts scored, Brier {_fmt(s.get('brier_model'), 3)} vs base rate "
             f"{_fmt(s.get('brier_base'), 3)} (skill {_fmt(s.get('skill'), 3)}) in {time.time() - started:.0f} s -> "
             f"{path.relative_to(settings.ROOT)}")
    return {"forecasts": f, "scores": s, "report": path}
