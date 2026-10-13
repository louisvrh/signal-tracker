"""H5's one look at the vault (S26): `python -m tracker vault H5`.

Runs exactly the test pre-registered in SIGNALS.md "H5-vault" (committed on its
own, before any vault data was read): H5's chokepoint events dated on or after
the vault start (2025-10-08), BDRY and Brent returns over 10, 20 and 40 trading
days, the cluster-respecting randomisation test of the diagnostics inside the
vault, primary BDRY at 20 days. Data are read as stored when the run starts
(`store.latest(as_of=now)`; weekly signals through `pit.compute`).

The run refuses to start if `hypotheses_tested` already has a vault look for H5,
and logs every test as one look (`uses_vault = true`, one shared timestamp),
including a crash. `--rehearse` runs the same code on pre-vault data only
(events and prices before 2025-10-08, window from 2022 as in the diagnostics),
logged as exploratory `DX5`: a check of the code, not a look at the vault.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tracker import settings, store
from tracker import research_trading as rt
from tracker.backtest import pit
from tracker.backtest import trading as tr
from tracker.backtest import vault as vt
from tracker.registry import Registry, load_registry
from tracker.research import REPORTS_DIR, VAULT_FROM, _fmt

HYPOTHESIS = "H5"
PREREGISTERED = "SIGNALS.md H5-vault, 2026-10-07, commit d954e4a"
HORIZONS = (10, 20, 40)
PRIMARY = ("transit counts", "BDRY", 20)
TRAINING_SIDE = 1  # BDRY rose more after H5's training events than on other days (reports/H5.md)
INFO_DAYS = 70  # pre-vault events this close to the vault start had horizons reaching into it (S24 dropped them)
CONTAMINATED = "contaminated by public knowledge of the 2026 Hormuz oil spike"
SIGNALS = (("transits_z", "transit counts", "portwatch_transits"),
           ("volume_z", "cargo volume", "portwatch_transit_volume"))
INSTRUMENTS = (("BDRY", "prices_freight"), ("Brent BZ=F", "prices_brent_future"))
REHEARSAL_ID = "DX5"
REHEARSAL_FROM = pd.Timestamp("2022-01-01", tz="UTC")  # the diagnostics' window start (CONTROLS.md §5)


class VaultAlreadyOpened(RuntimeError):
    """The hypothesis has already had its one look at the vault."""


def weekly_dates(end: pd.Timestamp, include_vault: bool) -> pd.DatetimeIndex:
    """Weekly test dates (Mondays 00:00 UTC) from 2020-01-06 to `end`, as in S24; vault dates only
    on the one look (`pit.walk_forward(include_vault=True)`)."""
    dates = pd.date_range(pd.Timestamp("2020-01-06", tz="UTC"), end, freq="W-MON")
    return pd.DatetimeIndex([f.test for f in pit.walk_forward(dates, VAULT_FROM, min_train=0,
                                                              include_vault=include_vault)])


def weekly_events(registry: Registry, dates: pd.DatetimeIndex, data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """H5's events (rule of `reports/H5.md`) on every test date, per signal label."""
    chokepoints = registry.countries_for("portwatch_transits")
    vint = store.Vintages.load(indicators=[ind for _, _, ind in SIGNALS], countries=chokepoints, data_dir=data_dir)
    rows = pit.compute(vint, registry, dates, series=[
        pit.ChainSignal(name, "yoy>rz(3y)", (ind,), tuple(chokepoints)) for name, _, ind in SIGNALS])
    return {label: rt.chokepoint_events(rows, name) for name, label, _ in SIGNALS}


def evaluate(events: dict[str, pd.DataFrame], px: pd.DataFrame, start: pd.Timestamp) -> list[dict[str, Any]]:
    """Every pre-registered test: signal x instrument x horizon, on events dated on or after `start`."""
    out = []
    for label, ev in events.items():
        chosen = ev[pd.DatetimeIndex(ev["as_of"]) >= start] if not ev.empty else ev
        for instrument, ind in INSTRUMENTS:
            prices = rt.price(px, ind, "WLD")
            for h in HORIZONS:
                r = vt.vault_test(pd.DatetimeIndex(chosen["as_of"]), prices, start, h)
                primary = (label, instrument, h) == PRIMARY
                r.update(signal=label, instrument=instrument, primary=primary, counts_for_success=primary,
                         note="primary" if primary else (CONTAMINATED if instrument.startswith("Brent")
                                                         else "information only"))
                out.append(r)
    return out


def info_events(events: pd.DataFrame, px: pd.DataFrame, vault_from: pd.Timestamp) -> list[dict[str, Any]]:
    """Returns after pre-vault events within INFO_DAYS of the vault start (S24 dropped their
    horizons): shown for information only, never counted."""
    if events.empty:
        return []
    dates = pd.DatetimeIndex(events["as_of"])
    near = events[(dates < vault_from) & (dates >= vault_from - pd.Timedelta(days=INFO_DAYS))]
    rows = []
    for e in near.itertuples():
        for instrument, ind in INSTRUMENTS:
            study = tr.event_study(pd.DatetimeIndex([e.as_of]), rt.price(px, ind, "WLD"), HORIZONS)
            rows.append({"chokepoint": e.chokepoint, "as_of": e.as_of, "z": e.z, "instrument": instrument,
                         **{f"r{r.horizon_days}": r.mean for r in study.itertuples()}})
    return rows


def _log(results: list[dict[str, Any]], hypothesis: str, params: dict[str, Any], status: str, uses_vault: bool,
         notes: str, tested_at: pd.Timestamp, dates: tuple[Any, Any]) -> None:
    for r in results:
        metrics = {k: r[k] for k in ("events", "episodes", "mean", "placebo_median", "p", "side", "welch_t",
                                     "welch_p", "other_days_mean", "window_days", "counts_for_success")}
        variant = f"vault: {r['signal']}: {r['instrument']}, {r['horizon_days']} days ({r['note']})"
        store.append_hypotheses_tested(pit.hypothesis_record(
            hypothesis, variant, params, None, dates, metrics, status=status, uses_vault=uses_vault,
            notes=notes, tested_at=tested_at))


def run_vault(rehearse: bool = False, progress: Callable[[str], None] = print,
              data_dir: Path | None = None) -> dict[str, Any]:
    """H5's one look at the vault (or, with `rehearse`, the same code on pre-vault data only)."""
    started = pd.Timestamp.now(tz="UTC").floor("s")
    if not rehearse:
        looks = pit.vault_looks(store.read_hypotheses_tested(data_dir), HYPOTHESIS)
        if not looks.empty:
            raise VaultAlreadyOpened(f"{HYPOTHESIS} already looked at the vault on {looks['tested_at'].min()}; "
                                     "each hypothesis gets exactly one look (PLAN §5.6)")
    start = REHEARSAL_FROM if rehearse else VAULT_FROM
    as_of = VAULT_FROM if rehearse else started
    params = {"preregistered": PREREGISTERED, "vault_from": str(VAULT_FROM), "window_start": str(start),
              "data_as_of": str(as_of), "seed": vt.SEED, "draws": vt.DRAWS, "horizons": list(HORIZONS),
              "primary": "transit counts, BDRY, 20 trading days", "test": "shifted real event dates, two-sided"}
    if rehearse:
        params["exploratory"] = True
    hypothesis = REHEARSAL_ID if rehearse else HYPOTHESIS
    status = "exploratory" if rehearse else "ok"
    notes = ("exploratory rehearsal of the H5-vault code on pre-vault data (vault closed)" if rehearse
             else f"S26 one-time vault look ({PREREGISTERED})")
    registry = load_registry()
    try:
        dates = weekly_dates(as_of, include_vault=not rehearse)
        progress(f"H5 vault{' rehearsal' if rehearse else ''}: weekly signals on {len(dates)} test dates "
                 f"({dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d})")
        events = weekly_events(registry, dates, data_dir)
        if rehearse:
            px = rt.prices(data_dir)  # as known at the vault start, dates before it only
        else:
            px = store.latest(indicators=list(rt.PRICE_IDS), as_of=as_of, data_dir=data_dir)
        results = evaluate(events, px, start)
        info = [] if rehearse else info_events(events["transit counts"], px, VAULT_FROM)
    except Exception as exc:  # a crash after the guard still counts as the look
        if not rehearse:
            store.append_hypotheses_tested(pit.hypothesis_record(
                hypothesis, "vault (crashed)", params, None, None, {}, status="failed", uses_vault=True,
                notes=f"{notes}: {type(exc).__name__}: {settings.redact(str(exc))[:300]}", tested_at=started))
        raise
    period = (start, max(dates.max(), start))
    _log(results, hypothesis, params, status, not rehearse, notes, started, period)
    primary = next(r for r in results if r["primary"])
    passed, reason = vt.verdict(primary, TRAINING_SIDE)
    out = {"results": results, "info": info, "primary": primary, "passed": passed, "reason": reason,
           "started": started, "as_of": as_of, "start": start, "dates": dates, "rehearse": rehearse,
           "events": events}
    path = write_report(out)
    progress(f"{'Rehearsal' if rehearse else 'Vault'}: {primary['events']} events at the primary horizon "
             f"({primary['episodes']} episodes); {reason} -> {path.relative_to(settings.ROOT)}")
    return out


# ------------------------------------------------------------------- report


def _pct(v: Any, digits: int = 1) -> str:
    return "–" if v is None or v != v else f"{100 * v:.{digits}f}%"


def write_report(out: dict[str, Any]) -> Path:
    results, primary, rehearse = out["results"], out["primary"], out["rehearse"]
    path = (REPORTS_DIR / "diagnostics" / "H5_vault_rehearsal.md") if rehearse else (REPORTS_DIR / "H5_vault.md")
    if rehearse:
        lines = ["# H5 vault code: rehearsal on pre-vault data (exploratory, DX5)", "",
                 f"*Run {out['started']:%Y-%m-%d %H:%M} UTC by `python -m tracker vault H5 --rehearse`. The code of the "
                 f"one-time vault test, run on the training period instead: events from {out['start']:%Y-%m-%d}, prices "
                 f"as known at the vault start and dated before it. Nothing on or after {VAULT_FROM:%Y-%m-%d} is read. "
                 "A check of the code against `reports/diagnostics/CONTROLS.md` §5, not a test of H5.*", ""]
    else:
        lines = ["# H5 vault test: chokepoint transits fall sharply, freight (BDRY) moves", "",
                 f"*Run {out['started']:%Y-%m-%d %H:%M} UTC by `python -m tracker vault H5`: H5's one look at the vault, "
                 f"exactly as pre-registered in {PREREGISTERED} (committed before any vault data was read). Vault: "
                 f"test dates and prices from {VAULT_FROM:%Y-%m-%d}, data as stored at the run (last weekly test date "
                 f"{out['dates'].max():%Y-%m-%d}). Logged in `hypotheses_tested` with `uses_vault = true`. "
                 "Research only, not financial advice.*", ""]
    lines += ["## Headline: the primary test", "",
              f"BDRY, transit-count events, {primary['horizon_days']} trading days (4 weeks). Test: the vault events' "
              "dates moved together by one random offset inside the vault window (1,000 placebo sets), two-sided.", "",
              "| Events (episodes) | Mean return after events | Placebo median | Side (training: above) | Randomisation p "
              "| Welch p (information) |", "|---|---|---|---|---|---|",
              f"| {primary['events']} ({primary['episodes']}) | {_pct(primary['mean'])} | {_pct(primary['placebo_median'])} "
              f"| {'above' if primary['side'] > 0 else 'below' if primary['side'] < 0 else '–'} | {vt.format_p(primary['p'])} "
              f"| {_fmt(primary['welch_p'], 3)} |", "",
              f"**Verdict.** {out['reason']}", ""]
    lines += ["## Every test (only the primary counts)", "",
              "| Signal | Instrument | Horizon | Events (episodes) | Mean after events | Placebo median | Other days "
              "| Randomisation p | Welch p | Status |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['signal']} | {r['instrument']} | {r['horizon_days']} days | {r['events']} ({r['episodes']}) "
                     f"| {_pct(r['mean'])} | {_pct(r['placebo_median'])} | {_pct(r['other_days_mean'])} "
                     f"| {vt.format_p(r['p'])} | {_fmt(r['welch_p'], 3)} | {r['note']} |")
    lines += ["", "## Events", ""]
    for label, ev in out["events"].items():
        chosen = ev[pd.DatetimeIndex(ev["as_of"]) >= out["start"]] if not ev.empty else ev
        lines.append(f"- {label}: " + (", ".join(f"{e.chokepoint} {pd.Timestamp(e.as_of):%Y-%m-%d} (z {e.z:.1f})"
                                                 for e in chosen.itertuples()) or "none"))
    longest = max(HORIZONS)
    transits = out["events"]["transit counts"]
    chosen = pd.DatetimeIndex(transits["as_of"]) if not transits.empty else pd.DatetimeIndex([])
    chosen = chosen[chosen >= out["start"]]
    for r in results:
        if r["signal"] == "transit counts" and r["instrument"] == "BDRY":
            missing = [d for d in chosen if d not in set(r["event_dates"])]
            if missing:
                lines.append(f"- Not counted at {r['horizon_days']} days (return not complete yet): "
                             + ", ".join(f"{pd.Timestamp(d):%Y-%m-%d}" for d in missing))
    if out["info"]:
        lines += ["", "## For information only (never counted)", "",
                  f"Pre-vault events within {INFO_DAYS} days of the vault start, whose horizons S24 dropped because they "
                  "reached into the vault:", "", "| Event | Instrument | 10 days | 20 days | 40 days |", "|---|---|---|---|---|"]
        lines += [f"| {i['chokepoint']} {pd.Timestamp(i['as_of']):%Y-%m-%d} (z {i['z']:.1f}) | {i['instrument']} "
                  f"| {_pct(i.get('r10'))} | {_pct(i.get('r20'))} | {_pct(i.get('r40'))} |" for i in out["info"]]
    lines += ["", "## How to read this", "",
              f"- **Episodes, not events, measure the evidence.** Events less than 8 weeks apart (any chokepoint) share "
              f"outcome days and count as one episode. At the primary horizon the vault held {primary['events']} "
              f"event(s) in {primary['episodes']} episode(s).",
              f"- **Brent is {CONTAMINATED}**, so it can't count as a success whatever it shows.",
              "- The randomisation test keeps the events' clustering, but with few episodes it can only say how unusual "
              "these few dates were within the window of BDRY's history"
              + (" (the training period here)." if rehearse else " (one year: the vault).") ,
              f"- **How small p can get.** The primary window has {primary['window_days']} trading days, so "
              f"{100 * vt.overlap_share(primary['horizon_days'], primary['window_days']):.0f}% of placebo sets are shifted by "
              f"fewer than {primary['horizon_days']} trading days and overlap the real events' outcome days. A real move "
              "that builds over the whole horizon shows up partly in those placebos too, so the shorter the window, "
              "the harder a small p-value is to reach.",
              f"- Window per horizon: the trading days from the window start whose return over that horizon is complete "
              f"(the {longest}-day window is the shortest). Each test uses its own generator (seed {vt.SEED}).",
              "- Prices are Yahoo adjusted closes; BZ=F is a continuous front-month future whose returns include roll jumps."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
