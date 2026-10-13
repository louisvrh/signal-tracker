"""The morning digest (S21): one short Telegram message a day, always sent.

Three kinds of message, chosen by `build`:

* **failure** first: when a `run_daily.ps1` step failed (named), with the
  failed sources, then the usual content if it could still be built;
* **full**: something is new since the last digest: data failures, newly
  stale series, divergence flags switching on or off, new anomalies
  (|z| > 2), plus the top divergences with a one-line reason each;
* **quiet**: a two-line "all quiet" message when nothing is new.

Every message states the dashboard's status. "New" is judged against the
state recorded with the last digest actually sent (the `digests` meta table);
the first digest treats flag events of the past week as new.

Pure: tables in, text out. Reading the store and sending are the CLI's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tracker import daily_signals as ds

TITLE = "Signal Tracker"
DIVERGENCE = {"d_mkt": "market", "d_off": "official"}
TOP_DIVERGENCES = 3
FIRST_DIGEST_DAYS = 7
MISSED_HOURS = 26  # a source with no fetch in this long missed its daily run
MAX_LINES = 6  # per list of new crossings, so the message stays short on a phone


@dataclass
class DigestInputs:
    """Everything a digest is built from.

    Attributes:
        now: when the digest is built (UTC).
        failed_steps: run_daily.ps1 steps that failed (empty when run by hand).
        dashboard: the dashboard's state: up, restarted, off, DOWN (or unknown).
        sources: per enabled source its latest fetch_log row: source, started_at, status, error.
        stale: stale indicators (indicator, last_date).
        quality_errors: open quality issues of severity error (source, indicator, country, detail).
        signals, flags: the `signals` step's tables (D46).
        last_state: the state saved with the last digest sent (None before the first).
        daily_run: built by run_daily.ps1 (else by hand).
    """

    now: pd.Timestamp
    failed_steps: list[str] = field(default_factory=list)
    dashboard: str = "unknown"
    sources: pd.DataFrame = field(default_factory=pd.DataFrame)
    stale: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality_errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    flags: pd.DataFrame = field(default_factory=pd.DataFrame)
    last_state: dict[str, Any] | None = None
    daily_run: bool = False


def _day(now: pd.Timestamp) -> str:
    return f"{pd.Timestamp(now):%a %d %b %Y}"


def source_problems(sources: pd.DataFrame, now: pd.Timestamp) -> list[str]:
    """One line per source whose latest fetch failed or that hasn't run for MISSED_HOURS."""
    lines = []
    for r in sources.itertuples():
        started = pd.Timestamp(r.started_at)
        if pd.isna(started) or now - started > pd.Timedelta(hours=MISSED_HOURS):
            when = "never" if pd.isna(started) else f"{started:%d %b %H:%M} UTC"
            lines.append(f"{r.source}: no fetch since {when}")
        elif r.status == "failed":
            error = str(r.error or "").strip().splitlines()[0][:120] if isinstance(r.error, str) else ""
            lines.append(f"{r.source}: FAILED{': ' + error if error else ''}")
    return lines


def state_of(inputs: DigestInputs) -> dict[str, Any]:
    """What the next digest compares against: active flags, anomalies and stale series."""
    active = ds.active_flags(inputs.flags, inputs.signals) if not inputs.flags.empty else pd.DataFrame()
    alerts = ds.anomalies(inputs.signals) if not inputs.signals.empty else pd.DataFrame()
    return {
        "flags": sorted(f"{r.country}:{r.signal}" for r in active.itertuples()) if not active.empty else [],
        "anomalies": sorted(f"{r.country}:{r.indicator}" for r in alerts.itertuples()) if not alerts.empty else [],
        "stale": sorted(inputs.stale["indicator"]) if not inputs.stale.empty else [],
    }


def _first_digest_flags(flags: pd.DataFrame, now: pd.Timestamp) -> tuple[set[str], set[str]]:
    """Flags switched on or off in the past week (the first digest's idea of "new")."""
    if flags.empty:
        return set(), set()
    now = pd.Timestamp(now)
    since = (now.tz_convert(None) if now.tzinfo else now).normalize() - pd.Timedelta(days=FIRST_DIGEST_DAYS)
    recent = flags[(flags["date"] >= since) & flags["event"].notna()]
    on = {f"{r.country}:{r.signal}" for r in recent.itertuples() if r.event == "on"}
    off = {f"{r.country}:{r.signal}" for r in recent.itertuples() if r.event == "off"}
    return on, off


def changes(inputs: DigestInputs, state: dict[str, Any]) -> dict[str, list[str]]:
    """What is new since the last digest: flags on and off, new anomalies, newly stale and recovered series."""
    last = inputs.last_state
    if last is None:
        on, off = _first_digest_flags(inputs.flags, inputs.now)
        on &= set(state["flags"])
        return {"flags_on": sorted(on), "flags_off": sorted(off - set(state["flags"])),
                "anomalies": state["anomalies"], "stale": state["stale"], "recovered": []}
    return {
        "flags_on": sorted(set(state["flags"]) - set(last.get("flags", []))),
        "flags_off": sorted(set(last.get("flags", [])) - set(state["flags"])),
        "anomalies": sorted(set(state["anomalies"]) - set(last.get("anomalies", []))),
        "stale": sorted(set(state["stale"]) - set(last.get("stale", []))),
        "recovered": sorted(set(last.get("stale", [])) - set(state["stale"])),
    }


def _flag_line(key: str, signals: pd.DataFrame, switched: str) -> str:
    country, signal = key.split(":")
    last = signals[(signals["country"] == country) & (signals["signal"] == signal)]
    value = f" ({last.sort_values('date')['value'].iloc[-1]:+.2f})" if not last.empty else ""
    return f"{country} {DIVERGENCE.get(signal, signal)} divergence flag {switched}{value}"


def _anomaly_line(key: str, signals: pd.DataFrame) -> str:
    country, indicator = key.split(":", 1)
    last = signals[(signals["country"] == country) & (signals["signal"] == f"z:{indicator}")]
    z = f" z {last.sort_values('date')['value'].iloc[-1]:+.2f}" if not last.empty else ""
    return f"{country} {indicator}{z}"


def failure_text(now: pd.Timestamp, failed_steps: list[str], dashboard: str, reason: str = "") -> str:
    """The shortest possible failure message: used when the full digest can't be built."""
    steps = ", ".join(failed_steps) if failed_steps else "none reported"
    text = f"{TITLE} · {_day(now)}: daily run FAILED at: {steps}. Dashboard: {dashboard}."
    if reason:
        text += f"\nThe digest itself failed too ({reason})."
    return text + "\nDetails: logs\\run_daily.log and logs\\tracker.log."


def build(inputs: DigestInputs) -> tuple[str, str, dict[str, Any]]:
    """The digest: (kind, text, state to save). Kind is failure, full or quiet."""
    state = state_of(inputs)
    new = changes(inputs, state)
    problems = source_problems(inputs.sources, inputs.now) if not inputs.sources.empty else []
    errors = len(inputs.quality_errors)
    n_flags = len(state["flags"])
    something_new = any(new.values()) or problems or errors or inputs.failed_steps \
        or inputs.dashboard not in ("up", "off", "unknown")

    if not something_new:
        stale = f" Stale (unchanged): {', '.join(state['stale'])}." if state["stale"] else ""
        text = (f"{TITLE} · {_day(inputs.now)}: all quiet.\n"
                f"{'Run OK' if inputs.daily_run else 'Checked by hand'}, dashboard {inputs.dashboard}, no new crossings. {n_flags} divergence flag(s) on, "
                f"unchanged.{stale}")
        return "quiet", text, state

    lines = []
    if inputs.failed_steps:
        kind = "failure"
        lines.append(f"{TITLE} · {_day(inputs.now)}: daily run FAILED at: {', '.join(inputs.failed_steps)}.")
    else:
        kind = "full"
        lines.append(f"{TITLE} · {_day(inputs.now)}")
        lines.append("Run OK" if inputs.daily_run else "Built by hand (not by the daily run)")
    lines.append(f"Dashboard: {inputs.dashboard}")
    if problems:
        lines += ["", "Data:"] + [f"- {p}" for p in problems]
    if errors:
        first = inputs.quality_errors.iloc[0]
        lines.append(f"Quality: {errors} error(s), e.g. {first['indicator']} {first['country']}: "
                     f"{str(first['detail'])[:80]}")
    if new["stale"]:
        lines.append(f"Newly stale: {', '.join(new['stale'])}")
    if new["recovered"]:
        lines.append(f"Fresh again: {', '.join(new['recovered'])}")
    crossings = (_capped([f"+ {_flag_line(k, inputs.signals, 'ON')}" for k in new["flags_on"]], "flags on")
                 + _capped([f"- {_flag_line(k, inputs.signals, 'off')}" for k in new["flags_off"]], "flags off")
                 + _capped([f"! anomaly: {_anomaly_line(k, inputs.signals)}"
                            for k in _by_size(new["anomalies"], inputs.signals)], "anomalies"))
    if crossings:
        lines += ["", "New since the last digest:"] + crossings
    active = ds.active_flags(inputs.flags, inputs.signals) if not inputs.flags.empty else pd.DataFrame()
    if not active.empty:
        lines += ["", f"Top divergences ({n_flags} flag(s) on):"]
        for n, r in enumerate(active.head(TOP_DIVERGENCES).itertuples(), start=1):
            lines.append(f"{n}. {_reason(r)}")
    elif inputs.signals.empty:
        lines.append("No signals yet (python -m tracker signals).")
    return kind, "\n".join(lines), state


def _capped(lines: list[str], what: str, limit: int = MAX_LINES) -> list[str]:
    """At most `limit` lines, then a count of the rest (the Overview page lists them all)."""
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"  ... and {len(lines) - limit} more {what} (Overview page)"]


def _by_size(keys: list[str], signals: pd.DataFrame) -> list[str]:
    """Anomaly keys (country:indicator), largest |z| first."""
    def size(key: str) -> float:
        country, indicator = key.split(":", 1)
        last = signals[(signals["country"] == country) & (signals["signal"] == f"z:{indicator}")]
        return float(last.sort_values("date")["value"].abs().iloc[-1]) if not last.empty else 0.0
    return sorted(keys, key=size, reverse=True)


def _reason(row: Any) -> str:
    """'CHN official -6.19 (12 wks): activity z -3.03 vs imf_exports z +3.16'."""
    info = json.loads(row.components) if isinstance(row.components, str) else {}
    if row.signal == "d_mkt":
        other = f"{info.get('etf', 'ETF')} vs URTH z {info.get('market_z', float('nan')):+.2f}"
    else:
        other = f"{info.get('official', 'official')} z {info.get('official_z', float('nan')):+.2f}"
    weeks = f" ({row.weeks} wk)" if row.weeks else ""
    return (f"{row.country} {DIVERGENCE[row.signal]} {row.value:+.2f}{weeks}: activity z "
            f"{info.get('activity_z', float('nan')):+.2f} vs {other}")
