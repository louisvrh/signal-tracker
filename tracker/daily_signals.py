"""The daily `signals` command (S19): composites, divergences and flags as of the latest data.

Reads the store as known now (`store.latest(as_of=now)`), runs the same
functions the backtest engine uses as panel signals
(`divergence.signals_frame`, PLAN D39), and replaces two tables:

* `data/signals/signals.parquet` (PLAN §4.2): computed_at, as_of, date,
  country, signal, value, components (JSON). The weekly history of every
  signal: activity, activity_trade, activity_energy, activity_z, market_z,
  official_z, d_mkt, d_off, and each component's z as `z:<indicator>`.
* `data/signals/flags.parquet`: the divergence flags' weekly states
  (persistence and hysteresis) with on/off events.

The weekly history is recomputed from today's data, so it shows revised
values; backtests rebuild it as it was known on each date instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from tracker import store
from tracker.registry import load_registry
from tracker.signals import divergence as dv

SIGNALS_TABLE = "signals"
FLAGS_TABLE = "flags"
ANOMALY_Z = 2.0  # |z| above this on a component is an anomaly alert (SIGNALS.md §4)


def latest_rows(signals: pd.DataFrame) -> pd.DataFrame:
    """Each (country, signal)'s most recent row."""
    if signals.empty:
        return signals
    return signals.sort_values("date").groupby(["country", "signal"], as_index=False).tail(1)


def top_divergences(signals: pd.DataFrame, flags: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The `n` most negative current divergences (D_mkt and D_off on the latest week), with flag
    state and components."""
    last_week = signals["date"].max() if not signals.empty else None
    d = signals[(signals["signal"].isin(["d_mkt", "d_off"])) & (signals["date"] == last_week)]
    if d.empty:
        return d.assign(flag=pd.Series(dtype=bool))
    state = flags[flags["date"] == last_week][["country", "signal", "flag"]] if not flags.empty else None
    if state is not None:
        d = d.merge(state, on=["country", "signal"], how="left")
    return d.sort_values("value").head(n).reset_index(drop=True)


def run_signals(data_dir: Path | None = None, as_of: Any = None) -> dict[str, Any]:
    """Compute and write every S19 signal as of `as_of` (default now). Returns counts, timings,
    the latest values and the top divergences."""
    started = time.time()
    registry = load_registry()
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    snapshot = store.latest(indicators=dv.needed_indicators(), as_of=as_of, data_dir=data_dir)
    frame = dv.signals_frame(snapshot, as_of, registry)
    computed_at = pd.Timestamp.now(tz="UTC")
    table = frame.assign(computed_at=computed_at, as_of=as_of)[
        ["computed_at", "as_of", "date", "country", "signal", "value", "components"]]
    flags = dv.flag_table(frame).assign(computed_at=computed_at)
    store.write_signal_table(SIGNALS_TABLE, table, data_dir)
    store.write_signal_table(FLAGS_TABLE, flags, data_dir)
    return {"rows": len(table), "flags": flags, "signals": table, "top": top_divergences(table, flags),
            "seconds": time.time() - started}


def describe(row: pd.Series) -> str:
    """One line on a divergence: its value, flag state and what drives it."""
    info = json.loads(row["components"]) if isinstance(row.get("components"), str) else {}
    name = {"d_mkt": "market", "d_off": "official"}[row["signal"]]
    flag = "FLAG ON" if bool(row.get("flag")) else "no flag"
    if row["signal"] == "d_mkt":
        why = (f"activity z {info.get('activity_z', float('nan')):+.2f} vs {info.get('etf', 'ETF')} 3-month return "
               f"relative to URTH, z {info.get('market_z', float('nan')):+.2f}")
    else:
        why = (f"activity z {info.get('activity_z', float('nan')):+.2f} vs latest official "
               f"({info.get('official', '?')}) z {info.get('official_z', float('nan')):+.2f}")
    return f"{row['country']} {name} divergence {row['value']:+.2f} ({flag}): {why}"


# ------------------------------------------- summaries for the Overview page and the digest


def latest_scores(signals: pd.DataFrame) -> pd.DataFrame:
    """Each country's scores on the latest week: trade, energy, activity, activity_z, with coverage
    (components counted) and the components themselves ({indicator: {z, w, date, cat}})."""
    columns = ["date", "trade", "energy", "activity", "activity_z", "coverage", "n_trade", "n_energy", "components"]
    if signals.empty:
        return pd.DataFrame(columns=columns)
    week = signals["date"].max()
    last = signals[signals["date"] == week]
    names = {"activity_trade": "trade", "activity_energy": "energy", "activity": "activity", "activity_z": "activity_z"}
    wide = last[last["signal"].isin(names)].pivot(index="country", columns="signal", values="value").rename(columns=names)
    out = wide.reindex(columns=["trade", "energy", "activity", "activity_z"])
    meta = last[last["signal"].isin(["activity_trade", "activity_energy"])].drop_duplicates("country").set_index("country")
    counts = meta["components"].map(json.loads) if not meta.empty else pd.Series(dtype=object)
    out["n_trade"] = counts.map(lambda d: d["n_trade"]).reindex(out.index).fillna(0).astype(int)
    out["n_energy"] = counts.map(lambda d: d["n_energy"]).reindex(out.index).fillna(0).astype(int)
    out["coverage"] = out["n_trade"] + out["n_energy"]
    detail = last[last["signal"] == "activity"].set_index("country")["components"].map(json.loads)
    out["components"] = detail.reindex(out.index)
    out["date"] = week
    return out[columns].sort_values("activity", na_position="last")


def movers(signals: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The countries whose activity score changed most since the week before (largest absolute
    change first): country, now, week_before, change."""
    act = signals[signals["signal"] == "activity"]
    if act.empty:
        return pd.DataFrame(columns=["country", "now", "week_before", "change"])
    week = act["date"].max()
    now = act[act["date"] == week].set_index("country")["value"]
    before = act[act["date"] == week - pd.Timedelta(days=7)].set_index("country")["value"]
    out = pd.DataFrame({"now": now, "week_before": before.reindex(now.index)}).dropna()
    out["change"] = out["now"] - out["week_before"]
    out = out.reindex(out["change"].abs().sort_values(ascending=False).index).head(n)
    return out.rename_axis("country").reset_index()


def active_flags(flags: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Divergence flags on in the latest week: country, signal, value, since (the week it switched
    on), weeks (on for), components and a one-line description."""
    columns = ["country", "signal", "value", "since", "weeks", "components", "description"]
    if flags.empty:
        return pd.DataFrame(columns=columns)
    week = flags["date"].max()
    on = flags[(flags["date"] == week) & flags["flag"]]
    rows = []
    for r in on.itertuples():
        history = flags[(flags["country"] == r.country) & (flags["signal"] == r.signal) & (flags["event"] == "on")]
        since = history["date"].max()
        detail = signals[(signals["country"] == r.country) & (signals["signal"] == r.signal)
                         & (signals["date"] == week)]
        comps = detail["components"].iloc[0] if not detail.empty else None
        row = {"country": r.country, "signal": r.signal, "value": r.value, "since": since,
               "weeks": int((week - since).days // 7) + 1 if pd.notna(since) else None, "components": comps}
        row["description"] = describe(pd.Series({**row, "flag": True}))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values("value").reset_index(drop=True)


def anomalies(signals: pd.DataFrame, threshold: float = ANOMALY_Z) -> pd.DataFrame:
    """Components with |z| above the threshold on the latest week: country, indicator, z."""
    last = signals[(signals["date"] == signals["date"].max()) & signals["signal"].str.startswith("z:")]
    out = last[last["value"].abs() > threshold]
    return pd.DataFrame({"country": out["country"], "indicator": out["signal"].str[2:], "z": out["value"]}) \
        .sort_values("z").reset_index(drop=True)
