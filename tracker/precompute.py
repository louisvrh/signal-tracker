"""Precomputed STL decompositions for the dashboard (S17c, PLAN D32).

MSTL takes about a second per daily series, too slow for the dashboard to fit
while a page loads. `python -m tracker decompose`, a step of run_daily.ps1,
fits every stored series with at least 3 years of data and writes the
seasonal components to data/signals/stl.parquet. The dashboard hands them to
the chain's `stl` step, which uses one only if it was fitted on exactly the
series the step receives, and otherwise fits afresh.

Each series is fitted on up to two inputs: the raw series (`variant` raw, for
a chain starting with `stl`) and, where the country's moving holidays change
it, the holiday-adjusted series (`variant` hol, for the "Seasonally adjusted"
preset `hol>stl`).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from tracker import store
from tracker.registry import Registry, load_registry
from tracker.signals import chain
from tracker.signals import transforms as tf

log = logging.getLogger(__name__)

STL_TABLE = "stl"
STL_COLUMNS = ["indicator", "country", "variant", "date", "observed", "seasonal"]
FREQUENCIES = {"D", "W", "M", "Q"}
MIN_DAYS = tf.MIN_STL_YEARS * 365


def decompose_series(values: pd.Series, ctx: chain.Context) -> dict[str, pd.DataFrame]:
    """STL of one series: {variant: frame with observed and seasonal, indexed by date}.

    Empty if the series is too short for STL. Pure: no file access.
    """
    raw = tf.clean(values)
    inputs = {"raw": raw}
    held = chain.apply("hol", raw, ctx).values
    if not (held.index.equals(raw.index) and held.to_numpy().tolist() == raw.to_numpy().tolist()):
        inputs["hol"] = held
    out = {}
    for variant, series in inputs.items():
        try:
            parts = tf.stl_components(series, ctx.frequency, ctx.calendar)
        except ValueError:
            return {}
        out[variant] = parts[["observed", "seasonal"]]
    return out


def default_workers() -> int:
    """Worker processes for the precompute: all cores but one, at most 8."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def precompute_stl(
    registry: Registry | None = None, data_dir: Path | None = None, workers: int | None = None
) -> dict[str, float]:
    """Fit STL to every stored series with 3+ years of data and write data/signals/stl.parquet.

    Series are fitted in `workers` processes (default `default_workers()`;
    1 fits them in this process). Returns counts: series fitted, of which
    holiday-adjusted too, series too short, rows written and seconds taken.
    """
    registry = registry or load_registry()
    workers = workers or default_workers()
    started = time.monotonic()
    obs = store.latest(data_dir=data_dir)
    keys: list[tuple[str, str]] = []
    tasks: list[tuple[pd.Series, chain.Context]] = []
    short = 0
    for (indicator, country), part in obs.groupby(["indicator", "country"], sort=True):
        if indicator not in registry.indicators or registry.indicator(indicator).frequency not in FREQUENCIES:
            continue
        values = pd.Series(part["value"].to_numpy(), index=pd.DatetimeIndex(part["date"])).sort_index()
        if (values.index.max() - values.index.min()).days < MIN_DAYS - 1:
            short += 1
            continue
        keys.append((indicator, country))
        tasks.append((values, chain.series_context(registry, indicator, country, values)))
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(decompose_series, *zip(*tasks), chunksize=4))
    else:
        results = [decompose_series(values, ctx) for values, ctx in tasks]
    frames: list[pd.DataFrame] = []
    fitted = holiday = 0
    for (indicator, country), parts in zip(keys, results):
        if not parts:
            short += 1
            continue
        fitted += 1
        holiday += "hol" in parts
        for variant, frame in parts.items():
            frames.append(frame.rename_axis("date").reset_index().assign(
                indicator=indicator, country=country, variant=variant)[STL_COLUMNS])
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STL_COLUMNS)
    path = store.write_signal_table(STL_TABLE, table, data_dir)
    seconds = time.monotonic() - started
    log.info("STL precomputed for %d series (%d also holiday-adjusted), %d too short: %d rows to %s in %.0f s",
             fitted, holiday, short, len(table), path, seconds)
    return {"fitted": fitted, "holiday_adjusted": holiday, "too_short": short, "rows": len(table),
            "seconds": seconds}


def stl_frames(table: pd.DataFrame) -> dict[tuple[str, str], list[pd.DataFrame]]:
    """Split the stl table into {(indicator, country): [frame per variant]} for chain contexts."""
    out: dict[tuple[str, str], list[pd.DataFrame]] = {}
    if table.empty:
        return out
    for (indicator, country, _), part in table.groupby(["indicator", "country", "variant"], sort=False):
        frame = part.set_index(pd.DatetimeIndex(part["date"]))[["observed", "seasonal"]]
        out.setdefault((indicator, country), []).append(frame)
    return out
