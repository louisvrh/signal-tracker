"""Event forecasts (S25; PLAN §5.6 item 7, SIGNALS.md E2).

E2: on each monthly test date, for each country with a D_off value (the
backtest engine's weekly end values, as known then), the event is "the next
official print is weaker than the last one": the YoY of the first month not
yet published is below the YoY of the last published month, in the same
official series D_off used (IP where a country has monthly IP, else official
exports; D46).

* Model: a logistic regression of the event on D_off, pooled over countries,
  refitted on each date on the events already resolved then (labels as known
  then), needing at least `MIN_EVENTS` of them.
* Benchmark: the base rate, the share of those resolved events that happened.
* Scored on the outcomes as known when the vault starts (Brier score,
  Brier skill score against the base rate, calibration bins).

Pure: frames and vintages in, frames out.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from tracker import store
from tracker.backtest import nowcast as nc
from tracker.backtest import pit
from tracker.signals import chain as chains
from tracker.signals import divergence as dv

if TYPE_CHECKING:
    from tracker.registry import Registry

MIN_EVENTS = 60
STALE_DAYS = 14  # a D_off end value older than this at the test date isn't used
BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def official_yoy(snap: pd.DataFrame, indicator: str, country: str, registry: "Registry") -> pd.Series:
    """The official series' YoY as known in a snapshot (Taiwan's mirror: complete months only, D46)."""
    part = snap[(snap["indicator"] == indicator) & (snap["country"] == country)]
    if part.empty:
        return pd.Series(dtype="float64")
    values = pit.as_series(part)
    if indicator == dv.MIRROR:
        rep = snap[(snap["indicator"] == dv.MIRROR_REPORTERS) & (snap["country"] == country)]
        values = dv.complete_mirror_months(values, pit.as_series(rep) if not rep.empty else None)
    return nc.official_yoy(values, chains.context_for(registry, indicator, country)).dropna()


def _latest_d_off(pit_rows: pd.DataFrame, test: pd.Timestamp) -> pd.DataFrame:
    rows = pit_rows[(pit_rows["signal"] == "d_off") & pit_rows["value"].notna()
                    & (pit_rows["as_of"] <= test) & (pit_rows["as_of"] > test - pd.Timedelta(days=STALE_DAYS))]
    return rows.sort_values("as_of").groupby("country").tail(1)


def _label(y: pd.Series, last: pd.Timestamp, nxt: pd.Timestamp) -> float:
    if nxt in y.index and last in y.index:
        return float(y[nxt] < y[last])
    return float("nan")


def _logit_p(train: pd.DataFrame, x: float) -> tuple[float, str]:
    """P(event | D_off = x) from a logistic fit on the training events; NaN with a reason if it can't be fitted."""
    if len(train) < MIN_EVENTS or train["label"].nunique() < 2:
        return float("nan"), f"{len(train)} resolved events"
    X = sm.add_constant(train["d_off"].to_numpy(), has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(train["label"].to_numpy(), X).fit(disp=0)
        return float(fit.predict(np.array([[1.0, x]]))[0]), ""
    except Exception as exc:  # e.g. perfect separation
        return float("nan"), type(exc).__name__


def e2_forecasts(pit_rows: pd.DataFrame, vintages: store.Vintages, registry: "Registry", test_dates: Iterable[Any],
                 truth_as_of: Any) -> pd.DataFrame:
    """One probability forecast per (country, next print), made on the first test date it is the
    next print; columns: test_date, country, official, d_off, last_month, next_month, p_model,
    p_base, n_train, note, outcome (as known at `truth_as_of`; NaN if not out by then)."""
    pit_rows = pit_rows.assign(as_of=pd.to_datetime(pit_rows["as_of"], utc=True))
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, pd.Timestamp]] = set()
    for test in sorted(pd.Timestamp(d) for d in test_dates):
        snap = pit.snapshot(vintages, test)
        cache: dict[tuple[str, str], pd.Series] = {}

        def yoy(indicator: str, country: str) -> pd.Series:
            key = (indicator, country)
            if key not in cache:
                cache[key] = official_yoy(snap, indicator, country, registry)
            return cache[key]

        # Training labels: earlier candidates whose next print is out by now (as known now).
        train = []
        for c in candidates:
            label = _label(yoy(c["official"], c["country"]), c["last_month"], c["next_month"])
            if not np.isnan(label):
                train.append({"d_off": c["d_off"], "label": label})
        train_df = pd.DataFrame(train, columns=["d_off", "label"])
        base = float(train_df["label"].mean()) if len(train_df) >= MIN_EVENTS else float("nan")
        for r in _latest_d_off(pit_rows, test).itertuples():
            info = json.loads(r.components) if isinstance(r.components, str) else {}
            official = info.get("official")
            if not official:
                continue
            y = yoy(official, r.country)
            if y.empty:
                continue
            last = y.index.max()
            nxt = last + pd.DateOffset(months=1)
            if (r.country, nxt) in seen:
                continue
            seen.add((r.country, nxt))
            p, note = _logit_p(train_df, float(r.value))
            candidates.append({"test_date": test, "country": r.country, "official": official, "d_off": float(r.value),
                               "last_month": last, "next_month": nxt, "p_model": p, "p_base": base,
                               "n_train": len(train_df), "note": note})
    out = pd.DataFrame(candidates)
    if out.empty:
        return out
    truth = pit.snapshot(vintages, truth_as_of)
    out["outcome"] = [_label(official_yoy(truth, o, c, registry), last, nxt)
                      for o, c, last, nxt in zip(out["official"], out["country"], out["last_month"], out["next_month"])]
    return out


def brier_scores(f: pd.DataFrame) -> dict[str, Any]:
    """Brier scores of the model and the base rate on the forecasts both made and that resolved,
    the Brier skill score (1 − model/base; above 0 = the model is better), halves and a paired t-test."""
    from scipy import stats

    s = f.dropna(subset=["p_model", "p_base", "outcome"])
    if s.empty:
        return {"n": 0}
    e_m, e_b = (s["p_model"] - s["outcome"]) ** 2, (s["p_base"] - s["outcome"]) ** 2
    dates = sorted(s["test_date"].unique())
    mid = dates[len(dates) // 2]
    halves = []
    for part in (s["test_date"] < mid, s["test_date"] >= mid):
        bm, bb = e_m[part].mean(), e_b[part].mean()
        halves.append(float(1 - bm / bb) if bb > 0 else float("nan"))
    t, p = stats.ttest_rel(e_m, e_b) if len(s) > 2 else (float("nan"), float("nan"))
    return {"n": int(len(s)), "countries": int(s["country"].nunique()), "test_dates": len(dates),
            "first_test": f"{min(dates):%Y-%m-%d}", "last_test": f"{max(dates):%Y-%m-%d}",
            "event_rate": float(s["outcome"].mean()), "brier_model": float(e_m.mean()), "brier_base": float(e_b.mean()),
            "brier_50pct": 0.25, "skill": float(1 - e_m.mean() / e_b.mean()) if e_b.mean() > 0 else float("nan"),
            "skill_first_half": halves[0], "skill_second_half": halves[1], "t": float(t), "p": float(p),
            "beats_base_both_halves": bool(halves[0] > 0 and halves[1] > 0)}


def calibration(f: pd.DataFrame, column: str = "p_model") -> pd.DataFrame:
    """Forecasts binned by probability: count, mean forecast and observed frequency per bin."""
    s = f.dropna(subset=[column, "outcome"])
    if s.empty:
        return pd.DataFrame(columns=["bin", "n", "mean_p", "observed"])
    bins = pd.cut(s[column], BINS, include_lowest=True)
    g = s.groupby(bins, observed=False)
    return pd.DataFrame({"bin": [str(b) for b in g.size().index], "n": g.size().to_numpy(),
                         "mean_p": g[column].mean().to_numpy(), "observed": g["outcome"].mean().to_numpy()})
