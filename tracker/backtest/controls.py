"""Engine controls (exploratory diagnostics, 2026-10-07): can the backtests find a signal that is
there, and do they find nothing where there is nothing?

* **Positive controls** run real series through the unchanged nowcast engine
  (`nowcast.run`): a monthly official series that is published earlier than
  another measure of the same thing, and a *synthetic* signal (the target's own
  final values plus noise of known size, published at month end), which also
  measures how much a test can detect with its sample.
* **Placebos** break the link on purpose: a hypothesis's signal shifted in time
  (`NowcastSpec.shift_months`) or taken from another country
  (`NowcastSpec.signal_from`).
* **Replication**: `replicate_nowcast` recomputes the primary forecasts with
  separate code (statsmodels OLS, its own monthly means, first-seen values and
  horizons) for countries without Lunar New Year or break rules; the two must
  agree.
* **Timing audit**: `audit_timing` checks from the raw vintages that each
  forecast's signal month was published and its target month wasn't, on the
  test date.
* **Trading and event checks**: perfect-foresight positions must make money
  (and the reverse lose it), which tests the sign and timing of the trading
  code; random event dates show how often the event study's t-test calls
  chance significant; a D_off replaced by the answer must give E2 skill.

Pure: vintages, registries and frames in; frames and numbers out.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from tracker import store
from tracker.backtest import events as ev
from tracker.backtest import nowcast as nc
from tracker.backtest import pit
from tracker.backtest import trading as tr
from tracker.signals import chain as chains
from tracker.signals import weather as wx

if TYPE_CHECKING:
    from tracker.registry import Registry

SYNTHETIC = "diag_synthetic"  # in-memory only: never written to the lake or config
# Placebo shifts in months: far enough from 0, 12 and 24 that base effects of a YoY can't line up.
SHIFT_CHOICES = tuple(range(14, 23)) + tuple(range(26, 35))


# ------------------------------------------------------------------ helpers


def derangement(items: Iterable[str], rng: np.random.Generator, pool: Iterable[str] | None = None) -> dict[str, str]:
    """A random map of each item to a different one: a permutation with no fixed point, or, with
    `pool`, a draw from `pool` without the item itself (used when sources and targets differ)."""
    items = list(items)
    if pool is not None:
        pool = list(pool)
        return {i: str(rng.choice([p for p in pool if p != i])) for i in items}
    if len(items) < 2:
        raise ValueError("a derangement needs at least two items")
    while True:
        perm = list(rng.permutation(items))
        if all(a != b for a, b in zip(items, perm)):
            return dict(zip(items, perm))


def placebo_shifts(rng: np.random.Generator, n: int) -> list[int]:
    """`n` different shifts (months) drawn from SHIFT_CHOICES."""
    return sorted(int(k) for k in rng.choice(SHIFT_CHOICES, size=n, replace=False))


def delayed(vintages: store.Vintages, indicator: str, days: int) -> store.Vintages:
    """A copy of the vintages in which `indicator` is published `days` later (a timing-relaxed control)."""
    rows = vintages.rows.copy()
    mask = rows["indicator"] == indicator
    rows.loc[mask, "available_at"] = rows.loc[mask, "available_at"] + pd.Timedelta(days=days)
    return store.Vintages(rows)


def with_synthetic(registry: "Registry", like: str) -> "Registry":
    """A copy of the registry with the in-memory `diag_synthetic` indicator, defined like `like`."""
    base = registry.indicator(like)
    synthetic = base.model_copy(update={"id": SYNTHETIC, "source": "diag", "role": "control",
                                        "countries": "watchlist", "breaks": {}, "known_gaps": {}})
    return dataclasses.replace(registry, indicators={**registry.indicators, SYNTHETIC: synthetic})


def synthetic_rows(truth: pd.DataFrame, targets: dict[str, str], noise: float, rng: np.random.Generator,
                   lag_days: int = 1, start: Any = None) -> pd.DataFrame:
    """A synthetic monthly signal per country: the target's level as known in `truth` (deliberately
    final values) times exp(noise · N(0, 1)), published `lag_days` after each month ends, from
    `start` (to mimic a hypothesis's own sample). Its YoY is the target's YoY plus noise of about
    141 · noise percentage points."""
    parts = []
    for country, indicator in targets.items():
        part = truth[(truth["indicator"] == indicator) & (truth["country"] == country)]
        if start is not None:
            part = part[part["date"] >= pd.Timestamp(start)]
        if part.empty:
            continue
        dates = pd.DatetimeIndex(part["date"])
        published = (dates + pd.offsets.MonthEnd(0)).tz_localize("UTC") + pd.Timedelta(days=lag_days)
        parts.append(pd.DataFrame({
            "date": dates, "country": country, "indicator": SYNTHETIC,
            "value": part["value"].to_numpy() * np.exp(noise * rng.standard_normal(len(part))),
            "unit": part["unit"].to_numpy(), "frequency": "M", "source": "diag",
            "fetched_at": published, "available_at": published, "vintage_type": "estimated"}))
    return pd.concat(parts, ignore_index=True) if parts else store.empty_frame()


def noise_rows(countries: Iterable[str], start: Any, end: Any, rng: np.random.Generator,
               persistent: bool = False, lag_days: int = 1) -> pd.DataFrame:
    """A pure-noise monthly signal per country, published `lag_days` after each month: the null
    against which the "significant win" criterion is calibrated. White: level 100 · exp(0.05 · N(0, 1)),
    so its YoY is independent noise of about 7 pp a month. Persistent: the log level is a random walk
    (steps of 0.01), so its YoY is a smooth 12-month sum like real YoY series (about 3.5 pp), which can
    line up with a smooth target by chance."""
    months = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    published = (months + pd.offsets.MonthEnd(0)).tz_localize("UTC") + pd.Timedelta(days=lag_days)
    parts = []
    for c in countries:
        z = rng.standard_normal(len(months))
        level = np.exp(np.cumsum(0.01 * z)) if persistent else np.exp(0.05 * z)
        parts.append(pd.DataFrame({"date": months, "country": c, "indicator": SYNTHETIC, "value": 100 * level,
                                   "unit": "index", "frequency": "M", "source": "diag", "fetched_at": published,
                                   "available_at": published, "vintage_type": "estimated"}))
    return pd.concat(parts, ignore_index=True) if parts else store.empty_frame()


def add_rows(vintages: store.Vintages, rows: pd.DataFrame) -> store.Vintages:
    """The vintages plus extra rows (in memory)."""
    return store.Vintages(pd.concat([vintages.rows, rows], ignore_index=True))


# -------------------------------------------------------------- replication


def _months(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def _monthly_mean(values: pd.Series, calendar: str, share: float) -> pd.Series:
    by_month = values.dropna().resample("MS")
    mean, count = by_month.mean(), by_month.count()
    if calendar == "weekdays":
        need = pd.Series([np.busday_count(m.date(), (m + pd.offsets.MonthBegin(1)).date()) for m in mean.index],
                         index=mean.index)
    else:
        need = pd.Series(mean.index.days_in_month, index=mean.index)
    return mean[count >= share * need].dropna()


def _plain_yoy(monthly: pd.Series) -> pd.Series:
    year_ago = pd.Series(monthly.to_numpy(), index=monthly.index + pd.DateOffset(years=1))
    both = pd.concat([monthly.rename("now"), year_ago.rename("then")], axis=1, sort=True).dropna()
    return (100 * (both["now"] / both["then"] - 1)).rename(None)


def plain_countries(registry: "Registry", spec: nc.NowcastSpec) -> list[str]:
    """Countries a plain monthly YoY treats exactly like the engine: no Lunar New Year combining and
    no known break in the signal or target."""
    out = []
    for c in spec.tested():
        ref = registry.countries.get(c)
        lny = ref is not None and "lunar_new_year" in ref.moving_holidays
        breaks = registry.breaks_for(spec.signal, c) or registry.breaks_for(spec.targets[c], c)
        if not lny and not breaks:
            out.append(c)
    return out


def replicate_nowcast(vintages: store.Vintages, registry: "Registry", spec: nc.NowcastSpec,
                      test_dates: Iterable[Any], truth_as_of: Any, min_train: int = nc.MIN_TRAIN) -> pd.DataFrame:
    """The primary (`arx`) and benchmark (`ar`) forecasts recomputed with separate code, for the
    countries `plain_countries` returns. Columns: test_date, country, month, h, arx, ar, y_true."""
    countries = plain_countries(registry, spec)
    first: dict[tuple[str, pd.Timestamp], float] = {}
    done: set[tuple[str, pd.Timestamp]] = set()
    rows = []
    signal = registry.indicator(spec.signal)

    def series(snap: pd.DataFrame, indicator: str, country: str) -> pd.Series:
        part = snap[(snap["indicator"] == indicator) & (snap["country"] == country)]
        return pd.Series(part["value"].to_numpy(), index=pd.DatetimeIndex(part["date"])).sort_index()

    for test in sorted(pd.Timestamp(d) for d in test_dates):
        snap = vintages.latest(as_of=test)
        for c in countries:
            raw = series(snap, spec.signal, c)
            if raw.empty:
                continue
            if spec.weather:
                extra = {dd: series(snap, dd, c) for dd in (wx.HDD, wx.CDD)}
                if any(s.empty for s in extra.values()):
                    continue
                raw = chains.apply("wx", raw, chains.context_for(registry, spec.signal, c, extra=extra)).values
            monthly = raw if signal.frequency == "M" else _monthly_mean(raw, signal.calendar, nc.MIN_MONTH_SHARE)
            x = _plain_yoy(monthly)
            y = _plain_yoy(series(snap, spec.targets[c], c))
            for month, value in x.items():
                first.setdefault((c, month), float(value))
            if x.empty or y.empty:
                continue
            m, last = x.index.max(), y.index.max()
            if m <= last or (c, m) in done:
                continue
            h = _months(m, last)
            train = [(first[(c, k)], y[k], y[k - pd.DateOffset(months=h)]) for k in y.index
                     if (c, k) in first and (k - pd.DateOffset(months=h)) in y.index]
            if len(train) < min_train:
                continue
            t = pd.DataFrame(train, columns=["x", "y", "y_lag"])
            arx = sm.OLS(t["y"], sm.add_constant(t[["x", "y_lag"]])).fit()
            ar = sm.OLS(t["y"], sm.add_constant(t[["y_lag"]])).fit()
            done.add((c, m))
            rows.append({"test_date": test, "country": c, "month": m, "h": h,
                         "arx": float(arx.params @ [1.0, x[m], y[last]]),
                         "ar": float(ar.params @ [1.0, y[last]])})
    out = pd.DataFrame(rows, columns=["test_date", "country", "month", "h", "arx", "ar"])
    truth = vintages.latest(as_of=truth_as_of)
    out["y_true"] = [_plain_yoy(series(truth, spec.targets[c], c)).get(m, np.nan)
                     for c, m in zip(out["country"], out["month"])]
    return out


def compare_replication(engine: pd.DataFrame, replica: pd.DataFrame) -> dict[str, Any]:
    """How far the engine's forecasts are from the replica's on the same country-months."""
    keys = ["country", "month"]
    e = engine[engine["country"].isin(replica["country"].unique())]
    both = e.merge(replica, on=keys, how="outer", suffixes=("", "_rep"), indicator=True)
    matched = both[both["_merge"] == "both"]
    return {
        "engine_rows": int(len(e)), "replica_rows": int(len(replica)), "matched": int(len(matched)),
        "only_engine": int((both["_merge"] == "left_only").sum()),
        "only_replica": int((both["_merge"] == "right_only").sum()),
        "same_test_date": bool((matched["test_date"] == matched["test_date_rep"]).all()),
        "same_h": bool((matched["h"] == matched["h_rep"]).all()),
        "max_abs_diff_arx": float((matched["arx"] - matched["arx_rep"]).abs().max()) if len(matched) else float("nan"),
        "max_abs_diff_ar": float((matched["ar"] - matched["ar_rep"]).abs().max()) if len(matched) else float("nan"),
        "max_abs_diff_truth": float((matched["y_true"] - matched["y_true_rep"]).abs().max()) if len(matched) else float("nan"),
    }


# ------------------------------------------------------------- timing audit


def audit_timing(forecasts: pd.DataFrame, vintages: store.Vintages, spec: nc.NowcastSpec,
                 registry: "Registry") -> dict[str, Any]:
    """From the raw vintages: on each forecast's test date, was its signal month published (the
    engine's share of its days, or the monthly value), its target month not yet published, and its
    last official month published? Counts of forecasts breaking each rule."""
    rows = vintages.rows
    first_pub = rows.groupby(["indicator", "country", "date"])["available_at"].min()
    never = pd.Timestamp.max.tz_localize("UTC")
    signal = registry.indicator(spec.signal)
    bad_signal = bad_target = bad_last = 0
    for f in forecasts.itertuples():
        target = spec.targets[f.country]
        source = spec.signal_from.get(f.country, f.country)
        test = pit._utc(f.test_date)
        month = pd.Timestamp(f.month)
        source_month = month - pd.DateOffset(months=spec.shift_months)
        if signal.frequency == "M":
            if first_pub.get((spec.signal, source, source_month), never) > test:
                bad_signal += 1
        else:
            try:
                pub = first_pub.loc[(spec.signal, source)]
            except KeyError:
                bad_signal += 1
                continue
            end = source_month + pd.DateOffset(months=1)
            days = pub[(pub.index >= source_month) & (pub.index < end)]
            expected = np.busday_count(source_month.date(), end.date()) if signal.calendar == "weekdays" \
                else source_month.days_in_month
            if (days <= test).sum() < nc.MIN_MONTH_SHARE * expected:
                bad_signal += 1
        if first_pub.get((target, f.country, month), never) <= test:
            bad_target += 1
        if first_pub.get((target, f.country, pd.Timestamp(f.last_month)), never) > test:
            bad_last += 1
    return {"forecasts": int(len(forecasts)), "signal_not_yet_published": bad_signal,
            "target_already_published": bad_target, "last_official_not_published": bad_last}


# --------------------------------------------------------- trading controls


def oracle_daily(returns: pd.DataFrame, costs: tr.Costs, anti: bool = False) -> dict[str, Any]:
    """Perfect one-day foresight through the real position code: the trade is switched on at the
    close before every day the pair gains (`anti`: loses). Its gross return on every day it is on
    must have the right sign; if positions were a day out, it would look like noise."""
    sign = -1.0 if anti else 1.0
    out = {}
    pos = {}
    for col in returns.columns:
        r = returns[col]
        wanted = (sign * r > 0).astype(float)
        signal = pd.Series(wanted.to_numpy()[1:], index=r.index[:-1])  # known at the previous close
        pos[col] = tr.positions_from_signal(signal, r.index)
    positions = pd.DataFrame(pos)
    net, on = tr.portfolio(positions, returns, costs)
    gross, _ = tr.portfolio(positions, returns, tr.Costs(0.0, 0.0))
    held = gross[on > 0]
    out.update(tr.metrics(net, on))
    out["share_of_on_days_with_expected_sign"] = float(((sign * held) > 0).mean()) if len(held) else float("nan")
    return out


def oracle_holds(returns: pd.DataFrame, test_dates: pd.DatetimeIndex, hold: int, costs: tr.Costs,
                 anti: bool = False) -> dict[str, Any]:
    """Perfect foresight at the hypotheses' own rhythm: on each weekly test date not inside an open
    trade, enter (through `fixed_holds`, at the next close, for `hold` days) only where that trade
    will gain (`anti`: lose). Every trade's gross return must then have the expected sign; if
    `fixed_holds` were a day out, small trades would flip."""
    sign = -1.0 if anti else 1.0
    days = returns.index
    pos = {}
    gross_trades: list[float] = []
    for col in returns.columns:
        r = returns[col]
        growth = np.cumprod(1 + r.to_numpy())
        entries = []
        free_from = 0
        for d in test_dates:
            d = pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tzinfo else pd.Timestamp(d)
            i = int(np.searchsorted(days.to_numpy(), d.normalize().to_datetime64(), side="left"))
            if i < free_from or i + hold >= len(days):
                continue
            if sign * (growth[i + hold] / growth[i] - 1) > 0:
                entries.append(days[i])
                free_from = i + hold
        pos[col] = tr.fixed_holds(pd.DatetimeIndex(entries), days, hold)
        gross_trades += tr.trades(pos[col], r)
    positions = pd.DataFrame(pos)
    net, on = tr.portfolio(positions, returns, costs)
    out = tr.metrics(net, on)
    out["trades"] = len(gross_trades)
    out["share_of_trades_with_expected_sign"] = float(np.mean([sign * t > 0 for t in gross_trades])) \
        if gross_trades else float("nan")
    return out


def random_event_rate(prices: pd.Series, n_events: int, start: Any, end: Any, rng: np.random.Generator,
                      draws: int = 500, horizons: tuple[int, ...] = (10, 20, 40), alpha: float = 0.05) -> dict[str, Any]:
    """How often `event_study`'s Welch t-test calls `n_events` random days significant: the share of
    draws with p < `alpha`, per horizon. A calibrated test gives about `alpha`."""
    p = prices.dropna().sort_index()
    days = p.index[(p.index >= pd.Timestamp(start)) & (p.index < pd.Timestamp(end))]
    hits = {h: 0 for h in horizons}
    means = {h: [] for h in horizons}
    for _ in range(draws):
        events = pd.DatetimeIndex(sorted(rng.choice(days, size=n_events, replace=False)))
        study = tr.event_study(events, p, horizons)
        for r in study.itertuples():
            hits[r.horizon_days] += int(r.p < alpha)
            means[r.horizon_days].append(r.mean)
    return {"draws": draws, "events": n_events,
            **{f"false_positive_rate_{h}d": hits[h] / draws for h in horizons},
            **{f"random_means_{h}d": np.array(means[h]) for h in horizons}}


def shifted_event_means(events: pd.DatetimeIndex, prices: pd.Series, start: Any, end: Any,
                        rng: np.random.Generator, draws: int = 1000,
                        horizons: tuple[int, ...] = (10, 20, 40)) -> dict[int, np.ndarray]:
    """Mean forward returns of the real event dates all moved by one random number of trading days
    (wrapping round inside [start, end)): a placebo that keeps the events' clustering, which a test
    treating them as independent ignores. Per horizon, one mean per draw."""
    p = prices.dropna().sort_index()
    days = p.index[(p.index >= pd.Timestamp(start)) & (p.index < pd.Timestamp(end))]
    base = []
    for e in events:
        e = pd.Timestamp(e).tz_localize(None) if pd.Timestamp(e).tzinfo else pd.Timestamp(e)
        i = int(np.searchsorted(days.to_numpy(), e.normalize().to_datetime64(), side="left"))
        if i < len(days):
            base.append(i)
    base_arr = np.array(base)
    out = {}
    for h in horizons:
        forward = (p.shift(-h) / p - 1).reindex(days).to_numpy()
        means = []
        for _ in range(draws):
            idx = (base_arr + int(rng.integers(1, len(days)))) % len(days)
            means.append(float(np.nanmean(forward[idx])))
        out[h] = np.array(means)
    return out


def randomisation_p(actual_mean: float, random_means: np.ndarray) -> float:
    """Two-sided randomisation p-value: the share of random event sets whose mean forward return is at
    least as far from the random sets' median as the real one."""
    centre = float(np.median(random_means))
    return float((np.abs(random_means - centre) >= abs(actual_mean - centre)).mean())


# ------------------------------------------------------------- E2 controls


def e2_used_rows(pit_rows: pd.DataFrame, forecasts: pd.DataFrame) -> dict[int, float]:
    """For each E2 forecast with a known outcome, the index of the D_off row it used, with the outcome."""
    rows = pit_rows.assign(as_of=pd.to_datetime(pit_rows["as_of"], utc=True))
    d = rows[(rows["signal"] == "d_off") & rows["value"].notna()]
    out = {}
    for f in forecasts.dropna(subset=["outcome"]).itertuples():
        test = pd.Timestamp(f.test_date)
        cand = d[(d["country"] == f.country) & (d["as_of"] <= test)
                 & (d["as_of"] > test - pd.Timedelta(days=ev.STALE_DAYS))]
        if len(cand):
            out[int(cand.sort_values("as_of").index[-1])] = float(f.outcome)
    return out


def e2_oracle_rows(pit_rows: pd.DataFrame, used: dict[int, float], rng: np.random.Generator,
                   noise: float = 1.0) -> pd.DataFrame:
    """D_off replaced by the answer: −1 where the next print turned out weaker, +1 where not, plus
    N(0, noise). The E2 code must then show clear skill."""
    rows = pit_rows.copy()
    for i, outcome in used.items():
        rows.loc[i, "value"] = (1.0 - 2.0 * outcome) + noise * rng.standard_normal()
    return rows


def e2_shuffled_rows(pit_rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """D_off values shuffled across all rows: any skill left would be an artefact."""
    rows = pit_rows.copy()
    mask = (rows["signal"] == "d_off") & rows["value"].notna()
    rows.loc[mask, "value"] = rng.permutation(rows.loc[mask, "value"].to_numpy())
    return rows
