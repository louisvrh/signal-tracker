"""Why the nowcasts failed (exploratory diagnostics, 2026-10-07; reports/diagnostics/WHY_FAILED.md).

Descriptive checks on the pre-registered nowcasts H1, H1b, H2 and H2b, none of
which tests a new hypothesis out of sample:

* **Alignment**: lead/lag correlations of signal and target YoY (a timing slip
  shows up as a peak away from lag 0) and the publication lags in the store.
* **Value versus volume**: how much of official export values' YoY is price
  rather than volume (CPB unit values and volumes), and whether port volumes
  track volume better than value.
* **Stability**: correlations and slopes by period and over rolling windows,
  to see whether the relationships changed (e.g. in 2022).
* **In sample versus out of sample**: the same models fitted on the whole
  sample (in-sample fit), the walk-forward forecasts with coefficients fitted
  on the whole sample instead ("oracle coefficients": look-ahead on purpose),
  and the real walk-forward forecasts. A signal with a stable relationship
  helps under oracle coefficients; if it helps only in sample, the
  relationship is too weak or unstable to estimate in real time.
* **Calendar composition**: how much of a month's average daily load moves
  with its number of weekend days, which calendar-adjusted IP doesn't.

Pure: snapshots, specs and forecast frames in; frames out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

from tracker.backtest import nowcast as nc

if TYPE_CHECKING:
    from tracker.registry import Registry

PERIODS = (("2020-21", "2020-01", "2021-12"), ("2022-23", "2022-01", "2023-12"), ("2024-25", "2024-01", "2025-12"))


def pairs(snap: pd.DataFrame, spec: nc.NowcastSpec, registry: "Registry") -> dict[str, pd.DataFrame]:
    """Per country, the signal's and target's monthly YoY (columns x, y) on the months both have,
    as known in one snapshot (the engine's own feature code)."""
    out = {}
    for country, (x, y) in nc.snapshot_features(snap, spec, registry).items():
        both = pd.concat([x.rename("x"), y.rename("y")], axis=1, sort=True).dropna()
        if len(both) >= 12:
            out[country] = both
    return out


def lead_lag(data: dict[str, pd.DataFrame], lags: Iterable[int] = range(-3, 4),
             drop_years: tuple[int, ...] = ()) -> pd.DataFrame:
    """Correlation of y_t with x_(t+k) for each lag k (k > 0: the signal k months later), per
    country, summarised across countries: median and the share of countries peaking at each k."""
    rows = []
    for country, d in data.items():
        x, y = d["x"], d["y"]
        corrs = {}
        for k in lags:
            shifted = pd.Series(x.to_numpy(), index=x.index - pd.DateOffset(months=k))
            both = pd.concat([y.rename("y"), shifted.rename("x")], axis=1, sort=True).dropna()
            if drop_years:
                both = both[~both.index.year.isin(drop_years)]
            corrs[k] = both["y"].corr(both["x"]) if len(both) >= 12 else np.nan
        best = max(corrs, key=lambda k: -np.inf if np.isnan(corrs[k]) else corrs[k])
        rows += [{"country": country, "lag": k, "corr": c, "peak": k == best} for k, c in corrs.items()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby("lag").agg(median_corr=("corr", "median"), countries=("corr", "count"),
                                 share_peaking=("peak", "mean")).reset_index()


def by_period(data: dict[str, pd.DataFrame], periods=PERIODS) -> pd.DataFrame:
    """Correlation and slope (y on x) per country and period, plus the share of y's variance the
    same-month signal explains."""
    rows = []
    for country, d in data.items():
        for label, lo, hi in periods:
            part = d.loc[lo:hi]
            if len(part) < 10 or part["x"].std() == 0:
                continue
            slope = float(np.polyfit(part["x"], part["y"], 1)[0])
            corr = float(part["x"].corr(part["y"]))
            rows.append({"country": country, "period": label, "months": len(part), "corr": corr, "slope": slope,
                         "r2": corr ** 2, "sd_x": float(part["x"].std()), "sd_y": float(part["y"].std())})
    return pd.DataFrame(rows)


def rolling_corr(data: dict[str, pd.DataFrame], window: int = 24) -> pd.DataFrame:
    """Rolling `window`-month correlation per country, and the median across countries by window end."""
    parts = []
    for country, d in data.items():
        r = d["x"].rolling(window, min_periods=window).corr(d["y"]).dropna()
        parts.append(pd.DataFrame({"date": r.index, "country": country, "corr": r.to_numpy()}))
    if not parts:
        return pd.DataFrame(columns=["date", "median_corr", "countries"])
    df = pd.concat(parts, ignore_index=True)
    return df.groupby("date").agg(median_corr=("corr", "median"), countries=("corr", "count")).reset_index()


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return beta, float(np.sqrt(np.mean(resid ** 2)))


def in_sample_vs_oos(forecasts: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per country: the in-sample RMSE ratio ARX/AR (whole sample, each forecast horizon h the
    country used), the out-of-sample ratio with whole-sample ("oracle") coefficients on the real
    forecasts' inputs, and the real walk-forward ratio. `data` = final-vintage pairs."""
    rows = []
    for country, f in forecasts.dropna(subset=["y_true"]).groupby("country"):
        d = data.get(country)
        if d is None or f.empty:
            continue
        sq_in_arx = sq_in_ar = 0.0
        n_in = 0
        oracle_arx, oracle_ar = [], []
        for h, part in f.groupby("h"):
            lag = pd.Series(d["y"].to_numpy(), index=d.index + pd.DateOffset(months=int(h)))
            t = pd.concat([d["x"].rename("x"), d["y"].rename("y"), lag.rename("y_lag")], axis=1, sort=True).dropna()
            if len(t) < 12:
                continue
            one = np.ones(len(t))
            b_arx, rmse_arx = _ols(np.column_stack([one, t["x"], t["y_lag"]]), t["y"].to_numpy())
            b_ar, rmse_ar = _ols(np.column_stack([one, t["y_lag"]]), t["y"].to_numpy())
            sq_in_arx += rmse_arx ** 2 * len(t) * len(part)
            sq_in_ar += rmse_ar ** 2 * len(t) * len(part)
            n_in += len(t) * len(part)
            for r in part.itertuples():
                oracle_arx.append((b_arx @ [1.0, r.x, r.y_last] - r.y_true) ** 2)
                oracle_ar.append((b_ar @ [1.0, r.y_last] - r.y_true) ** 2)
        if not oracle_arx:
            continue
        wf = np.sqrt(((f["arx"] - f["y_true"]) ** 2).mean()) / np.sqrt(((f["ar"] - f["y_true"]) ** 2).mean())
        rows.append({"country": country, "forecasts": len(f),
                     "in_sample_ratio": float(np.sqrt(sq_in_arx / sq_in_ar)) if sq_in_ar > 0 else np.nan,
                     "oracle_coef_ratio": float(np.sqrt(np.mean(oracle_arx) / np.mean(oracle_ar))),
                     "walk_forward_ratio": float(wf)})
    return pd.DataFrame(rows)


def pooled_ratios(forecasts: pd.DataFrame, data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """`in_sample_vs_oos` over all countries together (RMSE over all forecasts, not a mean of ratios)."""
    per = []
    for country, f in forecasts.dropna(subset=["y_true"]).groupby("country"):
        d = data.get(country)
        if d is None:
            continue
        for h, part in f.groupby("h"):
            lag = pd.Series(d["y"].to_numpy(), index=d.index + pd.DateOffset(months=int(h)))
            t = pd.concat([d["x"].rename("x"), d["y"].rename("y"), lag.rename("y_lag")], axis=1, sort=True).dropna()
            if len(t) < 12:
                continue
            one = np.ones(len(t))
            b_arx, rmse_arx = _ols(np.column_stack([one, t["x"], t["y_lag"]]), t["y"].to_numpy())
            b_ar, rmse_ar = _ols(np.column_stack([one, t["y_lag"]]), t["y"].to_numpy())
            for r in part.itertuples():
                per.append({"in_arx": rmse_arx ** 2, "in_ar": rmse_ar ** 2,
                            "or_arx": (b_arx @ [1.0, r.x, r.y_last] - r.y_true) ** 2,
                            "or_ar": (b_ar @ [1.0, r.y_last] - r.y_true) ** 2,
                            "wf_arx": (r.arx - r.y_true) ** 2, "wf_ar": (r.ar - r.y_true) ** 2})
    if not per:
        return {}
    p = pd.DataFrame(per).mean()
    return {"in_sample_ratio": float(np.sqrt(p["in_arx"] / p["in_ar"])),
            "oracle_coef_ratio": float(np.sqrt(p["or_arx"] / p["or_ar"])),
            "walk_forward_ratio": float(np.sqrt(p["wf_arx"] / p["wf_ar"])), "forecasts": len(per)}


def benchmark_from(forecasts: pd.DataFrame, data: dict[str, pd.DataFrame], start: str = "2022-01-01",
                   min_months: int = 12) -> dict[str, Any]:
    """How much the COVID years hurt the benchmark itself: on the same forecasts, the RMSE of the
    engine's AR benchmark, of the last published value, and of the same AR refitted only on months
    from `start` that were published by each forecast (final vintages). The model with the signal
    isn't refitted, so this tests no signal."""
    rows = []
    for r in forecasts.dropna(subset=["y_true"]).itertuples():
        d = data.get(r.country)
        if d is None:
            continue
        y = d["y"]
        lag = pd.Series(y.to_numpy(), index=y.index + pd.DateOffset(months=int(r.h)))
        t = pd.concat([y.rename("y"), lag.rename("y_lag")], axis=1, sort=True).dropna()
        t = t[(t.index >= pd.Timestamp(start)) & (t.index <= pd.Timestamp(r.last_month))]
        if len(t) < min_months:
            continue
        beta, _ = _ols(np.column_stack([np.ones(len(t)), t["y_lag"]]), t["y"].to_numpy())
        rows.append({"ar": r.ar, "rw": r.rw, "ar_recent": float(beta @ [1.0, r.y_last]), "y_true": r.y_true})
    if not rows:
        return {"forecasts": 0}
    f = pd.DataFrame(rows)

    def rmse(col: str) -> float:
        return float(np.sqrt(((f[col] - f["y_true"]) ** 2).mean()))

    return {"forecasts": len(f), "rmse_ar": rmse("ar"), "rmse_last_value": rmse("rw"),
            f"rmse_ar_from_{start[:4]}": rmse("ar_recent")}


def weekend_share_change(months: pd.DatetimeIndex) -> pd.Series:
    """Per month, its share of Saturdays and Sundays minus that of the same month a year earlier."""
    def share(m: pd.Timestamp) -> float:
        days = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D")
        return float((days.dayofweek >= 5).mean())
    months = pd.DatetimeIndex(months)
    return pd.Series([share(m) - share(m - pd.DateOffset(years=1)) for m in months], index=months)


def calendar_effect(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per country: how much the signal's YoY moves per point of weekend-share change, and the
    share of its variance that calendar composition explains (the target, calendar-adjusted IP,
    shouldn't move with it)."""
    rows = []
    for country, d in data.items():
        w = weekend_share_change(d.index) * 100
        for col in ("x", "y"):
            if d[col].std() == 0:
                continue
            slope = float(np.polyfit(w, d[col], 1)[0])
            corr = float(np.corrcoef(w, d[col])[0, 1])
            rows.append({"country": country, "series": "signal" if col == "x" else "target", "months": len(d),
                         "pp_per_weekend_point": slope, "r2": corr ** 2})
    return pd.DataFrame(rows)


def value_volume(value: pd.Series, volume: pd.Series, price: pd.Series, port: pd.Series | None,
                 drop_years: tuple[int, ...] = ()) -> dict[str, Any]:
    """Monthly YoY (%) of an official export value, its CPB volume and price, and port volumes:
    correlations, and the share of the value's YoY variance that volume and price each explain."""
    def yoy(s: pd.Series) -> pd.Series:
        s = s.dropna().sort_index()
        prev = pd.Series(s.to_numpy(), index=s.index + pd.DateOffset(years=1))
        return 100 * (s / prev.reindex(s.index) - 1)

    cols = {"value": yoy(value), "volume": yoy(volume), "price": yoy(price)}
    if port is not None:
        cols["port"] = yoy(port)
    d = pd.concat(cols, axis=1, sort=True).dropna()
    if drop_years:
        d = d[~d.index.year.isin(drop_years)]
    if len(d) < 12:
        return {"months": len(d)}
    out: dict[str, Any] = {"months": len(d), "first": f"{d.index.min():%Y-%m}", "last": f"{d.index.max():%Y-%m}",
                           "r2_value_on_volume": float(d["value"].corr(d["volume"]) ** 2),
                           "r2_value_on_price": float(d["value"].corr(d["price"]) ** 2),
                           "sd_value": float(d["value"].std()), "sd_volume": float(d["volume"].std()),
                           "sd_price": float(d["price"].std())}
    if "port" in d:
        out.update({"corr_port_volume": float(d["port"].corr(d["volume"])),
                    "corr_port_value": float(d["port"].corr(d["value"])),
                    "corr_port_price": float(d["port"].corr(d["price"]))})
    return out
