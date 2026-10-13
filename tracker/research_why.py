"""Why the nowcasts failed (exploratory diagnostics, 2026-10-07): `python -m tracker diagnose why`.

Reruns the pre-registered nowcasts H1, H1b, H2 and H2b exactly as S23 did (no
new model is tested out of sample), then runs `tracker/backtest/why.py`'s
descriptive checks on them: alignment and publication lags, value versus
volume (CPB World Trade Monitor, PLAN D53), stability over time, in sample
versus out of sample, and calendar composition. The numbers go to
`reports/diagnostics/why_failed_tables.md` (the plain-English reading is
`WHY_FAILED.md`, written from them); each check is logged to
`hypotheses_tested` as `DX2`, status `exploratory`. The vault stays closed.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from tracker import settings, store
from tracker import research as rs
from tracker.backtest import nowcast as nc
from tracker.backtest import pit
from tracker.backtest import why
from tracker.registry import Registry, load_registry
from tracker.research import VAULT_FROM, _fmt

DIAG_ID = "DX2"
STATUS = "exploratory"
# v1 (2026-10-07): the rolling-correlation table showed windows covering one or two countries.
# v2: only windows covering at least half the countries, and the benchmark refitted from 2022 (new).
RUN_VERSION = "v2"
NOTE = f"exploratory diagnostic of why the S23 nowcasts failed (not a hypothesis test), {RUN_VERSION}; vault closed"
OUT_DIR = settings.REPORTS_DIR / "diagnostics"
VALUE_VOLUME_COUNTRIES = ("USA", "GBR", "JPN", "CHN")


def _log(variant: str, params: dict[str, Any], metrics: dict[str, Any]) -> None:
    clean = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool, str, np.floating, np.integer))}
    store.append_hypotheses_tested(pit.hypothesis_record(
        DIAG_ID, variant, {**params, "vault_from": str(VAULT_FROM), "exploratory": True}, None, None, clean,
        status=STATUS, uses_vault=False, notes=NOTE))


def _pairs(snap: pd.DataFrame, spec: nc.NowcastSpec, registry: Registry) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for signal in spec.signal.split("|"):
        countries = tuple(c for c in spec.tested() if c in registry.countries_for(signal))
        if countries:
            out.update(why.pairs(snap, replace(spec, signal=signal, countries=countries), registry))
    return out


def timing(forecasts: pd.DataFrame, vintages: store.Vintages, spec: nc.NowcastSpec) -> dict[str, Any]:
    """How early each nowcast was: horizon h, days from the signal month's end to the test date, and
    from the test date to the target month's first publication (as stored)."""
    first_pub = vintages.rows.groupby(["indicator", "country", "date"])["available_at"].min()
    month_end = (forecasts["month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).dt.tz_localize("UTC")
    published = [first_pub.get((spec.targets[c], c, m), pd.NaT) for c, m in zip(forecasts["country"], forecasts["month"])]
    lead = (pd.to_datetime(pd.Series(published), utc=True) - forecasts["test_date"].reset_index(drop=True)).dt.days
    return {"forecasts": len(forecasts), "h_median": float(forecasts["h"].median()),
            "h_share": {int(k): round(float(v), 2) for k, v in forecasts["h"].value_counts(normalize=True).sort_index().items()},
            "days_after_month_end_median": float((forecasts["test_date"] - month_end).dt.days.median()),
            "days_before_target_published_median": float(lead.median())}


def publication_lags(vintages: store.Vintages, indicators: list[str]) -> pd.DataFrame:
    """Stored publication lags (available_at minus the period's end, days) per indicator and vintage type."""
    rows = vintages.rows[vintages.rows["indicator"].isin(indicators)]
    first = rows.sort_values("available_at").drop_duplicates(["indicator", "country", "date"]).copy()
    dates = pd.to_datetime(first["date"])
    end = (dates + pd.Timedelta(days=1)).where(first["frequency"] == "D", dates + pd.offsets.MonthBegin(1))
    first["lag"] = (first["available_at"].dt.tz_localize(None) - end).dt.days
    return (first.groupby(["indicator", "vintage_type"])
            .agg(values=("lag", "size"), median_days=("lag", "median"), min_days=("lag", "min"), max_days=("lag", "max"))
            .reset_index())


def value_volume_table(registry: Registry) -> pd.DataFrame:
    """Official export values (IMF IMTS, USD) against CPB volumes and prices and PortWatch port
    exports, as known when the vault starts, for the countries CPB shows individually."""
    v = store.Vintages.load(indicators=["imf_exports", "cpb_export_volume", "cpb_export_price", "portwatch_exports"])
    snap = v.latest(as_of=VAULT_FROM)

    def s(ind: str, c: str) -> pd.Series:
        part = snap[(snap["indicator"] == ind) & (snap["country"] == c)]
        return pit.as_series(part) if len(part) else pd.Series(dtype="float64")

    def port_monthly(c: str | None) -> pd.Series:
        if c is None:
            daily = [s("portwatch_exports", x) for x in registry.signal_countries("portwatch_exports")]
            total = pd.concat(daily, axis=1).dropna().sum(axis=1)
            return nc.monthly_mean(total)
        return nc.monthly_mean(s("portwatch_exports", c))

    rows = []
    for c in (*VALUE_VOLUME_COUNTRIES, "WLD"):
        vol, price = s("cpb_export_volume", c), s("cpb_export_price", c)
        value = s("imf_exports", c) if c != "WLD" else vol * price  # CPB's own value index for the world
        port = port_monthly(None if c == "WLD" else c)
        for label, drop in (("2020 on", ()), ("2022 on", (2020, 2021))):
            r = why.value_volume(value.loc["2019-01":], vol, price, port, drop_years=drop)
            rows.append({"country": c, "window": label, "value_source": "IMF IMTS (USD)" if c != "WLD" else
                         "CPB volume × price", **r})
    return pd.DataFrame(rows)


def analyse(hypothesis: str, spec: nc.NowcastSpec, registry: Registry, progress: Callable[[str], None]) -> dict[str, Any]:
    vintages = store.Vintages.load(indicators=rs.needed([spec]))
    forecasts = rs.run_spec(vintages, registry, spec)
    snap = vintages.latest(as_of=VAULT_FROM)
    data = _pairs(snap, spec, registry)
    result = {
        "score": nc.score(forecasts, "arx", "ar"),
        "timing": timing(forecasts, vintages, spec),
        "lags": publication_lags(vintages, sorted(set(spec.signal.split("|")) | set(spec.targets.values()))),
        "lead_lag": why.lead_lag(data),
        "lead_lag_ex": why.lead_lag(data, drop_years=(2020, 2021)),
        "periods": why.by_period(data),
        "rolling": why.rolling_corr(data),
        "is_oos": why.in_sample_vs_oos(forecasts, data),
        "is_oos_all": why.pooled_ratios(forecasts, data),
        "benchmark": why.benchmark_from(forecasts, data),
        "calendar": why.calendar_effect(data) if spec.weather else pd.DataFrame(),
    }
    _log(f"{hypothesis}: timing", {"hypothesis": hypothesis}, {**result["timing"], "h_share": str(result["timing"]["h_share"])})
    peak = result["lead_lag"].loc[result["lead_lag"]["median_corr"].idxmax()] if len(result["lead_lag"]) else None
    _log(f"{hypothesis}: lead/lag correlations", {"hypothesis": hypothesis},
         {"peak_lag": int(peak["lag"]) if peak is not None else None,
          "peak_median_corr": float(peak["median_corr"]) if peak is not None else None})
    per = result["periods"].groupby("period")["corr"].median().to_dict() if len(result["periods"]) else {}
    _log(f"{hypothesis}: correlation by period", {"hypothesis": hypothesis}, {f"median_corr_{k}": v for k, v in per.items()})
    _log(f"{hypothesis}: in sample vs out of sample", {"hypothesis": hypothesis}, result["is_oos_all"])
    _log(f"{hypothesis}: benchmark trained from 2022", {"hypothesis": hypothesis}, result["benchmark"])
    progress(f"  {hypothesis}: {len(forecasts)} forecasts; in-sample {_fmt(result['is_oos_all'].get('in_sample_ratio'))}, "
             f"oracle coefficients {_fmt(result['is_oos_all'].get('oracle_coef_ratio'))}, walk-forward "
             f"{_fmt(result['is_oos_all'].get('walk_forward_ratio'))}")
    return result


# ------------------------------------------------------------------- report


def _table(df: pd.DataFrame, columns: list[tuple[str, str, int | None]]) -> list[str]:
    """Markdown table of `columns` (column, header, digits; None = as text)."""
    lines = ["| " + " | ".join(h for _, h, _ in columns) + " |", "|" + "---|" * len(columns)]
    for r in df.to_dict("records"):
        cells = []
        for col, _, digits in columns:
            v = r.get(col)
            cells.append(str(v) if digits is None else _fmt(v, digits))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_tables(results: dict[str, dict[str, Any]], vv: pd.DataFrame, out_dir: Path = OUT_DIR) -> Path:
    lines = ["# Why the nowcasts failed: the numbers", "",
             f"*Run {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `python -m tracker diagnose why` "
             f"(`tracker/backtest/why.py`, {RUN_VERSION}). Exploratory diagnostics logged as `{DIAG_ID}`; vault closed "
             f"(data as known on {VAULT_FROM:%Y-%m-%d}). The plain-English reading is in `WHY_FAILED.md`.*", ""]
    lines += ["## 1. In sample versus out of sample", "",
              "RMSE ratio of the model with the signal (`arx`) to the same model without it (`ar`); below 1 = the signal "
              "helps. *In sample*: both fitted on every month before the vault (final vintages). *Oracle coefficients*: "
              "the real forecasts' inputs with those whole-sample coefficients (look-ahead on purpose). *Walk-forward*: "
              "the real result.", "",
              "| Hypothesis | Forecasts | In sample | Oracle coefficients | Walk-forward |", "|---|---|---|---|---|"]
    for h, r in results.items():
        a = r["is_oos_all"]
        lines.append(f"| {h} | {a.get('forecasts', 0)} | {_fmt(a.get('in_sample_ratio'))} | "
                     f"{_fmt(a.get('oracle_coef_ratio'))} | {_fmt(a.get('walk_forward_ratio'))} |")
    lines += ["", "### The benchmark without the COVID years", "",
              "On the same forecasts (those with at least 12 published months from 2022 to train on): the RMSE (pp) of "
              "the engine's AR benchmark (trained on every month since the signal starts, so on 2020-21 too), of the "
              "last published value, and of the same AR refitted only on months from 2022 (final vintages). The model "
              "with the signal isn't refitted.", "",
              "| Hypothesis | Forecasts | AR benchmark | Last published value | AR from 2022 |", "|---|---|---|---|---|"]
    for h, r in results.items():
        b = r["benchmark"]
        lines.append(f"| {h} | {b.get('forecasts', 0)} | {_fmt(b.get('rmse_ar'))} | {_fmt(b.get('rmse_last_value'))} | "
                     f"{_fmt(b.get('rmse_ar_from_2022'))} |")
    for h, r in results.items():
        lines += ["", f"### {h} by country", ""]
        lines += _table(r["is_oos"].sort_values("walk_forward_ratio"),
                        [("country", "Country", None), ("forecasts", "Forecasts", 0), ("in_sample_ratio", "In sample", 2),
                         ("oracle_coef_ratio", "Oracle coefficients", 2), ("walk_forward_ratio", "Walk-forward", 2)])
    lines += ["", "## 2. Correlation of signal and target YoY by period", "",
              "Median across countries of the same-month correlation (and slope of target on signal) within each period, "
              "final vintages.", "", "| Hypothesis | Period | Countries | Median correlation | Median slope | Median R² |",
              "|---|---|---|---|---|---|"]
    for h, r in results.items():
        p = r["periods"]
        if p.empty:
            continue
        for period, part in p.groupby("period"):
            lines.append(f"| {h} | {period} | {len(part)} | {_fmt(part['corr'].median())} | {_fmt(part['slope'].median())} "
                         f"| {_fmt(part['r2'].median())} |")
    for h, r in results.items():
        p = r["periods"]
        if p.empty:
            continue
        wide = p.pivot(index="country", columns="period", values="corr").reset_index()
        lines += ["", f"### {h}: correlation by country and period", ""]
        lines += _table(wide, [("country", "Country", None)] + [(c, c, 2) for c in wide.columns if c != "country"])
    lines += ["", "## 3. Rolling 24-month correlation (median across countries)", "",
              "Windows ending in June and December, and the last window that still covers at least half the countries "
              "(the newest months exist for only a few).", "",
              "| Window ending | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    covered = {h: r["rolling"][r["rolling"]["countries"] >= r["rolling"]["countries"].max() / 2]
               for h, r in results.items()}
    ends = sorted({d for c in covered.values() for d in c["date"] if d.month in (6, 12) or d == c["date"].max()})
    for d in ends:
        cells = []
        for h in results:
            row = covered[h][covered[h]["date"] == d]
            cells.append(_fmt(float(row["median_corr"].iloc[0])) if len(row) else "–")
        lines.append(f"| {d:%Y-%m} | " + " | ".join(cells) + " |")
    lines += ["", "## 4. Alignment: lead/lag correlations", "",
              "Median across countries of corr(target_t, signal_(t+k)); k > 0 = the signal k months later. A timing slip "
              "would put the peak away from k = 0. Share peaking = share of countries whose highest correlation is at k.", ""]
    for h, r in results.items():
        ll, lx = r["lead_lag"], r["lead_lag_ex"]
        if ll.empty:
            continue
        both = ll.merge(lx, on="lag", suffixes=("", "_ex"))
        lines += [f"### {h}", ""]
        lines += _table(both, [("lag", "k", 0), ("median_corr", "Median corr, all months", 2),
                               ("share_peaking", "Share peaking", 2), ("median_corr_ex", "Median corr, 2022 on", 2),
                               ("share_peaking_ex", "Share peaking, 2022 on", 2)])
        lines.append("")
    lines += ["## 5. Timing of the nowcasts and stored publication lags", "",
              "| Hypothesis | Forecasts | Horizon h (share of forecasts) | Test date, days after the signal month | "
              "Test date, days before the target month was published |", "|---|---|---|---|---|"]
    for h, r in results.items():
        t = r["timing"]
        lines.append(f"| {h} | {t['forecasts']} | " + ", ".join(f"{k}: {v:.2f}" for k, v in t["h_share"].items())
                     + f" | {_fmt(t['days_after_month_end_median'], 0)} | {_fmt(t['days_before_target_published_median'], 0)} |")
    lags = pd.concat([r["lags"] for r in results.values()]).drop_duplicates(["indicator", "vintage_type"])
    lines += ["", "Stored publication lags (first vintage's `available_at` minus the period's end, days):", ""]
    lines += _table(lags.sort_values(["indicator", "vintage_type"]),
                    [("indicator", "Indicator", None), ("vintage_type", "Vintage type", None), ("values", "Values", 0),
                     ("median_days", "Median", 0), ("min_days", "Min", 0), ("max_days", "Max", 0)])
    lines += ["", "## 6. Value versus volume (CPB World Trade Monitor)", "",
              "Monthly YoY of official export values in USD (IMF IMTS), CPB export volumes and USD unit values (prices), "
              "and PortWatch port exports (monthly mean of daily tonnes; for WLD, the sum over the 25 watchlist "
              "countries). R² = squared correlation with the value's YoY.", ""]
    lines += _table(vv, [("country", "Country", None), ("window", "Window", None), ("months", "Months", 0),
                         ("r2_value_on_volume", "R² value~volume", 2), ("r2_value_on_price", "R² value~price", 2),
                         ("sd_value", "s.d. value", 1), ("sd_volume", "s.d. volume", 1), ("sd_price", "s.d. price", 1),
                         ("corr_port_volume", "corr port~volume", 2), ("corr_port_value", "corr port~value", 2),
                         ("corr_port_price", "corr port~price", 2)])
    cal = [(h, r["calendar"]) for h, r in results.items() if not r["calendar"].empty]
    if cal:
        lines += ["", "## 7. Calendar composition (H2, H2b)", "",
                  "How much each series' YoY moves per percentage point of change in the month's share of weekend days "
                  "against a year earlier, and the R². The target is calendar-adjusted IP, so it shouldn't move.", ""]
        for h, c in cal:
            lines += [f"### {h}", ""]
            lines += _table(c.sort_values(["series", "country"]), [("country", "Country", None), ("series", "Series", None),
                                                                    ("months", "Months", 0), ("pp_per_weekend_point", "pp per point", 2),
                                                                    ("r2", "R²", 2)])
            lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "why_failed_tables.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_why(progress: Callable[[str], None] = print) -> dict[str, Any]:
    started = time.time()
    registry = load_registry()
    specs = rs.specs(registry)
    results = {}
    for hypothesis in ("H1", "H1b", "H2", "H2b"):
        results[hypothesis] = analyse(hypothesis, specs[hypothesis][0], registry, progress)
    vv = value_volume_table(registry)
    for r in vv.itertuples():
        _log(f"value vs volume: {r.country}, {r.window}", {"country": r.country, "window": r.window},
             {k: getattr(r, k) for k in ("months", "r2_value_on_volume", "r2_value_on_price", "corr_port_volume",
                                         "corr_port_value") if hasattr(r, k)})
    path = write_tables(results, vv)
    progress(f"Done in {time.time() - started:.0f} s -> {path.relative_to(settings.ROOT)}")
    return {"results": results, "value_volume": vv, "report": path}
