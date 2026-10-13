"""Hypothesis test runs (S23): read the vintages, run the pure tests, log them, write the reports.

`python -m tracker nowcast H1 H1b H2 H2b` runs the nowcast hypotheses with
`tracker/backtest/nowcast.py`, logs every variant to `hypotheses_tested`
(PLAN §4.2) and writes `reports/<id>.md`. The vault (the 12 months before the
first test run, PLAN §5.6) is never read: test dates end before it starts and
forecasts are scored on official values published before it.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from tracker import settings, store
from tracker.backtest import nowcast as nc
from tracker.backtest import pit
from tracker.registry import Registry, load_registry
from tracker.signals import weather as wx

# Fixed once, when the tests were first run on 2026-10-07 (pit.vault_start of that day), and
# passed to every later run so the vault doesn't slide as data arrives (D39).
VAULT_FROM = pd.Timestamp("2025-10-08", tz="UTC")
FIRST_TEST = pd.Timestamp("2020-01-15", tz="UTC")
TEST_DAY = 15  # monthly test dates on the 15th: last month's daily data are in by then
# v1 (2026-10-07): H1 only, then H1b crashed (logged as failed). v2: hit rates count only forecasts
# that call a direction (the last-value benchmark showed 0.00 before); models and data unchanged.
# v3: H1b rerun only; v2's sensitivity check read no Comtrade data, so it excluded every country.
RUN_VERSION = "v3"
REPORTS_DIR = settings.REPORTS_DIR
ESTIMATE_GAP_MONTHS = 6  # H1b sensitivity: IMF runs this far beyond the country's own Comtrade reports

H2_COUNTRIES = ("DEU", "ITA", "FRA", "POL", "ESP", "NLD", "GBR")
TITLES = {
    "H1": "PortWatch export volume YoY → Comtrade official exports YoY",
    "H1b": "PortWatch export volume YoY → IMF monthly exports YoY",
    "H2": "Weather-normalised total power demand → official industrial production YoY",
    "H2b": "Weekday overnight load predicts industrial production better than total load",
}


def test_dates() -> pd.DatetimeIndex:
    """Monthly test dates on the 15th (00:00 UTC), from FIRST_TEST to the last before the vault."""
    dates = pd.date_range(FIRST_TEST, VAULT_FROM, freq="MS") + pd.Timedelta(days=TEST_DAY - 1)
    folds = pit.walk_forward(dates, VAULT_FROM, min_train=0)  # the vault check, in one place
    return pd.DatetimeIndex([f.test for f in folds])


def _load_target(registry: Registry, country: str) -> str:
    return "eurostat_ip" if country in registry.countries_for("eurostat_ip") else "oecd_ip"


def specs(registry: Registry) -> dict[str, list[nc.NowcastSpec]]:
    """The pre-registered nowcasts (SIGNALS.md §5) and their ex-2020 versions."""
    pw = set(registry.signal_countries("portwatch_exports"))
    h1 = {c: "comtrade_exports" for c in registry.signal_countries("comtrade_exports") if c in pw}
    h1b = {c: "imf_exports" for c in registry.signal_countries("imf_exports") if c in pw}
    load = {c: _load_target(registry, c) for c in H2_COUNTRIES}
    total = {c: ("neso_load" if c == "GBR" else "entsoe_load") for c in H2_COUNTRIES}
    out = {
        "H1": [nc.NowcastSpec("H1", "portwatch_exports", h1)],
        "H1b": [nc.NowcastSpec("H1b", "portwatch_exports", h1b)],
        "H2": [_load_spec("H2", total, load)],
        "H2b": [_load_spec("H2b", {c: ("neso_overnight_load" if c == "GBR" else "entsoe_overnight_load")
                                   for c in H2_COUNTRIES}, load)],
    }
    for key, items in out.items():
        items.append(replace(items[0], exclude_2020=True))
    return out


def _load_spec(hypothesis: str, signals: dict[str, str], targets: dict[str, str]) -> nc.NowcastSpec:
    """Load signals differ by source (ENTSO-E or NESO), so one spec per signal indicator, merged later."""
    return nc.NowcastSpec(hypothesis, "|".join(sorted(set(signals.values()))), targets, weather=True,
                          countries=tuple(signals))


def run_spec(vintages: store.Vintages, registry: Registry, spec: nc.NowcastSpec) -> pd.DataFrame:
    """Forecasts for one spec; a spec whose signal names several indicators (ENTSO-E and NESO
    load) runs once per indicator on its own countries."""
    parts = []
    for signal in spec.signal.split("|"):
        countries = tuple(c for c in spec.tested() if c in registry.countries_for(signal))
        if countries:
            parts.append(nc.run(vintages, registry, replace(spec, signal=signal, countries=countries),
                                test_dates(), VAULT_FROM))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True).sort_values(["test_date", "country"]).reset_index(drop=True) \
        if parts else pd.DataFrame()


def likely_estimates(vintages: store.Vintages) -> dict[str, str]:
    """H1b sensitivity: countries whose recent IMF months are likely estimates, with the reason.

    As known when the vault starts: a country with no monthly Comtrade report at all, or whose IMF
    series runs ESTIMATE_GAP_MONTHS or more beyond its own latest Comtrade report, so its recent IMF
    months come from national releases or IMF staff estimates that can't be checked (D42)."""
    snap = pit.snapshot(vintages, VAULT_FROM, ["imf_exports", "comtrade_exports"])
    last = snap.groupby(["indicator", "country"])["date"].max().unstack("indicator")
    out = {}
    for country, row in last.iterrows():
        imf, com = row.get("imf_exports"), row.get("comtrade_exports")
        if pd.isna(imf):
            continue
        if pd.isna(com):
            out[country] = f"no monthly Comtrade data; IMF to {imf:%Y-%m}"
            continue
        gap = (imf.year - com.year) * 12 + imf.month - com.month
        if gap >= ESTIMATE_GAP_MONTHS:
            out[country] = f"IMF to {imf:%Y-%m}, own Comtrade reports to {com:%Y-%m} ({gap} months behind)"
    return out


# ------------------------------------------------------------------ variants


def variants(forecasts: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    """Every model-versus-benchmark comparison reported for one set of forecasts."""
    out = []
    for model, bench, note in (("arx", "ar", "primary: signal + last official value vs the same without the signal"),
                               ("x_only", "ar", "signal alone vs AR benchmark"),
                               ("pooled", "ar", "pooled ARX (intercept per country) vs AR benchmark"),
                               ("arx", "rw", "primary model vs last published value"),
                               ("ar", "rw", "AR benchmark vs last published value (benchmark check)")):
        metrics = nc.score(forecasts, model, bench)
        out.append({"sample": label, "model": model, "benchmark": bench, "note": note, **metrics})
    return out


def _log(hypothesis: str, variant: str, spec: nc.NowcastSpec, metrics: dict[str, Any],
         forecasts: pd.DataFrame) -> None:
    """One hypotheses_tested row per variant (PLAN §4.2). Every fold trains on all data published
    before its test date (expanding window), so both periods are given as the test dates' span."""
    params = {"signal": spec.signal, "targets": sorted(set(spec.targets.values())), "weather": spec.weather,
              "exclude_2020": spec.exclude_2020, "countries": list(spec.tested()), "vault_from": str(VAULT_FROM),
              "min_train": nc.MIN_TRAIN, "test_day": TEST_DAY, "window": "expanding, from the start of the data"}
    tests = list(forecasts["test_date"]) if not forecasts.empty else None
    store.append_hypotheses_tested(pit.hypothesis_record(
        hypothesis, variant, params, tests, tests,
        {k: v for k, v in metrics.items() if not isinstance(v, (list, dict)) and k not in ("note", "sample", "model",
                                                                                         "benchmark")},
        status="ok" if metrics.get("n") else "failed", uses_vault=False,
        notes=f"S23 nowcast {RUN_VERSION}; scored on official values published before the vault" if metrics.get("n")
        else "no out-of-sample forecast could be scored"))


# -------------------------------------------------------------------- report


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.{digits}f}"


def _variant_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Sample | Model vs benchmark | Forecasts | RMSE ratio | 1st half | 2nd half | Hit rate (benchmark) | DM p |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['sample']} | `{r['model']}` vs `{r['benchmark']}`: {r['note']} | {r.get('n', 0)} "
                     f"| {_fmt(r.get('rmse_ratio'))} | {_fmt(r.get('ratio_first_half'))} "
                     f"| {_fmt(r.get('ratio_second_half'))} | {_fmt(r.get('hit_rate'))} "
                     f"({_fmt(r.get('hit_rate_benchmark'))}) | {_fmt(r.get('dm_p'), 3)} |")
    return lines


def _headline(main: dict[str, Any], ex2020: dict[str, Any]) -> list[str]:
    rows = [("Forecasts (countries, test dates)", lambda m: f"{m.get('n', 0)} ({m.get('countries', 0)}, {m.get('test_dates', 0)})"),
            ("Test dates", lambda m: f"{m.get('first_test', '–')} to {m.get('last_test', '–')}"),
            ("RMSE, signal model / AR benchmark (pp)", lambda m: f"{_fmt(m.get('rmse'))} / {_fmt(m.get('rmse_benchmark'))}"),
            ("**RMSE ratio** (below 1 = signal helps)", lambda m: f"**{_fmt(m.get('rmse_ratio'))}**"),
            ("Ratio in 1st / 2nd half of test dates", lambda m: f"{_fmt(m.get('ratio_first_half'))} / {_fmt(m.get('ratio_second_half'))}"),
            ("Directional hit rate (benchmark)", lambda m: f"{_fmt(m.get('hit_rate'))} ({_fmt(m.get('hit_rate_benchmark'))})"),
            ("Diebold-Mariano stat (p)", lambda m: f"{_fmt(m.get('dm_stat'))} ({_fmt(m.get('dm_p'), 3)})")]
    lines = ["| Out of sample | All test months | Excluding 2020 |", "|---|---|---|"]
    lines += [f"| {name} | {fn(main)} | {fn(ex2020)} |" for name, fn in rows]
    return lines


def verdict(main: dict[str, Any], ex2020: dict[str, Any]) -> str:
    """A plain-English one-paragraph reading of the primary comparison."""
    if not main.get("n"):
        return "No out-of-sample forecasts could be scored."
    ratio, p = main.get("rmse_ratio", float("nan")), main.get("dm_p", float("nan"))
    both = main.get("beats_benchmark_both_halves") and ex2020.get("beats_benchmark_both_halves")
    if ratio < 1 and p < 0.05 and both:
        return (f"Promising: adding the signal cut the out-of-sample error by {100 * (1 - ratio):.0f}% against the "
                "AR benchmark, in both halves and without 2020, and the difference is statistically clear. "
                "Still tentative (short sample, estimated publication dates); the vault decides.")
    if ratio < 1 and both:
        return (f"Weakly positive: the signal lowered the error by {100 * (1 - ratio):.0f}% and did so in both halves "
                f"(also without 2020), but the Diebold-Mariano p-value ({p:.2f}) means it could be luck.")
    if ratio < 1:
        return (f"Mixed: overall {100 * (1 - ratio):.0f}% lower error, but not in both halves (or not without 2020), "
                "so it fails the graduation rule as it stands.")
    return (f"Not supported: forecasts with the signal were {100 * (ratio - 1):.0f}% worse than the AR benchmark "
            "out of sample.")


def write_report(hypothesis: str, specs_run: list[nc.NowcastSpec], forecasts: dict[str, pd.DataFrame],
                 rows: list[dict[str, Any]], appendix: pd.DataFrame, extra: list[str] | None = None,
                 out_dir: Path = REPORTS_DIR) -> Path:
    """reports/<hypothesis>.md: headline (out of sample only), every variant, per country, caveats, appendix."""
    main = next(r for r in rows if r["sample"] == "all" and r["model"] == "arx" and r["benchmark"] == "ar")
    ex = next(r for r in rows if r["sample"] == "ex-2020" and r["model"] == "arx" and r["benchmark"] == "ar")
    spec = specs_run[0]
    f_all = forecasts["all"]
    lines = [
        f"# {hypothesis}: {TITLES[hypothesis]}",
        "",
        f"*Run {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `python -m tracker nowcast {hypothesis}` "
        f"(S23, `tracker/backtest/nowcast.py`). Vault closed: test dates end before {VAULT_FROM:%Y-%m-%d} and "
        "forecasts are scored on official values published before then. Research only, not financial advice.*",
        "",
        "## Headline (out of sample)",
        "",
        *_headline(main, ex),
        "",
        f"**Verdict.** {verdict(main, ex)}",
        "",
        "## Method",
        "",
        f"- Signal: `{spec.signal.replace('|', '`, `')}`{', weather-normalised (`wx`, D45)' if spec.weather else ''}; "
        "daily values averaged per calendar month (at least 90% of days), YoY against the same month a year earlier. "
        f"Target: {', '.join(f'`{t}`' for t in sorted(set(spec.targets.values())))} YoY. Lunar New Year January-February "
        "combined and known breaks respected on both sides.",
        f"- Monthly test dates on the {TEST_DAY}th from {FIRST_TEST:%Y-%m}. On each, the data as known that day "
        "(`Vintages.latest(as_of)`, D39). A country-month is forecast once, on the first test date its signal month "
        "is complete and its official value isn't out yet.",
        f"- Models are refitted on every test date on months whose official value was already published, with each "
        f"month's signal as first computed (real-time), per country, needing {nc.MIN_TRAIN} months. "
        "Primary model `arx`: y = a + b·signal + c·(last published y); benchmark `ar`: the same without the signal "
        "(both direct forecasts h months ahead, h = months since the last published value).",
        "- Scored on the official YoY as known when the vault starts (the latest vintage then). Most of that history "
        "has estimated publication dates (backfilled in September 2026, PLAN §5.1), so results may flatter slightly.",
        "",
        "## Every variant tried",
        "",
        *_variant_table(rows),
    ]
    if extra:
        lines += ["", *extra]
    by_country = nc.score_by_country(f_all, "arx", "ar")
    if not by_country.empty:
        lines += ["", "## By country (primary model, all test months)", "",
                  "| Country | Forecasts | RMSE ratio | Hit rate | RMSE model / benchmark |", "|---|---|---|---|---|"]
        for r in by_country.sort_values("rmse_ratio").itertuples():
            lines.append(f"| {r.country} | {r.n} | {_fmt(r.rmse_ratio)} | {_fmt(r.hit_rate)} | "
                         f"{_fmt(r.rmse)} / {_fmt(r.rmse_benchmark)} |")
    lines += ["", "## Appendix: in-sample fit (not evidence)", "",
              "Official YoY on the same month's signal YoY, all months before the vault (latest vintages).", ""]
    if appendix.empty:
        lines.append("No data.")
    else:
        lines += ["| Country | Months | From | To | Correlation | Slope | R² |", "|---|---|---|---|---|---|---|"]
        for r in appendix.sort_values("corr", ascending=False).itertuples():
            lines.append(f"| {r.country} | {r.n} | {r.first} | {r.last} | {_fmt(r.corr)} | {_fmt(r.slope)} | {_fmt(r.r2)} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{hypothesis}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------- run


def needed(spec_list: list[nc.NowcastSpec]) -> list[str]:
    ids = set()
    for spec in spec_list:
        ids |= set(spec.signal.split("|")) | set(spec.targets.values())
        if spec.weather:
            ids |= {wx.HDD, wx.CDD}
    return sorted(ids)


def run_nowcasts(hypotheses: list[str], data_dir: Path | None = None,
                 progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Run, log and report each hypothesis; returns the primary metrics per hypothesis."""
    registry = load_registry()
    all_specs = specs(registry)
    summary: dict[str, Any] = {}
    total_forecasts: dict[str, pd.DataFrame] = {}
    for hypothesis in hypotheses:
        started = time.time()
        spec_list = all_specs[hypothesis]
        vintages = store.Vintages.load(indicators=needed(spec_list), data_dir=data_dir)
        forecasts = {"all": run_spec(vintages, registry, spec_list[0]),
                     "ex-2020": run_spec(vintages, registry, spec_list[1])}
        rows = variants(forecasts["all"], "all") + variants(forecasts["ex-2020"], "ex-2020")
        extra = []
        if hypothesis == "H1b":
            excluded = likely_estimates(store.Vintages.load(indicators=["imf_exports", "comtrade_exports"],
                                                            data_dir=data_dir))
            kept = forecasts["all"][~forecasts["all"]["country"].isin(excluded)]
            kept_ex = forecasts["ex-2020"][~forecasts["ex-2020"]["country"].isin(excluded)]
            sens = variants(kept, "sensitivity: likely-estimate countries out")[:1] \
                + variants(kept_ex, "sensitivity, ex-2020")[:1]
            rows += sens
            extra = ["## Sensitivity: countries whose recent IMF months are likely estimates", "",
                     f"Excluded (as known on {VAULT_FROM:%Y-%m-%d}; rule: no monthly Comtrade report, or IMF "
                     f"{ESTIMATE_GAP_MONTHS}+ months beyond the country's own Comtrade reports):", ""]
            extra += [f"- {c}: {why}" for c, why in sorted(excluded.items())] or ["- none"]
            extra += ["", "The sensitivity rows in the variant table above use the remaining countries."]
        if hypothesis in ("H2", "H2b"):
            total_forecasts[hypothesis] = forecasts["all"]
        appendix = _appendix(vintages, spec_list[0], registry)
        if hypothesis == "H2b" and "H2" in total_forecasts:
            extra = h2b_comparison(total_forecasts["H2"], forecasts["all"])
            rows += h2b_rows(total_forecasts["H2"], forecasts["all"])
        for r in rows:
            spec = spec_list[1] if r["sample"].endswith("ex-2020") or r["sample"] == "ex-2020" else spec_list[0]
            f = forecasts["ex-2020" if spec.exclude_2020 else "all"]
            _log(hypothesis, f"{r['sample']}: {r['model']} vs {r['benchmark']}", spec, r, f)
        path = write_report(hypothesis, spec_list, forecasts, rows, appendix, extra)
        primary = next(r for r in rows if r["sample"] == "all" and r["model"] == "arx" and r["benchmark"] == "ar")
        primary_ex = next(r for r in rows if r["sample"] == "ex-2020" and r["model"] == "arx" and r["benchmark"] == "ar")
        summary[hypothesis] = {"primary": primary, "ex2020": primary_ex, "report": path,
                               "seconds": time.time() - started, "variants": len(rows)}
        progress(f"{hypothesis}: {primary.get('n', 0)} forecasts, RMSE ratio {_fmt(primary.get('rmse_ratio'))} "
                 f"(ex-2020 {_fmt(primary_ex.get('rmse_ratio'))}), DM p {_fmt(primary.get('dm_p'), 3)}, "
                 f"{len(rows)} variants, {time.time() - started:.0f} s -> {path.relative_to(settings.ROOT)}")
    return summary


def _appendix(vintages: store.Vintages, spec: nc.NowcastSpec, registry: Registry) -> pd.DataFrame:
    snap = pit.snapshot(vintages, VAULT_FROM)
    parts = []
    for signal in spec.signal.split("|"):
        countries = tuple(c for c in spec.tested() if c in registry.countries_for(signal))
        if countries:
            parts.append(nc.in_sample(snap, replace(spec, signal=signal, countries=countries), registry))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def h2b_rows(total: pd.DataFrame, overnight: pd.DataFrame) -> list[dict[str, Any]]:
    """H2b's own test: overnight-load `arx` against total-load `arx` on the same country-months."""
    both = _paired(total, overnight)
    if both.empty:
        return []
    paired = both.rename(columns={"arx_overnight": "arx", "arx_total": "total_arx"})
    m = nc.score(paired, "arx", "total_arx")
    return [{"sample": "H2b test (paired months)", "model": "arx (overnight)", "benchmark": "arx (total)",
             "note": "overnight load vs total load, same country-months", **m}]


def _paired(total: pd.DataFrame, overnight: pd.DataFrame) -> pd.DataFrame:
    keys = ["country", "month"]
    t = total[keys + ["test_date", "arx", "y_true", "y_last"]].rename(columns={"arx": "arx_total"})
    o = overnight[keys + ["arx"]].rename(columns={"arx": "arx_overnight"})
    return t.merge(o, on=keys).dropna(subset=["y_true"])


def h2b_comparison(total: pd.DataFrame, overnight: pd.DataFrame) -> list[str]:
    both = _paired(total, overnight)
    if both.empty:
        return ["## H2b comparison", "", "No country-months forecast by both."]
    rmse_t = float(np.sqrt(((both["arx_total"] - both["y_true"]) ** 2).mean()))
    rmse_o = float(np.sqrt(((both["arx_overnight"] - both["y_true"]) ** 2).mean()))
    return ["## H2b comparison: overnight against total load", "",
            f"On the {len(both)} country-months both models forecast, the out-of-sample RMSE was {rmse_o:.2f} pp "
            f"with overnight load and {rmse_t:.2f} pp with total load (ratio {rmse_o / rmse_t:.2f}; below 1 = "
            "overnight better). The paired row in the variant table has the halves and the Diebold-Mariano test."]
