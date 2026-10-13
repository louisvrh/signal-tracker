"""Weather-model outputs and diagnostics (S18): data/signals/weather.parquet and reports/diagnostics/.

`python -m tracker weather` runs the weather-normalised demand model
(tracker/signals/weather.py) on every daily power-demand series, writes the
daily outputs (anomaly, normalised demand and both YoYs) to
`data/signals/weather.parquet`, and, unless told not to, a fit report:
`reports/diagnostics/weather_fit.md` (one table row per series) and one
Plotly chart page per country, `reports/diagnostics/weather_<ISO3>.html`.

The model itself is pure; this module does the reading and writing, like
`precompute.py`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tracker import settings, store
from tracker.registry import Registry, load_registry
from tracker.signals import transforms as tf
from tracker.signals import weather as wx
from tracker.signals.chain import context_for

log = logging.getLogger(__name__)

WEATHER_TABLE = "weather"
REPORT_DIR = settings.ROOT / "reports" / "diagnostics"
COLOURS = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2 (blue, orange)
SURFACE, GRID, INK, MUTED = "#fcfcfb", "#e6e5e0", "#0b0b0b", "#52514e"
# Fit verdicts: out-of-sample daily error (outside 2020) and weather left in the normalised YoY.
GOOD = (4.5, 0.05)
FAIR = (6.0, 0.12)


def daily_load_series(registry: Registry) -> list[tuple[str, str]]:
    """(indicator, country) for every daily power-demand series the model applies to
    (`signal_countries`, so countries excluded from power signals are left out, PLAN D22)."""
    out = []
    for ind in registry.indicators.values():
        if ind.signal_group == "power_demand" and ind.frequency == "D":
            out += [(ind.id, c) for c in registry.signal_countries(ind)]
    return out


def _series(obs: pd.DataFrame, indicator: str, country: str) -> pd.Series:
    part = obs[(obs["indicator"] == indicator) & (obs["country"] == country)]
    return pd.Series(part["value"].to_numpy(), index=pd.DatetimeIndex(part["date"])).sort_index()


def weather_share(yoy: pd.Series, hdd: pd.Series, cdd: pd.Series) -> float:
    """Share of a YoY series' variance explained by the YoY change in 28-day mean HDD and CDD
    (weekly sample). Near 0 for well-normalised demand."""
    def change(w: pd.Series) -> pd.Series:
        m = tf.clean(w).rolling("28D", min_periods=20).mean()
        return m - pd.Series(m.reindex(m.index - pd.Timedelta(days=tf.DAILY_LAG_DAYS)).to_numpy(), index=m.index)

    frame = pd.concat([tf.clean(yoy).rename("y"), change(hdd).rename("h"), change(cdd).rename("c")],
                      axis=1, sort=True).dropna()
    frame = frame[frame.index.dayofweek == 0]
    if len(frame) < 20:
        return float("nan")
    X = np.column_stack([np.ones(len(frame)), frame["h"], frame["c"]])
    beta = np.linalg.lstsq(X, frame["y"].to_numpy(), rcond=None)[0]
    resid = frame["y"].to_numpy() - X @ beta
    return float(1 - resid.var() / frame["y"].var())


def run_models(obs: pd.DataFrame, registry: Registry) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The model on every daily load series in `obs` (with the degree days in it).

    Returns (daily outputs, monthly fits, one summary row per series).
    """
    daily_parts, fit_parts, rows = [], [], []
    for indicator, country in daily_load_series(registry):
        load, hdd, cdd = (_series(obs, i, country) for i in (indicator, wx.HDD, wx.CDD))
        if load.empty or hdd.empty:
            continue
        ctx = context_for(registry, indicator, country)
        model = wx.normalise(load, hdd, cdd, ctx.holiday_calendar, ctx.calendar, ctx.breaks)
        if model.daily.empty:
            continue
        d = model.daily.copy()
        d["yoy_raw"] = tf.yoy_daily(d["actual"], calendar=ctx.calendar, breaks=ctx.breaks)
        d["yoy_normalised"] = wx.normalised_yoy(model, ctx.calendar, ctx.breaks)
        d = d.reset_index().assign(country=country, indicator=indicator)
        daily_parts.append(d)
        fit_parts.append(model.fits.assign(country=country, indicator=indicator))
        rows.append(_summary_row(indicator, country, d, model.fits, hdd, cdd))
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    fits = pd.concat(fit_parts, ignore_index=True) if fit_parts else pd.DataFrame()
    return daily, fits, pd.DataFrame(rows)


def _summary_row(indicator: str, country: str, d: pd.DataFrame, fits: pd.DataFrame,
                 hdd: pd.Series, cdd: pd.Series) -> dict:
    ex2020 = d[d["date"].dt.year != 2020]
    actual = d["actual"]
    yoy = d.set_index("date")
    mape = float((ex2020["anomaly"].abs() / ex2020["actual"]).mean() * 100)
    left = weather_share(yoy["yoy_normalised"], hdd, cdd)
    verdict = ("good" if mape < GOOD[0] and left < GOOD[1]
               else "fair" if mape < FAIR[0] and left < FAIR[1] else "poor")
    return {
        "indicator": indicator, "country": country, "first": d["date"].min().date(), "last": d["date"].max().date(),
        "r2_in_sample": float(fits["r2"].median()),
        "r2_out_of_sample": float(1 - (d["anomaly"] ** 2).sum() / ((actual - actual.mean()) ** 2).sum()),
        "mape": float((d["anomaly"].abs() / actual).mean() * 100), "mape_ex2020": mape,
        "hdd_pct": float((fits["b_hdd"] / fits["mean_load"]).median() * 100),
        "cdd_pct": float((fits["b_cdd"] / fits["mean_load"]).median() * 100),
        "weather_in_raw_yoy": weather_share(yoy["yoy_raw"], hdd, cdd), "weather_in_normalised_yoy": left,
        "sd_yoy_raw": float(yoy["yoy_raw"].std()), "sd_yoy_normalised": float(yoy["yoy_normalised"].std()),
        "verdict": verdict,
    }


# ------------------------------------------------------------------- report


def _weekly(s: pd.Series) -> pd.Series:
    return s.resample("W-MON", label="left", closed="left").mean().dropna()


def country_figure(country: str, name: str, daily: pd.DataFrame, fits: pd.DataFrame, registry: Registry):
    """Per country: for each load series, weekly actual vs expected load, raw vs normalised
    28-day YoY, and the weather coefficients of each monthly refit."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    indicators = sorted(daily.loc[daily["country"] == country, "indicator"].unique(),
                        key=lambda i: "overnight" in i)
    titles = []
    for ind in indicators:
        label = registry.indicator(ind).name
        titles += [f"{label}: weekly mean, actual and model ({registry.indicator(ind).unit})",
                   f"{label}: 28-day YoY, raw and weather-normalised (%)",
                   f"{label}: weather coefficients per refit (% of mean load per degree day)"]
    fig = make_subplots(rows=3 * len(indicators), cols=1, subplot_titles=titles, vertical_spacing=0.045)
    for n, ind in enumerate(indicators):
        d = daily[(daily["country"] == country) & (daily["indicator"] == ind)].set_index("date")
        f = fits[(fits["country"] == country) & (fits["indicator"] == ind)]
        rows = [
            [("Actual", _weekly(d["actual"])), ("Model (expected)", _weekly(d["expected"]))],
            [("Raw YoY", d["yoy_raw"].dropna()[::7]), ("Weather-normalised YoY", d["yoy_normalised"].dropna()[::7])],
            [("Heating (HDD)", (f["b_hdd"] / f["mean_load"] * 100).set_axis(f["month"])),
             ("Cooling (CDD)", (f["b_cdd"] / f["mean_load"] * 100).set_axis(f["month"]))],
        ]
        for r, pair in enumerate(rows):
            for k, (label, s) in enumerate(pair):
                fig.add_trace(go.Scatter(x=s.index, y=s.round(4).to_numpy(), name=label, mode="lines",
                                         line={"color": COLOURS[k], "width": 2},
                                         legendgroup=f"{r}{k}", showlegend=n == 0),
                              row=3 * n + r + 1, col=1)
    height = 330 * 3 * len(indicators)
    fig.update_layout(
        title={"text": f"{name} ({country}): weather-normalised power demand", "font": {"color": INK},
               "yref": "container", "y": 1 - 12 / height, "yanchor": "top"},
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, hovermode="x unified",
        font={"color": MUTED, "size": 12},
        legend={"orientation": "h", "yref": "container", "y": 1 - 50 / height, "yanchor": "top", "x": 0},
        margin={"l": 60, "r": 20, "t": 130, "b": 40},
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
    return fig


def _markdown(summary: pd.DataFrame, registry: Registry, seconds: float) -> str:
    def fmt(v, digits: int = 2) -> str:
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"

    lines = [
        "# Weather-normalised power demand: fit diagnostics",
        "",
        f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `python -m tracker weather` "
        f"({seconds:.0f} s). Model: `tracker/signals/weather.py` (PLAN §5.4, D45).",
        "",
        "Every value below is **out of sample** except the in-sample R²: each month is predicted by a model "
        "fitted on the 3 years before it (at least 1), so the errors are one-month-ahead errors. Degree days "
        "start in 2019, so outputs start in 2020.",
        "",
        "- **MAPE**: mean absolute daily error as % of actual load (all days; and outside 2020, the lockdowns).",
        "- **HDD, CDD**: median coefficient, as % of mean load per degree day.",
        "- **Weather in YoY**: share of the 28-day YoY's variance explained by the YoY change in 28-day mean "
        "HDD and CDD (weekly sample). Raw shows how much weather moves the raw YoY; normalised should be near 0.",
        f"- **Verdict**: good = MAPE outside 2020 under {GOOD[0]}% and under {GOOD[1]:.0%} weather left; "
        f"fair = under {FAIR[0]}% and {FAIR[1]:.0%}; poor otherwise.",
        "",
        "| Series | Country | From | R² in | R² out | MAPE | MAPE ex-2020 | HDD % | CDD % | Weather in raw YoY "
        "| Weather in normalised YoY | SD raw YoY | SD normalised YoY | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.indicator} | {r.country} | {r.first} | {fmt(r.r2_in_sample)} | {fmt(r.r2_out_of_sample)} "
            f"| {fmt(r.mape)} | {fmt(r.mape_ex2020)} | {fmt(r.hdd_pct)} | {fmt(r.cdd_pct)} "
            f"| {fmt(r.weather_in_raw_yoy * 100, 0)}% | {fmt(r.weather_in_normalised_yoy * 100, 0)}% "
            f"| {fmt(r.sd_yoy_raw, 1)} | {fmt(r.sd_yoy_normalised, 1)} | {r.verdict} |")
    countries = sorted(summary["country"].unique())
    lines += ["", "Charts per country: " + ", ".join(f"[{c}](weather_{c}.html)" for c in countries) + "."]
    return "\n".join(lines) + "\n"


def write_report(daily: pd.DataFrame, fits: pd.DataFrame, summary: pd.DataFrame, registry: Registry,
                 out_dir: Path = REPORT_DIR, seconds: float = 0.0) -> list[Path]:
    """Write weather_fit.md and one chart page per country; returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "weather_fit.md"]
    paths[0].write_text(_markdown(summary, registry, seconds), encoding="utf-8")
    for country in sorted(summary["country"].unique()):
        fig = country_figure(country, registry.countries[country].name, daily, fits, registry)
        path = out_dir / f"weather_{country}.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
        paths.append(path)
    return paths


def run_weather(report: bool = True, data_dir: Path | None = None, out_dir: Path = REPORT_DIR) -> dict:
    """Read the store, run every model, write data/signals/weather.parquet and (optionally) the report."""
    started = time.time()
    registry = load_registry()
    series = daily_load_series(registry)
    indicators = sorted({i for i, _ in series} | {wx.HDD, wx.CDD})
    countries = sorted({c for _, c in series})
    obs = store.latest(indicators=indicators, countries=countries, data_dir=data_dir)
    daily, fits, summary = run_models(obs, registry)
    computed_at = pd.Timestamp.now(tz="UTC")
    if not daily.empty:
        store.write_signal_table(WEATHER_TABLE, daily.assign(computed_at=computed_at), data_dir)
    paths = write_report(daily, fits, summary, registry, out_dir, time.time() - started) if report and len(summary) else []
    return {"series": len(summary), "rows": len(daily), "summary": summary, "paths": paths,
            "seconds": time.time() - started}
