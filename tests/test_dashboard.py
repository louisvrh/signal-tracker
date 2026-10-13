"""Dashboard: display transforms, figures, and every page loading (Streamlit AppTest).

The pages run against a small synthetic lake in the test's isolated data folder
(conftest), never the real data/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import charts
from tracker import quality, settings, store
from tracker.registry import load_registry

APP = str(settings.ROOT / "dashboard" / "app.py")
REG = load_registry()
T0 = pd.Timestamp("2026-07-01", tz="UTC")
DAILY = pd.date_range("2024-01-01", "2026-06-30", freq="D")
MONTHLY = pd.date_range("2023-01-01", "2026-05-01", freq="MS")


def obs(indicator: str, country: str, dates, values) -> pd.DataFrame:
    """Store-schema rows for one series (unit and frequency from the registry)."""
    ind = REG.indicator(indicator)
    df = pd.DataFrame({"date": pd.to_datetime(list(dates)), "value": np.asarray(values, dtype=float)})
    df["country"], df["indicator"], df["unit"] = country, indicator, ind.unit
    df["frequency"], df["source"] = ind.frequency, ind.source
    df["fetched_at"] = df["available_at"] = T0
    df["vintage_type"] = "live"
    return df[store.COLUMNS]


def weekly_pattern(dates: pd.DatetimeIndex, level: float = 100.0) -> np.ndarray:
    """Weekend dips on top of a flat level."""
    return level * np.where(dates.dayofweek >= 5, 0.6, 1.0)


@pytest.fixture
def small_lake() -> None:
    """DEU port calls and exports, official exports, IP and temperature; Suez transits; a fetch
    log; one quality warning."""
    rows = [
        obs("portwatch_exports", "DEU", DAILY, weekly_pattern(DAILY)),
        obs("portwatch_portcalls", "DEU", DAILY, weekly_pattern(DAILY, 50)),
        obs("portwatch_transits", "CHK_SUEZ", DAILY, np.full(len(DAILY), 40.0)),
        obs("comtrade_exports", "DEU", MONTHLY, np.linspace(100, 130, len(MONTHLY))),
        obs("eurostat_ip", "DEU", MONTHLY, np.linspace(95, 100, len(MONTHLY))),
        obs("openmeteo_temp_mean", "DEU", DAILY, 10 + 8 * np.sin(np.arange(len(DAILY)) / 58)),
    ]
    store.append_changes(pd.concat(rows, ignore_index=True), "r1")
    log = pd.DataFrame([{
        "run_id": "r1", "source": s, "started_at": T0, "finished_at": T0 + pd.Timedelta(seconds=30),
        "status": "success", "rows_fetched": 10, "rows_written": 10, "error": None,
    } for s in ("portwatch", "comtrade", "eurostat", "openmeteo")])
    store.append_fetch_log(log)
    issue = pd.DataFrame([{"source": "portwatch", "indicator": "portwatch_exports", "country": "DEU",
                           "check": "freshness", "severity": "warning", "detail": "latest 2026-06-30"}])
    quality.record(issue, now=T0)


def run_app(country: str | None = None, page: str = "views/country.py") -> AppTest:
    """The app on one page (the Country page unless told otherwise; Overview is the landing page, S20)."""
    at = AppTest.from_file(APP, default_timeout=60)
    if country:
        at.query_params["country"] = country
    at.switch_page(page)
    return at.run()


def chart_specs(at: AppTest) -> list[str]:
    return [el.proto.spec for el in at.get("plotly_chart")]


# ----------------------------------------------------------------- transforms


def test_monthly_yoy_is_percent_against_twelve_months_earlier() -> None:
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    values = pd.Series([100.0] * 12 + [110.0] * 12, index=dates)
    out = charts.yoy(values, "M", "USD")
    assert out.iloc[:12].isna().all()
    assert np.allclose(out.iloc[12:], 10.0)


def test_interval_units_use_differences() -> None:
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    temps = pd.Series([2.0] * 12 + [3.5] * 12, index=dates)
    assert np.allclose(charts.yoy(temps, "M", "degC").iloc[12:], 1.5)
    assert charts.yoy_label("degC", "D") == "YoY change (degC), 28-day avg"
    assert charts.yoy_label("USD", "M") == "YoY %"


def test_daily_yoy_keeps_weekdays_aligned() -> None:
    # Same weekly pattern each year, 5% higher in the second year: a 364-day
    # comparison of 28-day averages gives exactly 5% on every day.
    dates = pd.date_range("2024-01-01", periods=728, freq="D")  # two 364-day years
    values = pd.Series(weekly_pattern(dates) * np.where(np.arange(728) >= 364, 1.05, 1.0), index=dates)
    out = charts.yoy(values, "D", "metric tons")
    settled = out[dates >= dates[364 + 27]]
    assert np.allclose(settled, 5.0)
    assert out[dates < dates[364]].isna().all()


def test_percent_change_from_zero_or_negative_base_is_missing() -> None:
    dates = pd.date_range("2024-01-01", periods=13, freq="MS")
    values = pd.Series([0.0] + [5.0] * 11 + [7.0], index=dates)
    assert np.isnan(charts.yoy(values, "M", "TWh").iloc[12])


def test_holiday_windows_cover_whole_months_on_monthly_charts() -> None:
    windows = pd.DataFrame({"holiday": ["easter"], "start": [pd.Timestamp("2025-04-17")],
                            "end": [pd.Timestamp("2025-04-21")]})
    monthly = charts.period_windows(windows, "M")
    assert (monthly["start"].iloc[0], monthly["end"].iloc[0]) == (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-05-01"))
    daily = charts.period_windows(windows, "D")
    assert daily["end"].iloc[0] == pd.Timestamp("2025-04-22")


def test_panel_figure_legend_only_with_an_official_series() -> None:
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    s = pd.Series(np.arange(60.0), index=dates)
    windows = pd.DataFrame({"holiday": ["x", "y"], "start": [dates[5], pd.Timestamp("2030-01-01")],
                            "end": [dates[7], pd.Timestamp("2030-01-02")]})
    plain = charts.panel_figure(s, "Port exports", "tons", raw=s, windows=windows)
    assert not plain.layout.showlegend
    assert len(plain.layout.shapes) == 1  # the window outside the data's range is left out
    both = charts.panel_figure(s, "Port exports", "YoY %", official=s * 2, official_name="Goods exports", zero_line=True)
    assert both.layout.showlegend
    assert [t.name for t in both.data] == ["Port exports", "Goods exports"]
    assert len(both.layout.shapes) == 1  # the zero line


def test_latest_yoy_ranks_countries_and_puts_missing_last() -> None:
    dates = pd.date_range("2024-01-01", periods=13, freq="MS")
    rows = pd.concat([
        pd.DataFrame({"date": dates, "country": "AAA", "value": [100.0] * 12 + [105.0]}),
        pd.DataFrame({"date": dates, "country": "BBB", "value": [100.0] * 12 + [120.0]}),
        pd.DataFrame({"date": dates[:6], "country": "CCC", "value": [1.0] * 6}),
    ])
    table = charts.latest_yoy(rows, "M", "USD")
    assert table["country"].tolist() == ["BBB", "AAA", "CCC"]
    assert np.allclose(table["yoy"].iloc[:2], [20.0, 5.0]) and np.isnan(table["yoy"].iloc[2])
    assert table["yoy_date"].iloc[0] == dates[-1]
    assert table["latest_date"].iloc[2] == dates[5]


def test_seasonal_band_uses_the_five_years_before() -> None:
    dates = pd.date_range("2019-01-01", "2026-06-30", freq="D")
    values = pd.Series(dates.year.to_numpy() - 2000.0, index=dates)  # constant within each year
    band, years = charts.seasonal_band(values, 2026, smooth_days=1)
    assert years == [2021, 2022, 2023, 2024, 2025]
    day = pd.Timestamp("2001-03-15")
    assert (band.loc[day, "min"], band.loc[day, "max"], band.loc[day, "mean"], band.loc[day, "current"]) == (
        21.0, 25.0, 23.0, 26.0)
    assert pd.isna(band.loc[pd.Timestamp("2001-08-01"), "current"])  # 2026 not there yet
    assert len(band) == 365  # 29 February dropped
    short, years = charts.seasonal_band(values, 2021, smooth_days=1)
    assert years == [2019, 2020]


def test_band_position() -> None:
    idx = pd.date_range("2001-01-01", periods=3, freq="D")
    band = pd.DataFrame({"min": [1.0, 1.0, 1.0], "max": [3.0, 3.0, 3.0], "mean": [2.0, 2.0, 2.0],
                         "current": [2.5, 0.5, np.nan]}, index=idx)
    pos = charts.band_position(band)
    assert pos["ref"] == idx[1] and pos["position"] == "below range" and pos["vs_mean"] == -75.0
    assert charts.band_position(band.assign(current=[np.nan, 4.0, np.nan]))["position"] == "above range"
    assert charts.band_position(band.assign(current=np.nan)) is None


def test_every_seventh_day_ends_on_the_latest_day() -> None:
    s = pd.Series(np.arange(20.0), index=pd.date_range("2025-01-01", periods=20, freq="D"))
    thin = charts.every_seventh_day(s)
    assert thin.index[-1] == s.index[-1] and thin.tolist() == [5.0, 12.0, 19.0]


def test_ranking_and_heatmap_figures() -> None:
    bars = charts.ranking_figure(["A", "B"], [3.0, -2.0], "YoY %")
    assert list(bars.data[0].marker.color) == [charts.POSITIVE_COLOR, charts.NEGATIVE_COLOR]
    heat = charts.yoy_heatmap(pd.DataFrame({"Port calls": [5.0, -40.0]}, index=["A", "B"]), dark=True)
    assert heat.data[0].zmid == 0 and heat.data[0].colorscale[1][1] == charts.NEUTRAL_DARK


# ---------------------------------------------------------------------- pages


def test_country_page_shows_every_stored_indicator(small_lake) -> None:
    at = run_app("DEU")
    assert not at.exception
    assert at.selectbox[0].value == "DEU"
    specs = chart_specs(at)
    assert len(specs) == 5
    assert [s.value for s in at.subheader] == ["Trade", "Official statistics", "Weather"]
    assert any("Shaded: Easter" in c.value for c in at.caption)


def test_country_page_yoy_overlays_the_official_series(small_lake) -> None:
    at = run_app("DEU")
    at.radio[0].set_value(charts.YOY).run()
    assert not at.exception
    port = next(s for s in chart_specs(at) if "portwatch_exports" in s)
    assert "Goods exports (official)" in port


def test_country_selection_sticks_and_empty_countries_say_so(small_lake) -> None:
    at = run_app("DEU")
    for country in ["USA", "DEU", "JPN", "DEU"]:
        at.selectbox[0].select(country).run()
        assert not at.exception
        assert at.selectbox[0].value == country
        assert at.query_params["country"] == [country]
        if country == "DEU":
            assert len(chart_specs(at)) == 5
        else:
            assert "No data stored" in at.info[0].value


def test_data_health_page(small_lake) -> None:
    at = run_app()
    at.switch_page("views/data_health.py").run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Open warnings"] == "1"
    assert metrics["Open errors"] == "0"
    assert metrics["Last fetch run (UTC)"] == "01 Jul 00:00"
    assert [s.value for s in at.subheader] == ["Freshness per source", "Fetch log", "Open quality issues"]


def test_indicator_page_ranks_and_draws_every_country(small_lake) -> None:
    at = run_app()
    at.switch_page("views/indicator.py").run()
    assert not at.exception
    assert at.selectbox[0].value == "portwatch_exports"
    ranking = at.dataframe[0].value
    assert ranking["country"].tolist() == ["DEU"] and ranking["rank"].tolist() == [1]
    assert len(chart_specs(at)) == 2  # the ranking bars and DEU's panel
    at.selectbox[0].select("eurostat_ip").run()
    at.radio[0].set_value(charts.YOY).run()
    assert not at.exception
    assert at.query_params["indicator"] == ["eurostat_ip"]
    assert at.dataframe[0].value["yoy"].notna().all()


def test_shipping_page_bands_and_port_heatmap(small_lake) -> None:
    at = run_app()
    at.switch_page("views/shipping.py").run()
    assert not at.exception
    assert at.selectbox[1].value == 2026 and at.selectbox[1].options == ["2026", "2025"]
    summary = at.dataframe[0].value
    assert summary["chokepoint"].tolist() == ["Suez Canal"] and summary["position"].tolist() == ["in range"]
    assert any("2024–2025" in c.value for c in at.caption)
    specs = chart_specs(at)
    assert len(specs) == 1 + 1 + 2  # Suez band, port heatmap, DEU port calls and exports (no imports stored)
    at.selectbox[0].select("portwatch_tanker_transits").run()
    assert not at.exception
    assert "No chokepoint data" in at.info[0].value


def test_pages_load_on_an_empty_store() -> None:
    at = run_app()
    assert not at.exception
    assert "No data stored" in at.info[0].value
    for page, message in [("views/indicator.py", "No data stored yet."),
                          ("views/shipping.py", "No PortWatch data stored yet.")]:
        at.switch_page(page).run()
        assert not at.exception
        assert at.info[0].value == message
    at.switch_page("views/data_health.py").run()
    assert not at.exception
    assert at.success[0].value == "No open issues."


# ------------------------------------------------------------ remote access


def test_streamlit_stays_on_loopback_with_its_protections_on() -> None:
    """Remote access goes through a proxy (PLAN D38): Streamlit itself must
    listen on 127.0.0.1 only, keep CORS and XSRF protection, and use the same port
    as the launcher and the service script."""
    import re
    import tomllib

    config = tomllib.loads((settings.ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    server = config["server"]
    assert server["address"] == "127.0.0.1"
    assert server.get("enableCORS", True) and server.get("enableXsrfProtection", True)
    for script in ("run_dashboard.ps1", "dashboard_service.ps1", "register_dashboard_task.ps1", "check_dashboard.ps1"):
        text = (settings.ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert re.search(rf"\[int\]\$Port = {server['port']}\b", text), script
        assert "0.0.0.0" not in text, script
    # The daily run restarts the dashboard if nothing listens on that same port.
    daily = (settings.ROOT / "scripts" / "run_daily.ps1").read_text(encoding="utf-8")
    assert re.search(rf"\[int\]\$DashboardPort = {server['port']}\b", daily)
    assert "check_dashboard.ps1" in daily


# ------------------------------------------------------------ known structural breaks (PLAN D41)


def test_display_yoy_never_compares_across_a_break() -> None:
    dates = pd.date_range("2015-01-01", periods=36, freq="MS")
    values = pd.Series([100.0] * 12 + [117.0] * 24, index=dates)
    out = charts.yoy(values, "M", "MWh", breaks=[pd.Timestamp("2016-01-01")])
    assert out.iloc[12:24].isna().all() and np.allclose(out.iloc[24:], 0.0)
    daily = pd.date_range("2015-01-01", "2017-06-30", freq="D")
    d = charts.yoy(pd.Series(np.where(daily >= "2016-01-01", 117.0, 100.0), index=daily), "D", "MWh",
                   breaks=[pd.Timestamp("2016-01-01")])
    assert d["2016-01-01":"2017-01-25"].isna().all() and np.allclose(d["2017-01-26":], 0.0)  # 391 days on
    table = charts.latest_yoy(pd.DataFrame({"country": "NLD", "date": dates[:24], "value": values.iloc[:24]}),
                              "M", "MWh", {"NLD": [pd.Timestamp("2016-01-01")]})
    assert np.isnan(table.loc[0, "yoy"])  # the latest month is still within a year of the break


def test_panel_figure_marks_breaks_in_range_with_the_reason_on_hover() -> None:
    dates = pd.date_range("2015-01-01", periods=36, freq="MS")
    main = pd.Series(np.arange(36.0), index=dates)
    fig = charts.panel_figure(main, "Load", "MWh", breaks=[(pd.Timestamp("2016-01-01"), "Reporting <change>"),
                                                          (pd.Timestamp("2010-01-01"), "Out of range")])
    [marker] = [t for t in fig.data if t.name == "Known break"]
    assert marker.hovertemplate.startswith("Break 2016-01-01: Reporting &lt;change&gt;")
    assert (min(marker.y), max(marker.y)) == (0.0, 35.0) and not fig.layout.showlegend


@pytest.fixture
def break_lake() -> None:
    """Dutch daily load with the 2016 reporting step (a known break in indicators.yaml)."""
    days = pd.date_range("2015-01-01", "2017-12-31", freq="D")
    rows = [obs("entsoe_load", "NLD", days, np.where(days >= "2016-01-01", 332_000.0, 283_000.0))]
    store.append_changes(pd.concat(rows, ignore_index=True), "r1")


def test_country_page_marks_the_dutch_break(break_lake) -> None:
    at = run_app("NLD")
    assert not at.exception
    [spec] = chart_specs(at)
    assert "Known break" in spec and "TenneT" in spec
    assert any("known structural break" in c.value for c in at.caption)
    at.radio[0].set_value(charts.YOY).run()
    assert not at.exception and "Known break" in chart_specs(at)[0]


# ------------------------------------------------------------ replay (PLAN D43)


def trace_xy(spec: str, index: int = 0) -> pd.Series:
    """One trace of a Plotly chart spec as a series (y arrays may be base64-encoded)."""
    import base64
    import json

    trace = json.loads(spec)["data"][index]
    y = trace["y"]
    if isinstance(y, dict):
        y = np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"])
    return pd.Series(np.asarray(y, dtype=float), index=pd.to_datetime(trace["x"]))


@pytest.fixture
def vintage_lake() -> None:
    """German IP first published about 47 days after each month (downloaded 2024-08-01); on
    2024-09-10 March is revised from 100 to 110 and June and July arrive."""
    months = pd.date_range("2023-01-01", "2024-05-01", freq="MS")
    first = obs("eurostat_ip", "DEU", months, np.full(len(months), 100.0))
    first["available_at"] = (months + pd.offsets.MonthEnd(0) + pd.Timedelta(days=47)).tz_localize("UTC")
    first["fetched_at"] = pd.Timestamp("2024-08-01", tz="UTC")
    first["vintage_type"] = "estimated"
    store.append_changes(first, "r1")
    later = obs("eurostat_ip", "DEU", pd.to_datetime(["2024-03-01", "2024-06-01", "2024-07-01"]), [110.0, 101.0, 102.0])
    later["fetched_at"] = later["available_at"] = pd.Timestamp("2024-09-10", tz="UTC")
    store.append_changes(later, "r2")


def test_end_of_day_is_the_last_instant_of_the_utc_day() -> None:
    from dashboard import data
    assert data.end_of_day("2024-08-15") == pd.Timestamp("2024-08-15 23:59:59.999999", tz="UTC")


def test_replay_shows_the_data_as_known_on_the_chosen_day(vintage_lake) -> None:
    from datetime import date

    at = run_app("DEU")
    assert not at.exception and not at.warning
    assert trace_xy(chart_specs(at)[0])["2024-03-01"] == 110.0  # today's view: the revised value
    at.toggle(key="replay_on").set_value(True).run()
    at.slider(key="replay_date").set_value(date(2024, 8, 15)).run()
    assert not at.exception
    assert "Replay: Germany as known at the end of 15 August 2024" in at.warning[0].value
    spec = chart_specs(at)[0]
    shown = trace_xy(spec)
    assert shown["2024-03-01"] == 100.0  # the first print; the revision came on 10 September
    assert shown.index.max() == pd.Timestamp("2024-05-01")  # June was published on 16 August
    assert "as known 2024-08-15" in spec
    at.slider(key="replay_date").set_value(date(2024, 9, 20)).run()
    shown = trace_xy(chart_specs(at)[0])
    assert shown["2024-03-01"] == 110.0 and shown.index.max() == pd.Timestamp("2024-07-01")
    at.slider(key="replay_date").set_value(date(2023, 1, 5)).run()
    assert any("Nothing about Germany had been published" in i.value for i in at.info)


def test_replay_keeps_only_one_sided_chain_steps(vintage_lake) -> None:
    from datetime import date

    at = run_app("DEU")
    at.radio[0].set_value(charts.CHAIN).run()
    at.toggle(key="replay_on").set_value(True).run()
    at.slider(key="replay_date").set_value(date(2024, 8, 15)).run()
    assert not at.exception
    forced = at.toggle(key="wb_realtime_forced")
    assert forced.value and forced.disabled
    assert any("Real-time only: hidden `hol`, `stl`" in c.value for c in at.caption)  # the default hol>stl
    at.toggle(key="replay_on").set_value(False).run()
    assert not at.exception and not at.warning
    assert not any("Real-time only: hidden" in c.value for c in at.caption)


def test_replayed_weather_adjustment_uses_degree_days_as_known_then() -> None:
    """`wx` is one-sided, so replay runs it; its degree days must be the ones known on the
    replay day, not later revisions (here: the last 10 days' degree days revised 10 days on)."""
    from datetime import date

    from dashboard import data
    from tracker.signals import chain as chains

    rng = np.random.default_rng(3)
    doy = DAILY.dayofyear.to_numpy()
    temp = 12 - 9 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 3, len(DAILY))
    hdd, cdd = np.maximum(0, 18 - temp), np.maximum(0, temp - 22)
    load = 1e6 + 2e4 * hdd + 3e4 * cdd - 1e5 * (DAILY.dayofweek >= 5) + rng.normal(0, 5e3, len(DAILY))
    store.append_changes(pd.concat([obs("entsoe_load", "DEU", DAILY, load), obs("openmeteo_hdd18", "DEU", DAILY, hdd),
                                    obs("openmeteo_cdd22", "DEU", DAILY, cdd)], ignore_index=True), "r1")
    revised = obs("openmeteo_hdd18", "DEU", DAILY[-10:], hdd[-10:] + 5)
    revised["fetched_at"] = revised["available_at"] = T0 + pd.Timedelta(days=10)
    store.append_changes(revised, "r2")

    values = pd.Series(load, index=DAILY)
    then = data.chain_result("wx", "entsoe_load", "DEU", values, replay_day=date(2026, 7, 2))
    now = data.chain_result("wx", "entsoe_load", "DEU", values)
    ctx = chains.context_for(REG, "entsoe_load", "DEU", extra={"openmeteo_hdd18": pd.Series(hdd, index=DAILY),
                                                                "openmeteo_cdd22": pd.Series(cdd, index=DAILY)})
    expected = chains.apply("wx", values, ctx).values
    pd.testing.assert_series_equal(then.values, expected)
    assert not np.allclose(now.values.iloc[-10:], expected.iloc[-10:])  # today's view has the revision


# ------------------------------------------------------------- overview (S20)


def test_overview_is_the_landing_page_and_waits_for_signals() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Overview"
    assert "python -m tracker signals" in at.info[0].value


def test_overview_numbers_are_the_signals_output() -> None:
    """Heatmap cells, movers and flags come straight from the signals tables (S19), and each
    flag links to its Country page."""
    import json

    from tests.test_signals import synthetic_vintages
    from tracker import daily_signals as ds

    store.append_changes(synthetic_vintages(), "r1")
    ds.run_signals(as_of=pd.Timestamp("2024-08-05", tz="UTC"))
    signals, flags = store.read_signal_table("signals"), store.read_signal_table("flags")
    scores = ds.latest_scores(signals)

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and at.title[0].value == "Overview"
    spec = json.loads(next(s for s in chart_specs(at) if '"heatmap"' in s))["data"][0]
    z = spec["z"]
    if isinstance(z, dict):  # Plotly 6 may encode arrays as base64
        import base64
        z = np.frombuffer(base64.b64decode(z["bdata"]), dtype=z["dtype"]).reshape(z["shape"]
                                                                                  if isinstance(z["shape"], list)
                                                                                  else [int(n) for n in z["shape"].split(",")])
    expected = scores[["trade", "energy", "activity", "activity_z"]].to_numpy(dtype=float)
    assert np.allclose(np.asarray(z, dtype=float), expected, equal_nan=True)
    assert spec["x"] == ["Trade", "Energy", "Score", "Score z"]
    assert [y.split("(")[-1].rstrip(")") for y in spec["y"]] == list(scores.index)
    metrics = {m.label: m.value for m in at.metric}
    active = ds.active_flags(flags, signals)
    assert metrics["Market flags"] == str((active["signal"] == "d_mkt").sum())
    assert metrics["Official flags"] == str((active["signal"] == "d_off").sum())
    links = at.get("page_link")
    assert len(links) == len(active)
    for link, row in zip(links, active.itertuples()):
        assert link.proto.page.endswith("country") and f"country={row.country}" in link.proto.query_string


def test_old_country_links_still_open_the_country_page(small_lake) -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["country"] = "DEU"
    at.run()
    assert not at.exception
    assert at.title[0].value == "Country" and at.selectbox[0].value == "DEU"
