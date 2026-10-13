"""Workbench (S17c): presets, STL precompute, chain and spectrum charts, and the sidebar chain builder."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

from dashboard import charts, presets, workbench
from tests.test_dashboard import chart_specs, run_app, small_lake  # noqa: F401  (fixture)
from tracker import precompute, settings, store
from tracker.registry import load_registry
from tracker.signals import chain as ch
from tracker.signals import transforms as tf
from tracker.signals.moving_holidays import holiday_windows

REG = load_registry()


# -------------------------------------------------------------------- presets


def test_default_presets_exist_even_without_a_file(tmp_path) -> None:
    loaded, problems = presets.load_presets(tmp_path / "none.yaml")
    assert loaded == presets.DEFAULT_PRESETS and problems == []
    assert list(loaded) == ["Raw", "Smoothed 28d", "Seasonally adjusted", "Business-cycle band",
                            "Short cycle (2-18 months)"]
    for text in loaded.values():
        assert str(ch.parse(text)) == text  # canonical chain strings


def test_the_shipped_presets_file_matches_the_defaults() -> None:
    shipped = yaml.safe_load((settings.CONFIG_DIR / "presets.yaml").read_text(encoding="utf-8"))["presets"]
    assert {p["name"]: p["chain"] for p in shipped} == presets.DEFAULT_PRESETS


def test_save_and_delete_custom_presets(tmp_path) -> None:
    path = tmp_path / "presets.yaml"
    assert presets.save_preset("  My   chain ", "stl > sma(28d) > rz(3y)", path) == "stl>sma(28d)>rz(3y)"
    loaded, _ = presets.load_presets(path)
    assert loaded["My chain"] == "stl>sma(28d)>rz(3y)" and list(loaded)[:5] == list(presets.DEFAULT_PRESETS)
    presets.save_preset("My chain", "yoy", path)  # replaces
    assert presets.load_presets(path)[0]["My chain"] == "yoy"
    with pytest.raises(ValueError, match="default preset"):
        presets.save_preset("Raw", "yoy", path)
    with pytest.raises(ValueError, match="name"):
        presets.save_preset("  ", "yoy", path)
    with pytest.raises(ch.ChainError):
        presets.save_preset("Bad", "sma(28x)", path)
    presets.delete_preset("My chain", path)
    assert "My chain" not in presets.load_presets(path)[0]
    with pytest.raises(ValueError, match="can't be deleted"):
        presets.delete_preset("Raw", path)


def test_a_broken_preset_is_reported_and_kept_in_the_file(tmp_path) -> None:
    path = tmp_path / "presets.yaml"
    path.write_text("presets:\n  - name: Typo\n    chain: sma(28x)\n", encoding="utf-8")
    loaded, problems = presets.load_presets(path)
    assert "Typo" not in loaded and "preset 'Typo' skipped" in problems[0]
    presets.save_preset("Good", "yoy", path)
    names = [e["name"] for e in yaml.safe_load(path.read_text(encoding="utf-8"))["presets"]]
    assert names == list(presets.DEFAULT_PRESETS) + ["Typo", "Good"]  # the typo isn't lost


# -------------------------------------------------------------- STL precompute


def test_decompose_series_fits_raw_and_holiday_adjusted_inputs() -> None:
    dates = pd.date_range("2019-01-01", "2023-06-30", freq="D")
    s = pd.Series(100 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25), index=dates)
    windows = holiday_windows("DE", ["easter"], 2019, 2023)
    s[tf.holiday_days(windows, dates[0], dates[-1]).to_numpy()] *= 0.5
    ctx = ch.Context(frequency="D", unit="ships", holiday_windows=windows)
    parts = precompute.decompose_series(s, ctx)
    assert set(parts) == {"raw", "hol"}
    assert np.allclose(parts["hol"]["observed"], ch.apply("hol", s, ctx).values)
    no_holidays = precompute.decompose_series(s, ch.Context(frequency="D", unit="ships"))
    assert set(no_holidays) == {"raw"}
    assert precompute.decompose_series(s.iloc[:500], ctx) == {}  # under 3 years


def test_precompute_writes_a_table_the_stl_step_uses() -> None:
    monthly = pd.date_range("2019-01-01", "2026-06-01", freq="MS")
    rows = pd.DataFrame({"date": monthly, "value": 100 + 5 * np.sin(np.arange(len(monthly)))})
    ind = REG.indicator("comtrade_exports")
    rows = rows.assign(country="CHN", indicator=ind.id, unit=ind.unit, frequency=ind.frequency, source=ind.source,
                       fetched_at=pd.Timestamp("2026-07-01", tz="UTC"), available_at=pd.Timestamp("2026-07-01", tz="UTC"),
                       vintage_type="live")
    short = rows.iloc[:20].assign(country="DEU")
    store.append_changes(pd.concat([rows, short])[store.COLUMNS], "r1")
    result = precompute.precompute_stl(REG, workers=1)
    assert (result["fitted"], result["holiday_adjusted"], result["too_short"]) == (1, 1, 1)  # CHN: Lunar New Year
    table = store.read_signal_table(precompute.STL_TABLE)
    assert list(table.columns) == precompute.STL_COLUMNS and set(table["variant"]) == {"raw", "hol"}
    frames = precompute.stl_frames(table)[("comtrade_exports", "CHN")]
    values = pd.Series(rows["value"].to_numpy(), index=monthly)
    ctx = ch.series_context(REG, "comtrade_exports", "CHN", values)
    live = ch.apply("hol>stl", values, ctx).values
    fake = [f.assign(seasonal=f["seasonal"] + 1.0) for f in frames]  # marked, to see it being used
    used = ch.apply("hol>stl", values, ch.series_context(REG, "comtrade_exports", "CHN", values, {"stl": fake})).values
    assert np.allclose(used, live - 1.0)


# ---------------------------------------------------------------------- charts


def test_chain_figure_dashes_the_provisional_end_and_draws_what_was_removed() -> None:
    dates = pd.date_range("2015-01-01", periods=60, freq="MS")
    s = pd.Series(100 + 10 * np.sin(2 * np.pi * dates.month / 12) + np.arange(60.0), index=dates)
    result = ch.apply("stl", s, ch.Context(frequency="M", unit="TWh"))
    fig = charts.chain_figure(result, "Demand", "TWh")
    names = [t.name for t in fig.data]
    assert names == ["Raw", "stl", "Provisional", "Removed by stl"]
    assert fig.data[2].line.dash == "dash" and fig.data[2].x[0] == dates[-13]  # joins the solid line
    assert len(fig.data[2].x) == 13 and fig.data[3].yaxis == "y2"
    z = ch.apply("yoy>rz(1y)", s * np.linspace(1, 3, 60), ch.Context(frequency="M", unit="TWh"))
    zfig = charts.chain_figure(z, "Demand", "robust z")
    assert [t.name for t in zfig.data] == ["yoy>rz(1y)"]  # no raw (units changed), nothing removed
    assert max(np.abs(zfig.data[0].y)) <= charts.Z_CLIP


def test_axis_labels_follow_the_chain() -> None:
    assert charts.chain_axis_label(ch.parse("stl>sma(28d)"), "ships") == "ships"
    assert charts.chain_axis_label(ch.parse("stl>yoy"), "ships") == "YoY %"
    assert charts.chain_axis_label(ch.parse("yoy"), "degC") == "YoY change (degC)"
    assert charts.chain_axis_label(ch.parse("hp>yoy"), "TWh") == "YoY change (cycle (TWh))"
    assert charts.chain_axis_label(ch.parse("log>cf"), "USD") == "cycle (log USD)"
    assert charts.chain_axis_label(ch.parse("yoy>rz(3y)"), "USD") == "robust z"


def test_periodogram_finds_a_three_year_cycle_and_the_band_is_shaded() -> None:
    dates = pd.date_range("1996-01-01", periods=360, freq="MS")
    s = pd.Series(5 * np.sin(2 * np.pi * np.arange(360) / 36) + 0.01 * np.arange(360), index=dates)
    spec = charts.periodogram(s, "M")
    assert spec["period_years"].iloc[spec["share"].argmax()] == pytest.approx(3.0)
    assert spec["share"].sum() == pytest.approx(1.0)
    fig = charts.spectrum_figure(spec, spec, "x", band=charts.band_years(ch.parse("cf(1.5y,8y)")))
    assert fig.layout.xaxis.type == "log" and len(fig.layout.shapes) == 1
    assert charts.band_years(ch.parse("stl>sma(3m)")) is None


def test_hypothesis_line_quotes_the_chain() -> None:
    text = workbench.hypothesis_text("[indicator] DEU", ch.parse("stl>sma(28d)>rz(3y)"), date(2026, 9, 27))
    assert text.startswith("- **Hn** [indicator] DEU, transform `stl>sma(28d)>rz(3y)` → [target] | [lead time]")
    assert "Written: 2026-09-27." in text


# ------------------------------------------------------------------- the pages


def chain_app(country: str = "DEU"):
    at = run_app(country)
    at.radio[0].set_value(charts.CHAIN).run()
    assert not at.exception
    return at


def sidebar_button(at, label: str):
    return next(b for b in at.sidebar.button if b.label == label)


def test_every_preset_renders_on_the_country_page(small_lake) -> None:
    at = chain_app()
    preset = at.sidebar.selectbox[0]
    assert preset.value == workbench.DEFAULT_PRESET
    assert list(preset.options) == list(presets.DEFAULT_PRESETS)
    for name, text in presets.DEFAULT_PRESETS.items():
        at.sidebar.selectbox[0].select(name).run()
        assert not at.exception, name
        assert at.sidebar.text_input[0].value == text
        assert len(chart_specs(at)) == 5, name
    # Days on monthly data run as the nearest whole months (PLAN D37), and an expander says so.
    at.sidebar.selectbox[0].select("Smoothed 28d").run()
    labels = [e.label for e in at.expander]
    assert "Lengths converted on 2 chart(s)" in labels
    assert not any(label.startswith("Chain not applied") for label in labels)
    note = next(e for e in at.expander if e.label.startswith("Lengths converted")).markdown[0].value
    assert "`sma(28d)` → `sma(1m)` (monthly data)" in note
    specs = chart_specs(at)
    exports = json.loads(next(s for s in specs if "portwatch_exports" in s))
    assert [t["name"] for t in exports["data"]] == ["Raw", "sma(28d)", "Removed by sma(28d)"]
    ip = json.loads(next(s for s in specs if "eurostat_ip" in s))
    assert "sma(1m)" in [t["name"] for t in ip["data"]]
    assert "lengths converted" in json.dumps(ip["layout"]["title"])


def test_a_custom_three_step_chain_is_built_and_saved(small_lake) -> None:
    at = chain_app()
    at.sidebar.selectbox[0].select("Raw").run()
    sidebar_button(at, "Remove").click().run()  # the empty chain
    assert at.session_state[workbench.CHAIN_KEY] == ""
    for key in ["hol", "sma", "rz"]:
        at.sidebar.selectbox[-1].select(key).run()  # the "Add a step" box
        sidebar_button(at, "Add").click().run()
        assert not at.exception
    assert at.session_state[workbench.CHAIN_KEY] == "hol>sma>rz"
    sma_window = next(t for t in at.sidebar.text_input if t.label == "n")
    sma_window.set_value("14d").run()
    rz_window = next(t for t in at.sidebar.text_input if t.label == "window")
    rz_window.set_value("2y").run()
    assert at.session_state[workbench.CHAIN_KEY] == "hol>sma(14d)>rz(2y)"
    downs = [b for b in at.sidebar.button if b.label == "Down"]
    downs[0].click().run()  # move hol below sma
    assert at.session_state[workbench.CHAIN_KEY] == "sma(14d)>hol>rz(2y)"
    assert at.sidebar.selectbox[0].value == workbench.CUSTOM
    assert len(chart_specs(at)) == 5 and not at.exception

    next(t for t in at.sidebar.text_input if t.label == "Save as preset").set_value("Smooth z").run()
    sidebar_button(at, "Save").click().run()
    assert not at.exception
    assert at.sidebar.success[0].value.startswith("Saved preset 'Smooth z'")
    saved = yaml.safe_load(settings.PRESETS_FILE.read_text(encoding="utf-8"))["presets"]
    assert {"name": "Smooth z", "chain": "sma(14d)>hol>rz(2y)"} in saved
    assert at.sidebar.selectbox[0].value == "Smooth z"
    # A new session (a restarted dashboard) still has it.
    again = chain_app()
    assert "Smooth z" in again.sidebar.selectbox[0].options
    again.sidebar.selectbox[0].select("Smooth z").run()
    assert again.session_state[workbench.CHAIN_KEY] == "sma(14d)>hol>rz(2y)"


def test_typed_chains_bad_parameters_and_weather_are_explained(small_lake) -> None:
    at = chain_app()
    at.sidebar.text_input[0].set_value("yoy>ema(3m)").run()
    assert at.session_state[workbench.CHAIN_KEY] == "yoy>ema(3m)" and not at.exception
    at.sidebar.text_input[0].set_value("yoy|ema").run()
    assert "don't use '|'" in at.sidebar.error[0].value
    assert at.session_state[workbench.CHAIN_KEY] == "yoy>ema(3m)"  # unchanged
    next(t for t in at.sidebar.text_input if t.label == "n").set_value("28x").run()
    assert "is not a length" in at.sidebar.error[0].value
    # Weather adjustment (S18) can be added, but after `yoy` it has no load to model: every chart
    # says why the chain wasn't applied.
    at.sidebar.selectbox[-1].select("wx").run()
    sidebar_button(at, "Add").click().run()
    assert at.session_state[workbench.CHAIN_KEY] == "yoy>ema(3m)>wx" and not at.exception
    reasons = " ".join(m.value for e in at.expander if e.label.startswith("Chain not applied") for m in e.markdown)
    assert "weather adjustment needs a positive series" in reasons
    assert "only applies to power-demand indicators" in reasons


def test_real_time_only_hides_two_sided_steps(small_lake) -> None:
    at = chain_app()
    at.sidebar.text_input[0].set_value("hol>sma(28d)").run()
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert any("Real-time only: hidden `hol`" in c.value for c in at.caption)
    exports = json.loads(next(s for s in chart_specs(at) if "portwatch_exports" in s))
    assert [t["name"] for t in exports["data"]][1] == "sma(28d)"  # drawn without hol
    addable = at.sidebar.selectbox[-1].options
    assert not any(f"({k})" in o for k in ("stl", "hol", "cf", "prf") for o in addable)
    assert any("(sma)" in o for o in addable)
    assert "Hidden: real-time only is on." in [c.value for c in at.sidebar.caption]


def test_spectrum_view_and_hypothesis_popover(small_lake) -> None:
    at = chain_app()
    at.sidebar.text_input[0].set_value("cf(2m,18m)").run()
    at.sidebar.toggle[1].set_value(True).run()
    assert not at.exception
    spec = json.loads(next(s for s in chart_specs(at) if "portwatch_exports" in s))
    assert spec["layout"]["xaxis"]["type"] == "log" and len(spec["layout"]["shapes"]) == 1
    assert "transform `cf(2m,18m)`" in at.sidebar.code[0].value
    assert "[indicator] DEU" in at.sidebar.code[0].value


def test_indicator_page_chain_view(small_lake) -> None:
    at = run_app()
    at.switch_page("views/indicator.py").run()
    at.radio[0].set_value(charts.CHAIN).run()
    assert not at.exception
    assert at.sidebar.selectbox[0].value == workbench.DEFAULT_PRESET
    at.sidebar.selectbox[0].select("Smoothed 28d").run()
    assert not at.exception and len(chart_specs(at)) == 2  # ranking bars and DEU's panel
    assert "portwatch_exports [country]" in at.sidebar.code[0].value


def test_indicator_page_lists_converted_lengths(small_lake) -> None:
    at = run_app()
    at.switch_page("views/indicator.py").run()
    at.selectbox(key="indicator").select("eurostat_ip").run()
    at.radio[0].set_value(charts.CHAIN).run()
    at.sidebar.selectbox[0].select("Smoothed 28d").run()
    assert not at.exception
    note = next(e for e in at.expander if e.label == "Lengths converted on 1 chart(s)")
    assert note.markdown[0].value == "- `sma(28d)` → `sma(1m)` (monthly data): **Germany**"


def test_chain_figure_marks_known_breaks_in_the_main_panel() -> None:
    dates = pd.date_range("2013-01-01", periods=84, freq="MS")
    s = pd.Series(100 + 10 * np.sin(2 * np.pi * dates.month / 12) + np.arange(84.0), index=dates)
    result = ch.apply("stl", s, ch.Context(frequency="M", unit="TWh"))
    fig = charts.chain_figure(result, "Demand", "TWh", breaks=[(pd.Timestamp("2016-01-01"), "New definition")])
    [marker] = [t for t in fig.data if t.name == "Known break"]
    assert marker.yaxis == "y" and list(marker.x) == [pd.Timestamp("2016-01-01")] * 2
    assert "New definition" in marker.hovertemplate and marker.line.dash == "dash"
