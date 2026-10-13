"""Quality checks: freshness, missing countries, jumps, duplicates, units;
recording, open issues, and the quality/status/compact commands."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from tracker import quality, settings, store
from tracker.registry import load_registry

REG = load_registry()
SOURCES = settings.load_sources()
T0 = pd.Timestamp("2026-01-05", tz="UTC")
T1 = pd.Timestamp("2026-02-05", tz="UTC")


def obs(indicator: str, country: str, dates, values, fetched_at: pd.Timestamp = T0, unit: str | None = None) -> pd.DataFrame:
    """Store-schema rows for one series (unit and frequency from the registry)."""
    ind = REG.indicator(indicator)
    df = pd.DataFrame({"date": pd.to_datetime(list(dates)), "value": np.asarray(values, dtype=float)})
    df["country"] = country
    df["indicator"] = indicator
    df["unit"] = unit or ind.unit
    df["frequency"] = ind.frequency
    df["source"] = ind.source
    df["fetched_at"] = fetched_at
    df["available_at"] = fetched_at
    df["vintage_type"] = "live"
    return df[store.COLUMNS]


def noisy(n: int, level: float = 100.0, sd: float = 5.0, seed: int = 1) -> np.ndarray:
    return level + np.random.default_rng(seed).normal(0, sd, n)


def last_dates(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame([(REG.indicator(i).source, i, c, pd.Timestamp(d)) for i, c, d in rows],
                        columns=["source", "indicator", "country", "last_date"])


# ------------------------------------------------------------------ freshness


def test_freshness_uses_per_country_lag() -> None:
    # ember_demand: 30-day lag, but 60 days for ZAF (sources.yaml). The July
    # figure is due 1 Aug + lag + 7 days' grace.
    dates = last_dates([("ember_demand", "DEU", "2026-06-01"), ("ember_demand", "ZAF", "2026-06-01")])
    issues = quality.check_freshness(dates, REG, SOURCES, now=pd.Timestamp("2026-09-15", tz="UTC"))
    assert issues["country"].tolist() == ["DEU"]
    assert "due by 2026-09-07 (8 days late)" in issues["detail"].iloc[0]


def test_freshness_skips_disabled_sources_and_unlisted_countries() -> None:
    dates = last_dates([("ember_demand", "DEU", "2020-01-01"), ("ember_demand", "VNM", "2020-01-01")])
    now = pd.Timestamp("2026-09-15", tz="UTC")
    assert quality.check_freshness(dates, REG, SOURCES, now)["country"].tolist() == ["DEU"]  # VNM isn't expected
    disabled = {**SOURCES, "ember": {**SOURCES["ember"], "enabled": False}}
    assert quality.check_freshness(dates, REG, disabled, now).empty


def test_retired_indicators_are_never_late_or_missing() -> None:
    # fred_copper stopped being fetched on 2026-10-07 (PLAN D56); its last month stays July 2026.
    now = pd.Timestamp("2027-01-15", tz="UTC")
    dates = last_dates([("fred_copper", "WLD", "2026-07-01"), ("fred_brent", "WLD", "2026-07-01")])
    assert quality.check_freshness(dates, REG, SOURCES, now)["indicator"].tolist() == ["fred_brent"]
    missing = quality.check_missing_countries(last_dates([]), REG, SOURCES)
    assert "fred_brent" in set(missing["indicator"])
    assert not {"fred_copper", "fred_eu_gas"} & set(missing["indicator"])


# ---------------------------------------------------------- missing countries


def test_missing_countries_listed_per_country_or_once_per_indicator() -> None:
    present = [c for c in REG.countries_for("ember_demand") if c != "ZAF"]
    dates = last_dates([("ember_demand", c, "2026-07-01") for c in present])
    issues = quality.check_missing_countries(dates, REG, SOURCES)
    demand = issues[issues["indicator"] == "ember_demand"]
    assert demand["country"].tolist() == ["ZAF"]
    # An indicator with no data at all gets a single issue, not one per country.
    gen = issues[issues["indicator"] == "ember_generation"]
    assert len(gen) == 1 and gen["country"].isna().all() and "any of its 22 countries" in gen["detail"].iloc[0]


def test_no_data_expected_countries_are_not_missing_but_data_is_noticed() -> None:
    # ember_gen_nuclear lists ITA (no nuclear plants) in no_data_expected.
    present = [c for c in REG.countries_for("ember_gen_nuclear") if c not in REG.indicator("ember_gen_nuclear").no_data_expected]
    dates = last_dates([("ember_gen_nuclear", c, "2026-07-01") for c in present])
    issues = quality.check_missing_countries(dates, REG, SOURCES)
    assert issues[issues["indicator"] == "ember_gen_nuclear"].empty
    dates = last_dates([("ember_gen_nuclear", c, "2026-07-01") for c in present + ["ITA"]])
    issues = quality.check_missing_countries(dates, REG, SOURCES)
    nuclear = issues[issues["indicator"] == "ember_gen_nuclear"]
    assert nuclear[["country", "check", "severity"]].values.tolist() == [["ITA", "unexpected_data", "info"]]


# ----------------------------------------------------------------------- gaps


def test_gap_in_a_daily_series() -> None:
    days = pd.date_range("2025-01-01", "2025-02-28").delete(range(10, 15))  # 11-15 Jan missing
    issues = quality.check_gaps(obs("portwatch_portcalls", "DEU", days, noisy(len(days))), REG)
    assert issues["detail"].tolist() == ["5 missing days in 1 gap(s): 2025-01-11 to 2025-01-15 (5)"]
    assert issues["check"].tolist() == ["gap"] and issues["severity"].tolist() == ["warning"]
    full = pd.date_range("2025-01-01", "2025-02-28")
    assert quality.check_gaps(obs("portwatch_portcalls", "DEU", full, noisy(len(full))), REG).empty


def test_weekday_series_ignore_weekends_and_short_holiday_gaps() -> None:
    days = pd.bdate_range("2025-01-01", "2025-03-31")  # prices: calendar weekdays, gap_tolerance 3
    holidays = days.delete([10, 40, 41, 42])  # a 1-day and a 3-day closure: normal
    assert quality.check_gaps(obs("prices_country_etf", "DEU", holidays, noisy(len(holidays))), REG).empty
    outage = days.delete(range(30, 35))  # a whole trading week missing
    detail = quality.check_gaps(obs("prices_country_etf", "DEU", outage, noisy(len(outage))), REG)["detail"].tolist()
    assert detail == [f"5 missing weekdays in 1 gap(s): {days[30]:%Y-%m-%d} to {days[34]:%Y-%m-%d} (5)"]


def test_single_missing_weekday_is_shown_as_one_date() -> None:
    first, last = quality._gap_span(pd.Timestamp("2015-04-02"), pd.Timestamp("2015-04-06"), "D", "weekdays")
    assert first == last == pd.Timestamp("2015-04-03")  # Good Friday


def test_monthly_gaps_and_known_source_gaps() -> None:
    months = pd.date_range("2020-01-01", "2021-12-01", freq="MS")
    # ZAF 2020-12 is a known Comtrade gap (indicators.yaml known_gaps); 2021-06 is not.
    held = months[(months != "2020-12-01") & (months != "2021-06-01")]
    issues = quality.check_gaps(obs("comtrade_exports", "ZAF", held, noisy(len(held), 1e10, 1e8)), REG)
    assert issues["detail"].tolist() == ["1 missing months in 1 gap(s): 2021-06-01 (1)"]
    as_deu = quality.check_gaps(obs("comtrade_exports", "DEU", held, noisy(len(held), 1e10, 1e8)), REG)
    assert as_deu["detail"].tolist() == ["2 missing months in 2 gap(s): 2020-12-01 (1), 2021-06-01 (1)"]


# ---------------------------------------------------------------------- jumps


def test_jump_is_flagged_and_data_is_left_alone() -> None:
    values = noisy(120)
    values[80] = 1000.0  # a tenfold spike
    df = obs("portwatch_portcalls", "DEU", pd.date_range("2025-01-01", periods=120), values)
    before = df.copy()
    issues = quality.check_jumps(df)
    assert len(issues) == 2  # up to the spike and back down
    assert issues["detail"].str.contains("on 2025-03-22").all()
    assert set(issues["severity"]) == {"warning"}
    pd.testing.assert_frame_equal(df, before)


def test_ordinary_noise_and_moderate_moves_are_not_flagged() -> None:
    values = noisy(200)
    values[150:] *= 1.3  # a real 30% step up: large, but plausible
    df = obs("portwatch_portcalls", "DEU", pd.date_range("2025-01-01", periods=200), values)
    assert quality.check_jumps(df).empty


def test_temperatures_use_differences_not_ratios() -> None:
    values = noisy(120, level=5.0, sd=0.8)  # always positive, so ratios would be used by default
    values[60] = 45.0
    df = obs("openmeteo_temp_mean", "DEU", pd.date_range("2025-01-01", periods=120), values)
    assert quality.check_jumps(df)["detail"].str.contains(r"\(x").all()
    as_difference = quality.check_jumps(df, interval_indicators={"openmeteo_temp_mean"})
    assert len(as_difference) == 2 and as_difference["detail"].str.contains(r"\(change [+-]\d").all()
    values[60] = values[59] + 3.0  # a few degrees: ordinary weather
    df = obs("openmeteo_temp_mean", "DEU", pd.date_range("2025-01-01", periods=120), values)
    assert quality.check_jumps(df, interval_indicators={"openmeteo_temp_mean"}).empty


def test_zero_heavy_series_judged_on_their_active_days() -> None:
    rng = np.random.default_rng(3)
    values = np.where(rng.random(365) < 0.2, rng.uniform(0.5, 4.0, 365), 0.0)  # cooling degree days
    df = obs("openmeteo_cdd22", "GBR", pd.date_range("2025-01-01", periods=365), values)
    assert quality.check_jumps(df).empty
    values[200] = 60.0
    df = obs("openmeteo_cdd22", "GBR", pd.date_range("2025-01-01", periods=365), values)
    assert len(quality.check_jumps(df)) >= 1


def test_jumps_since_reports_only_new_values() -> None:
    values = noisy(120)
    values[30] = 1000.0  # old spike, already reviewed
    values[119] = 1000.0  # arrives in the new fetch
    df = obs("portwatch_portcalls", "DEU", pd.date_range("2025-01-01", periods=120), values)
    df.loc[119, "fetched_at"] = T1
    issues = quality.check_jumps(df, since=T0)
    assert len(issues) == 1 and "on 2025-04-30" in issues["detail"].iloc[0]
    assert len(quality.check_jumps(df)) == 3


def test_short_series_are_not_judged() -> None:
    df = obs("portwatch_portcalls", "DEU", pd.date_range("2025-01-01", periods=10), [100] * 9 + [1000])
    assert quality.check_jumps(df).empty


# ---------------------------------------------------------- duplicates, units


def test_duplicates_identical_warn_conflicting_error() -> None:
    base = obs("ember_demand", "DEU", ["2026-01-01", "2026-02-01"], [40.0, 38.0])
    identical = quality.check_duplicates(pd.concat([base, base.iloc[[0]]], ignore_index=True))
    assert identical["severity"].tolist() == ["warning"] and "1 duplicate rows over 1 keys" in identical["detail"].iloc[0]
    clash = pd.concat([base, base.iloc[[1]].assign(value=39.0)], ignore_index=True)
    assert quality.check_duplicates(clash)["severity"].tolist() == ["error"]
    assert quality.check_duplicates(base).empty


def test_unit_change_and_registry_mismatch_are_errors() -> None:
    old = obs("eurostat_ip", "DEU", ["2026-01-01"], [100.0], fetched_at=T0)
    new = obs("eurostat_ip", "DEU", ["2026-02-01"], [101.0], fetched_at=T1, unit="index 2025=100")
    issues = quality.check_units(pd.concat([old, new], ignore_index=True), REG)
    assert issues["severity"].tolist() == ["error"]
    assert "'index 2021=100' from 2026-01-05, 'index 2025=100' from 2026-02-05" in issues["detail"].iloc[0]
    assert "differs from the registry" in quality.check_units(new, REG)["detail"].iloc[0]
    assert quality.check_units(old, REG).empty


# ------------------------------------------------------- run, record, display


def only(sources: dict, *names: str) -> dict:
    return {n: {**cfg, "enabled": n in names} for n, cfg in sources.items()}


def test_run_record_and_open_issues(tmp_path) -> None:
    data = tmp_path / "data"
    cfg = only(SOURCES, "ember")
    rows = [obs("ember_demand", c, pd.date_range("2023-01-01", "2026-07-01", freq="MS"), noisy(43, 30, 0.5))
            for c in REG.countries_for("ember_demand") if c != "ZAF"]
    store.append_changes(pd.concat(rows, ignore_index=True), "r1", data)
    now = pd.Timestamp("2026-09-01", tz="UTC")

    first = quality.run_checks(REG, cfg, data, now=now)
    assert set(first["check"]) == {"missing_country"}  # ZAF demand, other Ember indicators
    t1 = quality.record(first, data, now=now)
    assert quality.last_run_time(store.read_quality_issues(data)) == t1
    shown = quality.open_issues(store.read_quality_issues(data), now=now)
    assert len(shown) == len(first)

    # A fetcher reports an issue; then a later run finds nothing: the state
    # issues close, the fetcher issue since the previous run stays visible.
    fetch_issue = pd.DataFrame([{"source": "ember", "indicator": "ember_demand", "country": "DEU", "check": "no_price_data",
                                 "severity": "warning", "detail": "x", "checked_at": now + pd.Timedelta(hours=1)}])
    store.append_quality_issues(fetch_issue, data)
    quality.record(first.iloc[0:0], data, now=now + pd.Timedelta(hours=2))
    shown = quality.open_issues(store.read_quality_issues(data), now=now)
    assert shown["check"].tolist() == ["no_price_data"]
    markers = store.read_quality_issues(data).query("check == 'quality_run'").sort_values("checked_at")
    assert markers["detail"].tolist() == [quality.summary_text(first), "0 errors, 0 warnings, 0 info"]


def test_first_run_checks_full_history_later_runs_only_new_values(tmp_path) -> None:
    data = tmp_path / "data"
    cfg = only(SOURCES, "portwatch")
    values = noisy(120)
    values[30] = 1000.0
    store.append_changes(obs("portwatch_portcalls", "DEU", pd.date_range("2025-01-01", periods=120), values), "r1", data)
    now = T0 + pd.Timedelta(days=1)
    jumps = lambda df: df[df["check"] == "jump"]  # noqa: E731
    first = quality.run_checks(REG, cfg, data, now=now)
    assert len(jumps(first)) == 2
    quality.record(first, data, now=now)
    assert jumps(quality.run_checks(REG, cfg, data, now=now)).empty
    history = store.history(data_dir=data)
    assert len(history) == 120 and history["value"].max() == 1000.0  # flagged, never dropped


def test_quality_command_and_status_section(capsys: pytest.CaptureFixture[str]) -> None:
    from tracker.__main__ import main

    assert main(["status"]) == 0
    assert "Quality (never run" in capsys.readouterr().out
    assert main(["quality", "--dry-run"]) == 0
    assert "dry run, nothing recorded" in capsys.readouterr().out
    assert store.read_quality_issues().empty
    assert main(["quality"]) == 0
    assert "Quality checks recorded" in capsys.readouterr().out
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Quality (last check" in out and "missing_country" in out
    assert main(["quality", "--list"]) == 0
    assert "Open quality issues" in capsys.readouterr().out


def test_compact_command_merges_files_and_keeps_every_row(capsys: pytest.CaptureFixture[str]) -> None:
    from tracker.__main__ import main

    dates = pd.date_range("2026-01-01", "2026-06-01", freq="MS")
    store.append_changes(obs("ember_demand", "DEU", dates, noisy(6, 30, 1)), "r1")
    store.append_changes(obs("ember_demand", "DEU", dates, noisy(6, 30, 1, seed=2), fetched_at=T1), "r2")
    store.append_changes(obs("ember_demand", "FRA", dates, noisy(6, 40, 1)), "r3")
    folder = store.source_dir("ember")
    before = store.history()
    assert len(list(folder.glob("*.parquet"))) == 3

    assert main(["compact"]) == 0
    assert "unchanged" in capsys.readouterr().out
    assert len(list(folder.glob("*.parquet"))) == 1
    pd.testing.assert_frame_equal(before, store.history())


def test_duplicate_files_in_the_lake_are_reported(tmp_path) -> None:
    data = tmp_path / "data"
    store.append_changes(obs("ember_demand", "DEU", ["2026-01-01", "2026-02-01"], [40.0, 38.0]), "r1", data)
    original = next(store.source_dir("ember", data).glob("*.parquet"))
    shutil.copy(original, original.with_name("restored_copy.parquet"))  # e.g. a restore on top
    issues = quality.run_checks(REG, only(SOURCES, "ember"), data, now=T1)
    dup = issues[issues["check"] == "duplicate"]
    assert dup["severity"].tolist() == ["warning"] and dup["country"].tolist() == ["DEU"]
