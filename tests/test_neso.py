"""NESO fetcher tests: settlement periods to UTC, CSV parsing, the yearly and update files
combined, and local days (saved fixtures, no network)."""

from __future__ import annotations

import gzip
import io

import pandas as pd
import pytest

from tests.helpers import FakeResponse, RoutedSession, fixture_text, no_wait
from tracker import settings, store
from tracker.fetchers.base import FetchError
from tracker.fetchers.neso import NesoFetcher, interval_starts, parse_demand_csv, parse_settlement_dates
from tracker.registry import load_registry

CONFIG = settings.load_sources()["neso"]
TZ = "Europe/London"
FILES = {  # resource id in the fixture's download URL -> fixture
    "cc505e45-65ae-4819-9b90-1fbb06880293": "historic_2015_spring.csv",  # 27-30 March 2015, old date format
    "8a4a771c-3929-4e56-93ad-cdf13219dea5": "historic_2026_sep.csv",     # 7-9 September 2026
    "177f6fa4-ae49-4182-81ea-0c6b35f26ca6": "update_sep.csv",            # 8-10 September actuals, 11th forecast
}


def handler_for(files: dict[str, str]):
    def handler(url: str, params: dict) -> FakeResponse:
        if url.endswith("/package_show"):
            name = {"historic-demand-data": "package_historic.json", "daily-demand-update": "package_update.json"}
            return FakeResponse(200, fixture_text("neso", name[params["id"]]))
        for resource, fixture in files.items():
            if resource in url:
                return FakeResponse(200, content=fixture_text("neso", fixture).encode("utf-8"))
        return FakeResponse(404, "not found")
    return handler


def make(tmp_path, files: dict[str, str] = FILES) -> tuple[NesoFetcher, RoutedSession]:
    session = RoutedSession(handler_for(files))
    f = NesoFetcher(registry=load_registry(), config=CONFIG, raw_dir=tmp_path / "raw", session=session)
    no_wait(f)
    f.download_interval_s = 0
    return f, session


def expected_days(fixture: str) -> pd.Series:
    """MWh per settlement date straight from a fixture: half-hour MW x 0.5 h, actuals only."""
    df = pd.read_csv(io.StringIO(fixture_text("neso", fixture)))
    if "FORECAST_ACTUAL_INDICATOR" in df:
        df = df[df["FORECAST_ACTUAL_INDICATOR"] == "A"]
    total = df["ND"] + df["EMBEDDED_WIND_GENERATION"] + df["EMBEDDED_SOLAR_GENERATION"]
    return (total * 0.5).groupby(parse_settlement_dates(df["SETTLEMENT_DATE"]).to_numpy()).sum()


# ------------------------------------------------------------ parsing


def test_settlement_dates_in_every_format_the_files_use() -> None:
    out = parse_settlement_dates(pd.Series(["01-JAN-2015", "2026-09-08", "29-Mar-2015", "01-Jan-23", "31-Dec-23"]))
    assert list(out) == [pd.Timestamp(d) for d in ("2015-01-01", "2026-09-08", "2015-03-29", "2023-01-01",
                                                     "2023-12-31")]
    with pytest.raises(FetchError, match="unreadable"):
        parse_settlement_dates(pd.Series(["2026/09/08"]))


def test_demand_is_national_demand_plus_embedded_generation_and_forecasts_are_dropped() -> None:
    df = parse_demand_csv(fixture_text("neso", "update_sep.csv"))
    assert df["settlement_date"].max() == pd.Timestamp("2026-09-10")  # the 11th is forecast only
    assert len(df) == 3 * 48
    raw = pd.read_csv(io.StringIO(fixture_text("neso", "update_sep.csv"))).iloc[0]
    assert df["value"].iloc[0] == raw["ND"] + raw["EMBEDDED_WIND_GENERATION"] + raw["EMBEDDED_SOLAR_GENERATION"]


def test_a_csv_without_the_demand_columns_is_an_error() -> None:
    with pytest.raises(FetchError, match="lacks columns"):
        parse_demand_csv("SETTLEMENT_DATE,SETTLEMENT_PERIOD,TSD\n2026-01-01,1,100\n")


def test_settlement_periods_count_half_hours_from_local_midnight() -> None:
    dates = pd.Series(pd.to_datetime(["2015-03-29", "2015-03-29", "2015-10-25", "2015-10-25", "2026-09-08"]))
    starts = interval_starts(dates, pd.Series([1, 46, 1, 50, 1]), TZ)
    assert list(starts) == [pd.Timestamp(s, tz="UTC") for s in (
        "2015-03-29 00:00", "2015-03-29 22:30",   # spring: 46 periods, midnight is GMT
        "2015-10-24 23:00", "2015-10-25 23:30",   # autumn: 50 periods, midnight is BST
        "2026-09-07 23:00")]


# ------------------------------------------------------------ fetching


def test_spring_clock_change_and_weekday_nights(tmp_path) -> None:
    f, session = make(tmp_path)
    df = f.fetch("2015-03-27", "2015-03-30")
    daily = df[df["indicator"] == "neso_load"].set_index("date")["value"]
    expected = expected_days("historic_2015_spring.csv")
    assert list(daily.index) == list(expected.index)
    assert daily.to_numpy() == pytest.approx(expected.to_numpy())
    nights = df[df["indicator"] == "neso_overnight_load"]
    assert [d.day for d in nights["date"]] == [27, 30]  # Friday and Monday, not the weekend
    assert (df["country"] == "GBR").all() and set(df["unit"]) == {"MWh", "MW"}
    downloads = [url for url, _, _ in session.calls if "download" in url]
    assert len(downloads) == 1  # the yearly file covers the range: no update file
    assert f.issues == []


def test_the_update_file_adds_the_days_after_the_yearly_file(tmp_path) -> None:
    f, session = make(tmp_path)
    df = f.fetch("2026-09-07", "2026-09-10")
    daily = df[df["indicator"] == "neso_load"].set_index("date")["value"]
    assert [d.day for d in daily.index] == [7, 8, 9, 10]
    settled, update = expected_days("historic_2026_sep.csv"), expected_days("update_sep.csv")
    assert daily.iloc[:3].to_numpy() == pytest.approx(settled.to_numpy())
    assert daily.iloc[3] == pytest.approx(update[pd.Timestamp("2026-09-10")])
    assert len(df[df["indicator"] == "neso_overnight_load"]) == 4  # Monday to Thursday
    raw = sorted(p.name for p in (tmp_path / "raw").rglob("*.csv.gz"))
    assert len(raw) == 2 and any("demand_data_update" in n for n in raw)
    text = gzip.decompress(next((tmp_path / "raw").rglob("*update.csv.gz")).read_bytes()).decode("utf-8")
    assert text == fixture_text("neso", "update_sep.csv")


def test_a_day_with_missing_half_hours_is_reported_not_stored(tmp_path) -> None:
    lines = fixture_text("neso", "historic_2015_spring.csv").splitlines()
    holed = [l for l in lines if not l.startswith("28-MAR-2015,1")]  # periods 1 and 10-19 of the 28th
    tests_dir = tmp_path / "fx"
    tests_dir.mkdir()
    (tests_dir / "holed.csv").write_text("\n".join(holed) + "\n", encoding="utf-8")
    session = RoutedSession(lambda url, params: (
        FakeResponse(200, fixture_text("neso", "package_historic.json")) if url.endswith("/package_show")
        else FakeResponse(200, content=(tests_dir / "holed.csv").read_bytes())))
    f = NesoFetcher(registry=load_registry(), config=CONFIG, raw_dir=tmp_path / "raw", session=session)
    no_wait(f)
    f.download_interval_s = 0
    df = f.fetch("2015-03-27", "2015-03-30")
    days = df.loc[df["indicator"] == "neso_load", "date"].dt.day.tolist()
    assert days == [27, 29, 30]
    assert [i["check"] for i in f.issues] == ["missing_hours"]
    assert "2015-03-28: 18.5 of 24 hours" in f.issues[0]["detail"]


def test_rows_match_the_store_and_a_rerun_writes_nothing(tmp_path) -> None:
    f, _ = make(tmp_path)
    df = f.fetch("2026-09-07", "2026-09-10")
    data_dir = tmp_path / "data"
    assert store.append_changes(df, "run1", data_dir) == len(df) == 8
    again, _ = make(tmp_path)
    assert store.append_changes(again.fetch("2026-09-07", "2026-09-10"), "run2", data_dir) == 0
    assert set(df["vintage_type"]) == {"estimated"}


def test_missing_yearly_resources_fail_clearly(tmp_path) -> None:
    session = RoutedSession(lambda url, params: FakeResponse(
        200, '{"success": true, "result": {"resources": [{"name": "FAQ", "format": "DOC", "url": "x"}]}}'))
    f = NesoFetcher(registry=load_registry(), config=CONFIG, raw_dir=tmp_path / "raw", session=session)
    no_wait(f)
    with pytest.raises(FetchError, match="Historic Demand Data"):
        f.fetch("2015-03-27", "2015-03-30")
