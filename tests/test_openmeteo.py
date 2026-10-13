"""Open-Meteo fetcher tests: weighting, degree days, budget and chunk planning (no network)."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from tests.helpers import FakeResponse, RoutedSession, fixture_text, no_wait
from tracker import settings, store
from tracker.fetchers.base import FetchError
from tracker.fetchers.openmeteo import OpenMeteoFetcher, call_weight, country_weather, parse_locations
from tracker.registry import City, load_registry

REG = load_registry()
CONFIG = {**settings.load_sources()["openmeteo"], "max_weighted_calls_per_minute": 1e9}


def synthetic(url: str, params: dict) -> FakeResponse:
    """DEU gets the saved fixture; any other request gets 10 °C everywhere."""
    lats = params["latitude"].split(",")
    if params["timezone"] == "Europe/Berlin" and params["start_date"] == "2024-01-01" and params["end_date"] == "2024-01-10":
        return FakeResponse(200, fixture_text("openmeteo", "deu_2024-01.json"))
    days = pd.date_range(params["start_date"], params["end_date"]).strftime("%Y-%m-%d").tolist()
    locs = [{"daily": {"time": days, "temperature_2m_mean": [10.0] * len(days)}} for _ in lats]
    return FakeResponse(200, json.dumps(locs if len(locs) > 1 else locs[0]))


def make(tmp_path, session, config=None) -> OpenMeteoFetcher:
    f = OpenMeteoFetcher(registry=REG, config=config or CONFIG, raw_dir=tmp_path / "raw",
                         session=session, data_dir=tmp_path / "data")
    no_wait(f)
    return f


def test_call_weight() -> None:
    assert call_weight(1, 7) == 1 and call_weight(5, 14) == 5 and call_weight(5, 15) == 10
    assert call_weight(5, 365) == 5 * 27


def test_parse_locations_accepts_dict_or_list() -> None:
    cities = [City(name="A", lat=0, lon=0, weight=1.0)]
    one = {"daily": {"time": ["2024-01-01"], "temperature_2m_mean": [3.5]}}
    assert parse_locations(one, cities)["temp"].tolist() == [3.5]
    assert parse_locations([one], cities)["temp"].tolist() == [3.5]
    with pytest.raises(FetchError, match="expected 2 locations"):
        parse_locations([one], cities * 2)
    with pytest.raises(FetchError, match="out of allowed range"):
        parse_locations({"error": True, "reason": "Parameter 'end_date' is out of allowed range"}, cities)


def test_degree_days_are_computed_per_city_then_weighted() -> None:
    temps = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 2), "city": ["cold", "hot"], "temp": [8.0, 30.0]})
    out = country_weather(temps, {"cold": 0.5, "hot": 0.5}, 18.0, 22.0).iloc[0]
    assert out["temp"] == 19.0
    assert out["hdd"] == 5.0  # 0.5 * (18 - 8); the weighted mean (19 °C) alone would give 0
    assert out["cdd"] == 4.0  # 0.5 * (30 - 22)


def test_day_with_a_missing_city_is_dropped() -> None:
    temps = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        "city": ["a", "b", "a", "b"], "temp": [1.0, 2.0, 3.0, None],
    })
    out = country_weather(temps, {"a": 0.5, "b": 0.5}, 18, 22)
    assert out["date"].tolist() == [pd.Timestamp("2024-01-01")]


def test_fetch_germany_from_fixture(tmp_path) -> None:
    session = RoutedSession(synthetic)
    df = make(tmp_path, session).fetch("2024-01-01", "2024-01-10")
    deu = df[df["country"] == "DEU"].pivot(index="date", columns="indicator", values="value")
    assert len(deu) == 10
    assert (deu["openmeteo_hdd18"] >= 0).all() and (deu["openmeteo_cdd22"] == 0).all()  # January in Germany
    assert deu["openmeteo_temp_mean"].between(-20, 15).all()
    assert set(df["country"]) == set(REG.watchlist())
    params = next(p for _, p, _ in session.calls if p["timezone"] == "Europe/Berlin")
    assert len(params["latitude"].split(",")) == 5 and params["daily"] == "temperature_2m_mean"


def test_end_is_capped_at_yesterday(tmp_path) -> None:
    session = RoutedSession(synthetic)
    fetcher = make(tmp_path, session)
    today = pd.Timestamp.now(tz="UTC").date()
    fetcher.fetch(today - pd.Timedelta(days=3), today)
    assert {p["end_date"] for _, p, _ in session.calls} == {(today - pd.Timedelta(days=1)).isoformat()}


def test_budget_stops_early_and_marks_partial(tmp_path) -> None:
    session = RoutedSession(synthetic)
    fetcher = make(tmp_path, session, {**CONFIG, "max_weighted_calls_per_run": 300})
    fetcher.fetch("2023-01-01", "2024-12-31")
    assert fetcher.partial
    # One year for a 5-city country costs 135; the budget allows two such chunks.
    assert len(session.calls) == 2
    assert all(p["start_date"].startswith("2024") for _, p, _ in session.calls)  # newest year first


def test_complete_years_are_skipped(tmp_path) -> None:
    fetcher = make(tmp_path, RoutedSession(synthetic))
    first = fetcher.fetch("2023-01-01", "2023-12-31")
    store.append_changes(first, "r1", tmp_path / "data")
    session = RoutedSession(synthetic)
    make(tmp_path, session).fetch("2023-01-01", "2024-01-31")
    assert {p["start_date"][:4] for _, p, _ in session.calls} == {"2024"}  # 2023 already stored


def test_incomplete_country_after_a_complete_one_gets_its_whole_year(tmp_path) -> None:
    # Regression: a complete country inside the refetch window used to shorten
    # the start date of every later country in the same year (ZAF, added last,
    # got only the last 30 days of 2026 and the gap would never have filled).
    fetcher = make(tmp_path, RoutedSession(synthetic))
    days = pd.date_range("2024-01-01", "2024-12-31")
    rows = fetcher.frame("openmeteo_temp_mean", pd.DataFrame({"date": days, "country": "DEU", "value": 10.0}))
    store.append_changes(rows, "r1", tmp_path / "data")
    chunks = fetcher._plan_chunks(["DEU", "ZAF"], date(2024, 1, 1), date(2024, 12, 31), date(2024, 12, 1))
    assert chunks == [("DEU", date(2024, 12, 1), date(2024, 12, 31)), ("ZAF", date(2024, 1, 1), date(2024, 12, 31))]


def test_rerun_writes_zero_rows(tmp_path) -> None:
    data_dir = tmp_path / "data"
    first = make(tmp_path, RoutedSession(synthetic)).fetch("2024-01-01", "2024-01-10")
    assert store.append_changes(first, "r1", data_dir) == len(first) > 0
    second = make(tmp_path, RoutedSession(synthetic)).fetch("2024-01-01", "2024-01-10")
    assert store.append_changes(second, "r2", data_dir) == 0
