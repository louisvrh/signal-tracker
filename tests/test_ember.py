"""Ember fetcher tests using saved API responses (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.helpers import FakeResponse, RoutedSession, fixture_text, no_wait
from tracker import settings, store
from tracker.fetchers.base import FetchError
from tracker.fetchers.ember import EmberFetcher, parse_records
from tracker.registry import load_registry

CONFIG = settings.load_sources()["ember"]


def handler(url: str, params: dict) -> FakeResponse:
    if params.get("entity_code") != "DEU":
        return FakeResponse(200, '{"stats": {}, "data": []}')
    name = "demand_DEU.json" if "electricity-demand" in url else "generation_DEU.json"
    return FakeResponse(200, fixture_text("ember", name))


def make(tmp_path, session, monkeypatch) -> EmberFetcher:
    monkeypatch.setenv("EMBER_API_KEY", "emberkey123")
    f = EmberFetcher(registry=load_registry(), config=CONFIG, raw_dir=tmp_path / "raw", session=session)
    no_wait(f)
    return f


def test_fetch_demand_and_generation(tmp_path, monkeypatch) -> None:
    df = make(tmp_path, RoutedSession(handler), monkeypatch).fetch("2024-01-01", "2024-03-31")
    assert list(df.columns) == store.COLUMNS
    assert set(df["country"]) == {"DEU"} and set(df["frequency"]) == {"M"}
    inds = set(df["indicator"])
    assert {"ember_demand", "ember_generation", "ember_gen_coal", "ember_gen_solar"} <= inds
    jan = df[df["date"] == "2024-01-01"].set_index("indicator")["value"]
    assert jan["ember_demand"] > 0 and jan["ember_generation"] > jan["ember_gen_coal"]
    assert sorted(df["date"].unique()) == list(pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))


def test_requests_use_month_bounds_and_key(tmp_path, monkeypatch) -> None:
    session = RoutedSession(handler)
    make(tmp_path, session, monkeypatch).fetch("2024-01-15", "2024-03-31")
    url, params, _ = session.calls[0]
    # end_date is exclusive at Ember, so March needs end_date=2024-04.
    assert params["start_date"] == "2024-01" and params["end_date"] == "2024-04"
    assert params["api_key"] == "emberkey123"
    gen = [p for u, p, _ in session.calls if "generation" in u]
    assert "Total generation" in gen[0]["series"]


def test_per_country_lag_is_used(tmp_path, monkeypatch) -> None:
    fetcher = make(tmp_path, RoutedSession(handler), monkeypatch)
    lags = {"DEU": 30, "MYS": 120}
    data = pd.DataFrame({"date": ["2020-01-01", "2020-01-01"], "country": ["DEU", "MYS"], "value": [1.0, 2.0]})
    out = fetcher.frame("ember_demand", data, lag_by_country=lags).set_index("country")["available_at"]
    assert out["DEU"] == pd.Timestamp("2020-03-02", tz="UTC")  # 2020-02-01 + 30 days
    assert out["MYS"] == pd.Timestamp("2020-05-31", tz="UTC")  # 2020-02-01 + 120 days


def test_bad_payload_raises() -> None:
    with pytest.raises(FetchError):
        parse_records({"detail": "Not authenticated"}, "demand_twh")


def test_rerun_writes_zero_rows(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    first = make(tmp_path, RoutedSession(handler), monkeypatch).fetch("2024-01-01", "2024-03-31")
    assert store.append_changes(first, "r1", data_dir) == len(first) > 0
    second = make(tmp_path, RoutedSession(handler), monkeypatch).fetch("2024-01-01", "2024-03-31")
    assert store.append_changes(second, "r2", data_dir) == 0
