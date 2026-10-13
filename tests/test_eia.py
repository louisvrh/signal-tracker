"""EIA fetcher tests: local-day aggregation, DST, gaps, and fixtures (no network)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tests.helpers import FakeResponse, RoutedSession, fixture_text, no_wait
from tracker import settings, store
from tracker.fetchers.base import FetchError
from tracker.fetchers.eia import EIAFetcher, hourly_to_daily, parse_rows
from tracker.registry import load_registry

CONFIG = settings.load_sources()["eia"]
TZ = "America/Chicago"


def hourly(start_utc_end: str, hours: int, value: float = 1000.0, drop: tuple[int, ...] = ()) -> pd.DataFrame:
    """Hour-ending UTC periods starting at `start_utc_end`, constant value."""
    ends = pd.date_range(start_utc_end, periods=hours, freq="h")
    df = pd.DataFrame({"period": ends.strftime("%Y-%m-%dT%H"), "value": value})
    return df.drop(index=list(drop)).reset_index(drop=True)


def handler(url: str, params: dict) -> FakeResponse:
    if "region-data" in url:
        return FakeResponse(200, fixture_text("eia", "hourly_us48.json"))
    return FakeResponse(200, fixture_text("eia", "sales_ind.json"))


def make(tmp_path, session, monkeypatch) -> EIAFetcher:
    monkeypatch.setenv("EIA_API_KEY", "eiakey12345")
    f = EIAFetcher(registry=load_registry(), config=CONFIG, raw_dir=tmp_path / "raw", session=session)
    no_wait(f)
    return f


# ------------------------------------------------------------- aggregation


def test_hour_ending_timestamps_map_to_the_right_local_day() -> None:
    # Hour ending 06:00 UTC on 2 Jan = 23:00-24:00 CST on 1 Jan.
    df = pd.DataFrame({"period": ["2024-01-02T06"], "value": [5.0]})
    daily, overnight, incomplete = hourly_to_daily(df, TZ)
    assert incomplete["date"].tolist() == [pd.Timestamp("2024-01-01")]


def test_complete_day_sums_24_hours() -> None:
    # Local 2024-01-03 (CST = UTC-6) is hour-endings 07:00 UTC 3 Jan .. 06:00 UTC 4 Jan.
    daily, _, incomplete = hourly_to_daily(hourly("2024-01-03T07", 24), TZ)
    assert daily["date"].tolist() == [pd.Timestamp("2024-01-03")]
    assert daily["value"].item() == 24_000
    assert incomplete.empty


def test_dst_days_have_23_and_25_hours() -> None:
    spring, _, inc1 = hourly_to_daily(hourly("2024-03-10T07", 23), TZ)  # 10 Mar 2024: clocks go forward
    assert spring["value"].tolist() == [23_000] and inc1.empty
    autumn, _, inc2 = hourly_to_daily(hourly("2024-11-03T06", 25), TZ)  # 3 Nov 2024: clocks go back
    assert autumn["value"].tolist() == [25_000] and inc2.empty


def test_missing_hours_are_flagged_not_averaged() -> None:
    daily, overnight, incomplete = hourly_to_daily(hourly("2024-01-03T07", 24, drop=(2, 10)), TZ)
    assert daily.empty
    assert incomplete[["hours", "expected"]].values.tolist() == [[22, 24]]
    assert overnight.empty  # hour index 2 is 02:00-03:00 local, inside the overnight window


def test_overnight_is_weekday_mean_of_00_to_05_local() -> None:
    values = list(range(24))  # value = local hour number
    df = hourly("2024-01-03T07", 24)  # Wednesday
    df["value"] = values
    _, overnight, _ = hourly_to_daily(df, TZ)
    assert overnight["value"].item() == pytest.approx((0 + 1 + 2 + 3 + 4) / 5)
    _, weekend, _ = hourly_to_daily(hourly("2024-01-06T07", 24), TZ)  # Saturday
    assert weekend.empty


# ------------------------------------------------------------------- fetch


def test_fetch_from_fixtures(tmp_path, monkeypatch) -> None:
    df = make(tmp_path, RoutedSession(handler), monkeypatch).fetch("2024-03-08", "2024-03-12")
    assert list(df.columns) == store.COLUMNS
    load = df[df["indicator"] == "eia_us48_load"].set_index("date")["value"]
    assert list(load.index) == list(pd.date_range("2024-03-08", "2024-03-12"))
    night = df[df["indicator"] == "eia_us48_overnight_load"]
    assert [d.day_name() for d in night["date"]] == ["Friday", "Monday", "Tuesday"]
    sales = df[df["indicator"] == "eia_industrial_sales"]
    assert len(sales) == 3 and set(sales["frequency"]) == {"M"}
    assert sales["value"].iloc[0] == pytest.approx(82517.33102)
    assert set(df["country"]) == {"USA"}


def test_request_parameters(tmp_path, monkeypatch) -> None:
    session = RoutedSession(handler)
    make(tmp_path, session, monkeypatch).fetch("2024-03-08", "2024-03-12")
    url, p, _ = session.calls[0]
    assert url.endswith("electricity/rto/region-data/data/")
    assert p["facets[respondent][]"] == "US48" and p["facets[type][]"] == "D" and p["frequency"] == "hourly"
    assert p["start"] == "2024-03-07T00" and p["end"] == "2024-03-13T23"  # padded by a day
    assert p["length"] == 5000 and p["api_key"] == "eiakey12345"


def test_pagination(tmp_path, monkeypatch) -> None:
    rows = [{"period": f"2024-01-01T{h:02d}", "value": "1"} for h in range(3)]

    def paged(url, params):
        return FakeResponse(200, json.dumps({"response": {"total": "7001", "data": rows}}))

    session = RoutedSession(paged)
    fetcher = make(tmp_path, session, monkeypatch)
    fetcher._paged("electricity/rto/region-data", {}, "value", "x")
    assert [p["offset"] for _, p, _ in session.calls] == [0, 5000]


def test_gaps_become_quality_issues(tmp_path, monkeypatch) -> None:
    payload = json.loads(fixture_text("eia", "hourly_us48.json"))
    payload["response"]["data"] = [r for r in payload["response"]["data"] if r["period"] != "2024-03-11T12"]

    def gappy(url, params):
        if "region-data" in url:
            return FakeResponse(200, json.dumps(payload))
        return FakeResponse(200, fixture_text("eia", "sales_ind.json"))

    fetcher = make(tmp_path, RoutedSession(gappy), monkeypatch)
    df = fetcher.fetch("2024-03-08", "2024-03-12")
    load_days = df.loc[df["indicator"] == "eia_us48_load", "date"]
    assert pd.Timestamp("2024-03-11") not in set(load_days)
    assert [i["check"] for i in fetcher.issues] == ["missing_hours"]
    assert "2024-03-11" in fetcher.issues[0]["detail"]


def test_api_error_payload() -> None:
    with pytest.raises(FetchError, match="API_KEY_INVALID"):
        parse_rows({"error": {"code": "API_KEY_INVALID"}}, "value")


def test_rerun_writes_zero_rows(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    first = make(tmp_path, RoutedSession(handler), monkeypatch).fetch("2024-03-08", "2024-03-12")
    assert store.append_changes(first, "r1", data_dir) == len(first) > 0
    second = make(tmp_path, RoutedSession(handler), monkeypatch).fetch("2024-03-08", "2024-03-12")
    assert store.append_changes(second, "r2", data_dir) == 0
