"""ENTSO-E fetcher tests: interval weighting, local days, areas and request
windows, and saved fixtures (no network)."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest
from entsoe.mappings import lookup_area

from tests.helpers import FakeResponse, RoutedSession, fixture_text, no_wait
from tracker import settings, store
from tracker.fetchers.base import FetchError
from tracker.fetchers.entsoe import AreaPeriod, EntsoeFetcher, area_periods, parse_load_xml, request_windows
from tracker.fetchers.load import daily_load, interval_hours
from tracker.registry import load_registry

CONFIG = settings.load_sources()["entsoe"]
TZ = "Europe/Berlin"
TOKEN = "entsoetoken123"
DE_FIXTURE = "load_DE_2024-03-30_dst.xml"   # 2024-03-30..04-01 Berlin days, 15-minute, spring DST on 31 March
FR_FIXTURE = "load_FR_2019-06-03_hourly.xml"  # 2019-06-03..04 Paris days, hourly
DE_CODE = lookup_area("DE").code
FR_CODE = lookup_area("FR").code


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def xml(text: str, status: int = 200) -> FakeResponse:
    return FakeResponse(status, text, headers={"content-type": "text/xml"})


def handler(url: str, params: dict) -> FakeResponse:
    area = params["outBiddingZone_Domain"]
    if area == DE_CODE:
        return xml(fixture_text("entsoe", DE_FIXTURE))
    if area == FR_CODE:
        return xml(fixture_text("entsoe", FR_FIXTURE))
    return xml(fixture_text("entsoe", "no_data_TR.xml"))  # "No matching data found" acknowledgement


def make(tmp_path, session, monkeypatch, config: dict | None = None) -> EntsoeFetcher:
    monkeypatch.setenv("ENTSOE_API_TOKEN", TOKEN)
    f = EntsoeFetcher(registry=load_registry(), config=config or CONFIG, raw_dir=tmp_path / "raw", session=session)
    no_wait(f)
    return f


def quarter_hours(start_utc: str, n: int, value: float = 1000.0, drop: tuple[int, ...] = ()) -> pd.Series:
    idx = pd.date_range(start_utc, periods=n, freq="15min", tz="UTC")
    return pd.Series(value, index=idx).drop(idx[list(drop)])


# ------------------------------------------------------------- aggregation


def test_interval_hours_use_the_nearer_neighbour_capped_at_an_hour() -> None:
    q = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    assert interval_hours(q).tolist() == [0.25, 0.25, 0.25]
    apart = pd.DatetimeIndex([ts("2024-01-01T00:00"), ts("2024-01-01T03:00")])
    assert interval_hours(apart).tolist() == [1.0, 1.0]  # a gap doesn't stretch a value
    assert interval_hours(pd.DatetimeIndex([], tz="UTC")).size == 0


def test_15_minute_day_is_weighted_by_quarter_hours() -> None:
    s = quarter_hours("2024-01-09T23:00", 96)  # local Wednesday 10 Jan 2024 (CET = UTC+1)
    daily, overnight, incomplete = daily_load(s.index, s.to_numpy(), TZ)
    assert daily["date"].tolist() == [pd.Timestamp("2024-01-10")]
    assert daily["value"].item() == pytest.approx(24_000)
    assert overnight["value"].item() == pytest.approx(1000)
    assert incomplete.empty


def test_resolution_change_within_a_day() -> None:
    hourly = pd.date_range("2024-01-09T23:00", periods=12, freq="h", tz="UTC")     # local 00:00-12:00
    quarter = pd.date_range("2024-01-10T11:00", periods=48, freq="15min", tz="UTC")  # local 12:00-24:00
    daily, _, incomplete = daily_load(hourly.append(quarter), [1000.0] * 12 + [2000.0] * 48, TZ)
    assert daily["value"].item() == pytest.approx(12 * 1000 + 12 * 2000)
    assert incomplete.empty


def test_a_missing_quarter_hour_flags_the_day_instead_of_averaging() -> None:
    s = quarter_hours("2024-01-09T23:00", 96, drop=(40,))
    daily, overnight, incomplete = daily_load(s.index, s.to_numpy(), TZ)
    assert daily.empty
    assert incomplete[["hours", "expected"]].values.tolist() == [[23.75, 24]]
    assert len(overnight) == 1  # quarter-hour 40 is 10:00 local, outside 00:00-05:00


def test_a_day_without_any_value_is_reported() -> None:
    s = quarter_hours("2024-01-09T23:00", 96 * 3, drop=tuple(range(96, 192)))  # 10 and 12 Jan, nothing on 11 Jan
    daily, _, incomplete = daily_load(s.index, s.to_numpy(), TZ)
    assert daily["date"].tolist() == [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-12")]
    assert incomplete.values.tolist() == [[pd.Timestamp("2024-01-11"), 0.0, 24.0]]


def test_missing_values_count_as_gaps() -> None:
    s = quarter_hours("2024-01-09T23:00", 96)
    s.iloc[5] = np.nan  # 01:15 local
    daily, overnight, incomplete = daily_load(s.index, s.to_numpy(), TZ)
    assert daily.empty and overnight.empty and len(incomplete) == 1


def test_european_dst_days_have_23_and_25_hours() -> None:
    spring = quarter_hours("2024-03-30T23:00", 92)  # 31 March 2024, clocks go forward
    autumn = quarter_hours("2024-10-26T22:00", 100)  # 27 October 2024, clocks go back
    for s, hours in ((spring, 23), (autumn, 25)):
        daily, _, incomplete = daily_load(s.index, s.to_numpy(), TZ)
        assert daily["value"].tolist() == [hours * 1000] and incomplete.empty


def test_overnight_uses_local_hours() -> None:
    idx = pd.date_range("2024-01-09T23:00", periods=24, freq="h", tz="UTC")  # local Wednesday 10 Jan
    _, overnight, _ = daily_load(idx, idx.hour.astype(float), TZ)  # value = UTC hour
    assert overnight["value"].item() == pytest.approx((23 + 0 + 1 + 2 + 3) / 5)  # local 00-05 = UTC 23-04
    weekend = pd.date_range("2024-01-12T23:00", periods=24, freq="h", tz="UTC")  # Saturday
    assert daily_load(weekend, np.ones(24), TZ)[1].empty


# ------------------------------------------------------- areas and windows


def test_area_periods_from_a_code_or_pieces() -> None:
    assert area_periods("DE") == [AreaPeriod("DE")]
    pieces = area_periods([{"area": "DE_AT_LU", "to": "2018-10-01"}, {"area": "DE_LU", "from": "2018-10-01"}])
    assert pieces == [AreaPeriod("DE_AT_LU", None, ts("2018-10-01")), AreaPeriod("DE_LU", ts("2018-10-01"), None)]
    with pytest.raises(FetchError, match="must be after"):
        area_periods([{"area": "X", "from": "2020-01-01", "to": "2019-01-01"}])


def test_request_windows_split_at_years_and_bidding_zone_changes() -> None:
    periods = area_periods([{"area": "DE_AT_LU", "to": "2018-10-01"}, {"area": "DE_LU", "from": "2018-10-01"}])
    windows = request_windows(periods, ts("2017-06-01"), ts("2019-03-01"))
    assert windows == [
        ("DE_AT_LU", ts("2017-06-01"), ts("2018-01-01")),
        ("DE_AT_LU", ts("2018-01-01"), ts("2018-10-01")),
        ("DE_LU", ts("2018-10-01"), ts("2019-01-01")),
        ("DE_LU", ts("2019-01-01"), ts("2019-03-01")),
    ]
    long = request_windows([AreaPeriod("DE")], ts("2015-01-01"), ts("2026-09-29"))
    assert all(e - s <= pd.Timedelta(days=366) for _, s, e in long)  # never more than a year per call
    assert long[0][1] == ts("2015-01-01") and long[-1][2] == ts("2026-09-29")


def test_sources_yaml_areas_override_countries_yaml(tmp_path, monkeypatch) -> None:
    session = RoutedSession(handler)
    assert make(tmp_path, session, monkeypatch).areas_for("FRA") == [AreaPeriod("FR")]
    custom = dict(CONFIG, areas={"DEU": "DE_LU"})
    assert make(tmp_path, session, monkeypatch, custom).areas_for("DEU") == [AreaPeriod("DE_LU")]


# ------------------------------------------------------------------- fetch


def test_fetch_from_fixtures(tmp_path, monkeypatch) -> None:
    fetcher = make(tmp_path, RoutedSession(handler), monkeypatch)
    df = fetcher.fetch("2024-03-30", "2024-04-01")
    assert list(df.columns) == store.COLUMNS
    assert set(df["country"]) == {"DEU"}  # FR's fixture is from 2019; the rest have no data
    load = df[df["indicator"] == "entsoe_load"].set_index("date")["value"]
    assert list(load.index) == list(pd.date_range("2024-03-30", "2024-04-01"))
    # Independent check: MW x 0.25 h summed over each Berlin day; 31 March has 23 hours.
    s = parse_load_xml(fixture_text("entsoe", DE_FIXTURE))
    local_day = s.index.tz_convert(TZ).tz_localize(None).normalize()
    assert (local_day == pd.Timestamp("2024-03-31")).sum() == 92
    expected = (s * 0.25).groupby(local_day).sum()
    assert load.to_numpy() == pytest.approx(expected.loc[load.index].to_numpy())
    night = df[df["indicator"] == "entsoe_overnight_load"]
    assert [d.day_name() for d in night["date"]] == ["Monday"]  # 30 and 31 March are a weekend
    assert set(df.loc[df["indicator"] == "entsoe_load", "unit"]) == {"MWh"}
    assert fetcher.issues == []


def test_hourly_fixture(tmp_path, monkeypatch) -> None:
    df = make(tmp_path, RoutedSession(handler), monkeypatch).fetch("2019-06-03", "2019-06-04")
    fra = df[df["country"] == "FRA"]
    load = fra[fra["indicator"] == "entsoe_load"].set_index("date")["value"]
    s = parse_load_xml(fixture_text("entsoe", FR_FIXTURE))
    local_day = s.index.tz_convert("Europe/Paris").tz_localize(None).normalize()
    assert load.to_numpy() == pytest.approx(s.groupby(local_day).sum().loc[load.index].to_numpy())
    assert len(fra[fra["indicator"] == "entsoe_overnight_load"]) == 2  # Monday and Tuesday


def test_request_parameters_and_raw_copy(tmp_path, monkeypatch) -> None:
    session = RoutedSession(handler)
    make(tmp_path, session, monkeypatch).fetch("2024-03-30", "2024-04-01")
    params = [p for _, p, _ in session.calls]
    de = [p for p in params if p["outBiddingZone_Domain"] == DE_CODE]
    assert len(de) == 1
    assert de[0]["documentType"] == "A65" and de[0]["processType"] == "A16"
    assert de[0]["periodStart"] == "202403290000" and de[0]["periodEnd"] == "202404030000"  # padded by a day
    assert de[0]["securityToken"] == TOKEN
    registry = load_registry()
    codes = {lookup_area(registry.countries[c].entsoe_code).code for c in registry.countries_for("entsoe_load")}
    assert {p["outBiddingZone_Domain"] for p in params} == codes
    saved = sorted((tmp_path / "raw").rglob("*load_DE_*.xml.gz"))
    assert len(saved) == 1
    assert gzip.decompress(saved[0].read_bytes()).decode("utf-8") == fixture_text("entsoe", DE_FIXTURE)


def test_requests_split_at_the_new_year(tmp_path, monkeypatch) -> None:
    session = RoutedSession(handler)
    make(tmp_path, session, monkeypatch).fetch("2023-12-30", "2024-01-02")
    de = [(p["periodStart"], p["periodEnd"]) for _, p, _ in session.calls if p["outBiddingZone_Domain"] == DE_CODE]
    assert de == [("202312290000", "202401010000"), ("202401010000", "202401040000")]


def test_missing_intervals_become_quality_issues(tmp_path, monkeypatch) -> None:
    # Cut the fixture's last two hours (22:00-24:00 local on 1 April).
    text = fixture_text("entsoe", DE_FIXTURE).replace(
        "            <end>2024-04-01T22:00Z</end>", "            <end>2024-04-01T20:00Z</end>")

    def gappy(url, params):
        return xml(text) if params["outBiddingZone_Domain"] == DE_CODE else handler(url, params)

    fetcher = make(tmp_path, RoutedSession(gappy), monkeypatch)
    df = fetcher.fetch("2024-03-30", "2024-04-01")
    assert pd.Timestamp("2024-04-01") not in set(df.loc[df["indicator"] == "entsoe_load", "date"])
    assert pd.Timestamp("2024-04-01") in set(df.loc[df["indicator"] == "entsoe_overnight_load", "date"])
    assert [(i["check"], i["country"]) for i in fetcher.issues] == [("missing_hours", "DEU")]
    assert "2024-04-01: 22 of 24 hours" in fetcher.issues[0]["detail"]


def test_bad_token_fails_cleanly_without_the_url(tmp_path, monkeypatch) -> None:
    def unauthorized(url, params):
        return FakeResponse(401, "<html><body>Unauthorized</body></html>", headers={"content-type": "text/html"})

    with pytest.raises(FetchError, match="HTTP 401") as exc:
        make(tmp_path, RoutedSession(unauthorized), monkeypatch).fetch("2024-03-30", "2024-04-01")
    assert "token" not in str(exc.value) and TOKEN not in str(exc.value)
    assert exc.value.__suppress_context__  # the HTTPError (whose message has the URL) isn't chained


def test_server_errors_are_retried(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}

    def flaky(url, params):
        calls["n"] += 1
        return FakeResponse(503, "busy") if calls["n"] == 1 else handler(url, params)

    df = make(tmp_path, RoutedSession(flaky), monkeypatch).fetch("2024-03-30", "2024-04-01")
    assert len(df[df["indicator"] == "entsoe_load"]) == 3


def test_rerun_writes_zero_rows(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    first = make(tmp_path, RoutedSession(handler), monkeypatch).fetch("2024-03-30", "2024-04-01")
    assert store.append_changes(first, "r1", data_dir) == len(first) > 0
    second = make(tmp_path, RoutedSession(handler), monkeypatch).fetch("2024-03-30", "2024-04-01")
    assert store.append_changes(second, "r2", data_dir) == 0
