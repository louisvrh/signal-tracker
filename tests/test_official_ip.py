"""Eurostat and OECD industrial-production fetchers (saved SDMX-CSV, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.helpers import FakeResponse, RoutedSession, fixture_text, no_wait
from tracker import settings, store
from tracker.fetchers.base import FetchError
from tracker.fetchers.eurostat import EurostatFetcher
from tracker.fetchers.oecd import OECDFetcher
from tracker.fetchers.sdmx import parse_sdmx_csv
from tracker.registry import load_registry

SOURCES = settings.load_sources()


def eurostat(tmp_path, session) -> EurostatFetcher:
    f = EurostatFetcher(registry=load_registry(), config=SOURCES["eurostat"], raw_dir=tmp_path / "raw", session=session)
    no_wait(f)
    return f


def oecd(tmp_path, session, config=None) -> OECDFetcher:
    cfg = config or {**SOURCES["oecd"], "series": SOURCES["oecd"]["series"][:1]}
    f = OECDFetcher(registry=load_registry(), config=cfg, raw_dir=tmp_path / "raw", session=session)
    no_wait(f)
    return f


def eurostat_handler(url, params):
    return FakeResponse(200, fixture_text("eurostat", "sts_inpr_m.csv"))


def oecd_handler(url, params):
    return FakeResponse(200, fixture_text("oecd", "indserv_bte.csv"))


def test_parse_sdmx_csv_maps_areas_and_drops_others() -> None:
    text = "geo,TIME_PERIOD,OBS_VALUE\nDE,2024-01,90.1\nXX,2024-01,1\nFR,2024-02,\n"
    out = parse_sdmx_csv(text, "geo", {"DE": "DEU", "FR": "FRA"})
    assert out.to_dict("records") == [{"date": pd.Timestamp("2024-01-01"), "country": "DEU", "value": 90.1}]
    assert parse_sdmx_csv("", "geo", {}).empty


def test_parse_sdmx_csv_rejects_ambiguous_keys() -> None:
    text = "geo,TIME_PERIOD,OBS_VALUE\nDE,2024-01,90\nDE,2024-01,91\n"
    with pytest.raises(FetchError, match="more than one series"):
        parse_sdmx_csv(text, "geo", {"DE": "DEU"})


def test_eurostat_fetch(tmp_path) -> None:
    session = RoutedSession(eurostat_handler)
    df = eurostat(tmp_path, session).fetch("2024-01-01", "2024-03-31")
    assert list(df.columns) == store.COLUMNS
    assert set(df["country"]) == {"DEU", "FRA", "ITA", "ESP", "NLD", "POL"}
    assert len(df) == 18 and set(df["indicator"]) == {"eurostat_ip"}
    url, params, _ = session.calls[0]
    assert url.endswith("/data/sts_inpr_m/M.PRD.B-D.SCA.I21.DE+ES+FR+IT+NL+PL")
    assert params == {"format": "SDMX-CSV", "startPeriod": "2024-01", "endPeriod": "2024-03"}
    # Estimated availability: period end + 47 days.
    jan = df[(df["country"] == "DEU") & (df["date"] == "2024-01-01")]["available_at"].item()
    assert jan == pd.Timestamp("2024-03-19", tz="UTC")


def test_oecd_fetch_uses_one_request_for_all_countries(tmp_path) -> None:
    session = RoutedSession(oecd_handler)
    df = oecd(tmp_path, session).fetch("2024-01-01", "2024-03-31")
    assert len(session.calls) == 1
    url, params, _ = session.calls[0]
    assert "DSD_STES@DF_INDSERV,/BRA+CAN+GBR+IND+JPN+KOR+MEX+TUR.M.PRVM.IX.BTE.Y._Z._Z.N" in url
    assert params["format"] == "csvfile"
    assert set(df["country"]) == {"JPN", "KOR", "CAN"} and set(df["indicator"]) == {"oecd_ip"}
    # KOR uses its per-country lag (55) instead of the default (75).
    avail = df[df["date"] == "2024-01-01"].set_index("country")["available_at"]
    assert avail["CAN"] - avail["KOR"] == pd.Timedelta(days=20)


def test_oecd_rate_limit_is_retried(tmp_path) -> None:
    replies = [FakeResponse(429, "Too many requests"), FakeResponse(200, fixture_text("oecd", "indserv_bte.csv"))]
    session = RoutedSession(lambda url, params: replies.pop(0))
    df = oecd(tmp_path, session).fetch("2024-01-01", "2024-03-31")
    assert len(session.calls) == 2 and not df.empty


def test_rerun_writes_zero_rows(tmp_path) -> None:
    data_dir = tmp_path / "data"
    for make, handler in [(eurostat, eurostat_handler), (oecd, oecd_handler)]:
        first = make(tmp_path, RoutedSession(handler)).fetch("2024-01-01", "2024-03-31")
        assert store.append_changes(first, "r1", data_dir) == len(first) > 0
        second = make(tmp_path, RoutedSession(handler)).fetch("2024-01-01", "2024-03-31")
        assert store.append_changes(second, "r2", data_dir) == 0
