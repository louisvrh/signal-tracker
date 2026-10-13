"""Prices fetcher tests. yfinance.download is replaced by a fake; Stooq is
served through a fake session. No network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yfinance

from tests.helpers import FakeResponse, RoutedSession, fixture_text, no_wait
from tracker import settings, store
from tracker.fetchers.base import FetchError
from tracker.fetchers.prices import (
    PricesFetcher,
    closes_to_long,
    completed_bars,
    parse_stooq_csv,
    stooq_symbol,
    yahoo_symbol,
)
from tracker.registry import load_registry

REG = load_registry()
CONFIG = settings.load_sources()["prices"]
DAYS = pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-08"])  # Fri, Sat, Mon


class FakeYF:
    """Returns yfinance-shaped data (MultiIndex columns: Price x Ticker)."""

    def __init__(self, missing: set[str] = frozenset()) -> None:
        self.missing = missing
        self.calls: list[dict] = []

    def __call__(self, symbols, **kw):
        self.calls.append(dict(kw, symbols=list(symbols)))
        cols = pd.MultiIndex.from_product([["Close", "Open"], symbols], names=["Price", "Ticker"])
        data = np.arange(len(DAYS) * len(cols), dtype=float).reshape(len(DAYS), len(cols)) + 1
        df = pd.DataFrame(data, index=pd.DatetimeIndex(DAYS, name="Date"), columns=cols)
        for s in self.missing:
            df[("Close", s)] = np.nan
        return df


@pytest.fixture()
def fake_yf(monkeypatch):
    fake = FakeYF()
    monkeypatch.setattr(yfinance, "download", fake)
    return fake


def stooq_html(url, params):
    return FakeResponse(200, "<!DOCTYPE html><html><body><noscript>This site requires JavaScript</noscript></body></html>")


def make(tmp_path, session=None) -> PricesFetcher:
    f = PricesFetcher(registry=REG, config=CONFIG, raw_dir=tmp_path / "raw", session=session or RoutedSession(stooq_html))
    no_wait(f)
    return f


def test_symbol_mapping() -> None:
    assert yahoo_symbol("USDEUR") == "EUR=X" and yahoo_symbol("EWG") == "EWG" and yahoo_symbol("HG=F") == "HG=F"
    assert stooq_symbol("EWG") == "ewg.us" and stooq_symbol("USDJPY") == "usdjpy" and stooq_symbol("EXV7.DE") == "exv7.de"
    assert stooq_symbol("HG=F") is None and stooq_symbol("HG=F", {"HG=F": "hg.f"}) == "hg.f"


def test_instruments_cover_etfs_fx_benchmark_and_sectors() -> None:
    inst = PricesFetcher(registry=REG, config=CONFIG, raw_dir=None).instruments()
    by_ind = {}
    for i in inst:
        by_ind.setdefault(i["indicator"], []).append(i)
    n_countries = len(REG.watchlist())
    assert n_countries == 25
    assert len(by_ind["prices_country_etf"]) == n_countries
    assert len(by_ind["prices_fx"]) == n_countries - 1  # every country but the USA
    zaf = {i["indicator"]: i["ticker"] for i in inst if i["country"] == "ZAF"}
    assert zaf == {"prices_country_etf": "EZA", "prices_fx": "USDZAR"}
    eur = [i["country"] for i in by_ind["prices_fx"] if i["ticker"] == "USDEUR"]
    assert set(eur) == {"DEU", "FRA", "ITA", "ESP", "NLD"}
    assert by_ind["prices_benchmark"][0]["ticker"] == "URTH"
    assert any(i["ticker"] == "EXV7.DE" for i in by_ind["prices_eu_chemicals"])


def test_fetch_maps_symbols_drops_weekends_and_stamps_22utc(tmp_path, fake_yf) -> None:
    df = make(tmp_path).fetch("2024-01-01", "2024-01-08")
    assert list(df.columns) == store.COLUMNS
    assert "EUR=X" in fake_yf.calls[0]["symbols"] and fake_yf.calls[0]["auto_adjust"] is True
    assert pd.Timestamp("2024-01-06") not in set(df["date"])  # Saturday dropped
    fri = df[(df["date"] == "2024-01-05")]
    assert set(fri["available_at"]) == {pd.Timestamp("2024-01-05 22:00", tz="UTC")}
    eur = df[(df["indicator"] == "prices_fx") & (df["date"] == "2024-01-05")].set_index("country")["value"]
    assert eur["DEU"] == eur["FRA"]  # one download, several countries
    assert set(df.loc[df["indicator"] == "prices_benchmark", "country"]) == {"WLD"}


def test_todays_bar_is_dropped_until_22utc() -> None:
    closes = pd.DataFrame({"date": pd.to_datetime(["2024-01-04", "2024-01-05"]), "ticker": "EWG", "value": [1.0, 2.0]})
    before = completed_bars(closes, pd.Timestamp("2024-01-05 21:59", tz="UTC"))
    assert before["date"].tolist() == [pd.Timestamp("2024-01-04")]
    after = completed_bars(closes, pd.Timestamp("2024-01-05 22:00", tz="UTC"))
    assert after["available_at"].tolist() == [pd.Timestamp("2024-01-04 22:00", tz="UTC"), pd.Timestamp("2024-01-05 22:00", tz="UTC")]


def test_missing_tickers_fall_back_to_stooq_and_report(tmp_path, monkeypatch) -> None:
    fake = FakeYF(missing={"EWG"})
    monkeypatch.setattr(yfinance, "download", fake)
    csv = "Date,Open,High,Low,Close,Volume\n2024-01-05,1,1,1,30.5,100\n2024-01-08,1,1,1,31.0,100\n"
    session = RoutedSession(lambda url, params: FakeResponse(200, csv if params["s"] == "ewg.us" else "<html>"))
    fetcher = make(tmp_path, session)
    df = fetcher.fetch("2024-01-01", "2024-01-08")
    deu = df[(df["indicator"] == "prices_country_etf") & (df["country"] == "DEU")]
    assert deu["value"].tolist() == [30.5, 31.0]
    assert [c[1]["s"] for c in session.calls] == ["ewg.us"]


def test_stooq_bot_check_is_reported_not_bypassed(tmp_path, monkeypatch) -> None:
    fake = FakeYF(missing={"EWG"})
    monkeypatch.setattr(yfinance, "download", fake)
    fetcher = make(tmp_path)
    df = fetcher.fetch("2024-01-01", "2024-01-08")
    assert "DEU" not in set(df.loc[df["indicator"] == "prices_country_etf", "country"])
    assert [i["check"] for i in fetcher.issues] == ["no_price_data"]
    with pytest.raises(FetchError, match="JavaScript"):
        parse_stooq_csv("<!DOCTYPE html><html>")


def test_total_failure_raises(tmp_path, monkeypatch) -> None:
    def broken(*a, **k):
        raise RuntimeError("Too Many Requests. Rate limited.")

    monkeypatch.setattr(yfinance, "download", broken)
    with pytest.raises(FetchError, match="no provider returned any data"):
        make(tmp_path).fetch("2024-01-01", "2024-01-08")


def test_closes_to_long() -> None:
    wide = pd.DataFrame({"A": [1.0, None], "B": [3.0, 4.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    long = closes_to_long(wide)
    assert len(long) == 3 and set(long["symbol"]) == {"A", "B"}


def test_float32_jitter_is_not_a_revision(tmp_path, monkeypatch) -> None:
    """Yahoo's adjusted closes differ by ~1e-7 between downloads (seen live)."""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(yfinance, "download", FakeYF())
    first = make(tmp_path).fetch("2024-01-01", "2024-01-08")
    store.append_changes(first, "r1", data_dir, rel_tol=CONFIG["change_tolerance"])

    def jittery(symbols, **kw):
        df = FakeYF()(symbols, **kw)
        return df * (1 + 3e-7)

    monkeypatch.setattr(yfinance, "download", jittery)
    second = make(tmp_path).fetch("2024-01-01", "2024-01-08")
    assert store.append_changes(second, "r2", data_dir, rel_tol=CONFIG["change_tolerance"]) == 0


def test_rerun_writes_zero_rows(tmp_path, fake_yf) -> None:
    data_dir = tmp_path / "data"
    first = make(tmp_path).fetch("2024-01-01", "2024-01-08")
    assert store.append_changes(first, "r1", data_dir) == len(first) > 0
    second = make(tmp_path).fetch("2024-01-01", "2024-01-08")
    assert store.append_changes(second, "r2", data_dir) == 0
