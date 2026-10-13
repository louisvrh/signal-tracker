"""CPB World Trade Monitor fetcher tests: the .xlsx reader, the series parsed and the two requests
(saved fixtures: the July 2026 workbook trimmed to its last 14 months, and the release page's
download links; no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.helpers import FakeResponse, RoutedSession, fixture_text
from tracker import settings
from tracker.fetchers.base import FetchError
from tracker.fetchers.cpb import CPBFetcher, latest_xlsx_url, parse_trade_sheet, read_xlsx_sheet
from tracker.registry import load_registry

CONFIG = settings.load_sources()["cpb"]
SERIES = {s["code"]: (s["indicator"], s["country"]) for s in CONFIG["series"]}
BOOK = (settings.FIXTURES_DIR / "cpb" / "world-trade-monitor.xlsx").read_bytes()
PAGE = fixture_text("cpb", "latest_release.html")
PAGE_URL = "https://www.cpb.nl/en/world-trade-monitor/cpb-world-trade-monitor-july-2026"


def test_the_xlsx_reader_resolves_shared_strings_and_numbers() -> None:
    cells = read_xlsx_sheet(BOOK, "trade_out")
    assert cells[1]["B"] == "CPB WORLD TRADE MONITOR"
    assert cells[8]["C"] == "tgz_w1_qnmi_sn" and float(cells[8]["D"]) == pytest.approx(21254.12, abs=0.01)
    with pytest.raises(FetchError, match="no sheet"):
        read_xlsx_sheet(BOOK, "missing")
    with pytest.raises(FetchError, match="not a readable"):
        read_xlsx_sheet(b"not a zip", "trade_out")


def test_parse_keeps_the_configured_series_for_every_month() -> None:
    out = parse_trade_sheet(BOOK, SERIES)
    assert set(out["indicator"]) == {"cpb_export_volume", "cpb_export_price", "cpb_import_volume", "cpb_import_price"}
    assert set(out["country"]) == {"WLD", "USA", "GBR", "JPN", "CHN"}
    assert out["date"].max() == pd.Timestamp("2026-07-01") and out["date"].min() == pd.Timestamp("2025-06-01")
    assert len(out) == len(SERIES) * 14
    assert (out["value"] > 0).all() and out["value"].between(30, 300).all()  # index levels, base 2021=100


def test_a_missing_series_is_a_clear_error() -> None:
    with pytest.raises(FetchError, match="not found"):
        parse_trade_sheet(BOOK, {**SERIES, "xgz_zz_qnmi_sn": ("cpb_export_volume", "ZZZ")})


def test_the_excel_link_is_found_on_the_release_page() -> None:
    assert latest_xlsx_url(PAGE, PAGE_URL) == "https://www.cpb.nl/system/files/cpbmedia/CPB-world-trade-monitor-july-2026.xlsx"
    with pytest.raises(FetchError, match="no World Trade Monitor"):
        latest_xlsx_url("<html></html>", PAGE_URL)


def test_two_requests_raw_copies_and_store_rows(tmp_path) -> None:
    def handler(url: str, params: dict) -> FakeResponse:
        if url.endswith(".xlsx"):
            return FakeResponse(200, content=BOOK)
        return FakeResponse(200, PAGE)

    session = RoutedSession(handler)
    f = CPBFetcher(registry=load_registry(), config=CONFIG, raw_dir=tmp_path / "raw", session=session)
    f.min_interval_s = 0
    df = f.fetch("2015-01-01", "2026-10-07")
    assert [c[0] for c in session.calls] == [CONFIG["latest_url"],
                                             "https://www.cpb.nl/system/files/cpbmedia/CPB-world-trade-monitor-july-2026.xlsx"]
    assert len(df) == len(SERIES) * 14 and (df["source"] == "cpb").all() and (df["frequency"] == "M").all()
    assert set(df["unit"]) == {"index 2021=100"}
    # Estimated availability: the month's end + 57 days, capped at the fetch time (then "live").
    estimate = (df["date"] + pd.DateOffset(months=1) + pd.Timedelta(days=57)).dt.tz_localize("UTC")
    assert (df["available_at"] == estimate.where(estimate < f.fetched_at, f.fetched_at)).all()
    assert (df.loc[estimate < f.fetched_at, "vintage_type"] == "estimated").all()
    raw = sorted(p.name for p in (tmp_path / "raw").rglob("*"))
    assert any(n.endswith("latest_release.html") for n in raw) and any(n.endswith(".xlsx") for n in raw)
    # A window narrower than the file keeps only its months.
    assert set(f.fetch("2026-01-01", "2026-03-31")["date"].dt.month) == {1, 2, 3}
