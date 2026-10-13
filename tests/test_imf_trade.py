"""IMF fetcher tests: IMTS trade and PCPS commodity prices; the queries, SDMX-CSV parsing and
store rows (saved fixtures, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.helpers import FakeResponse, RoutedSession, fixture_text, no_wait
from tracker import settings, store
from tracker.fetchers.base import FetchError
from tracker.fetchers.imf_trade import CSV_ACCEPT, IMFTradeFetcher, parse_imts_csv
from tracker.registry import load_registry

CONFIG = settings.load_sources()["imf"]
TRADE_ONLY = {k: v for k, v in CONFIG.items() if k != "commodity_prices"}  # the IMTS request alone
CODES = {"XG_FOB_USD": "imf_exports", "MG_CIF_USD": "imf_imports"}


def make(tmp_path, text: str | None = None, config: dict = TRADE_ONLY) -> tuple[IMFTradeFetcher, RoutedSession]:
    body = fixture_text("imf", "imts_monthly.csv") if text is None else text
    prices = fixture_text("imf", "pcps_monthly.csv")
    session = RoutedSession(lambda url, params: FakeResponse(200, prices if "/PCPS/" in url else body))
    f = IMFTradeFetcher(registry=load_registry(), config=config, raw_dir=tmp_path / "raw", session=session)
    no_wait(f)
    return f, session


def test_parse_keeps_monthly_totals_and_drops_blank_and_other_rows() -> None:
    out = parse_imts_csv(fixture_text("imf", "imts_monthly.csv"), CODES, {"CHN", "DEU", "VNM", "MEX"})
    assert set(out["country"]) == {"CHN", "DEU", "VNM"}  # MEX only had FOB imports, which aren't used
    assert set(out["indicator"]) == {"imf_exports", "imf_imports"}
    assert out["date"].min() == pd.Timestamp("2025-10-01") and out["date"].max() == pd.Timestamp("2026-05-01")
    assert len(out) == 3 * 2 * 8
    chn = out[(out["country"] == "CHN") & (out["indicator"] == "imf_exports")].set_index("date")["value"]
    assert chn[pd.Timestamp("2026-05-01")] == pytest.approx(376_782_582_803)  # US dollars, not millions


def test_parse_errors_are_clear() -> None:
    with pytest.raises(FetchError, match="lacks columns"):
        parse_imts_csv("COUNTRY,VALUE\nCHN,1\n", CODES, {"CHN"})
    assert parse_imts_csv("", CODES, {"CHN"}).empty


def test_one_request_for_every_country_and_flow(tmp_path) -> None:
    f, session = make(tmp_path)
    df = f.fetch("2025-10-01", "2026-05-31")
    [(url, params, headers)] = session.calls
    key = url.rsplit("/", 1)[-1]
    countries, indicators, partner, freq = key.split(".")
    assert url.startswith("https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/IMTS/+/")
    assert set(countries.split("+")) == set(load_registry().watchlist())
    assert indicators == "XG_FOB_USD+MG_CIF_USD" and partner == "G001" and freq == "M"
    assert params == {"c[TIME_PERIOD]": "ge:2025-10+le:2026-05"} and headers == {"Accept": CSV_ACCEPT}
    assert set(df["indicator"]) == {"imf_exports", "imf_imports"} and set(df["unit"]) == {"USD"}
    assert (df["frequency"] == "M").all() and (df["source"] == "imf").all()
    raw = list((tmp_path / "raw").rglob("*imts_monthly.csv"))
    assert len(raw) == 1 and raw[0].read_text(encoding="utf-8") == fixture_text("imf", "imts_monthly.csv")


def test_publication_lag_sets_estimated_availability_and_reruns_write_nothing(tmp_path) -> None:
    f, _ = make(tmp_path)
    df = f.fetch("2025-10-01", "2026-05-31")
    oct_row = df[(df["country"] == "DEU") & (df["date"] == "2025-10-01")].iloc[0]
    assert oct_row["available_at"] == pd.Timestamp("2025-11-01", tz="UTC") + pd.Timedelta(days=122)
    data_dir = tmp_path / "data"
    assert store.append_changes(df, "run1", data_dir) == len(df) == 48
    again, _ = make(tmp_path)
    assert store.append_changes(again.fetch("2025-10-01", "2026-05-31"), "run2", data_dir) == 0


def test_an_empty_result_gives_no_rows(tmp_path) -> None:
    empty = "\n".join(fixture_text("imf", "imts_monthly.csv").splitlines()[:1] +
                      ["dataflow,IMF.STA:IMTS(1.0.0),R,,,,,,,,,,,,,,,"]) + "\n"
    f, _ = make(tmp_path, empty)
    assert f.fetch("2030-01-01", "2030-02-28").empty


# ------------------------------------------------------- commodity prices (PCPS, PLAN D56)


def test_commodity_prices_come_in_a_second_request_and_are_stored_as_world(tmp_path) -> None:
    f, session = make(tmp_path, config=CONFIG)
    df = f.fetch("2026-01-01", "2026-09-30")
    assert len(session.calls) == 2
    url, params, headers = session.calls[1]
    assert url == "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.RES/PCPS/+/G001.PCOPP+PNGASEU.USD.M"
    assert params == {"c[TIME_PERIOD]": "ge:2026-01+le:2026-09"} and headers == {"Accept": CSV_ACCEPT}
    prices = df[df["indicator"].isin(["imf_copper", "imf_eu_gas"])]
    assert len(prices) == 18 and set(prices["country"]) == {"WLD"} and (prices["source"] == "imf").all()
    assert set(prices.loc[prices["indicator"] == "imf_copper", "unit"]) == {"USD per metric ton"}
    copper = prices[prices["indicator"] == "imf_copper"].set_index("date")["value"]
    gas = prices[prices["indicator"] == "imf_eu_gas"].set_index("date")["value"]
    # The same values FRED had through July 2026 (PLAN D55); August and September are new.
    assert copper[pd.Timestamp("2026-07-01")] == pytest.approx(13542.82, abs=0.01)
    assert gas[pd.Timestamp("2026-07-01")] == pytest.approx(17.93)
    assert gas.index.max() == pd.Timestamp("2026-09-01")
    assert set(df["indicator"]) == {"imf_exports", "imf_imports", "imf_copper", "imf_eu_gas"}
    assert len(list((tmp_path / "raw").rglob("*pcps_monthly.csv"))) == 1


def test_commodity_prices_use_their_own_publication_lag(tmp_path) -> None:
    f, _ = make(tmp_path, config=CONFIG)
    df = f.fetch("2026-01-01", "2026-09-30")
    jan = df[(df["indicator"] == "imf_copper") & (df["date"] == "2026-01-01")].iloc[0]
    assert jan["available_at"] == pd.Timestamp("2026-02-01", tz="UTC") + pd.Timedelta(days=10)
    data_dir = tmp_path / "data"
    store.append_changes(df, "run1", data_dir)
    again, _ = make(tmp_path, config=CONFIG)
    assert store.append_changes(again.fetch("2026-01-01", "2026-09-30"), "run2", data_dir) == 0
