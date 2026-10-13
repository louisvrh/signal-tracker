"""Comtrade fetcher tests. `comtradeapicall.getFinalData` is replaced by a fake
that serves saved records, so no network calls are made."""

from __future__ import annotations

import json
from datetime import date

import comtradeapicall
import pandas as pd
import pytest

from tests.helpers import fixture_text, no_wait
from tracker import settings, store
from tracker.fetchers.base import FetchError
from tracker.fetchers.comtrade import (
    ComtradeFetcher,
    mirror_totals,
    month_periods,
    parse_totals,
    split_combined_jan_feb,
)
from tracker.registry import load_registry

CONFIG = settings.load_sources()["comtrade"]
TOTALS = pd.DataFrame(json.loads(fixture_text("comtrade", "totals.json")))
MIRROR = pd.DataFrame(json.loads(fixture_text("comtrade", "mirror.json")))


class FakeComtrade:
    """Stand-in for getFinalData: returns fixtures, or prints an error and
    returns None like the real package does."""

    def __init__(self, failures: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self.failures = list(failures or [])

    def __call__(self, key, **kw):
        self.calls.append(dict(kw, key=key))
        if self.failures:
            print(self.failures.pop(0))
            return None
        source = MIRROR if kw["partnerCode"] == "490" else TOTALS
        return source[source["period"].astype(str).isin(kw["period"].split(","))].copy()


@pytest.fixture()
def fake(monkeypatch):
    f = FakeComtrade()
    monkeypatch.setattr(comtradeapicall, "getFinalData", f)
    monkeypatch.setattr(comtradeapicall, "getReference", lambda cat: pd.DataFrame({"id": [97, 842], "isGroup": [True, False]}))
    monkeypatch.setenv("COMTRADE_SUBSCRIPTION_KEY", "ctkey-123456")
    return f


def make(tmp_path, config=None) -> ComtradeFetcher:
    f = ComtradeFetcher(registry=load_registry(), config=config or CONFIG, raw_dir=tmp_path / "raw")
    no_wait(f)
    return f


def test_month_periods() -> None:
    assert month_periods(date(2023, 11, 15), date(2024, 2, 1)) == ["202311", "202312", "202401", "202402"]


def test_parse_totals_maps_comtrade_codes_to_iso3() -> None:
    reg = load_registry()
    code_to_iso = {reg.countries[c].comtrade_reporter: c for c in ["USA", "DEU", "CHN"]}
    out = parse_totals(TOTALS, code_to_iso)
    assert set(out["country"]) == {"USA", "DEU", "CHN"}  # USA is 842 in Comtrade, not 840
    assert len(out) == 12 and set(out["flow"]) == {"X", "M"}


def test_mirror_flips_flows_and_counts_reporters() -> None:
    out = mirror_totals(MIRROR, exclude_reporters={0, 97, 490})
    jan_x = out[(out["date"] == "2024-01-01") & (out["flow"] == "X")]
    jan = MIRROR["period"].astype(str) == "202401"
    partners_imports = MIRROR[jan & (MIRROR["flowCode"] == "M")]["primaryValue"].sum()
    assert jan_x["value"].item() == pytest.approx(partners_imports)
    assert jan_x["reporters"].item() > 50


def test_mirror_excludes_group_reporters() -> None:
    doubled = pd.concat([MIRROR, MIRROR.assign(reporterCode=97)])
    assert mirror_totals(doubled, {97})["value"].sum() == pytest.approx(mirror_totals(MIRROR, set())["value"].sum())


def test_combined_jan_feb_is_split_evenly() -> None:
    totals = pd.DataFrame({
        "date": pd.to_datetime(["2020-02-01", "2020-03-01"]), "country": "CHN", "flow": "X", "value": [300.0, 200.0],
    })
    out, notes = split_combined_jan_feb(totals, ["CHN"], {"202001", "202002", "202003"})
    by_month = out.set_index("date")["value"]
    assert by_month["2020-01-01"] == by_month["2020-02-01"] == 150.0
    assert by_month["2020-01-01"] + by_month["2020-02-01"] == 300.0
    assert notes == [{"country": "CHN", "flow": "X", "year": 2020}]
    # Not applied when January wasn't requested (e.g. a refetch window starting in February).
    _, notes = split_combined_jan_feb(totals.copy(), ["CHN"], {"202002", "202003"})
    assert notes == []


def test_fetch_builds_reported_and_mirror_rows(tmp_path, fake) -> None:
    df = make(tmp_path).fetch("2024-01-01", "2024-02-29")
    assert list(df.columns) == store.COLUMNS
    assert {"comtrade_exports", "comtrade_imports", "comtrade_mirror_exports",
            "comtrade_mirror_imports", "comtrade_mirror_reporters"} == set(df["indicator"])
    assert set(df.loc[df["indicator"].str.startswith("comtrade_mirror"), "country"]) == {"TWN"}
    assert "TWN" not in set(df.loc[df["indicator"] == "comtrade_exports", "country"])
    # Per-country lag: China (400 days) is stamped far later than the USA (default 60).
    exp = df[(df["indicator"] == "comtrade_exports") & (df["date"] == "2024-01-01")].set_index("country")["available_at"]
    assert exp["CHN"] - exp["USA"] == pd.Timedelta(days=340)


def test_call_parameters(tmp_path, fake) -> None:
    make(tmp_path).fetch("2024-01-01", "2024-02-29")
    totals, mirror = fake.calls
    assert totals["period"] == "202401,202402" and totals["partnerCode"] == "0" and totals["flowCode"] == "X,M"
    assert "842" in totals["reporterCode"].split(",") and "251" in totals["reporterCode"].split(",")
    assert mirror["reporterCode"] is None and mirror["partnerCode"] == "490"
    assert totals["cmdCode"] == "TOTAL" and totals["freqCode"] == "M" and totals["key"] == "ctkey-123456"


def test_twelve_periods_per_call_newest_first(tmp_path, fake) -> None:
    make(tmp_path).fetch("2022-01-01", "2024-02-29")
    periods = [c["period"] for c in fake.calls if c["partnerCode"] == "0"]
    assert [len(p.split(",")) for p in periods] == [2, 12, 12]
    assert periods[0] == "202401,202402"


def test_budget_marks_run_partial(tmp_path, fake) -> None:
    fetcher = make(tmp_path, {**CONFIG, "max_calls_per_run": 2})
    fetcher.fetch("2022-01-01", "2024-02-29")
    assert fetcher.partial and len(fake.calls) == 2


def test_rate_limit_is_retried(tmp_path, fake) -> None:
    fake.failures = ['{"statusCode": 429, "message": "Rate limit is exceeded."}']
    df = make(tmp_path).fetch("2024-01-01", "2024-02-29")
    assert not df.empty and len(fake.calls) == 3


def test_auth_error_fails_without_leaking_key(tmp_path, fake) -> None:
    fake.failures = ['{"statusCode": 401, "message": "Access denied due to invalid subscription key ctkey-123456."}']
    with pytest.raises(FetchError) as exc:
        make(tmp_path).fetch("2024-01-01", "2024-02-29")
    assert "invalid subscription key" in str(exc.value) and "ctkey-123456" not in str(exc.value)


def test_raw_copy_saved(tmp_path, fake) -> None:
    make(tmp_path).fetch("2024-01-01", "2024-02-29")
    raws = sorted(p.name for p in (tmp_path / "raw").rglob("*.json"))
    assert len(raws) == 2 and any("mirror" in r for r in raws)


def test_rerun_writes_zero_rows(tmp_path, fake) -> None:
    data_dir = tmp_path / "data"
    first = make(tmp_path).fetch("2024-01-01", "2024-02-29")
    assert store.append_changes(first, "r1", data_dir) == len(first) > 0
    second = make(tmp_path).fetch("2024-01-01", "2024-02-29")
    assert store.append_changes(second, "r2", data_dir) == 0
