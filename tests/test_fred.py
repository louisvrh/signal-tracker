"""FRED fetcher tests using saved fixtures (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tracker import settings, store
from tracker.fetchers.base import BaseFetcher, FetchError, estimate_availability, period_end
from tracker.fetchers.fred import FredFetcher, parse_observations
from tracker.registry import load_registry

FIXTURES = settings.FIXTURES_DIR / "fred"


class FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")

    def json(self) -> Any:
        return json.loads(self.text)


class FakeSession:
    """Returns fixture bodies by series_id; records every call."""

    def __init__(self, responses: dict[str, list[FakeResponse]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, headers=None, timeout=None) -> FakeResponse:
        self.calls.append(dict(params or {}))
        queue = self.responses[params["series_id"]]
        return queue.pop(0) if len(queue) > 1 else queue[0]


def fixture(name: str) -> FakeResponse:
    return FakeResponse(200, (FIXTURES / name).read_text(encoding="utf-8"))


CONFIG = {
    "key_env": "FRED_API_KEY",
    "series": [
        {"series_id": "INDPRO", "indicator": "fred_indpro", "country": "USA", "vintages": True},
        {"series_id": "DCOILBRENTEU", "indicator": "fred_brent", "country": "WLD", "vintages": False},
    ],
}


def make_fetcher(tmp_path: Path, session: FakeSession, monkeypatch: pytest.MonkeyPatch) -> FredFetcher:
    monkeypatch.setenv("FRED_API_KEY", "testkey0000")
    fetcher = FredFetcher(registry=load_registry(), config=CONFIG, raw_dir=tmp_path / "raw", session=session)
    fetcher.min_interval_s = 0
    fetcher.backoff_min_s = fetcher.backoff_max_s = 0
    return fetcher


def default_session() -> FakeSession:
    return FakeSession({"INDPRO": [fixture("indpro_vintages.json")], "DCOILBRENTEU": [fixture("brent_observations.json")]})


def test_parse_observations_drops_missing() -> None:
    payload = json.loads((FIXTURES / "brent_observations.json").read_text(encoding="utf-8"))
    df = parse_observations(payload)
    assert len(df) == 8  # 11 observations, 3 holidays marked '.'
    assert df["value"].dtype == "float64"
    assert pd.Timestamp("2024-12-25") not in set(df["date"])


def test_fetch_returns_store_schema_with_vintages(tmp_path, monkeypatch) -> None:
    df = make_fetcher(tmp_path, default_session(), monkeypatch).fetch("2024-01-01", "2025-01-05")
    assert list(df.columns) == store.COLUMNS
    indpro = df[df["indicator"] == "fred_indpro"]
    assert set(indpro["vintage_type"]) == {"source_vintage"}
    jan = indpro[indpro["date"] == "2024-01-01"].sort_values("available_at")
    assert len(jan) == 8  # eight real-time periods for January 2024 in the fixture
    # First vintage published 2024-02-15, treated as known from the next midnight UTC.
    assert jan["available_at"].iloc[0] == pd.Timestamp("2024-02-16", tz="UTC")
    assert jan["value"].iloc[0] == pytest.approx(102.5739)
    brent = df[df["indicator"] == "fred_brent"]
    assert set(brent["vintage_type"]) == {"estimated"}
    assert set(brent["country"]) == {"WLD"} and set(brent["frequency"]) == {"D"}


def test_request_parameters(tmp_path, monkeypatch) -> None:
    session = default_session()
    make_fetcher(tmp_path, session, monkeypatch).fetch("2015-01-01", "2025-01-05")
    indpro, brent = session.calls
    assert indpro["realtime_start"] == "2015-01-01" and indpro["realtime_end"] == "9999-12-31"
    assert indpro["output_type"] == 1 and indpro["observation_start"] == "2015-01-01"
    assert "realtime_start" not in brent
    assert brent["file_type"] == "json"


def test_raw_responses_saved_without_key(tmp_path, monkeypatch) -> None:
    make_fetcher(tmp_path, default_session(), monkeypatch).fetch("2024-01-01", "2025-01-05")
    raw = list((tmp_path / "raw").rglob("*.json"))
    assert len(raw) == 2
    assert all("testkey0000" not in p.read_text(encoding="utf-8") for p in raw)


def test_second_fetch_writes_zero_rows(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    first = make_fetcher(tmp_path, default_session(), monkeypatch).fetch("2024-01-01", "2025-01-05")
    assert store.append_changes(first, "run1", data_dir) == len(first) > 0
    second = make_fetcher(tmp_path, default_session(), monkeypatch).fetch("2024-01-01", "2025-01-05")
    assert store.append_changes(second, "run2", data_dir) == 0


def test_as_of_sees_first_published_indpro(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    df = make_fetcher(tmp_path, default_session(), monkeypatch).fetch("2024-01-01", "2025-01-05")
    store.append_changes(df, "run1", data_dir)
    as_of = store.latest(indicators=["fred_indpro"], as_of="2024-03-01", data_dir=data_dir)
    assert as_of["date"].tolist() == [pd.Timestamp("2024-01-01")]
    assert as_of["value"].item() == pytest.approx(102.5739)


def test_retries_on_server_errors(tmp_path, monkeypatch) -> None:
    session = FakeSession({
        "INDPRO": [FakeResponse(503, "busy"), FakeResponse(429, "slow down"), fixture("indpro_vintages.json")],
        "DCOILBRENTEU": [fixture("brent_observations.json")],
    })
    df = make_fetcher(tmp_path, session, monkeypatch).fetch("2024-01-01", "2025-01-05")
    assert len(session.calls) == 4 and not df.empty


def test_bad_request_fails_without_leaking_key(tmp_path, monkeypatch) -> None:
    body = '{"error_code":400,"error_message":"Bad Request. The value for variable api_key is not registered: testkey0000"}'
    session = FakeSession({"INDPRO": [FakeResponse(400, body)], "DCOILBRENTEU": [FakeResponse(400, body)]})
    with pytest.raises(FetchError) as exc:
        make_fetcher(tmp_path, session, monkeypatch).fetch("2024-01-01")
    assert "HTTP 400" in str(exc.value) and "testkey0000" not in str(exc.value)
    assert len(session.calls) == 1  # 4xx is not retried


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch) -> None:
    fetcher = make_fetcher(tmp_path, default_session(), monkeypatch)
    monkeypatch.setenv("FRED_API_KEY", "")
    with pytest.raises(FetchError, match="FRED_API_KEY is not set"):
        fetcher.fetch("2024-01-01")


def test_period_end_and_estimated_availability() -> None:
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]))
    assert period_end(dates, "M").tolist() == [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]
    fetched = pd.Timestamp("2024-03-10", tz="UTC")
    avail, vintage = estimate_availability(dates, "M", 16, fetched)
    assert avail.tolist() == [pd.Timestamp("2024-02-17", tz="UTC"), fetched]
    assert vintage.tolist() == ["estimated", "live"]


def test_unregistered_indicator_rejected(tmp_path, monkeypatch) -> None:
    class Bad(BaseFetcher):
        source = "fred"

        def _fetch(self, start, end):
            df = store.empty_frame()
            row = {c: None for c in store.COLUMNS}
            row.update(indicator="fred_unknown", source="fred")
            return pd.concat([df, pd.DataFrame([row])])

    with pytest.raises(FetchError, match="not defined in config/indicators.yaml"):
        Bad(registry=load_registry(), config={}, raw_dir=tmp_path).fetch("2024-01-01")


def test_retired_series_are_not_requested(tmp_path, monkeypatch) -> None:
    # fred_copper is retired in the real registry (PLAN D56): its history stays, nothing new is fetched.
    assert load_registry().indicator("fred_copper").retired.replaced_by == "imf_copper"
    session = default_session()
    fetcher = make_fetcher(tmp_path, session, monkeypatch)
    fetcher.config = {**CONFIG, "series": CONFIG["series"] + [
        {"series_id": "PCOPPUSDM", "indicator": "fred_copper", "country": "WLD", "vintages": False}]}
    df = fetcher.fetch("2024-01-01", "2025-01-05")
    assert [c["series_id"] for c in session.calls] == ["INDPRO", "DCOILBRENTEU"]
    assert "fred_copper" not in set(df["indicator"])

