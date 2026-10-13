"""PortWatch fetcher tests using saved ArcGIS responses (no network)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tests.helpers import FakeResponse, RoutedSession, fixture_text, no_wait
from tracker import settings, store
from tracker.fetchers.base import FetchError, RetryableHTTPError
from tracker.fetchers.portwatch import PortWatchFetcher, to_long
from tracker.registry import load_registry

CONFIG = settings.load_sources()["portwatch"]


def handler(url: str, params: dict) -> FakeResponse:
    if "Daily_Ports_Data" in url:
        return FakeResponse(200, fixture_text("portwatch", "ports_page.json"))
    return FakeResponse(200, fixture_text("portwatch", "chokepoints_page.json"))


def make(tmp_path, session) -> PortWatchFetcher:
    f = PortWatchFetcher(registry=load_registry(), config=CONFIG, raw_dir=tmp_path / "raw", session=session)
    no_wait(f)
    return f


def test_fetch_country_and_chokepoint_rows(tmp_path) -> None:
    df = make(tmp_path, RoutedSession(handler)).fetch("2025-01-01", "2025-01-03")
    assert list(df.columns) == store.COLUMNS
    deu = df[(df["country"] == "DEU") & (df["date"] == "2025-01-01")].set_index("indicator")["value"]
    assert deu["portwatch_portcalls"] == 35
    assert deu["portwatch_imports"] == 125376 and deu["portwatch_exports"] == 194036
    suez = df[(df["country"] == "CHK_SUEZ") & (df["date"] == "2025-01-01")].set_index("indicator")["value"]
    assert suez["portwatch_transits"] == 32 and suez["portwatch_tanker_transits"] == 9
    assert suez["portwatch_transit_volume"] == 1125034
    assert {"CHK_SUEZ", "CHK_HORMUZ", "DEU", "TWN"} <= set(df["country"])
    assert set(df["frequency"]) == {"D"} and set(df["vintage_type"]) == {"estimated"}


def test_queries_aggregate_server_side(tmp_path) -> None:
    session = RoutedSession(handler)
    make(tmp_path, session).fetch("2025-01-01", "2025-01-03")
    ports = next(p for u, p, _ in session.calls if "Daily_Ports_Data" in u)
    assert ports["groupByFieldsForStatistics"] == "ISO3,date"
    assert "'DEU'" in ports["where"] and "DATE '2025-01-01'" in ports["where"]
    assert {s["onStatisticField"] for s in json.loads(ports["outStatistics"])} == {"portcalls", "import", "export"}
    chk = next(p for u, p, _ in session.calls if "Daily_Chokepoints_Data" in u)
    assert "'chokepoint6'" in chk["where"]


def test_pagination_follows_full_pages(tmp_path) -> None:
    page = {"features": [{"attributes": {"ISO3": "DEU", "date": "2025-01-01", "portcalls": 1, "import": 2, "export": 3}}] * 1000}
    last = {"features": [{"attributes": {"ISO3": "DEU", "date": "2025-01-02", "portcalls": 1, "import": 2, "export": 3}}]}

    def paged(url, params):
        if "Daily_Chokepoints_Data" in url:
            return FakeResponse(200, json.dumps({"features": []}))
        return FakeResponse(200, json.dumps(page if params["resultOffset"] == 0 else last))

    session = RoutedSession(paged)
    fetcher = make(tmp_path, session)
    attrs = fetcher._query_all("Daily_Ports_Data", {"where": "1=1"}, "ports")
    assert len(attrs) == 1001
    assert [p["resultOffset"] for u, p, _ in session.calls] == [0, 1000]


def test_arcgis_error_payload_raises(tmp_path) -> None:
    def err(url, params):
        return FakeResponse(200, json.dumps({"error": {"code": 400, "message": "Invalid query"}}))

    with pytest.raises(FetchError, match="Invalid query"):
        make(tmp_path, RoutedSession(err)).fetch("2025-01-01", "2025-01-03")


# ------------------------------------------------------------ retry and resume

UNABLE = json.dumps({"error": {"code": 400, "message": "", "details": ["Unable to perform query. Please check your parameters."]}})


def feature_page(offset: int, n: int = 1000) -> str:
    """A ports page whose rows are numbered by their offset."""
    rows = [{"attributes": {"ISO3": "DEU", "date": "2025-01-01", "portcalls": offset + i, "import": 0, "export": 0}}
            for i in range(n)]
    return json.dumps({"features": rows})


def quick(fetcher: PortWatchFetcher, attempts: int = 3) -> PortWatchFetcher:
    fetcher.query_attempts = attempts
    fetcher.query_backoff_min_s = fetcher.query_backoff_max_s = 0
    return fetcher


def three_pages(fail_at: int | None = None, changed: bool = False):
    """Pages at offsets 0 and 1000 (full) and 2000 (last, 5 rows); optionally
    failing at one offset, or with page 1000 changed (data shifted)."""
    def handler(url, params):
        offset = params["resultOffset"]
        if offset == fail_at:
            return FakeResponse(200, UNABLE)
        if offset == 1000 and changed:
            return FakeResponse(200, feature_page(5000))
        return FakeResponse(200, feature_page(offset, 1000 if offset < 2000 else 5))
    return handler


def test_unable_to_perform_query_is_retried(tmp_path) -> None:
    replies = iter([UNABLE, UNABLE, feature_page(0, 3)])
    session = RoutedSession(lambda url, params: FakeResponse(200, next(replies)))
    attrs = quick(make(tmp_path, session))._query_all("Daily_Ports_Data", {"where": "1=1"}, "ports")
    assert len(attrs) == 3 and len(session.calls) == 3
    saved = sorted(p.name for p in (tmp_path / "raw").rglob("*.json"))
    assert sum("_error" in n for n in saved) == 2  # error bodies are kept as raw copies too
    assert not (tmp_path / "raw" / "resume").exists() or not any((tmp_path / "raw" / "resume").iterdir())


def test_retries_are_bounded_and_other_errors_are_not_retried(tmp_path) -> None:
    session = RoutedSession(lambda url, params: FakeResponse(200, UNABLE))
    with pytest.raises(RetryableHTTPError, match="Unable to perform query"):
        quick(make(tmp_path, session), attempts=3)._query_all("Daily_Ports_Data", {"where": "1=1"}, "ports")
    assert len(session.calls) == 3
    invalid = RoutedSession(lambda url, params: FakeResponse(200, json.dumps({"error": {"code": 400, "message": "Invalid query"}})))
    with pytest.raises(FetchError):
        quick(make(tmp_path, invalid))._query_all("Daily_Ports_Data", {"where": "1=1"}, "ports")
    assert len(invalid.calls) == 1


def test_failed_query_resumes_from_the_failed_page(tmp_path) -> None:
    key = {"countries": ["DEU"], "start": "2019-01-01"}
    first = RoutedSession(three_pages(fail_at=2000))
    with pytest.raises(RetryableHTTPError):
        quick(make(tmp_path, first))._query_all("Daily_Ports_Data", {"where": "end 2026-09-27"}, "ports", key)
    assert [p["resultOffset"] for _, p, _ in first.calls] == [0, 1000, 2000, 2000, 2000]

    # Next run (maybe another day, so a later end date): same key, resumes.
    second = RoutedSession(three_pages())
    attrs = quick(make(tmp_path, second))._query_all("Daily_Ports_Data", {"where": "end 2026-09-28"}, "ports", key)
    assert [p["resultOffset"] for _, p, _ in second.calls] == [1000, 2000]  # re-check last saved page, then go on
    assert {p["where"] for _, p, _ in second.calls} == {"end 2026-09-27"}  # finishes the original query
    assert attrs["portcalls"].tolist() == list(range(2000)) + list(range(2000, 2005))
    assert not list((tmp_path / "raw" / "resume").glob("*.json"))  # done: checkpoint removed


def test_resume_starts_over_if_the_data_shifted(tmp_path) -> None:
    key = {"countries": ["DEU"], "start": "2019-01-01"}
    with pytest.raises(RetryableHTTPError):
        quick(make(tmp_path, RoutedSession(three_pages(fail_at=2000))))._query_all("Daily_Ports_Data", {"where": "w"}, "ports", key)
    session = RoutedSession(three_pages(changed=True))
    quick(make(tmp_path, session))._query_all("Daily_Ports_Data", {"where": "w"}, "ports", key)
    assert [p["resultOffset"] for _, p, _ in session.calls] == [1000, 0, 1000, 2000]


def test_old_checkpoints_are_ignored(tmp_path) -> None:
    key = {"countries": ["DEU"], "start": "2019-01-01"}
    with pytest.raises(RetryableHTTPError):
        quick(make(tmp_path, RoutedSession(three_pages(fail_at=2000))))._query_all("Daily_Ports_Data", {"where": "w"}, "ports", key)
    checkpoint = next((tmp_path / "raw" / "resume").glob("*.json"))
    state = json.loads(checkpoint.read_text())
    state["started_at"] = (pd.Timestamp(state["started_at"]) - pd.Timedelta(days=8)).isoformat()
    checkpoint.write_text(json.dumps(state))
    session = RoutedSession(three_pages())
    quick(make(tmp_path, session))._query_all("Daily_Ports_Data", {"where": "w"}, "ports", key)
    assert [p["resultOffset"] for _, p, _ in session.calls] == [0, 1000, 2000]


def test_rerun_after_a_chokepoint_failure_does_not_redownload_ports(tmp_path) -> None:
    def chokepoints_down(url, params):
        return FakeResponse(200, UNABLE) if "Daily_Chokepoints_Data" in url else handler(url, params)

    with pytest.raises(RetryableHTTPError):
        quick(make(tmp_path, RoutedSession(chokepoints_down))).fetch("2025-01-01", "2025-01-03")
    session = RoutedSession(handler)
    df = quick(make(tmp_path, session)).fetch("2025-01-01", "2025-01-03")
    assert len(df) == 36  # same rows as a clean run
    assert sum("Daily_Ports_Data" in u for u, _, _ in session.calls) == 1  # just the check of its saved last page
    assert not list((tmp_path / "raw" / "resume").glob("*.json"))


def test_to_long_drops_missing_values() -> None:
    attrs = pd.DataFrame({"date": ["2025-01-01", "2025-01-02"], "ISO3": ["DEU", "DEU"], "export": [5, None]})
    out = to_long(attrs, "ISO3", [{"field": "export", "indicator": "portwatch_exports"}])
    assert len(out) == 1 and out["value"].item() == 5


def test_rerun_writes_zero_rows(tmp_path) -> None:
    data_dir = tmp_path / "data"
    first = make(tmp_path, RoutedSession(handler)).fetch("2025-01-01", "2025-01-03")
    assert store.append_changes(first, "r1", data_dir) == len(first) == 36
    second = make(tmp_path, RoutedSession(handler)).fetch("2025-01-01", "2025-01-03")
    assert store.append_changes(second, "r2", data_dir) == 0
