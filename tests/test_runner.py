"""Runner and status tests: failure isolation, resume logic, freshness."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tracker import fetchers, store
from tracker.fetchers.base import BaseFetcher
from tracker.registry import load_registry
from tracker.runner import resume_start, run_all, summary_table
from tracker.status import expected_next_by, indicator_freshness, source_status

REGISTRY = load_registry()


class OkFetcher(BaseFetcher):
    """Fake source returning two daily Brent-like rows."""

    source = "fred"
    calls: list[date] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(registry=REGISTRY, config={}, raw_dir=None, data_dir=kwargs.get("data_dir"))

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        OkFetcher.calls.append(start)
        data = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "country": "WLD", "value": [70.0, 71.0]})
        return self.frame("fred_brent", data)


class FailingFetcher(BaseFetcher):
    """Fake source that fails like a bad API key would."""

    source = "broken"

    def __init__(self, **kwargs) -> None:
        super().__init__(registry=REGISTRY, config={}, raw_dir=None)

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        raise RuntimeError("HTTP 400: api_key sekret-value-123 is not registered")


@pytest.fixture()
def fake_sources(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(fetchers.FETCHERS, "fred", "tests.test_runner:OkFetcher")
    monkeypatch.setitem(fetchers.FETCHERS, "broken", "tests.test_runner:FailingFetcher")
    monkeypatch.setenv("COMTRADE_SUBSCRIPTION_KEY", "sekret-value-123")  # so redact() knows it
    OkFetcher.calls = []
    from tracker import settings

    real = {name: {**cfg, "enabled": False} for name, cfg in settings.load_sources().items()}
    return {
        **real,  # keep every real source (disabled) so registry validation still passes
        "broken": {"enabled": True},
        "fred": {"enabled": True, "refetch_window_days": 30},
        "disabled": {"enabled": False},
    }


def test_failure_is_isolated(tmp_path, fake_sources) -> None:
    results = run_all(data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY)
    by_source = {r.source: r for r in results}
    assert set(by_source) == {"broken", "fred"}  # disabled source not run
    assert by_source["broken"].status == "failed"
    assert by_source["fred"].status == "success" and by_source["fred"].rows_written == 2
    log = store.read_fetch_log(tmp_path)
    assert sorted(log["status"]) == ["failed", "success"]
    assert log["run_id"].nunique() == 1
    error = log.loc[log["source"] == "broken", "error"].item()
    assert "sekret-value-123" not in error and "***" in error
    assert "sekret-value-123" not in summary_table(results)


def test_unknown_fetcher_fails_cleanly(tmp_path, fake_sources) -> None:
    fake_sources["nofetcher"] = {"enabled": True}
    results = run_all(data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY)
    nof = next(r for r in results if r.source == "nofetcher")
    assert nof.status == "failed" and "no fetcher registered" in nof.error
    assert next(r for r in results if r.source == "fred").status == "success"


def test_single_source_and_unknown_source(tmp_path, fake_sources) -> None:
    results = run_all(only="fred", data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY)
    assert [r.source for r in results] == ["fred"]
    with pytest.raises(KeyError, match="not in config/sources.yaml"):
        run_all(only="nope", data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY)


def test_resume_from_last_success_minus_window(tmp_path, fake_sources) -> None:
    today = date(2026, 9, 27)
    run_all(only="fred", data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY, today=today)
    assert OkFetcher.calls[-1] == date(2015, 1, 1)  # first run: backfill start
    run_all(only="fred", data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY, today=today)
    last_ok = store.read_fetch_log(tmp_path)["started_at"].min().date()
    assert OkFetcher.calls[-1] == last_ok - pd.Timedelta(days=30)
    run_all(only="fred", since=date(2020, 1, 1), data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY)
    assert OkFetcher.calls[-1] == date(2020, 1, 1)
    run_all(only="fred", backfill=True, data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY)
    assert OkFetcher.calls[-1] == date(2015, 1, 1)


def test_resume_from_backfill_start_option() -> None:
    log = pd.DataFrame({"source": ["x"], "status": ["success"], "started_at": pd.to_datetime(["2026-09-01"], utc=True)})
    cfg = {"refetch_window_days": 10, "resume_from": "backfill_start", "backfill_start": "2019-01-01"}
    assert resume_start("x", cfg, REGISTRY, log, date(2015, 1, 1), date(2026, 9, 27)) == date(2019, 1, 1)


def test_resume_ignores_failed_runs() -> None:
    log = pd.DataFrame({
        "source": ["x", "x"], "status": ["success", "failed"],
        "started_at": pd.to_datetime(["2026-09-01", "2026-09-20"], utc=True),
    })
    start = resume_start("x", {"refetch_window_days": 10}, REGISTRY, log, date(2015, 1, 1), date(2026, 9, 27))
    assert start == date(2026, 8, 22)


class IssueFetcher(OkFetcher):
    """Fake source that also reports a data-quality issue."""

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        self.report_issue("fred_brent", "WLD", "missing_hours", "warning", "2024-01-01: 20 of 24 hours")
        return super()._fetch(start, end)


def test_fetcher_issues_are_written_to_quality_issues(tmp_path, fake_sources, monkeypatch) -> None:
    monkeypatch.setitem(fetchers.FETCHERS, "fred", "tests.test_runner:IssueFetcher")
    run_all(only="fred", data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY)
    issues = store.read_quality_issues(tmp_path)
    assert issues[["source", "check", "country"]].values.tolist() == [["fred", "missing_hours", "WLD"]]


def test_rerun_writes_zero_rows(tmp_path, fake_sources) -> None:
    first = run_all(only="fred", data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY)
    second = run_all(only="fred", data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY)
    assert first[0].rows_written == 2 and second[0].rows_written == 0


def test_expected_next_by() -> None:
    # Monthly, lag 17: August data (2026-08-01) -> September due 2026-10-01 + 17 + 7 grace.
    assert expected_next_by(pd.Timestamp("2026-08-01"), "M", 17) == pd.Timestamp("2026-10-25", tz="UTC")
    assert expected_next_by(pd.Timestamp("2026-09-22"), "D", 7) == pd.Timestamp("2026-10-04", tz="UTC")


def test_status_reports_fresh_and_stale(tmp_path, fake_sources) -> None:
    run_all(data_dir=tmp_path, sources_config=fake_sources, registry=REGISTRY)
    early = pd.Timestamp("2024-01-05", tz="UTC")
    table = source_status(REGISTRY, fake_sources, tmp_path, now=early).set_index("source")
    assert table.loc["broken", "last_run"] == "failed" and table.loc["broken", "freshness"] == "no data"
    assert table.loc["disabled", "freshness"] == "never run"
    assert not table.loc["disabled", "enabled"]
    assert table.loc["fred", "rows_in_lake"] == 2
    assert table.loc["fred", "latest_data"] == date(2024, 1, 3)
    fresh = indicator_freshness(REGISTRY, tmp_path, now=early).set_index("indicator")
    assert not fresh.loc["fred_brent", "stale"]
    assert fresh.loc["fred_indpro", "stale"]  # registered but no data
    late = source_status(REGISTRY, fake_sources, tmp_path, now=pd.Timestamp("2024-03-01", tz="UTC"))
    assert late.set_index("source").loc["fred", "freshness"].startswith("STALE")


def test_retired_indicators_are_never_stale(tmp_path) -> None:
    # fred_copper is retired (PLAN D56): stored history, no next period due, not stale even with no data.
    fetcher = OkFetcher(data_dir=tmp_path)
    rows = fetcher.frame("fred_copper", pd.DataFrame({"date": ["2026-07-01"], "country": "WLD", "value": [13542.8]}))
    store.append_changes(rows, "run1", tmp_path)
    fresh = indicator_freshness(REGISTRY, tmp_path, now=pd.Timestamp("2027-06-01", tz="UTC")).set_index("indicator")
    assert fresh.loc["fred_copper", "retired"] and not fresh.loc["fred_copper", "stale"]
    assert fresh.loc["fred_copper", "last_date"] == date(2026, 7, 1) and fresh.loc["fred_copper", "next_due"] is None
    assert fresh.loc["fred_eu_gas", "retired"] and not fresh.loc["fred_eu_gas", "stale"]  # no data at all
    assert not fresh.loc["imf_copper", "retired"] and fresh.loc["imf_copper", "stale"]


def test_cli_fetch_and_status(tmp_path, fake_sources, monkeypatch, capsys) -> None:
    from tracker import settings
    from tracker.__main__ import main

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "load_sources", lambda config_dir=None: fake_sources)
    assert main(["fetch"]) == 1  # one source failed
    out = capsys.readouterr().out
    assert "broken" in out and "failed" in out and "success" in out
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "freshness" in out and "Registry:" in out
