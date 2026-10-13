"""Tests for the append-only store: write rule, vintages, as_of reads, compaction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tracker import store

T0 = pd.Timestamp("2024-01-10", tz="UTC")
T1 = pd.Timestamp("2024-02-10", tz="UTC")
T2 = pd.Timestamp("2024-03-10", tz="UTC")


def obs(
    rows: list[tuple[str, str, float]],
    fetched_at: pd.Timestamp = T0,
    available_at: pd.Timestamp | list[pd.Timestamp] | None = None,
    indicator: str = "test_metric",
    vintage_type: str = "estimated",
    frequency: str = "M",
) -> pd.DataFrame:
    """Build an observations frame from (date, country, value) tuples."""
    df = pd.DataFrame(rows, columns=["date", "country", "value"])
    df["indicator"] = indicator
    df["unit"] = "index"
    df["frequency"] = frequency
    df["source"] = indicator.split("_", 1)[0]
    df["fetched_at"] = fetched_at
    df["available_at"] = available_at if available_at is not None else fetched_at
    df["vintage_type"] = vintage_type
    return df[store.COLUMNS]


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


# ------------------------------------------------------------------ empty store


def test_empty_store_latest_returns_empty_frame_with_schema(data_dir: Path) -> None:
    df = store.latest(data_dir=data_dir)
    assert df.empty
    assert list(df.columns) == store.COLUMNS
    assert str(df["date"].dtype) == "datetime64[us]"
    assert store.latest(indicators=["test_metric"], as_of="2020-01-01", data_dir=data_dir).empty


def test_empty_store_meta_and_compact(data_dir: Path) -> None:
    assert store.read_fetch_log(data_dir).empty
    assert store.read_quality_issues(data_dir).empty
    assert store.history(data_dir=data_dir).empty
    assert store.source_summary(data_dir).empty
    report = store.compact(data_dir)
    assert (report["rows"] == 0).all()


def test_append_empty_frame_writes_nothing(data_dir: Path) -> None:
    assert store.append_changes(obs([]), "run0", data_dir) == 0
    assert not store.observations_dir(data_dir).exists()


# ------------------------------------------------------------------ write rule


def test_new_rows_are_written(data_dir: Path) -> None:
    df = obs([("2023-01-01", "DEU", 100.0), ("2023-02-01", "DEU", 101.0), ("2023-01-01", "FRA", 99.0)])
    assert store.append_changes(df, "run1", data_dir) == 3
    files = list(store.source_dir("test", data_dir).glob("*.parquet"))
    assert [f.name for f in files] == ["run1.parquet"]
    out = store.latest(data_dir=data_dir)
    assert len(out) == 3
    assert out.loc[(out["country"] == "DEU") & (out["date"] == "2023-02-01"), "value"].item() == 101.0


def test_unchanged_rows_are_skipped(data_dir: Path) -> None:
    df = obs([("2023-01-01", "DEU", 100.0), ("2023-02-01", "DEU", 101.0)])
    store.append_changes(df, "run1", data_dir)
    again = obs([("2023-01-01", "DEU", 100.0), ("2023-02-01", "DEU", 101.0)], fetched_at=T1)
    assert store.append_changes(again, "run2", data_dir) == 0
    assert len(list(store.source_dir("test", data_dir).glob("*.parquet"))) == 1


def test_float_noise_is_not_a_revision(data_dir: Path) -> None:
    store.append_changes(obs([("2023-01-01", "DEU", 0.1 + 0.2)]), "run1", data_dir)
    assert store.append_changes(obs([("2023-01-01", "DEU", 0.3)], fetched_at=T1), "run2", data_dir) == 0


def test_change_tolerance_can_be_raised_per_call(data_dir: Path) -> None:
    store.append_changes(obs([("2023-01-01", "DEU", 41.87)]), "run1", data_dir)
    jitter = obs([("2023-01-01", "DEU", 41.87 * (1 + 2e-7))], fetched_at=T1)
    assert store.append_changes(jitter, "run2", data_dir, rel_tol=1e-6) == 0
    assert store.append_changes(jitter.assign(value=41.87 * 0.99), "run3", data_dir, rel_tol=1e-6) == 1
    # With the default tolerance the same jitter would be a revision.
    assert store.append_changes(obs([("2023-01-01", "DEU", 41.87 * 0.99 * (1 + 2e-7))], fetched_at=T2), "run4", data_dir) == 1


def test_only_new_and_changed_rows_of_a_mixed_batch_are_written(data_dir: Path) -> None:
    store.append_changes(obs([("2023-01-01", "DEU", 100.0), ("2023-02-01", "DEU", 101.0)]), "run1", data_dir)
    batch = obs(
        [("2023-01-01", "DEU", 100.0), ("2023-02-01", "DEU", 102.0), ("2023-03-01", "DEU", 103.0)],
        fetched_at=T1,
    )
    assert store.append_changes(batch, "run2", data_dir) == 2
    assert len(store.history(data_dir=data_dir)) == 4


def test_revised_value_creates_new_vintage(data_dir: Path) -> None:
    store.append_changes(obs([("2023-01-01", "DEU", 100.0)], available_at=T0), "run1", data_dir)
    revised = obs([("2023-01-01", "DEU", 100.5)], fetched_at=T1, available_at=T1, vintage_type="live")
    assert store.append_changes(revised, "run2", data_dir) == 1
    hist = store.history(data_dir=data_dir)
    assert hist["value"].tolist() == [100.0, 100.5]
    assert hist["available_at"].tolist() == [T0, T1]


def test_estimated_revision_is_stamped_when_seen(data_dir: Path) -> None:
    """A revised value can't have been public before we downloaded it."""
    est = pd.Timestamp("2023-03-15", tz="UTC")
    store.append_changes(obs([("2023-01-01", "DEU", 100.0)], available_at=est), "run1", data_dir)
    revised = obs([("2023-01-01", "DEU", 99.0)], fetched_at=T1, available_at=est)
    assert store.append_changes(revised, "run2", data_dir) == 1
    hist = store.history(data_dir=data_dir)
    new = hist.iloc[-1]
    assert new["available_at"] == T1
    assert new["vintage_type"] == "live"
    # Before T1 we still see the original value; re-sending the revision is a no-op.
    assert store.latest(as_of=T1 - pd.Timedelta(days=1), data_dir=data_dir)["value"].item() == 100.0
    again = obs([("2023-01-01", "DEU", 99.0)], fetched_at=T2, available_at=est)
    assert store.append_changes(again, "run3", data_dir) == 0


def test_available_at_is_capped_at_fetched_at(data_dir: Path) -> None:
    df = obs([("2024-01-01", "DEU", 1.0)], fetched_at=T0, available_at=T0 + pd.Timedelta(days=30))
    store.append_changes(df, "run1", data_dir)
    assert store.latest(data_dir=data_dir)["available_at"].item() == T0


# ------------------------------------------------------------- source vintages


def alfred_like(vintages: list[tuple[str, float]], fetched_at: pd.Timestamp = T2) -> pd.DataFrame:
    """One observation (2023-01-01, USA) with several true vintages."""
    frames = [
        obs(
            [("2023-01-01", "USA", value)],
            fetched_at=fetched_at,
            available_at=pd.Timestamp(when, tz="UTC"),
            indicator="fred_indpro",
            vintage_type="source_vintage",
        )
        for when, value in vintages
    ]
    return pd.concat(frames, ignore_index=True)


def test_source_vintages_are_all_kept_and_rerun_is_noop(data_dir: Path) -> None:
    df = alfred_like([("2023-02-15", 100.0), ("2023-03-15", 100.4), ("2023-04-14", 100.1)])
    assert store.append_changes(df, "run1", data_dir) == 3
    assert store.append_changes(df.assign(fetched_at=T2 + pd.Timedelta(days=1)), "run2", data_dir) == 0


def test_new_source_vintage_is_appended(data_dir: Path) -> None:
    store.append_changes(alfred_like([("2023-02-15", 100.0), ("2023-03-15", 100.4)]), "run1", data_dir)
    later = alfred_like([("2023-02-15", 100.0), ("2023-03-15", 100.4), ("2023-06-15", 99.8)])
    assert store.append_changes(later, "run2", data_dir) == 1
    assert store.latest(data_dir=data_dir)["value"].item() == 99.8


def test_latest_as_of_follows_source_vintages(data_dir: Path) -> None:
    store.append_changes(
        alfred_like([("2023-02-15", 100.0), ("2023-03-15", 100.4), ("2023-04-14", 100.1)]), "run1", data_dir
    )

    def value_as_of(when: str) -> float | None:
        df = store.latest(indicators=["fred_indpro"], as_of=when, data_dir=data_dir)
        return None if df.empty else df["value"].item()

    assert value_as_of("2023-02-14") is None  # not yet published
    assert value_as_of("2023-02-15") == 100.0  # boundary is inclusive
    assert value_as_of("2023-03-20") == 100.4
    assert value_as_of("2023-12-31") == 100.1
    assert store.latest(data_dir=data_dir)["value"].item() == 100.1


# ------------------------------------------------------------------ read rule


def test_latest_returns_newest_vintage(data_dir: Path) -> None:
    store.append_changes(obs([("2023-01-01", "DEU", 1.0)], fetched_at=T0, vintage_type="live"), "r1", data_dir)
    store.append_changes(obs([("2023-01-01", "DEU", 2.0)], fetched_at=T1, vintage_type="live"), "r2", data_dir)
    store.append_changes(obs([("2023-01-01", "DEU", 3.0)], fetched_at=T2, vintage_type="live"), "r3", data_dir)
    out = store.latest(data_dir=data_dir)
    assert len(out) == 1 and out["value"].item() == 3.0


def test_latest_as_of_returns_what_was_known(data_dir: Path) -> None:
    store.append_changes(obs([("2023-01-01", "DEU", 1.0)], fetched_at=T0, vintage_type="live"), "r1", data_dir)
    store.append_changes(
        obs([("2023-01-01", "DEU", 2.0), ("2023-02-01", "DEU", 5.0)], fetched_at=T1, vintage_type="live"),
        "r2",
        data_dir,
    )
    before = store.latest(as_of=T0, data_dir=data_dir)
    assert before["value"].tolist() == [1.0]
    between = store.latest(as_of=T1 - pd.Timedelta(seconds=1), data_dir=data_dir)
    assert between["value"].tolist() == [1.0]
    after = store.latest(as_of=T1, data_dir=data_dir)
    assert after["value"].tolist() == [2.0, 5.0]
    assert store.latest(as_of="2023-06-01", data_dir=data_dir).empty


def test_latest_filters(data_dir: Path) -> None:
    rows = [(f"2023-{m:02d}-01", c, float(m)) for m in range(1, 7) for c in ("DEU", "FRA")]
    store.append_changes(obs(rows), "r1", data_dir)
    store.append_changes(obs([("2023-01-01", "DEU", 7.0)], indicator="other_metric"), "r1", data_dir)
    out = store.latest(indicators=["test_metric"], countries=["FRA"], start="2023-02-01", end="2023-04-01", data_dir=data_dir)
    assert out["country"].unique().tolist() == ["FRA"]
    assert out["date"].min() == pd.Timestamp("2023-02-01") and out["date"].max() == pd.Timestamp("2023-04-01")
    assert store.latest(indicators=["other_metric"], data_dir=data_dir)["value"].tolist() == [7.0]
    assert store.latest(indicators=["missing_metric"], data_dir=data_dir).empty
    assert set(store.latest(data_dir=data_dir)["source"]) == {"test", "other"}


def test_timestamps_round_trip_as_utc(data_dir: Path) -> None:
    store.append_changes(obs([("2023-01-01", "DEU", 1.0)]), "r1", data_dir)
    out = store.latest(data_dir=data_dir)
    assert str(out["fetched_at"].dtype) == "datetime64[us, UTC]"
    assert out["fetched_at"].item() == T0


# ----------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.drop(columns=["unit"]), "missing=['unit']"),
        (lambda d: d.assign(extra=1), "unexpected=['extra']"),
        (lambda d: d.assign(frequency="X"), "frequency"),
        (lambda d: d.assign(vintage_type="guess"), "vintage_type"),
        (lambda d: d.assign(country="Germany"), "country"),
        (lambda d: d.assign(indicator="other_metric"), "must start with '<source>_'"),
        (lambda d: d.assign(value="abc"), "value"),
        (lambda d: d.assign(value=float("nan")), "missing values"),
        (lambda d: d.assign(fetched_at=pd.Timestamp("2024-01-01")), "timezone-aware"),
        (lambda d: d.assign(date=pd.Timestamp("2023-01-01 12:00")), "time part"),
        (lambda d: pd.concat([d, d]), "duplicate"),
    ],
)
def test_validation_errors_are_specific(mutate, message: str) -> None:
    df = mutate(obs([("2023-01-01", "DEU", 1.0)]))
    with pytest.raises(store.SchemaError, match=message.replace("[", r"\[").replace("]", r"\]")):
        store.validate(df)


def test_chokepoint_and_world_codes_are_valid() -> None:
    store.validate(obs([("2023-01-01", "CHK_SUEZ", 1.0), ("2023-01-01", "WLD", 2.0)]))


# ------------------------------------------------------------------ compaction


def test_compaction_changes_no_query_results(data_dir: Path) -> None:
    for i, when in enumerate([T0, T1, T2]):
        rows = [(f"2023-{m:02d}-01", c, float(m + i * (m % 2))) for m in range(1, 13) for c in ("DEU", "FRA")]
        store.append_changes(obs(rows, fetched_at=when, vintage_type="live"), f"r{i}", data_dir)
    store.append_changes(obs([("2023-01-01", "DEU", 1.0)], indicator="other_metric"), "r0", data_dir)

    queries = [
        dict(),
        dict(as_of=T0),
        dict(as_of=T1 + pd.Timedelta(hours=1)),
        dict(countries=["FRA"], start="2023-03-01"),
    ]
    before = [store.latest(data_dir=data_dir, **q) for q in queries]
    hist_before = store.history(data_dir=data_dir)
    files_before = len(list(store.source_dir("test", data_dir).glob("*.parquet")))
    assert files_before == 3

    report = store.compact(data_dir, include_meta=False)
    assert report.set_index("folder").loc["source=test", "rows"] == len(hist_before[hist_before["source"] == "test"])
    assert len(list(store.source_dir("test", data_dir).glob("*.parquet"))) == 1

    after = [store.latest(data_dir=data_dir, **q) for q in queries]
    for b, a in zip(before, after):
        pd.testing.assert_frame_equal(b, a)
    pd.testing.assert_frame_equal(hist_before, store.history(data_dir=data_dir))

    # Appending after compaction still follows the write rule.
    same = obs([("2023-01-01", "DEU", 3.0)], fetched_at=T2, vintage_type="live")  # latest value (i=2)
    assert store.append_changes(same, "r9", data_dir) == 0
    assert store.append_changes(same.assign(value=4.0), "r9", data_dir) == 1


# ------------------------------------------------------------------ meta tables


def test_fetch_log_round_trip(data_dir: Path) -> None:
    row = {
        "run_id": "r1", "source": "fred", "started_at": T0, "finished_at": T0 + pd.Timedelta(seconds=5),
        "status": "success", "rows_fetched": 10, "rows_written": 3, "error": None,
    }
    store.append_fetch_log(pd.DataFrame([row]), data_dir)
    store.append_fetch_log(pd.DataFrame([{**row, "run_id": "r2", "status": "failed", "error": "boom"}]), data_dir)
    log = store.read_fetch_log(data_dir)
    assert log["run_id"].tolist() == ["r1", "r2"]
    assert log["rows_written"].tolist() == [3, 3]
    assert log["error"].isna().tolist() == [True, False]
    store.compact(data_dir)
    assert len(store.read_fetch_log(data_dir)) == 2


def test_quality_issues_round_trip(data_dir: Path) -> None:
    issue = {
        "checked_at": T0, "source": "fred", "indicator": "fred_indpro", "country": "USA",
        "check": "stale", "severity": "warning", "detail": "3 days late",
    }
    store.append_quality_issues(pd.DataFrame([issue]), data_dir)
    assert store.read_quality_issues(data_dir)["check"].tolist() == ["stale"]


def test_meta_missing_columns_rejected(data_dir: Path) -> None:
    with pytest.raises(store.SchemaError, match="missing columns"):
        store.append_fetch_log(pd.DataFrame([{"run_id": "r1"}]), data_dir)


def test_run_ids_are_unique_and_sortable() -> None:
    a = store.new_run_id(pd.Timestamp("2026-01-01 07:00", tz="UTC"))
    b = store.new_run_id(pd.Timestamp("2026-01-02 07:00", tz="UTC"))
    assert a.startswith("20260101T070000Z_") and a < b
    assert store.new_run_id() != store.new_run_id()


def test_meta_rows_written_in_one_clock_tick_keep_their_order(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = pd.Timestamp("2026-09-30 12:00", tz="UTC")
    monkeypatch.setattr(store.pd.Timestamp, "now", classmethod(lambda cls, tz=None: frozen))
    for n in range(20):
        store.append_fetch_log(pd.DataFrame([{"run_id": f"r{n:02d}", "source": "fred", "started_at": T0,
                                              "finished_at": T0, "status": "success", "rows_fetched": 1,
                                              "rows_written": 1, "error": None}]), data_dir)
    assert store.read_fetch_log(data_dir)["run_id"].tolist() == [f"r{n:02d}" for n in range(20)]
