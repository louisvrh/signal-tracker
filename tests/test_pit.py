"""Tests for the point-in-time backtest engine (S22): as-of reads, end values, walk-forward,
the hypotheses log and, above all, the leak test."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tracker import store
from tracker.backtest import pit
from tracker.registry import Indicator, Registry, load_registry
from tracker.signals import chain as chains

COUNTRIES = ["DEU", "CHN"]
DAY = pd.Timedelta(days=1)


def indicator(ind_id: str, frequency: str, unit: str = "GWh", **kw) -> Indicator:
    return Indicator(id=ind_id, source="test", name=ind_id, unit=unit, frequency=frequency, category="energy",
                     role="alt", higher_is_stronger=True, default_transform="level", publication_lag_days=3,
                     refetch_window_days=30, countries=COUNTRIES, **kw)


@pytest.fixture(scope="module")
def registry() -> Registry:
    real = load_registry()
    inds = {i.id: i for i in (indicator("test_daily", "D"), indicator("test_monthly", "M", unit="index"),
                              indicator("test_official", "M", unit="index"))}
    return Registry(indicators=inds, countries={c: real.countries[c] for c in COUNTRIES})


def frame(dates, country: str, ind: str, values, available, fetched=None, vintage="estimated") -> pd.DataFrame:
    available = pd.DatetimeIndex(available)
    return pd.DataFrame({
        "date": pd.DatetimeIndex(dates), "country": country, "indicator": ind, "value": np.asarray(values, float),
        "unit": "x", "frequency": "D" if ind == "test_daily" else "M", "source": "test",
        "fetched_at": pd.DatetimeIndex(fetched) if fetched is not None else available,
        "available_at": available, "vintage_type": vintage,
    })


def clean_vintages(seed: int = 1) -> pd.DataFrame:
    """Daily alt data (published 3 days on), a monthly alt series and an official monthly series
    published ~40 days after the month and revised 30 days later: a vintage history like the lake's."""
    rng = np.random.default_rng(seed)
    parts = []
    for n, country in enumerate(COUNTRIES):
        days = pd.date_range("2015-01-01", "2021-12-31", freq="D")
        t = np.arange(len(days))
        level = 1000 + 50 * n + 0.02 * t + 60 * np.sin(2 * np.pi * t / 365.25) + 25 * (days.dayofweek < 5)
        daily = level + rng.normal(0, 8, len(days))
        parts.append(frame(days, country, "test_daily", daily, days + 3 * DAY + pd.Timedelta(hours=7), vintage="live"))

        months = pd.date_range("2010-01-01", "2021-11-01", freq="MS")
        m = np.arange(len(months))
        cycle = 100 + 0.1 * m + 3 * np.sin(2 * np.pi * m / 40) + 2 * np.sin(2 * np.pi * m / 12)
        month_end = months + pd.offsets.MonthEnd(0)
        parts.append(frame(months, country, "test_monthly", cycle + rng.normal(0, 0.5, len(m)), month_end + 10 * DAY))
        first = cycle + rng.normal(0, 0.8, len(m))
        published = month_end + 40 * DAY
        parts.append(frame(months, country, "test_official", first, published))
        parts.append(frame(months, country, "test_official", first * 1.01, published + 30 * DAY, vintage="live"))
    df = pd.concat(parts, ignore_index=True)
    for col in ("fetched_at", "available_at"):
        df[col] = pd.to_datetime(df[col]).dt.tz_localize("UTC")
    return df


def poison(vintages: pd.DataFrame, cut: pd.Timestamp, seed: int = 99) -> pd.DataFrame:
    """The same history up to `cut`, with everything published after it poisoned.

    Every value published after `cut` is replaced by a huge random number;
    every value known at `cut` gets a poisoned revision one second after it;
    and poisoned new periods (a year of them) are added after the last date.
    """
    rng = np.random.default_rng(seed)
    out = vintages.copy()
    future = out["available_at"] > cut
    out.loc[future, "value"] = rng.uniform(-1e9, 1e9, int(future.sum()))
    known = out[~future].drop_duplicates(store.KEY, keep="last").copy()
    known["available_at"] = known["fetched_at"] = cut + pd.Timedelta(seconds=1)
    known["value"] = -1e9
    known["vintage_type"] = "live"
    extra = []
    for (ind, country), part in out.groupby(["indicator", "country"]):
        freq = "D" if ind == "test_daily" else "MS"
        dates = pd.date_range(part["date"].max() + DAY, periods=365 if freq == "D" else 12, freq=freq)
        stamp = max(cut, part["available_at"].max()) + DAY
        extra.append(frame(dates, country, ind, rng.uniform(1e8, 1e9, len(dates)), [stamp.tz_localize(None)] * len(dates)))
    extra_df = pd.concat(extra, ignore_index=True)
    for col in ("fetched_at", "available_at"):
        extra_df[col] = pd.to_datetime(extra_df[col]).dt.tz_localize("UTC")
    return pd.concat([out, known, extra_df], ignore_index=True)


# ------------------------------------------------------------ as-of reads


def lake(data_dir: Path) -> store.Vintages:
    """A small lake written through the store: estimated rows, a live revision and ALFRED-style vintages."""
    t0, t1 = pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-03-10", tz="UTC")
    base = frame(["2023-11-01", "2023-12-01", "2023-11-01"], "DEU", "test_monthly", [100, 101, 99],
                 [pd.Timestamp("2023-12-20"), pd.Timestamp("2024-01-05"), pd.Timestamp("2023-12-20")])
    base.loc[2, "country"] = "CHN"
    base["fetched_at"] = t0
    base["available_at"] = base["available_at"].dt.tz_localize("UTC")
    store.append_changes(base, "run1", data_dir)
    revised = base.copy()
    revised["value"] = [100.5, 101, 99]
    revised["fetched_at"] = revised["available_at"] = t1
    store.append_changes(revised, "run2", data_dir)
    sv = frame(["2023-10-01"] * 3, "DEU", "test_official", [1.0, 1.2, 1.1],
               ["2023-11-15", "2023-12-15", "2024-02-15"], vintage="source_vintage")
    sv["fetched_at"] = t1
    sv["available_at"] = sv["available_at"].dt.tz_localize("UTC")
    store.append_changes(sv, "run3", data_dir)
    return store.Vintages.load(data_dir=data_dir)


def test_vintages_match_store_latest_on_every_date(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    vintages = lake(data_dir)
    moments = [None, "2023-01-01", "2023-11-15", "2023-12-19T23:59:59", "2023-12-20", "2024-01-05", "2024-01-09",
               "2024-02-15", "2024-03-09", "2024-03-10", "2024-03-10T00:00:01", pd.Timestamp("2024-06-01", tz="UTC")]
    for as_of in moments:
        for filters in ({}, {"indicators": ["test_monthly"]}, {"countries": ["CHN"]},
                        {"start": "2023-11-01", "end": "2023-11-30"}):
            expected = store.latest(as_of=as_of, data_dir=data_dir, **filters)
            pd.testing.assert_frame_equal(vintages.latest(as_of=as_of, **filters), expected)


def test_vintages_of_an_empty_lake(tmp_path: Path) -> None:
    vintages = store.Vintages.load(data_dir=tmp_path / "data")
    assert len(vintages) == 0
    assert list(vintages.latest(as_of="2024-01-01").columns) == store.COLUMNS


def test_vintages_match_store_latest_on_a_revised_history(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    df = clean_vintages()
    df = df[df["date"] >= "2021-01-01"]
    first = df.drop_duplicates(store.KEY, keep="first")
    store.append_changes(first.assign(fetched_at=pd.Timestamp("2022-06-01", tz="UTC")), "run1", data_dir)
    revised = df[df["vintage_type"] == "live"].drop_duplicates(store.KEY, keep="last")
    revised = revised[revised["indicator"] == "test_official"]
    store.append_changes(revised.assign(fetched_at=pd.Timestamp("2022-07-01", tz="UTC")), "run2", data_dir)
    vintages = store.Vintages.load(data_dir=data_dir)
    assert len(vintages) > len(first)
    for as_of in list(pit.as_of_dates("2021-01-01", "2022-08-01", "MS")) + [pd.Timestamp("2022-07-01", tz="UTC")]:
        pd.testing.assert_frame_equal(vintages.latest(as_of=as_of), store.latest(as_of=as_of, data_dir=data_dir))


def test_as_of_dates_are_utc_midnights() -> None:
    dates = pit.as_of_dates("2024-01-01", "2024-01-31", "W-MON")
    assert list(dates.day) == [1, 8, 15, 22, 29]
    assert str(dates.tz) == "UTC" and (dates == dates.normalize()).all()


# ------------------------------------------------------------ end values


def test_series_signals_use_only_the_end_value_of_the_snapshot(registry: Registry) -> None:
    vintages = store.Vintages(clean_vintages())
    as_of = pd.Timestamp("2020-06-15", tz="UTC")
    snap = pit.snapshot(vintages, as_of)
    sig = pit.ChainSignal("cycle", "cf(1.5y,4y)", ("test_monthly",))
    out = pit.series_signals(snap, as_of, [sig], registry)
    assert set(out["country"]) == set(COUNTRIES)
    for row in out.itertuples():
        values = pit.as_series(snap[(snap["indicator"] == "test_monthly") & (snap["country"] == row.country)])
        full = chains.apply("cf(1.5y,4y)", values, chains.series_context(registry, "test_monthly", row.country, values))
        assert row.date == values.index[-1] == row.last_date
        assert row.value == pytest.approx(full.values.iloc[-1])
        assert row.provisional  # a band-pass's end always revises
        assert row.vintage_type == "estimated"
    assert out["last_date"].max() == pd.Timestamp("2020-05-01")  # May was out on 10 June, June isn't yet


def test_a_chain_that_cannot_run_yet_gives_a_reason_not_a_crash(registry: Registry) -> None:
    vintages = store.Vintages(clean_vintages())
    early = pd.Timestamp("2016-01-01", tz="UTC")
    signals = [pit.ChainSignal("sa", "stl", ("test_daily",)), pit.ChainSignal("wx", "wx", ("test_daily",))]
    out = pit.series_signals(pit.snapshot(vintages, early), early, signals, registry)
    assert out["value"].isna().all()
    assert out.loc[out["signal"] == "sa", "problem"].str.contains("3 years").all()
    assert out.loc[out["signal"] == "wx", "problem"].str.contains("weather", case=False).all()


def test_chain_signals_reject_typos_at_once() -> None:
    with pytest.raises(chains.ChainError):
        pit.ChainSignal("bad", "sma(28x)", ("test_daily",))


def test_preset_signals_and_countries_follow_the_registry(registry: Registry) -> None:
    signals = pit.preset_signals({"Raw": "level", "Smoothed": "sma(3m)"}, ["test_monthly"])
    assert [s.name for s in signals] == ["Raw", "Smoothed"]
    as_of = pd.Timestamp("2021-01-04", tz="UTC")
    out = pit.compute(store.Vintages(clean_vintages()), registry, [as_of], series=signals)
    assert set(out["country"]) == set(COUNTRIES) and len(out) == 4
    only = pit.compute(store.Vintages(clean_vintages()), registry, [as_of],
                       series=[pit.ChainSignal("Raw", "level", ("test_monthly",), ("CHN",))])
    assert list(only["country"]) == ["CHN"]


def test_unchanged_series_reuse_results_without_changing_them(registry: Registry) -> None:
    vintages = store.Vintages(clean_vintages())
    dates = pit.as_of_dates("2020-01-06", "2020-03-30", "W-MON")
    sig = [pit.ChainSignal("sa", "hol>stl", ("test_monthly", "test_official"))]  # official: revised in between
    cached = pit.compute(vintages, registry, dates, series=sig)
    fresh = pd.concat([pit.series_signals(pit.snapshot(vintages, d), d, sig, registry) for d in dates],
                      ignore_index=True)
    pd.testing.assert_frame_equal(cached, fresh)
    monthly = cached[cached["indicator"] == "test_monthly"]
    assert monthly.groupby("country")["last_date"].nunique().max() == 4  # four months over 13 weeks


def test_a_revision_without_a_new_period_is_recomputed(registry: Registry) -> None:
    df = clean_vintages()
    old = df[(df["indicator"] == "test_official") & (df["country"] == "DEU") & (df["date"] == "2019-06-01")]
    published = pd.Timestamp("2020-01-20 12:00", tz="UTC")  # between two test dates, no new month
    revision = old.tail(1).assign(value=old["value"].iloc[-1] * 1.5, available_at=published, fetched_at=published)
    vintages = store.Vintages(pd.concat([df, revision], ignore_index=True))
    dates = pd.to_datetime(["2020-01-20", "2020-01-21"], utc=True)
    out = pit.compute(vintages, registry, dates, series=[pit.ChainSignal("lin", "lin", ("test_official",), ("DEU",))])
    assert out["last_date"].nunique() == 1
    assert out["value"].iloc[0] != pytest.approx(out["value"].iloc[1])


def test_real_time_history_keeps_each_period_as_last_seen_at_its_end() -> None:
    pit_rows = pd.DataFrame({
        "as_of": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-02-03", "2020-02-10"], utc=True),
        "signal": "s", "indicator": "i", "country": "DEU",
        "date": pd.to_datetime(["2019-11-01", "2019-11-01", "2019-12-01", "2019-12-01"]),
        "value": [1.0, 1.5, 2.0, np.nan],
    })
    at = pit.real_time_history(pit_rows, "2020-02-10", "s", "DEU")
    assert at.to_dict() == {pd.Timestamp("2019-11-01"): 1.5, pd.Timestamp("2019-12-01"): 2.0}
    assert pit.real_time_history(pit_rows, "2020-01-10", "s", "DEU").to_dict() == {pd.Timestamp("2019-11-01"): 1.0}


# ------------------------------------------------------------ panel hook (S19)


def test_panel_signals_see_the_snapshot_and_keep_each_countrys_end(registry: Registry) -> None:
    seen = []

    def spread(snap: pd.DataFrame, as_of: pd.Timestamp, reg: Registry) -> pd.DataFrame:
        seen.append((as_of, snap["available_at"].max()))
        m = snap[snap["indicator"] == "test_monthly"]
        return m.assign(value=m["value"] - m.groupby("date")["value"].transform("mean"))[["date", "country", "value"]]

    dates = pit.as_of_dates("2020-01-06", "2020-02-24", "W-MON")
    out = pit.compute(store.Vintages(clean_vintages()), registry, dates, panel={"spread": spread})
    assert all(latest <= as_of for as_of, latest in seen)
    assert len(out) == len(dates) * 2 and out["indicator"].isna().all()
    assert (out["date"] == out["last_date"]).all()
    assert out.groupby("as_of")["value"].sum().abs().max() < 1e-9  # two countries around their mean


def test_s19_panel_signals_plug_in_from_their_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    # S19's composite.py and divergence.py (their engine tests are in test_signals.py).
    assert {"activity", "activity_z", "d_mkt", "d_off"} <= set(pit.available_panel_signals())
    fake = types.ModuleType("tracker.signals.composite")
    fake.PANEL_SIGNALS = {"activity_score": lambda snap, as_of, reg: pd.DataFrame()}
    monkeypatch.setitem(sys.modules, "tracker.signals.composite", fake)
    monkeypatch.setitem(sys.modules, "tracker.signals.divergence", types.ModuleType("tracker.signals.divergence"))
    assert list(pit.available_panel_signals()) == ["activity_score"]


# ------------------------------------------------------------ walk-forward and the vault


def test_walk_forward_trains_only_on_earlier_dates_and_keeps_out_the_vault() -> None:
    dates = pit.as_of_dates("2019-01-01", "2023-12-01", "MS")
    vault = pit.vault_start("2023-12-01")
    assert vault == pd.Timestamp("2022-12-02", tz="UTC")
    folds = pit.walk_forward(dates, vault, min_train=24)
    assert folds[0].test == pd.Timestamp("2021-01-01", tz="UTC") and len(folds[0].train) == 24
    assert folds[-1].test == pd.Timestamp("2022-12-01", tz="UTC")
    for f in folds:
        assert f.train.max() < f.test < vault and not f.in_vault
        assert (f.train < vault).all()
    final = pit.walk_forward(dates, vault, min_train=24, include_vault=True)
    in_vault = [f for f in final if f.in_vault]
    assert len(in_vault) == 12 and all(f.train.max() < f.test for f in final)


def test_walk_forward_embargo_window_and_step() -> None:
    dates = pit.as_of_dates("2020-01-01", "2021-12-01", "MS")
    vault = pd.Timestamp("2030-01-01", tz="UTC")
    folds = pit.walk_forward(dates, vault, min_train=6, embargo=2)
    first = folds[0]
    assert first.test == pd.Timestamp("2020-09-01", tz="UTC")
    assert first.train.max() == pd.Timestamp("2020-06-01", tz="UTC")  # July and August embargoed
    rolling = pit.walk_forward(dates, vault, min_train=6, window=6)
    assert all(len(f.train) == 6 for f in rolling)
    stepped = pit.walk_forward(dates, vault, min_train=6, step=3)
    assert [f.test.month for f in stepped][:3] == [7, 10, 1]
    early, late = pit.split_halves(folds)
    assert early[-1].test < late[0].test and abs(len(early) - len(late)) <= 1


# ------------------------------------------------------------ hypotheses_tested


def test_hypothesis_runs_are_logged_and_vault_looks_found(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    folds = pit.walk_forward(pit.as_of_dates("2019-01-01", "2023-12-01", "MS"), "2022-12-02", min_train=24)
    ok = pit.hypothesis_record("H1", "yoy", {"chain": "yoy", "lead": 1}, folds[0].train,
                               [f.test for f in folds], {"rmse_ratio": 0.91, "n": len(folds)},
                               tested_at="2026-09-30T10:00:00")
    failed = pit.hypothesis_record("H1", "stl", {"chain": "hol>stl"}, None, None, {}, status="failed",
                                   notes="STL needs 3 years", tested_at="2026-09-30T11:00:00")
    vault = pit.hypothesis_record("H2", "final", {}, None, None, {"rmse_ratio": 1.1}, uses_vault=True,
                                  tested_at="2026-09-30T12:00:00")
    for row in (ok, failed, vault):
        store.append_hypotheses_tested(row, data_dir)
    log = store.read_hypotheses_tested(data_dir)
    assert list(log["status"]) == ["ok", "failed", "ok"]
    assert log.loc[0, "train_period"] == "2019-01-01/2020-12-01"
    assert log.loc[0, "test_period"] == "2021-01-01/2022-12-01"
    assert json.loads(log.loc[0, "metrics"]) == {"n": 24, "rmse_ratio": 0.91}
    assert pit.vault_looks(log, "H1").empty and len(pit.vault_looks(log, "H2")) == 1


# ------------------------------------------------------------ the leak test

CUTS = [pd.Timestamp("2019-03-11 12:00", tz="UTC"), pd.Timestamp("2020-02-03", tz="UTC"),
        pd.Timestamp("2021-06-30 23:59:59", tz="UTC")]
LEAK_SIGNALS = [
    # one-sided and two-sided steps, daily and monthly, alt and official series
    pit.ChainSignal("level", "level", ("test_daily", "test_monthly", "test_official")),
    pit.ChainSignal("yoy_rz", "yoy>rz(3y)", ("test_monthly", "test_official")),
    pit.ChainSignal("yoy_ema", "yoy>ema(3m)", ("test_daily",)),
    pit.ChainSignal("smooth", "csma(28d)>yoy", ("test_daily",)),
    pit.ChainSignal("sa", "hol>stl", ("test_monthly", "test_official")),
    pit.ChainSignal("band", "cf(1.5y,4y)", ("test_monthly",)),
    pit.ChainSignal("ranks", "prf", ("test_daily", "test_official")),
    pit.ChainSignal("detrend", "lin", ("test_monthly",)),
    pit.ChainSignal("hamilton", "ham", ("test_official",)),
    pit.ChainSignal("hp", "hp", ("test_monthly",)),
]


def spread_panel(snap: pd.DataFrame, as_of: pd.Timestamp, reg: Registry) -> pd.DataFrame:
    """A stand-in for S19: each country's official YoY minus the cross-country average."""
    rows = []
    for country in COUNTRIES:
        part = snap[(snap["indicator"] == "test_official") & (snap["country"] == country)]
        yoy = chains.apply("yoy", pit.as_series(part), chains.Context(frequency="M")).values
        rows.append(pd.DataFrame({"date": yoy.index, "country": country, "value": yoy.to_numpy()}))
    df = pd.concat(rows, ignore_index=True)
    df["value"] = df["value"] - df.groupby("date")["value"].transform("mean")
    return df


def toy_nowcast(pit_rows: pd.DataFrame, vintages: store.Vintages, fold: pit.Fold, country: str) -> float:
    """Predict the official YoY of the latest month without one from the daily alt signal's real-time
    history, with a regression fitted on the months whose official value is known at the test date."""
    feature = pit.real_time_history(pit_rows, fold.test, "smooth", country, "test_daily")
    feature = feature.groupby(feature.index.to_period("M").to_timestamp()).last()
    official = pit.snapshot(vintages, fold.test, ["test_official"], [country])
    target = chains.apply("yoy", pit.as_series(official), chains.Context(frequency="M")).values
    pairs = pd.concat([feature.rename("x"), target.rename("y")], axis=1, sort=True).dropna()
    pairs = pairs[pairs.index <= fold.train.max().tz_localize(None)]
    slope, intercept = np.polyfit(pairs["x"], pairs["y"], 1)
    unknown = feature[feature.index > target.index.max()]
    return float(slope * unknown.iloc[-1] + intercept)


def run_engine(vintages: store.Vintages, registry: Registry, cut: pd.Timestamp) -> dict[str, object]:
    """Every result the engine produces for a test at `cut`."""
    dates = list(pit.as_of_dates(cut - pd.DateOffset(months=27), cut, "MS")) + [cut]
    pit_rows = pit.compute(vintages, registry, dates, series=LEAK_SIGNALS, panel={"spread": spread_panel})
    folds = pit.walk_forward(dates, pit.vault_start("2021-12-31"), min_train=12, include_vault=True)
    nowcasts = {(f.test, c): toy_nowcast(pit_rows, vintages, f, c) for f in folds[-3:] for c in COUNTRIES}
    return {"snapshot": pit.snapshot(vintages, cut), "signals": pit_rows, "nowcasts": nowcasts}


@pytest.mark.parametrize("cut", CUTS, ids=lambda c: f"{c:%Y-%m-%d}")
def test_leak_poisoned_future_never_changes_any_result(registry: Registry, cut: pd.Timestamp) -> None:
    """Everything published after the test date is replaced by huge random values, every known value
    gets a poisoned revision one second later, and poisoned future periods are appended. No result
    for the test date may change: snapshot, one- and two-sided chain end values, the panel hook,
    real-time histories and a walk-forward nowcast."""
    clean = clean_vintages()
    poisoned = poison(clean, cut)
    assert (poisoned["value"].abs() >= 1e8).sum() > 1000  # the poison is really there
    a = run_engine(store.Vintages(clean), registry, cut)
    b = run_engine(store.Vintages(poisoned), registry, cut)
    pd.testing.assert_frame_equal(a["snapshot"], b["snapshot"])
    pd.testing.assert_frame_equal(a["signals"], b["signals"])
    assert a["nowcasts"] == b["nowcasts"]
    signals = a["signals"]
    assert signals["value"].notna().sum() > 0.8 * len(signals)
    assert set(signals["signal"]) == {s.name for s in LEAK_SIGNALS} | {"spread"}
    assert signals["provisional"].any() and not signals["provisional"].all()
    assert (signals["last_date"] < cut.tz_localize(None)).all()


def test_leak_check_catches_an_engine_that_slices_instead_of_recomputing(registry: Registry) -> None:
    """The poison is detectable: reading a chain's value at the test date from a chain run on all
    the data as stored today (the classic look-ahead) differs between clean and poisoned data, for
    two-sided chains and, through later revisions, even for the plain level."""
    cut = CUTS[1]
    clean, poisoned = clean_vintages(), poison(clean_vintages(), cut)

    def sliced(v: pd.DataFrame, text: str) -> float:
        values = pit.as_series(store.Vintages(v).latest(indicators=["test_monthly"], countries=["DEU"]))
        known_end = pit.snapshot(store.Vintages(v), cut, ["test_monthly"], ["DEU"])["date"].max()
        ctx = chains.series_context(registry, "test_monthly", "DEU", values)
        return float(chains.apply(text, values, ctx).values.loc[known_end])

    for text in ("level", "cf(1.5y,4y)", "hol>stl", "lin"):
        assert sliced(clean, text) != pytest.approx(sliced(poisoned, text))
