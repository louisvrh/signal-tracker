"""The one-time vault test of an event study (SIGNALS.md "H5-vault"): windows, episodes,
the randomisation test inside the vault, the verdict and the one-look guard (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracker import store
from tracker.backtest import pit
from tracker.backtest import vault as vt
from tracker.research import VAULT_FROM
from tracker.research_vault import VaultAlreadyOpened, run_vault, weekly_dates

START = pd.Timestamp("2025-01-01")


def prices(jumps: dict[str, float] | None = None, seed: int = 3) -> pd.Series:
    """A business-day price path; each date in `jumps` is followed by that log rise on the next day."""
    days = pd.bdate_range("2024-01-01", "2025-12-31")
    steps = np.random.default_rng(seed).normal(0, 0.01, len(days))
    for d, size in (jumps or {}).items():
        steps[days.searchsorted(pd.Timestamp(d)) + 1] += size
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=days)


def test_complete_days_start_at_start_and_leave_out_the_last_horizon() -> None:
    p = prices()
    days = vt.complete_days(p, START, 20)
    assert days[0] == pd.Timestamp("2025-01-01") and days[-1] == p.index[-21]
    assert vt.complete_days(p.iloc[:5], START, 20).empty


def test_episodes_join_events_less_than_eight_weeks_apart() -> None:
    dates = ["2024-01-08", "2023-11-13", "2023-11-20", "2024-03-25", "2025-01-27"]
    # 13 Nov, 20 Nov and 8 Jan chain into one episode; 25 Mar is 11 weeks after 8 Jan.
    assert vt.episodes(dates) == [1, 1, 1, 2, 3]
    assert vt.episodes(["2024-01-01", "2024-02-26"]) == [1, 2]  # exactly 8 weeks: a new episode
    assert vt.episodes([]) == []


def test_a_real_effect_is_found_on_the_right_side() -> None:
    # Two years of window. Shifts of under 10 days still catch the jumps, which bounds p from below
    # (about 9 of ~510 offsets), as it would in the vault.
    events = ["2024-06-03", "2025-02-03", "2025-08-04"]
    r = vt.vault_test(pd.DatetimeIndex(events, tz="UTC"), prices({e: 0.25 for e in events}), "2024-01-01", 10)
    assert r["events"] == 3 and r["episodes"] == 3
    assert r["mean"] > 0.15 and r["side"] == 1 and r["p"] < 0.04
    assert r["welch_p"] < 0.05 and r["other_days_mean"] < r["mean"]
    passed, reason = vt.verdict(r, training_side=1)
    assert passed and reason.startswith("Passed")
    passed, reason = vt.verdict(r, training_side=-1)
    assert not passed and "training: below" in reason


def test_noise_is_not_significant_and_the_seed_fixes_the_p_value() -> None:
    events = pd.DatetimeIndex(["2025-03-03", "2025-06-02"], tz="UTC")
    a = vt.vault_test(events, prices(seed=11), START, 20)
    b = vt.vault_test(events, prices(seed=11), START, 20)
    assert a["p"] == b["p"] and a["p"] > 0.05
    assert not vt.verdict(a)[0]


def test_only_events_in_the_window_with_a_complete_return_count() -> None:
    p = prices()
    events = pd.DatetimeIndex(["2024-11-04", "2025-03-03", "2025-12-22"], tz="UTC")  # before; in; return incomplete
    r = vt.vault_test(events, p, START, 20)
    assert r["events"] == 1 and r["event_dates"] == [pd.Timestamp("2025-03-03", tz="UTC")]
    assert r["window_first"] == pd.Timestamp("2025-01-01") and r["window_last"] == p.index[-21]


def test_no_event_is_not_testable_and_not_a_pass() -> None:
    r = vt.vault_test(pd.DatetimeIndex([], tz="UTC"), prices(), START, 20)
    assert r["events"] == 0 and r["episodes"] == 0 and np.isnan(r["p"])
    passed, reason = vt.verdict(r)
    assert not passed and reason.startswith("Not testable")


def test_weekly_dates_reach_into_the_vault_only_on_the_one_look() -> None:
    end = pd.Timestamp("2026-10-07", tz="UTC")
    closed, opened = weekly_dates(end, include_vault=False), weekly_dates(end, include_vault=True)
    assert closed.max() < VAULT_FROM <= opened.max()
    assert closed[0] == opened[0] == pd.Timestamp("2020-01-06", tz="UTC")
    assert opened.max() == pd.Timestamp("2026-10-05", tz="UTC")


def test_a_second_look_at_the_vault_is_refused(tmp_path) -> None:
    store.append_hypotheses_tested(pit.hypothesis_record("H5", "vault: earlier look", {}, None, None, {},
                                                         uses_vault=True), tmp_path)
    with pytest.raises(VaultAlreadyOpened, match="already looked at the vault"):
        run_vault(data_dir=tmp_path, progress=lambda _: None)


def test_p_values_read_below_the_draws_resolution_and_overlap_share() -> None:
    assert vt.format_p(0.0) == "< 0.001" and vt.format_p(0.0123) == "0.012" and vt.format_p(float("nan")) == "–"
    assert vt.overlap_share(20, 230) == pytest.approx(38 / 229)
    assert vt.overlap_share(40, 30) == 1.0
