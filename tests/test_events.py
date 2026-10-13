"""Event forecasts (E2) and the prediction journal (S25)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tracker import journal, store
from tracker.backtest import events as ev
from tracker.registry import load_registry

REG = load_registry()

SAMPLE = """# Prediction journal

<!--
## 2026-01-01
- Claim: an example inside a comment is never scored.
- Probability: 99%
- Resolves: 2026-02-01 | Outcome: yes | Brier: ___
-->

## Predictions

## 2026-11-02
- Claim: Germany's October industrial production (released December) will be negative year-on-year.
- Probability: 70%
- Basis: overnight load -2.3z for 5 weeks; PortWatch exports -1.1z
- Resolves: 2026-12-08 | Outcome: yes | Brier: ___

## 2026-11-09
- Claim: Japan's September exports will be weaker than August's.
- Probability: 0.4
- Basis: port imports z -3.7
- Resolves: 2026-11-20 | Outcome: no | Brier: ___

## 2026-11-16
- Claim: China's November exports YoY above 5%.
- Probability: 55%
- Basis: none
- Resolves: 2026-12-10 | Outcome: ___ | Brier: ___

## 2026-12-01
- Claim: something far ahead.
- Probability: 120%
- Resolves: 2027-06-01 | Outcome: maybe | Brier: ___
"""


# ------------------------------------------------------------------ journal


def test_journal_scores_a_sample_entry(tmp_path) -> None:
    entries, problems = journal.parse(SAMPLE)
    assert [e.made.strftime("%Y-%m-%d") for e in entries] == ["2026-11-02", "2026-11-09", "2026-11-16", "2026-12-01"]
    table = journal.score(entries, today=pd.Timestamp("2026-12-15"))
    assert table.loc[0, "brier"] == pytest.approx(0.09)  # (0.7 - 1)^2
    assert table.loc[1, "brier"] == pytest.approx(0.16)  # (0.4 - 0)^2
    assert table["status"].tolist() == ["scored", "scored", "awaiting outcome", "open"]
    assert any("120%" in p for p in problems) and any("maybe" in p for p in problems)
    s = journal.summary(table)
    assert s["scored"] == 2 and s["brier"] == pytest.approx(0.125) and s["brier_base_rate"] == pytest.approx(0.25)

    path = tmp_path / "JOURNAL.md"
    path.write_text(SAMPLE, encoding="utf-8")
    assert journal.write_scores(path, entries) == 2
    text = path.read_text(encoding="utf-8")
    assert "Outcome: yes | Brier: 0.090" in text and "Outcome: no | Brier: 0.160" in text
    assert "Outcome: yes | Brier: ___\n-->" in text  # the commented example is untouched
    assert text.count("Brier: ___") == 3
    assert journal.write_scores(path, journal.parse(text)[0]) == 0  # already filled: nothing changes


def test_journal_score_command(tmp_path, capsys) -> None:
    from tracker.__main__ import main

    path = tmp_path / "JOURNAL.md"
    path.write_text(SAMPLE, encoding="utf-8")
    assert main(["journal-score", "--file", str(path)]) == 0
    out = capsys.readouterr().out
    assert "2 scored: mean Brier 0.125" in out and "0.090" in out
    assert "Brier: ___" in path.read_text(encoding="utf-8").split("## Predictions")[1]  # not written without --write
    assert main(["journal-score", "--file", str(path), "--write"]) == 0
    assert "Filled 2 Brier field(s)" in capsys.readouterr().out


def test_the_real_journal_parses() -> None:
    entries, problems = journal.parse(journal.JOURNAL_FILE.read_text(encoding="utf-8"))
    assert problems == []  # the template's example is commented out


# --------------------------------------------------------------------- E2

COUNTRIES = ("DEU", "FRA", "ITA", "ESP", "NLD")
MONTHS = pd.date_range("2018-01-01", "2025-06-01", freq="MS")


def world(seed: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official IP published 47 days after each month, and weekly D_off end values."""
    rng = np.random.default_rng(seed)
    vint, pit_rows = [], []
    mondays = pd.date_range("2023-01-02", "2025-06-30", freq="W-MON", tz="UTC")
    for c in COUNTRIES:
        level = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(MONTHS))))
        available = (MONTHS + pd.offsets.MonthEnd(0) + pd.Timedelta(days=47)).tz_localize("UTC")
        vint.append(pd.DataFrame({"date": MONTHS, "country": c, "indicator": "eurostat_ip", "value": level,
                                  "unit": "index", "frequency": "M", "source": "eurostat", "fetched_at": available,
                                  "available_at": available, "vintage_type": "estimated"}))
        d = rng.normal(0, 1.5, len(mondays))
        pit_rows.append(pd.DataFrame({"as_of": mondays, "signal": "d_off", "country": c, "value": d,
                                      "components": json.dumps({"official": "eurostat_ip"})}))
    return pd.concat(vint, ignore_index=True), pd.concat(pit_rows, ignore_index=True)


TESTS = pd.date_range("2023-01-15", "2024-12-15", freq="MS", tz="UTC") + pd.Timedelta(days=14)
TRUTH = pd.Timestamp("2025-01-01", tz="UTC")


def test_e2_forecasts_each_print_once_and_trains_only_on_resolved_events(monkeypatch) -> None:
    monkeypatch.setattr(ev, "MIN_EVENTS", 20)
    vint, pit_rows = world()
    f = ev.e2_forecasts(pit_rows, store.Vintages(vint), REG, TESTS, TRUTH)
    assert not f.duplicated(["country", "next_month"]).any()
    published = f["next_month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=47)
    assert (f["test_date"].dt.tz_localize(None) < published).all()  # forecast before the print
    assert f["n_train"].is_monotonic_increasing
    fitted = f[f["p_model"].notna()]
    assert len(fitted) > 50 and (fitted["n_train"] >= 20).all() and fitted["p_model"].between(0, 1).all()
    known = published < TRUTH.tz_localize(None)
    assert f.loc[known, "outcome"].notna().all() and f.loc[~known, "outcome"].isna().all()
    s = ev.brier_scores(f)
    assert s["n"] > 40 and 0 < s["brier_base"] < 0.35  # noise world: no skill expected
    cal = ev.calibration(f)
    assert cal["n"].sum() == f.dropna(subset=["p_model", "outcome"]).shape[0]


def test_e2_never_sees_anything_published_after_the_vault(monkeypatch) -> None:
    monkeypatch.setattr(ev, "MIN_EVENTS", 20)
    vint, pit_rows = world()
    poisoned = vint.copy()
    late = poisoned["available_at"] > TRUTH
    poisoned.loc[late, "value"] = 1e6
    revisions = poisoned[~late].copy()
    revisions["available_at"] = revisions["fetched_at"] = TRUTH + pd.Timedelta(seconds=1)
    revisions["value"] = 1.0
    poisoned = pd.concat([poisoned, revisions], ignore_index=True)
    later_pit = pit_rows.assign(as_of=pit_rows["as_of"] + pd.Timedelta(days=3650), value=-99.0)
    a = ev.e2_forecasts(pit_rows, store.Vintages(vint), REG, TESTS, TRUTH)
    b = ev.e2_forecasts(pd.concat([pit_rows, later_pit]), store.Vintages(poisoned), REG, TESTS, TRUTH)
    pd.testing.assert_frame_equal(a, b)


def test_a_logit_needs_enough_events_of_both_kinds() -> None:
    few = pd.DataFrame({"d_off": [0.0] * 5, "label": [1.0, 0.0, 1.0, 0.0, 1.0]})
    p, note = ev._logit_p(few, 0.0)
    assert np.isnan(p) and "5 resolved" in note
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 400)
    y = (rng.uniform(size=400) < 1 / (1 + np.exp(2 * x))).astype(float)  # weaker prints when D_off is low
    p_low, _ = ev._logit_p(pd.DataFrame({"d_off": x, "label": y}), -2.0)
    p_high, _ = ev._logit_p(pd.DataFrame({"d_off": x, "label": y}), 2.0)
    assert p_low > 0.9 and p_high < 0.1
