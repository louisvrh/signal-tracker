"""Signal Lab (S26): report headlines, statuses from SUMMARY.md, the per-hypothesis table and the page."""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import lab
from tracker import settings, store
from tracker.backtest import pit

APP = str(settings.ROOT / "dashboard" / "app.py")
T0 = pd.Timestamp("2026-10-07 16:00", tz="UTC")

H3 = """# H3: Countries with a market-divergence flag: the country ETF underperforms URTH

*Run note.*

## Headline (out of sample: fixed rules, nothing fitted)

| | Strategy |
|---|---|
| Sharpe | -0.51 |

**Verdict.** Not supported: lost money after costs (Sharpe -0.51, 38 trades).

## Every variant tried

| a | b |
"""
H5 = """# H5: Chokepoint transits fall sharply: freight (BDRY) and Brent move

## Headline (out of sample: fixed rules, nothing fitted)

**Verdict.** 12 events. Not supported.

## Events
"""
H5_VAULT = """# H5 vault test: chokepoint transits fall sharply, freight (BDRY) moves

## Headline: the primary test

| Events | p |
|---|---|
| 7 (3) | 0.458 |

**Verdict.** Failed: the mean is below the placebo median (training: above) and p 0.458 is not below 0.05.

## Every test
"""
E3 = """# E3: the system's probabilities against prediction-market prices

## Headline

Not testable on history: no market priced these prints before the vault.

## What was checked
"""
SUMMARY = """# Summary

| Hypothesis | Status | Why |
|---|---|---|
| **H3** | failed | lost money |
| H5 | lead (failed the vault) | freight |
| E3 | not testable | no markets |
"""


def record(hid: str, variant: str, status: str = "ok", uses_vault: bool = False,
           at: pd.Timestamp = T0) -> pd.DataFrame:
    return pit.hypothesis_record(hid, variant, {}, None, None, {"p": 0.5}, status=status, uses_vault=uses_vault,
                                 tested_at=at)


def test_split_report_finds_title_headline_and_verdict() -> None:
    parts = lab.split_report(H3)
    assert parts["title"] == "Countries with a market-divergence flag: the country ETF underperforms URTH"
    assert parts["headline"].startswith("| | Strategy |") and "Every variant" not in parts["headline"]
    assert parts["verdict"] == "Not supported: lost money after costs (Sharpe -0.51, 38 trades)."
    # No **Verdict.** paragraph: the headline's first paragraph that isn't a table.
    assert lab.split_report(E3)["verdict"].startswith("Not testable on history")
    assert lab.split_report("") == {"title": "", "headline": "", "verdict": ""}


def test_statuses_come_from_the_summary_table() -> None:
    assert lab.summary_statuses(SUMMARY) == {"H3": "failed", "H5": "lead (failed the vault)", "E3": "not testable"}


def test_table_has_one_row_per_hypothesis_without_diagnostics() -> None:
    log = pd.concat([record("H5", "transit counts"), record("H5", "vault: primary", uses_vault=True,
                                                             at=T0 + pd.Timedelta(hours=7)),
                     record("H3", "primary"), record("DX1", "control", status="exploratory"),
                     record("H1b", "crashed", status="failed")], ignore_index=True)
    reports = {"H3": H3, "H5": H5, "H5_vault": H5_VAULT, "E3": E3, "SUMMARY": SUMMARY}
    table = lab.hypothesis_table(log, reports, SUMMARY).set_index("hypothesis")
    assert table.index.tolist() == ["H1b", "H3", "H5", "E3"]  # natural order; no DX1, SUMMARY or H5_vault rows
    assert table.loc["H5", "vault"] == "opened 2026-10-07: failed" and table.loc["H3", "vault"] == "closed"
    assert table.loc["H5", "runs"] == 2 and table.loc["E3", "runs"] == 0 and table.loc["H1b", "runs"] == 1
    assert table.loc["H5", "status"] == "lead (failed the vault)" and table.loc["H1b", "status"] == ""
    assert table.loc["H5", "last_run"] == T0 + pd.Timedelta(hours=7)
    assert lab.sort_key("H10") > lab.sort_key("H2b") > lab.sort_key("H2") and lab.sort_key("E1") > lab.sort_key("H6")


@pytest.fixture
def lab_reports(tmp_path, monkeypatch):
    folder = tmp_path / "reports"
    folder.mkdir()
    for name, text in {"H3": H3, "H5": H5, "H5_vault": H5_VAULT, "E3": E3, "SUMMARY": SUMMARY}.items():
        (folder / f"{name}.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(settings, "REPORTS_DIR", folder)
    store.append_hypotheses_tested(pd.concat([
        record("H5", "transit counts"), record("H5", "vault: primary", uses_vault=True),
        record("H3", "primary"), record("DX1", "control", status="exploratory")], ignore_index=True))


def test_signal_lab_page(lab_reports) -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.switch_page("views/signal_lab.py")
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"Hypotheses": "3", "Test runs logged": "3", "Vault looks used": "1", "Passed": "0"}
    assert at.dataframe[0].value["hypothesis"].tolist() == ["H3", "H5", "E3"]
    assert len(at.dataframe[1].value) == 3  # the run log, diagnostics hidden
    at.selectbox(key="lab_hypothesis").select("H5").run()
    assert not at.exception
    assert any("Vault test" in m.value for m in at.markdown)
    at.toggle(key="lab_exploratory").set_value(True).run()
    assert len(at.dataframe[1].value) == 4


def test_signal_lab_on_an_empty_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "none")
    at = AppTest.from_file(APP, default_timeout=60)
    at.switch_page("views/signal_lab.py")
    at.run()
    assert not at.exception
    assert at.info[0].value == "No hypothesis tested yet."
