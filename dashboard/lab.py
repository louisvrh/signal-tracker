"""Signal Lab helpers: report headlines, statuses from reports/SUMMARY.md and the per-hypothesis
table built from `hypotheses_tested` (pure functions; the page and the loaders do the I/O)."""

from __future__ import annotations

import re

import pandas as pd

VAULT_SUFFIX = "_vault"
EXPLORATORY = ("exploratory", "exploratory-failed")
_ID = re.compile(r"^([A-Z])(\d+)([a-z]*)$")


def sort_key(hypothesis_id: str) -> tuple:
    """H1, H1b, H2, ..., H6, then E1, E2, E3; anything else last."""
    m = _ID.match(hypothesis_id)
    if not m:
        return (9, 0, hypothesis_id)
    return ({"H": 0, "E": 1}.get(m.group(1), 2), int(m.group(2)), m.group(3))


def split_report(text: str) -> dict[str, str]:
    """A report's title (first line, without its id), headline section and verdict sentence.

    The headline is the section under the first `## Headline...` heading; the verdict is its
    `**Verdict.**` paragraph, or the headline's first paragraph when there is none.
    """
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip() if lines else ""
    title = title.split(": ", 1)[1] if ": " in title else title
    headline: list[str] = []
    inside = False
    for line in lines[1:]:
        if line.startswith("## "):
            if inside:
                break
            inside = line.startswith("## Headline")
            continue
        if inside:
            headline.append(line)
    body = "\n".join(headline).strip()
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    verdict = next((p for p in paragraphs if p.startswith("**Verdict.**")), None)
    if verdict is None:
        verdict = next((p for p in paragraphs if not p.startswith("|")), "")
    return {"title": title, "headline": body, "verdict": verdict.replace("**Verdict.**", "").strip()}


def summary_statuses(text: str) -> dict[str, str]:
    """Status per hypothesis from the status table in reports/SUMMARY.md (rows `| <id> | <status> | ...`)."""
    out = {}
    for line in text.splitlines():
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and _ID.match(cells[0]):
            out.setdefault(cells[0], cells[1])
    return out


def hypothesis_table(log: pd.DataFrame, reports: dict[str, str], summary: str = "") -> pd.DataFrame:
    """One row per hypothesis with a report or logged runs (exploratory diagnostics left out):
    title, status (SUMMARY.md), verdict, vault state, runs logged and the last run."""
    statuses = summary_statuses(summary)
    tests = log[~log["status"].isin(EXPLORATORY)] if not log.empty else log
    ids = {name for name in reports if _ID.match(name)} | (set(tests["hypothesis_id"]) if not tests.empty else set())
    rows = []
    for hid in sorted(ids, key=sort_key):
        parts = split_report(reports[hid]) if hid in reports else {"title": "", "verdict": ""}
        mine = tests[tests["hypothesis_id"] == hid] if not tests.empty else tests
        looks = mine[mine["uses_vault"].astype(bool)] if not mine.empty else mine
        vault = "closed"
        if not looks.empty:
            verdict = split_report(reports.get(hid + VAULT_SUFFIX, ""))["verdict"]
            outcome = verdict.split(":", 1)[0].lower() if verdict else "see the log"
            vault = f"opened {pd.Timestamp(looks['tested_at'].min()):%Y-%m-%d}: {outcome}"
        rows.append({"hypothesis": hid, "title": parts["title"], "status": statuses.get(hid, ""),
                     "verdict": parts["verdict"], "vault": vault, "runs": len(mine),
                     "last_run": mine["tested_at"].max() if not mine.empty else pd.NaT})
    return pd.DataFrame(rows, columns=["hypothesis", "title", "status", "verdict", "vault", "runs", "last_run"])
