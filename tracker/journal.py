"""The prediction journal (S25; PLAN §9): parse `docs/JOURNAL.md`, score resolved entries by Brier score.

Entry format (PLAN §9):

    ## 2026-11-02
    - Claim: Germany's October industrial production (released December) will be negative year-on-year.
    - Probability: 70%
    - Basis: overnight load -2.3z for 5 weeks; PortWatch exports -1.1z
    - Resolves: 2026-12-08 | Outcome: ___ | Brier: ___

Outcome is `yes` or `no` (also `1`/`0`, `true`/`false`); `___` means not
known yet. The Brier score of one prediction is (probability − outcome)²:
0 is perfect, and always saying 50% scores 0.25. Text inside an HTML comment
(`<!-- ... -->`) is ignored, so the file can carry a commented example.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tracker import settings

JOURNAL_FILE = settings.ROOT / "docs" / "JOURNAL.md"
HEADER = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\s*$")
FIELD = re.compile(r"^-\s*(Claim|Probability|Basis|Resolves)\s*:\s*(.*)$", re.IGNORECASE)
OUTCOMES = {"yes": 1, "y": 1, "1": 1, "true": 1, "no": 0, "n": 0, "0": 0, "false": 0}
BLANK = re.compile(r"^_+$|^$")


@dataclass
class Entry:
    """One journal prediction."""

    made: pd.Timestamp
    claim: str
    probability: float | None
    basis: str
    resolves: pd.Timestamp | None
    outcome: int | None
    brier_written: str
    line: int  # 0-based line of the Resolves line, for writing the score back

    @property
    def brier(self) -> float | None:
        if self.probability is None or self.outcome is None:
            return None
        return (self.probability - self.outcome) ** 2


def _strip_comments(text: str) -> str:
    """Blank out HTML comments, keeping line numbers."""
    return re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.DOTALL)


def _probability(text: str) -> float | None:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(%?)\s*$", text)
    if not m:
        return None
    value = float(m.group(1))
    value = value / 100 if m.group(2) or value > 1 else value
    return value if 0 <= value <= 1 else None


def parse(text: str) -> tuple[list[Entry], list[str]]:
    """Every entry in the journal, and a list of problems (lines that don't parse)."""
    lines = _strip_comments(text).splitlines()
    entries, problems = [], []
    current: dict | None = None

    def close() -> None:
        if current is None:
            return
        if "resolves" not in current:
            problems.append(f"entry {current['made']:%Y-%m-%d}: no 'Resolves:' line")
            return
        entries.append(Entry(current["made"], current.get("claim", ""), current.get("probability"),
                             current.get("basis", ""), current.get("resolves"), current.get("outcome"),
                             current.get("brier", ""), current["line"]))

    for n, raw in enumerate(lines):
        line = raw.strip()
        header = HEADER.match(line)
        if header:
            close()
            current = {"made": pd.Timestamp(header.group(1))}
            continue
        field = FIELD.match(line)
        if not field or current is None:
            continue
        name, value = field.group(1).lower(), field.group(2).strip()
        if name == "claim":
            current["claim"] = value
        elif name == "basis":
            current["basis"] = value
        elif name == "probability":
            current["probability"] = _probability(value)
            if current["probability"] is None:
                problems.append(f"entry {current['made']:%Y-%m-%d}: probability '{value}' isn't a number from 0 to 100%")
        elif name == "resolves":
            parts = [p.strip() for p in value.split("|")]
            current["line"] = n
            try:
                current["resolves"] = pd.Timestamp(parts[0])
            except ValueError:
                current["resolves"] = None
                problems.append(f"entry {current['made']:%Y-%m-%d}: resolve date '{parts[0]}' isn't a date")
            for part in parts[1:]:
                key, _, val = part.partition(":")
                key, val = key.strip().lower(), val.strip()
                if key == "outcome":
                    if BLANK.match(val):
                        current["outcome"] = None
                    elif val.lower() in OUTCOMES:
                        current["outcome"] = OUTCOMES[val.lower()]
                    else:
                        current["outcome"] = None
                        problems.append(f"entry {current['made']:%Y-%m-%d}: outcome '{val}' isn't yes or no")
                elif key == "brier":
                    current["brier"] = val
    close()
    return entries, problems


def score(entries: list[Entry], today: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per entry: made, resolves, probability, outcome, brier and a status
    (scored, awaiting outcome, open)."""
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today)
    rows = []
    for e in entries:
        if e.brier is not None:
            status = "scored"
        elif e.resolves is not None and e.resolves <= today:
            status = "awaiting outcome"
        else:
            status = "open"
        rows.append({"made": e.made.date(), "resolves": e.resolves.date() if e.resolves is not None else None,
                     "claim": e.claim, "probability": e.probability, "outcome": e.outcome, "brier": e.brier,
                     "status": status})
    return pd.DataFrame(rows, columns=["made", "resolves", "claim", "probability", "outcome", "brier", "status"])


def summary(table: pd.DataFrame) -> dict[str, float | int]:
    """Mean Brier score of the scored entries, against always saying 50% (0.25) and against always
    predicting the share of outcomes that happened (the base rate, known only in hindsight)."""
    scored = table[table["status"] == "scored"]
    if scored.empty:
        return {"scored": 0}
    base = float(scored["outcome"].mean())
    return {"scored": int(len(scored)), "brier": float(scored["brier"].mean()),
            "brier_50pct": 0.25, "brier_base_rate": float(((base - scored["outcome"]) ** 2).mean()),
            "base_rate": base}


def write_scores(path: Path, entries: list[Entry]) -> int:
    """Fill each scored entry's `Brier: ___` with its score; returns how many were filled.
    Only blank Brier fields are touched; nothing else in the file changes."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    filled = 0
    for e in entries:
        if e.brier is None or not BLANK.match(e.brier_written.strip()):
            continue
        line = lines[e.line]
        new = re.sub(r"(Brier\s*:\s*)_*\s*$", lambda m: f"{m.group(1)}{e.brier:.3f}", line.rstrip("\r\n"))
        if new != line.rstrip("\r\n"):
            lines[e.line] = new + line[len(line.rstrip("\r\n")):]
            filled += 1
    if filled:
        path.write_text("".join(lines), encoding="utf-8")
    return filled
