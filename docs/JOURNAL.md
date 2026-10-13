# Prediction journal

Your dated, probability-weighted predictions, written **before** the outcome is known (PLAN §9). Weekly: after
reading the Overview page, add 1-3 entries. Monthly: fill in the outcomes of the entries that have resolved and run

```powershell
.venv\Scripts\python -m tracker journal-score           # table of every entry and the mean Brier score
.venv\Scripts\python -m tracker journal-score --write   # also fill in "Brier: ___" for entries with an outcome
```

Rules that keep the record honest:

- Never edit a claim or probability after the date in its heading. If you change your mind, add a new entry.
- `Outcome` is `yes` or `no`; leave `___` until the resolve date has passed and the number is published.
- The Brier score is (probability − outcome)²: 0 is perfect; always saying 50% scores 0.25.
- After at least 3 months of entries, a signal that keeps scoring better than 0.25 (and than the base rate) is the
  forward evidence the graduation rule asks for (PLAN §5.6).

Format (one block per prediction; the example below is inside a comment, so it isn't scored):

<!--
## 2026-11-02
- Claim: Germany's October industrial production (released December) will be negative year-on-year.
- Probability: 70%
- Basis: overnight load -2.3z for 5 weeks; PortWatch exports -1.1z
- Resolves: 2026-12-08 | Outcome: ___ | Brier: ___
-->

## Predictions
