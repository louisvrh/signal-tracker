# H5 vault code: rehearsal on pre-vault data (exploratory, DX5)

*Run 2026-10-07 23:09 UTC by `python -m tracker vault H5 --rehearse`. The code of the one-time vault test, run on the training period instead: events from 2022-01-01, prices as known at the vault start and dated before it. Nothing on or after 2025-10-08 is read. A check of the code against `reports/diagnostics/CONTROLS.md` §5, not a test of H5.*

## Headline: the primary test

BDRY, transit-count events, 20 trading days (4 weeks). Test: the vault events' dates moved together by one random offset inside the vault window (1,000 placebo sets), two-sided.

| Events (episodes) | Mean return after events | Placebo median | Side (training: above) | Randomisation p | Welch p (information) |
|---|---|---|---|---|---|
| 12 (5) | 18.6% | -1.4% | above | < 0.001 | 0.030 |

**Verdict.** Passed: the mean return after the vault events is above the placebo median, as in training, with p < 0.001, below 0.05.

## Every test (only the primary counts)

| Signal | Instrument | Horizon | Events (episodes) | Mean after events | Placebo median | Other days | Randomisation p | Welch p | Status |
|---|---|---|---|---|---|---|---|---|---|
| transit counts | BDRY | 10 days | 12 (5) | 9.9% | -0.8% | -0.6% | 0.015 | 0.027 | information only |
| transit counts | BDRY | 20 days | 12 (5) | 18.6% | -1.4% | -0.8% | < 0.001 | 0.030 | primary |
| transit counts | BDRY | 40 days | 12 (5) | 30.6% | -3.2% | -1.3% | 0.008 | 0.032 | information only |
| transit counts | Brent BZ=F | 10 days | 12 (5) | -0.6% | 0.1% | -0.0% | 0.743 | 0.736 | contaminated by public knowledge of the 2026 Hormuz oil spike |
| transit counts | Brent BZ=F | 20 days | 12 (5) | 0.8% | -0.3% | -0.2% | 0.707 | 0.626 | contaminated by public knowledge of the 2026 Hormuz oil spike |
| transit counts | Brent BZ=F | 40 days | 12 (5) | -0.2% | -0.8% | -0.8% | 0.900 | 0.812 | contaminated by public knowledge of the 2026 Hormuz oil spike |
| cargo volume | BDRY | 10 days | 10 (4) | 4.2% | -0.7% | -0.6% | 0.222 | 0.172 | information only |
| cargo volume | BDRY | 20 days | 10 (4) | 15.3% | -0.8% | -0.7% | 0.013 | 0.059 | information only |
| cargo volume | BDRY | 40 days | 10 (4) | 23.0% | -1.6% | -1.1% | 0.034 | 0.131 | information only |
| cargo volume | Brent BZ=F | 10 days | 10 (4) | 1.8% | 0.1% | -0.1% | 0.383 | 0.215 | contaminated by public knowledge of the 2026 Hormuz oil spike |
| cargo volume | Brent BZ=F | 20 days | 10 (4) | 2.1% | -0.4% | -0.3% | 0.264 | 0.240 | contaminated by public knowledge of the 2026 Hormuz oil spike |
| cargo volume | Brent BZ=F | 40 days | 10 (4) | 4.2% | -0.8% | -0.8% | 0.164 | 0.151 | contaminated by public knowledge of the 2026 Hormuz oil spike |

## Events

- transit counts: CHK_BAB_EL_MANDEB 2024-01-08 (z -2.8), CHK_BOSPORUS 2022-03-28 (z -2.1), CHK_DOVER 2023-11-13 (z -2.1), CHK_DOVER 2024-02-26 (z -2.3), CHK_GIBRALTAR 2024-02-26 (z -2.8), CHK_HORMUZ 2025-10-06 (z -2.4), CHK_KOREA 2023-03-13 (z -2.2), CHK_KOREA 2023-10-30 (z -3.5), CHK_KOREA 2025-01-27 (z -2.0), CHK_PANAMA 2023-11-20 (z -2.3), CHK_SUEZ 2023-08-21 (z -2.1), CHK_SUEZ 2024-01-15 (z -2.4), CHK_TAIWAN 2022-05-16 (z -2.1)
- cargo volume: CHK_BAB_EL_MANDEB 2024-01-01 (z -2.1), CHK_BOSPORUS 2022-04-11 (z -2.2), CHK_GIBRALTAR 2024-02-12 (z -2.8), CHK_HORMUZ 2023-01-30 (z -2.1), CHK_HORMUZ 2025-10-06 (z -2.0), CHK_KOREA 2023-01-02 (z -2.2), CHK_KOREA 2023-10-30 (z -2.0), CHK_PANAMA 2023-11-27 (z -2.5), CHK_SUEZ 2024-01-08 (z -2.1), CHK_TAIWAN 2022-05-09 (z -2.2), CHK_TAIWAN 2024-09-02 (z -2.2)
- Not counted at 10 days (return not complete yet): 2025-10-06
- Not counted at 20 days (return not complete yet): 2025-10-06
- Not counted at 40 days (return not complete yet): 2025-10-06

## How to read this

- **Episodes, not events, measure the evidence.** Events less than 8 weeks apart (any chokepoint) share outcome days and count as one episode. At the primary horizon the vault held 12 event(s) in 5 episode(s).
- **Brent is contaminated by public knowledge of the 2026 Hormuz oil spike**, so it can't count as a success whatever it shows.
- The randomisation test keeps the events' clustering, but with few episodes it can only say how unusual these few dates were within the window of BDRY's history (the training period here).
- **How small p can get.** The primary window has 924 trading days, so 4% of placebo sets are shifted by fewer than 20 trading days and overlap the real events' outcome days. A real move that builds over the whole horizon shows up partly in those placebos too, so the shorter the window, the harder a small p-value is to reach.
- Window per horizon: the trading days from the window start whose return over that horizon is complete (the 40-day window is the shortest). Each test uses its own generator (seed 20261007).
- Prices are Yahoo adjusted closes; BZ=F is a continuous front-month future whose returns include roll jumps.
