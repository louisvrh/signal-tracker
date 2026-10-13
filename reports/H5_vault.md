# H5 vault test: chokepoint transits fall sharply, freight (BDRY) moves

*Run 2026-10-07 23:12 UTC by `python -m tracker vault H5`: H5's one look at the vault, exactly as pre-registered in SIGNALS.md H5-vault, 2026-10-07, commit d954e4a (committed before any vault data was read). Vault: test dates and prices from 2025-10-08, data as stored at the run (last weekly test date 2026-10-05). Logged in `hypotheses_tested` with `uses_vault = true`. Research only, not financial advice.*

## Headline: the primary test

BDRY, transit-count events, 20 trading days (4 weeks). Test: the vault events' dates moved together by one random offset inside the vault window (1,000 placebo sets), two-sided.

| Events (episodes) | Mean return after events | Placebo median | Side (training: above) | Randomisation p | Welch p (information) |
|---|---|---|---|---|---|
| 7 (3) | 3.0% | 6.2% | below | 0.458 | 0.470 |

**Verdict.** Failed: the mean is below the placebo median (training: above) and p 0.458 is not below 0.05.

## Every test (only the primary counts)

| Signal | Instrument | Horizon | Events (episodes) | Mean after events | Placebo median | Other days | Randomisation p | Welch p | Status |
|---|---|---|---|---|---|---|---|---|---|
| transit counts | BDRY | 10 days | 7 (3) | -0.6% | 2.7% | 2.9% | 0.245 | 0.261 | information only |
| transit counts | BDRY | 20 days | 7 (3) | 3.0% | 6.2% | 6.3% | 0.458 | 0.470 | primary |
| transit counts | BDRY | 40 days | 7 (3) | 7.0% | 12.3% | 12.5% | 0.082 | 0.237 | information only |
| transit counts | Brent BZ=F | 10 days | 7 (3) | 7.1% | 3.1% | 2.6% | 0.332 | 0.450 | contaminated by public knowledge of the 2026 Hormuz oil spike |
| transit counts | Brent BZ=F | 20 days | 7 (3) | 12.4% | 5.1% | 5.4% | 0.186 | 0.280 | contaminated by public knowledge of the 2026 Hormuz oil spike |
| transit counts | Brent BZ=F | 40 days | 7 (3) | 22.8% | 10.6% | 10.7% | 0.014 | 0.309 | contaminated by public knowledge of the 2026 Hormuz oil spike |
| cargo volume | BDRY | 10 days | 9 (3) | 3.9% | 2.5% | 2.7% | 0.559 | 0.586 | information only |
| cargo volume | BDRY | 20 days | 8 (3) | 8.5% | 6.3% | 6.1% | 0.539 | 0.476 | information only |
| cargo volume | BDRY | 40 days | 7 (3) | 12.9% | 11.7% | 12.3% | 0.737 | 0.922 | information only |
| cargo volume | Brent BZ=F | 10 days | 9 (3) | -0.0% | 2.5% | 2.8% | 0.452 | 0.218 | contaminated by public knowledge of the 2026 Hormuz oil spike |
| cargo volume | Brent BZ=F | 20 days | 8 (3) | 6.9% | 6.1% | 5.6% | 0.948 | 0.759 | contaminated by public knowledge of the 2026 Hormuz oil spike |
| cargo volume | Brent BZ=F | 40 days | 7 (3) | 15.8% | 10.5% | 11.0% | 0.625 | 0.683 | contaminated by public knowledge of the 2026 Hormuz oil spike |

## Events

- transit counts: CHK_BOSPORUS 2026-08-03 (z -2.2), CHK_DOVER 2026-02-23 (z -2.4), CHK_DOVER 2026-06-22 (z -2.0), CHK_HORMUZ 2025-12-08 (z -2.3), CHK_KOREA 2026-02-02 (z -2.4), CHK_MALACCA 2026-04-13 (z -2.2), CHK_TAIWAN 2025-11-24 (z -2.6)
- cargo volume: CHK_BOSPORUS 2026-02-02 (z -2.1), CHK_BOSPORUS 2026-08-10 (z -2.2), CHK_DOVER 2026-03-09 (z -2.0), CHK_DOVER 2026-09-14 (z -2.5), CHK_HORMUZ 2025-12-08 (z -2.1), CHK_HORMUZ 2026-03-23 (z -3.7), CHK_MALACCA 2026-04-06 (z -2.3), CHK_TAIWAN 2025-11-17 (z -2.5), CHK_TAIWAN 2026-08-17 (z -2.1)

## For information only (never counted)

Pre-vault events within 70 days of the vault start, whose horizons S24 dropped because they reached into the vault:

| Event | Instrument | 10 days | 20 days | 40 days |
|---|---|---|---|---|
| CHK_HORMUZ 2025-10-06 (z -2.4) | BDRY | 4.8% | 0.7% | 17.6% |
| CHK_HORMUZ 2025-10-06 (z -2.4) | Brent BZ=F | -6.8% | -0.9% | -4.6% |

## How to read this

- **Episodes, not events, measure the evidence.** Events less than 8 weeks apart (any chokepoint) share outcome days and count as one episode. At the primary horizon the vault held 7 event(s) in 3 episode(s).
- **Brent is contaminated by public knowledge of the 2026 Hormuz oil spike**, so it can't count as a success whatever it shows.
- The randomisation test keeps the events' clustering, but with few episodes it can only say how unusual these few dates were within the window of BDRY's history (one year: the vault).
- **How small p can get.** The primary window has 230 trading days, so 17% of placebo sets are shifted by fewer than 20 trading days and overlap the real events' outcome days. A real move that builds over the whole horizon shows up partly in those placebos too, so the shorter the window, the harder a small p-value is to reach.
- Window per horizon: the trading days from the window start whose return over that horizon is complete (the 40-day window is the shortest). Each test uses its own generator (seed 20261007).
- Prices are Yahoo adjusted closes; BZ=F is a continuous front-month future whose returns include roll jumps.
