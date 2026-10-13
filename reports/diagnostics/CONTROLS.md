# Engine controls (exploratory)

*Run 2026-10-07 22:25 UTC by `python -m tracker diagnose controls` (`tracker/backtest/controls.py`). Exploratory diagnostics, not hypothesis tests: every check is logged to `hypotheses_tested` as `DX1` with status `exploratory`. Vault closed (test dates before 2025-10-08, outcomes and prices known before then).*

## Verdict

**The engine's code passed every check. Placebos H2: a placebo beat the benchmark significantly; §2 has the investigation.** Every check, what it should show and what it showed:

| Check | Should show | Showed | Pass |
|---|---|---|---|
| Positive A: Comtrade → IMF exports | RMSE ratio < 0.8, both halves < 1 | 0.33 (0.30 / 0.46) | yes |
| Positive B: EIA industrial sales → INDPRO | skill, if the relationship holds outside 2020-21 | no honest forecasts possible; delayed: ratio 2.13; from 2022 the signal's in-sample t is 0.80 | n/a |
| Synthetic signal, low noise | RMSE ratio well below 1 in both settings | H1 setting 0.12, H2 setting 0.32 | yes |
| Placebos H1 (13) | none beats the benchmark significantly | 0 of 13 significant wins; median ratio 1.10; persistent noise won 0 of 10 times, so at least 0 win(s) by chance has probability 1.00 | yes |
| Placebos H2 (13) | none beats the benchmark significantly | 1 of 13 significant wins; median ratio 1.17; persistent noise won 0 of 40 times, so at least 1 win(s) by chance has probability 0.15 (investigated in §2) | **no** |
| Replication H1 | identical forecasts | 742 matched, max diff 5e-14 | yes |
| Timing audit H1 | 0 violations | 893 forecasts, 0 violations | yes |
| Replication H2 | identical forecasts | 183 matched, max diff 1e-14 | yes |
| Timing audit H2 | 0 violations | 214 forecasts, 0 violations | yes |
| Trading: H3 pairs (short ETF, long URTH): daily oracle | a gain; all days on winning | annual 122.3%, Sharpe 14.14, 1.00 of days on with the expected sign | yes |
| Trading: H3 pairs (short ETF, long URTH): daily anti-oracle | a loss; all days on losing | annual -93.3%, Sharpe -54.39, 1.00 of days on with the expected sign | yes |
| Trading: H3 pairs (short ETF, long URTH): 63-day oracle | a gain; all trades winning | annual 13.7%, Sharpe 1.58, 1.00 of trades with the expected sign | yes |
| Trading: H3 pairs (short ETF, long URTH): 63-day anti-oracle | a loss; all trades losing | annual -19.5%, Sharpe -2.82, 1.00 of trades with the expected sign | yes |
| Trading: H4 (short COPX): daily oracle | a gain; all days on winning | annual 483.9%, Sharpe 8.92, 1.00 of days on with the expected sign | yes |
| Trading: H4 (short COPX): daily anti-oracle | a loss; all days on losing | annual -91.2%, Sharpe -11.16, 1.00 of days on with the expected sign | yes |
| Trading: H4 (short COPX): 63-day oracle | a gain; all trades winning | annual 5.9%, Sharpe 0.35, 1.00 of trades with the expected sign | yes |
| Trading: H4 (short COPX): 63-day anti-oracle | a loss; all trades losing | annual -29.4%, Sharpe -0.97, 1.00 of trades with the expected sign | yes |
| H5 event-study t-test on random days | about 5% significant at p < 0.05 | up to 10% (see §5) | n/a |
| E2 with the answer as D_off | large skill | skill 0.561 | yes |
| E2 with shuffled D_off | no skill | skill -0.012 | yes |

n/a: not a pass/fail test of the code (see its section).

RMSE ratio = out-of-sample RMSE of the model with the signal (`arx`) over the same model without it (`ar`); below 1 means the signal helps.

## 1. Positive controls

| Control | Forecasts | RMSE ratio | 1st / 2nd half | DM p | Signal-alone ratio |
|---|---|---|---|---|---|
| A: Comtrade exports → IMF exports (same country) | 861 | 0.33 | 0.30 / 0.46 | 0.051 | 0.32 |
| B: EIA industrial power sales → INDPRO, INDPRO delayed 75 days | 42 | 2.13 | 2.23 / 1.93 | 0.000 | 2.56 |

**A** passes: Comtrade's number for a month is out 1-4 months before the IMF's for most countries, and adding it cuts the IMF nowcast's error by 67%. Timing audit: 0 forecasts used an unpublished signal month, 0 forecast a target month already out.

| Country | Forecasts | RMSE ratio | RMSE with / without (pp) |
|---|---|---|---|
| DEU | 67 | 0.02 | 0.25 / 12.49 |
| IND | 67 | 0.02 | 0.62 / 29.78 |
| JPN | 67 | 0.12 | 1.25 / 10.45 |
| BRA | 66 | 0.14 | 2.14 / 15.51 |
| AUS | 67 | 0.20 | 2.31 / 11.57 |
| USA | 67 | 0.26 | 3.07 / 11.86 |
| IDN | 61 | 0.26 | 3.50 / 13.38 |
| NLD | 68 | 0.35 | 2.51 / 7.11 |
| MYS | 61 | 0.40 | 6.33 / 16.01 |
| CAN | 67 | 0.41 | 6.12 / 15.07 |
| MEX | 67 | 0.49 | 10.94 / 22.42 |
| POL | 68 | 0.58 | 6.09 / 10.56 |
| GBR | 68 | 0.73 | 7.42 / 10.15 |

**B is not a usable positive control, and its failure is the data's, not the engine's.** With honest timing the engine makes 0 forecasts: EIA publishes industrial sales about 57 days after the month, INDPRO comes out about 17 days after it, so the signal never arrives first. With INDPRO artificially delayed by 75 days the engine makes 42 one-month-ahead nowcasts from 2022, and the signal makes them worse (ratio 2.13). The in-sample fit shows why: the 0.85 correlation comes from the 2020-21 collapse and rebound; from 2022 on the two barely move together and the signal adds nothing even in sample:

| Window | Months | Correlation | In-sample RMSE, AR / ARX (pp) | Signal coefficient (t) |
|---|---|---|---|---|
| 2020 on | 67 | 0.85 | 2.95 / 2.51 | 0.62 (5.09) |
| 2021-07 on | 49 | 0.47 | 1.03 / 1.04 | 0.04 (0.59) |
| 2022 on | 43 | 0.28 | 1.03 / 1.04 | 0.06 (0.80) |

### Synthetic signal of known quality (power check)

The target's own final values times log-normal noise, published one day after each month ends, over each hypothesis's own sample (H1 from 2019 like PortWatch, H2 from 2020 like weather-normalised load). This deliberately uses final values, so it measures how much the test design can detect, not anything about the real signals.

| Setting | YoY noise (pp) | Median target YoY s.d. (pp) | Forecasts | RMSE ratio | 1st / 2nd half | DM p |
|---|---|---|---|---|---|---|
| H1 setting | 1.4 | 17.2 | 820 | 0.12 | 0.10 / 0.16 | 0.000 |
| H1 setting | 4.2 | 17.2 | 820 | 0.36 | 0.30 / 0.48 | 0.000 |
| H1 setting | 14.1 | 17.2 | 820 | 0.81 | 0.74 / 0.99 | 0.158 |
| H2 setting | 1.4 | 9.4 | 208 | 0.32 | 0.26 / 0.46 | 0.001 |
| H2 setting | 4.2 | 9.4 | 208 | 0.76 | 0.65 / 1.05 | 0.050 |
| H2 setting | 14.1 | 9.4 | 208 | 1.19 | 1.12 / 1.41 | 0.000 |

## 2. Placebos

Each hypothesis's own signal, broken on purpose: shifted in time 8 times by a random 14-34 months (not near 12 or 24, so YoY base effects can't line up; each month takes the value from k months earlier, wrapping round from the end of the series so the sample keeps its length), and taken from a randomly chosen other country (5 draws). The forecast months and horizons stay the same. A placebo should not beat the benchmark.

### H1

| Signal | Forecasts | RMSE ratio | 1st / 2nd half | DM p | Signal-alone ratio |
|---|---|---|---|---|---|
| none (the real signal) | 820 | 1.11 | 1.08 / 1.19 | 0.001 | 1.29 |
| shifted 14 months | 806 | 1.12 | 1.13 / 1.09 | 0.007 | 1.26 |
| shifted 15 months | 806 | 1.13 | 1.13 / 1.14 | 0.001 | 1.29 |
| shifted 16 months | 805 | 1.14 | 1.11 / 1.21 | 0.004 | 1.35 |
| shifted 21 months | 809 | 1.13 | 1.11 / 1.18 | 0.000 | 1.35 |
| shifted 30 months | 809 | 1.03 | 1.01 / 1.09 | 0.218 | 1.30 |
| shifted 31 months | 810 | 1.06 | 1.05 / 1.11 | 0.004 | 1.30 |
| shifted 32 months | 811 | 1.13 | 1.13 / 1.13 | 0.000 | 1.36 |
| shifted 33 months | 811 | 1.10 | 1.09 / 1.14 | 0.000 | 1.34 |
| wrong country #1 | 804 | 1.04 | 0.97 / 1.22 | 0.433 | 1.29 |
| wrong country #2 | 792 | 1.09 | 1.07 / 1.15 | 0.033 | 1.26 |
| wrong country #3 | 800 | 1.06 | 1.00 / 1.22 | 0.285 | 1.34 |
| wrong country #4 | 796 | 1.08 | 1.05 / 1.18 | 0.001 | 1.25 |
| wrong country #5 | 792 | 1.16 | 1.17 / 1.15 | 0.012 | 1.35 |

Placebo ratios: median 1.10, range 1.03-1.16. The real signal's 1.11 is inside that range.

Wrong-country draws: #1 USA←BRA, CAN←FRA, MEX←POL, BRA←IND, GBR←DEU, DEU←GBR, FRA←NLD, ITA←JPN, ESP←USA, NLD←TUR, POL←AUS, TUR←KOR, SAU←THA, IND←SGP, CHN←MYS, JPN←ZAF, KOR←CAN, THA←CHN, MYS←MEX, SGP←IDN, IDN←ESP, AUS←SAU, ZAF←ITA; #2 USA←DEU, CAN←MYS, MEX←CAN, BRA←IND, GBR←AUS, DEU←ESP, FRA←IDN, ITA←SAU, ESP←USA, NLD←BRA, POL←ITA, TUR←FRA, SAU←NLD, IND←KOR, CHN←THA, JPN←SGP, KOR←MEX, THA←CHN, MYS←ZAF, SGP←GBR, IDN←POL, AUS←TUR, ZAF←JPN; #3 USA←GBR, CAN←IND, MEX←FRA, BRA←AUS, GBR←ESP, DEU←ITA, FRA←MYS, ITA←MEX, ESP←NLD, NLD←TUR, POL←BRA, TUR←SGP, SAU←CAN, IND←KOR, CHN←ZAF, JPN←IDN, KOR←DEU, THA←JPN, MYS←THA, SGP←SAU, IDN←CHN, AUS←POL, ZAF←USA; #4 USA←TUR, CAN←IND, MEX←JPN, BRA←CHN, GBR←FRA, DEU←IDN, FRA←USA, ITA←MEX, ESP←AUS, NLD←DEU, POL←CAN, TUR←GBR, SAU←NLD, IND←SAU, CHN←SGP, JPN←THA, KOR←POL, THA←KOR, MYS←ZAF, SGP←ESP, IDN←BRA, AUS←MYS, ZAF←ITA; #5 USA←ITA, CAN←THA, MEX←IND, BRA←FRA, GBR←DEU, DEU←GBR, FRA←POL, ITA←ESP, ESP←AUS, NLD←CAN, POL←MEX, TUR←CHN, SAU←IDN, IND←USA, CHN←SAU, JPN←KOR, KOR←BRA, THA←MYS, MYS←NLD, SGP←ZAF, IDN←TUR, AUS←SGP, ZAF←JPN

### H2

| Signal | Forecasts | RMSE ratio | 1st / 2nd half | DM p | Signal-alone ratio |
|---|---|---|---|---|---|
| none (the real signal) | 206 | 1.26 | 1.13 / 1.63 | 0.076 | 1.27 |
| shifted 14 months | 205 | 1.13 | 1.13 / 1.13 | 0.001 | 1.28 |
| shifted 15 months | 205 | 0.93 | 0.92 / 0.95 | 0.024 | 1.10 |
| shifted 20 months | 204 | 0.93 | 0.93 / 0.93 | 0.261 | 1.08 |
| shifted 26 months | 204 | 1.12 | 1.15 / 0.99 | 0.238 | 1.30 |
| shifted 29 months | 204 | 1.11 | 1.16 / 0.89 | 0.607 | 1.37 |
| shifted 31 months | 205 | 1.13 | 1.16 / 0.99 | 0.210 | 1.31 |
| shifted 32 months | 205 | 1.23 | 1.27 / 1.07 | 0.320 | 1.33 |
| shifted 34 months | 205 | 1.17 | 1.22 / 1.01 | 0.092 | 1.44 |
| wrong country #1 | 207 | 1.60 | 1.49 / 1.93 | 0.000 | 1.54 |
| wrong country #2 | 206 | 1.63 | 1.55 / 1.84 | 0.000 | 1.63 |
| wrong country #3 | 206 | 1.95 | 2.08 / 1.40 | 0.104 | 2.01 |
| wrong country #4 | 206 | 1.92 | 2.02 / 1.56 | 0.091 | 1.99 |
| wrong country #5 | 206 | 1.65 | 1.62 / 1.72 | 0.002 | 1.72 |

Placebo ratios: median 1.17, range 0.93-1.95. The real signal's 1.26 is inside that range.

**H2, shifted 15 months: a significant win.** Ratio by country and the gain over the benchmark (sum of squared-error reductions, positive = the placebo did better) by forecast year:

| Country | Ratio | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| DEU | 0.53 | 144.5 | 193.5 | -28.5 |
| ESP | 0.83 | 134.3 | 41.1 | -21.7 |
| FRA | 0.94 | 13.3 | 18.0 | -7.5 |
| GBR | 1.04 | -2.2 | -0.3 | -0.7 |
| ITA | 0.99 | -85.3 | 93.2 | 4.8 |
| NLD | 1.18 | 35.4 | -129.3 | -4.0 |
| POL | 0.95 | 38.3 | 80.0 | -9.9 |

Investigated in v3 (NOTES, "Engine controls"): its timing audit is clean (no forecast used an unpublished value), and the replication shows the arithmetic is right. In v3 the win came from Germany and Spain in 2023-24. The AR benchmark is fitted on the same months as the model with the signal, and with the wrapped shift those include the 2021 rebound, which makes the benchmark poor (on the forecasts both versions made, its error was 2.80 pp with the rebound months against 1.63 pp without). A smooth load series whose 2022 energy-shock fall lines up, 15 months later, with German and Spanish IP's 2023-24 fall corrects part of that bias. That is a coincidence of two trends against a weak benchmark, not a leak, and it fades in 2025.

Wrong-country draws: #1 DEU←NLD, ITA←GBR, FRA←ESP, POL←ITA, ESP←POL, NLD←DEU, GBR←FRA; #2 DEU←ESP, ITA←GBR, FRA←DEU, POL←FRA, ESP←ITA, NLD←POL, GBR←NLD; #3 DEU←FRA, ITA←NLD, FRA←ITA, POL←GBR, ESP←DEU, NLD←ESP, GBR←POL; #4 DEU←POL, ITA←NLD, FRA←GBR, POL←ESP, ESP←ITA, NLD←FRA, GBR←DEU; #5 DEU←ITA, ITA←POL, FRA←ESP, POL←FRA, ESP←NLD, NLD←GBR, GBR←DEU

### Calibration: how often does pure noise win?

A placebo can beat the benchmark by luck. To know how often, pure-noise signals (a random series per country, published at each month's end, over each hypothesis's own sample) went through the same engine. Two kinds: *white* noise (independent from month to month) and *persistent* noise (a random-walk level, so its YoY is smooth like a real YoY series, and can line up with a smooth target by chance over a short sample). A *significant win* is a lower error than the benchmark with a Diebold-Mariano p below 0.05.

| Setting | Noise draws | Significant wins | Median ratio | Range | Share of ratios below 1 |
|---|---|---|---|---|---|
| H1, white noise | 10 | 0 | 1.06 | 1.03-1.14 | 0.00 |
| H1, persistent noise | 10 | 0 | 1.28 | 1.19-1.39 | 0.00 |
| H2, white noise | 40 | 0 | 1.06 | 1.00-1.24 | 0.00 |
| H2, persistent noise | 40 | 0 | 1.15 | 1.02-1.48 | 0.00 |

The placebos are real, smooth series moved in time or place, so the verdict table judges them against persistent noise: it asks whether they scored more significant wins than that rate makes likely (probability of at least that many below 0.05 = fail). It was revised after run v3, whose rule was "no significant win at all": there, one of 26 placebos (H2 shifted 15 months, ratio 0.93, DM p 0.02) crossed the bar. That was investigated rather than explained away (NOTES, "Engine controls"): the shift only moves values already published on the test date, the timing audit is clean, and this calibration measures how often chance alone does it.

## 3. Replication and timing audit

`replicate_nowcast` recomputes the primary and benchmark forecasts with separate code (its own monthly means and YoY, first-seen signal values, horizons, statsmodels OLS) for every country without Lunar New Year or break rules. `audit_timing` checks from the raw vintages (first publication time of each value) that on each test date the signal month was out and the target month wasn't.

| Hypothesis | Forecasts compared | Only in one | Same dates, horizons | Largest difference (pp) | Signal not yet out | Target already out | Last official not out |
|---|---|---|---|---|---|---|---|
| H1 | 742 | 0 | yes | 5.0e-14 | 0 of 893 | 0 | 0 |
| H2 | 183 | 0 | yes | 1.4e-14 | 0 of 214 | 0 | 0 |

The leak test (`tests/test_pit.py`: values published after each test date replaced by huge random numbers must change nothing) passed in this run's pytest.

## 4. Trading code: perfect foresight

Positions built with the real position code (`positions_from_signal`, `fixed_holds`) from foresight of the pair's return: daily (switched on at the close before each gaining day) and at the hypotheses' own rhythm (weekly Mondays, 63-day holds, entered only where the next 63 days gain). Central costs. If signs or timing were wrong, the oracle would not make money.

| Check | Days | Trades | Annual return | Sharpe | Days on (daily) or trades (holds) with the expected sign |
|---|---|---|---|---|---|
| H3 pairs (short ETF, long URTH): daily oracle | 442 | 1 | 122.3% | 14.14 | 1.00 |
| H3 pairs (short ETF, long URTH): daily anti-oracle | 442 | 1 | -93.3% | -54.39 | 1.00 |
| H3 pairs (short ETF, long URTH): 63-day oracle | 442 | 111 | 13.7% | 1.58 | 1.00 |
| H3 pairs (short ETF, long URTH): 63-day anti-oracle | 442 | 98 | -19.5% | -2.82 | 1.00 |
| H4 (short COPX): daily oracle | 943 | 228 | 483.9% | 8.92 | 1.00 |
| H4 (short COPX): daily anti-oracle | 943 | 227 | -91.2% | -11.16 | 1.00 |
| H4 (short COPX): 63-day oracle | 943 | 7 | 5.9% | 0.35 | 1.00 |
| H4 (short COPX): 63-day anti-oracle | 943 | 8 | -29.4% | -0.97 | 1.00 |

## 5. Event study (H5): is the t-test calibrated?

500 sets of 12 random trading days (2022 to the vault) put through the same `event_study` on BDRY. A calibrated test calls about 5% of them significant at p < 0.05. Two randomisation p-values compare H5's real mean return with placebo event sets instead of using the t-test: independent random days, and the real event dates all moved by one random offset (which keeps their clustering: most fall in the same few months of the 2023-24 Red Sea crisis).

| Horizon | H5 mean return | Welch p (reported) | Random sets with p < 0.05 | Randomisation p, random days | Randomisation p, shifted real dates |
|---|---|---|---|---|---|
| 10 days | 9.9% | 0.039 | 0.08 | 0.004 | 0.010 |
| 20 days | 18.6% | 0.043 | 0.09 | 0.002 | 0.003 |
| 40 days | 30.6% | 0.050 | 0.10 | 0.000 | 0.005 |

The t-test calls up to 10% of random event sets significant, so it is somewhat too generous in general (overlapping windows, skewed returns). For H5's own events, though, the randomisation p-values are smaller than the t-test's: freight returns are very skewed, which costs a t-test power. With the clustering kept (shifted real dates), 20 days would pass H5's Bonferroni bar of p < 0.0042. The events are still essentially one episode (the 2023-24 Red Sea crisis), so this strengthens H5 as a lead to watch forward but is not a result: it is exploratory, chosen after H5's pre-registered test was seen, and doesn't replace it.

## 6. Event forecasts (E2)

| D_off used | Forecasts | Brier (model / base rate) | Skill | 1st / 2nd half |
|---|---|---|---|---|
| real D_off | 410 | 0.250 / 0.252 | 0.005 | 0.004 / 0.006 |
| oracle D_off (the answer + noise) | 410 | 0.110 / 0.252 | 0.561 | 0.535 / 0.590 |
| shuffled D_off | 410 | 0.255 / 0.252 | -0.012 | -0.019 / -0.005 |
