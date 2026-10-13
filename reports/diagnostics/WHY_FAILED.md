# Why H1, H1b, H2 and H2b failed

*Exploratory diagnostics, 2026-10-07. Numbers come from `why_failed_tables.md` (`python -m tracker diagnose why`) and `CONTROLS.md` (`python -m tracker diagnose controls`); every check is logged in `hypotheses_tested` as `DX1` or `DX2` with status `exploratory`. Nothing here is a new hypothesis test, and the vault (data published from 2025-10-08) stayed closed. Research only, not financial advice.*

## The short answer

The tests themselves work, so the failures aren't bugs: the engine found strong signals where they exist, found nothing in pure noise and in all but one of 26 misplaced signals (that one is explained in `CONTROLS.md` §2), and an independent re-implementation gave identical forecasts. The signals failed for two different reasons:

- **Power demand (H2, H2b) tracked industrial production only during the COVID collapse and rebound.** In 2020-21 the two moved together (median correlation 0.82). From 2022 they didn't (0.11 in 2022-23, 0.02 in 2024-25), and in Germany and the Netherlands they moved in opposite directions. Even a model told the best coefficients for the whole period in hindsight forecasts 40% worse with the power signal than without it in 2023-25. The relationship broke; it wasn't a matter of fitting it badly.
- **Port export volumes (H1, H1b) were only ever weakly related to official exports in dollars, and the link has faded.** The median correlation fell from 0.55 (2020-21) to 0.34 (2022-23) and 0.17 (2024-25). In sample, the port signal improves the fit by only 6-8%; with hindsight coefficients it improves the 2022-25 forecasts by nothing; estimating it in real time then costs 11-16%. Signals shifted to the wrong dates or taken from the wrong country do about as well as the real one. Part of the reason is that dollar export values have been driven by prices and the dollar since 2022, not volumes.

## Explanations, ranked by the strength of the evidence

### 1. H2/H2b: the power-industry link was a COVID artefact and broke in 2022 (strong evidence)

- Correlation of weather-normalised load YoY with IP YoY, median of 7 countries: **0.82 in 2020-21, 0.11 in 2022-23, 0.02 in 2024-25** (overnight load: 0.73, 0.06, −0.06). The rolling 24-month correlation was 0.69 at the end of 2022 and around zero or below from mid-2023.
- From 2022, the correlation is negative in Germany (−0.42, −0.32), the Netherlands (−0.83, −0.41) and France in 2022-23 (−0.45). Only Poland and Italy kept a positive link in 2022-23, and neither did in 2024-25.
- **In sample versus out of sample:** fitted on the whole sample, the signal looks useful (RMSE ratio 0.77 for total load, 0.82 for overnight load), but that fit is the 2021 rebound. Applying those whole-sample coefficients to the real 2023-25 forecasts gives **1.40**, worse than the real walk-forward result (1.26). So the problem isn't overfitting a stable relationship; in the test period there was no relationship to find.
- Lead and lag: over all months the correlation peaks at lag 0 (0.58), as it should. From 2022 there's no peak at all (0.10 at lag 0, slightly negative at leads).
- A likely mechanism (not tested here): in the 2022 energy-price shock, energy-intensive industry (chemicals, metals, paper, cement) cut electricity use far more than its share of IP's value added, and IP is weighted towards low-energy sectors (pharmaceuticals, machinery, vehicles). Rooftop solar and electrification (heat pumps, cars) add load that has nothing to do with industry. Power demand became a measure of energy-intensive industry and household electrification rather than of industry as a whole.

### 2. H1/H1b: a weak, fading link that can't pay for its own estimation (strong evidence)

- Median correlation of port export YoY with official export YoY: **0.55 (2020-21), 0.34 (2022-23), 0.17 (2024-25)** for Comtrade (H1); 0.54, 0.29, 0.18 for IMF (H1b). Most of the 2020-21 correlation is the common COVID swing.
- **In sample versus out of sample:** whole-sample fit 0.94 (H1) and 0.92 (H1b), i.e. the signal explains a few per cent of what the last official number doesn't; with those coefficients in hindsight the real forecasts improve by **0.00 (H1, ratio 1.00) and get 3% worse (H1b, 1.03)**; with real-time estimates they get 11% and 16% worse. A weak signal plus parameters estimated on two or three years of data is worse than no signal. That's estimation cost, not a hidden good signal.
- **Placebos:** the same port signal shifted to the wrong months, or taken from another country, scores about the same as the real one (`CONTROLS.md`, §2), so the real signal adds nothing a random series of the same shape doesn't.
- A few countries did better with the signal (H1: Turkey 0.71, Thailand 0.80, Korea 0.91; H1b: Thailand 0.83, Vietnam and Turkey 0.86), but with 20-40 forecasts each, that is what luck looks like across 24 countries.

### 3. H1/H1b: official exports are values in dollars, and since 2022 prices drive them (moderate evidence)

From the CPB World Trade Monitor (new source, PLAN D53), from 2022 on:

| | Share of the YoY variation in export values explained by volumes | ... by prices (USD unit values) | Port exports' correlation with volumes | ... with values |
|---|---|---|---|---|
| USA | 24% | 79% | 0.58 | 0.47 |
| China | 16% | 31% | 0.34 | 0.11 |
| Japan | 51% | 37% | 0.05 | −0.04 |
| UK | 82% | 0% | 0.13 | 0.09 |
| World | 44% | 90% | 0.18 | 0.31 |

(Volume and price shares overlap because they are correlated; each is the R² on its own.) For the USA and the world, price swings (energy and commodity prices, the dollar) now account for most of the movement in export values, and port volumes track CPB volumes better than values in the USA, China, Japan and the UK. So comparing tonnes with dollars throws away part of whatever signal there is. But port volumes track volumes only weakly too (correlations 0.05-0.58), so a volume target alone wouldn't rescue H1. This covers only the four countries CPB shows individually.

### 4. Short, COVID-dominated training samples (strong evidence for H2, weak for H1)

- Every model is trained on the months since its signal starts: PortWatch from 2019 (YoY from 2020), weather-normalised load from 2020 (YoY from 2021, first forecast March 2023). The first two to three years of training data are the COVID collapse, rebound and 2022 shock, which teach coefficients that don't hold afterwards.
- The benchmark suffers too. For H2 the AR benchmark's error was 4.23 pp, twice that of simply repeating the last published value (2.11 pp). Refitted only on months from 2022, the same AR model would have made 2.18 pp on the same forecasts: the 2021 rebound in the training data doubled the benchmark's error, and the model with the signal is trained on the same months. The signal lost even against that weak benchmark. (The one engine placebo that beat the benchmark, `CONTROLS.md` §2, did so against this weakness.) For H1, dropping the COVID years doesn't help the benchmark (12.20 pp against 9.77): too few months remain for a model fitted per country.
- The test design is not too weak to find a real signal: a synthetic signal (the target's own final values plus noise of known size, `CONTROLS.md` §1) cut the error by 24% in H2's own sample even with noise of 4 percentage points, about the size of IP's own YoY swings. The real signals just carry less information than that.

### 5. Coverage and definitions (weak to moderate evidence)

- **H1:** port data see seaborne cargo only. Countries with much trade by road, rail or pipeline (Canada, Mexico, Poland) or entrepôt re-exports (Singapore) have zero or negative correlations in some periods (Canada −0.43 and −0.23 in 2020-23; Singapore −0.54 in 2022-23).
- **H2:** IP includes mining and energy supply (Eurostat NACE B-D), so the Netherlands' IP probably moved with gas extraction (the Groningen field was run down and closed over these years), which has nothing to do with power use; NLD's load-IP correlation is −0.57 over the whole sample (not checked against NLD's IP by sector). IP is seasonally and calendar-adjusted and load isn't, but that matters little: the month's weekend-day composition explains at most 5% of the total-load signal's variance and nothing of the overnight signal's.

### 6. Timing, publication lags and bugs (ruled out)

- The engine passed every control (`CONTROLS.md`): a real positive control (Comtrade predicting IMF exports, error −67%), synthetic signals, placebos, an independent re-implementation (identical forecasts) and a timing audit from the raw publication times (0 forecasts used a signal month that wasn't out or forecast a target month that was).
- Signal and target are aligned: the lead/lag correlations peak at lag 0 for every hypothesis (all months).
- Nowcasts were made 14 days after the signal month, 33 (H2), 76 (H1) and 108 (H1b) days before the official number. Publication lags of the official series are estimated for backfilled history (IMF 122 days, Eurostat 47, OECD 75, Comtrade per country), mostly on the late side (IMF's true lag was between 92 and 122 days, D42; Eurostat publishes after about 45). A later assumed release makes the benchmark's last known value older, which favours the signal, so the timing assumptions, if anything, flattered H1 and H2.

## Refined hypotheses you could pre-register (proposals only; not added to SIGNALS.md, not tested)

All three were shaped by looking at 2020-2025 data in this diagnosis, so a test on that same period wouldn't be blind. A clean test needs data nobody has looked at: forward data (journal), or each hypothesis's single look at the vault (from 2025-10-08).

1. **H2c: weekday overnight load → energy-intensive production.** Weather-normalised weekday overnight load YoY → YoY of the production index of energy-intensive manufacturing (Eurostat `sts_inpr_m`, NACE C17 paper, C20 chemicals, C23 non-metallic minerals and C24 basic metals, combined with fixed weights to be chosen when it's written, e.g. their electricity use) | 1-2 months | positive | nowcast, trained without 2020-2022 (COVID and the energy shock). Rationale: industrial electricity is dominated by these sectors, while total IP is weighted towards sectors that use little power, which is the likeliest reason the 2022 break happened; and training without 2020-21 halved even the benchmark's error (§4). Needs a new Eurostat series (four NACE divisions; a fetcher change, no new source), and with training from 2023 the first forecasts would come in 2025, so in practice it is a test on the vault and forward data.
2. **H1c: port export volumes → CPB export volumes.** PortWatch export volume YoY → CPB merchandise export volume YoY (USA, GBR, JPN, CHN; world against the 25-country port total) | 1-2 months (CPB publishes about 8 weeks after the month) | positive | nowcast, pooled across the five series. Rationale: removes prices and the dollar, which drive most of the dollar value since 2022. Caveats: only five series; in-sample correlations since 2022 were modest (0.05-0.58), so expect a small effect at best.
3. **H1d: a model with fewer parameters.** The change in the official export YoY since the last published month regressed on the change in the port YoY over the same months, pooled across countries, with no intercept; benchmark: the last published value | 1-3 months | positive | nowcast. Rationale: one slope estimated on ~20 countries instead of three per country cuts the estimation cost that turned a small in-sample gain into an out-of-sample loss, and the last published value was as good a benchmark as the AR model. Caveat: S23's pooled variant (ratio 1.01, 0.94 in the first half) was seen before proposing this.

Decide which (if any) to pre-register; once written in SIGNALS.md, they can be tested as S23 was.
