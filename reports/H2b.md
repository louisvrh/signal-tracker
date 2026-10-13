# H2b: Weekday overnight load predicts industrial production better than total load

*Run 2026-10-07 15:53 UTC by `python -m tracker nowcast H2b` (S23, `tracker/backtest/nowcast.py`). Vault closed: test dates end before 2025-10-08 and forecasts are scored on official values published before then. Research only, not financial advice.*

## Headline (out of sample)

| Out of sample | All test months | Excluding 2020 |
|---|---|---|
| Forecasts (countries, test dates) | 208 (7, 30) | 208 (7, 30) |
| Test dates | 2023-03-15 to 2025-08-15 | 2023-03-15 to 2025-08-15 |
| RMSE, signal model / AR benchmark (pp) | 5.28 / 4.19 | 5.28 / 4.19 |
| **RMSE ratio** (below 1 = signal helps) | **1.26** | **1.26** |
| Ratio in 1st / 2nd half of test dates | 1.13 / 1.64 | 1.13 / 1.64 |
| Directional hit rate (benchmark) | 0.57 (0.56) | 0.57 (0.56) |
| Diebold-Mariano stat (p) | 1.60 (0.110) | 1.60 (0.110) |

**Verdict.** Not supported: forecasts with the signal were 26% worse than the AR benchmark out of sample.

## Method

- Signal: `entsoe_overnight_load`, `neso_overnight_load`, weather-normalised (`wx`, D45); daily values averaged per calendar month (at least 90% of days), YoY against the same month a year earlier. Target: `eurostat_ip`, `oecd_ip` YoY. Lunar New Year January-February combined and known breaks respected on both sides.
- Monthly test dates on the 15th from 2020-01. On each, the data as known that day (`Vintages.latest(as_of)`, D39). A country-month is forecast once, on the first test date its signal month is complete and its official value isn't out yet.
- Models are refitted on every test date on months whose official value was already published, with each month's signal as first computed (real-time), per country, needing 24 months. Primary model `arx`: y = a + b·signal + c·(last published y); benchmark `ar`: the same without the signal (both direct forecasts h months ahead, h = months since the last published value).
- Scored on the official YoY as known when the vault starts (the latest vintage then). Most of that history has estimated publication dates (backfilled in September 2026, PLAN §5.1), so results may flatter slightly.

## Every variant tried

| Sample | Model vs benchmark | Forecasts | RMSE ratio | 1st half | 2nd half | Hit rate (benchmark) | DM p |
|---|---|---|---|---|---|---|---|
| all | `arx` vs `ar`: primary: signal + last official value vs the same without the signal | 208 | 1.26 | 1.13 | 1.64 | 0.57 (0.56) | 0.110 |
| all | `x_only` vs `ar`: signal alone vs AR benchmark | 208 | 1.26 | 1.10 | 1.70 | 0.58 (0.56) | 0.133 |
| all | `pooled` vs `ar`: pooled ARX (intercept per country) vs AR benchmark | 208 | 1.50 | 1.53 | 1.36 | 0.57 (0.56) | 0.036 |
| all | `arx` vs `rw`: primary model vs last published value | 208 | 2.51 | 2.46 | 2.59 | 0.57 (–) | 0.000 |
| all | `ar` vs `rw`: AR benchmark vs last published value (benchmark check) | 208 | 1.99 | 2.18 | 1.58 | 0.56 (–) | 0.001 |
| ex-2020 | `arx` vs `ar`: primary: signal + last official value vs the same without the signal | 208 | 1.26 | 1.13 | 1.64 | 0.57 (0.56) | 0.110 |
| ex-2020 | `x_only` vs `ar`: signal alone vs AR benchmark | 208 | 1.26 | 1.10 | 1.70 | 0.58 (0.56) | 0.133 |
| ex-2020 | `pooled` vs `ar`: pooled ARX (intercept per country) vs AR benchmark | 208 | 1.50 | 1.53 | 1.36 | 0.57 (0.56) | 0.036 |
| ex-2020 | `arx` vs `rw`: primary model vs last published value | 208 | 2.51 | 2.46 | 2.59 | 0.57 (–) | 0.000 |
| ex-2020 | `ar` vs `rw`: AR benchmark vs last published value (benchmark check) | 208 | 1.99 | 2.18 | 1.58 | 0.56 (–) | 0.001 |
| H2b test (paired months) | `arx (overnight)` vs `arx (total)`: overnight load vs total load, same country-months | 206 | 1.00 | 1.00 | 1.00 | 0.57 (0.59) | 0.976 |

## H2b comparison: overnight against total load

On the 206 country-months both models forecast, the out-of-sample RMSE was 5.30 pp with overnight load and 5.30 pp with total load (ratio 1.00; below 1 = overnight better). The paired row in the variant table has the halves and the Diebold-Mariano test.

## By country (primary model, all test months)

| Country | Forecasts | RMSE ratio | Hit rate | RMSE model / benchmark |
|---|---|---|---|---|
| POL | 30 | 0.96 | 0.53 | 5.68 / 5.90 |
| NLD | 30 | 1.08 | 0.63 | 3.16 / 2.91 |
| ITA | 30 | 1.18 | 0.57 | 7.37 / 6.27 |
| FRA | 30 | 1.44 | 0.50 | 3.62 / 2.51 |
| DEU | 30 | 1.50 | 0.57 | 5.71 / 3.80 |
| ESP | 30 | 1.56 | 0.60 | 6.41 / 4.10 |
| GBR | 28 | 2.80 | 0.61 | 3.24 / 1.16 |

## Appendix: in-sample fit (not evidence)

Official YoY on the same month's signal YoY, all months before the vault (latest vintages).

| Country | Months | From | To | Correlation | Slope | R² |
|---|---|---|---|---|---|---|
| ITA | 55 | 2021-01 | 2025-07 | 0.73 | 2.15 | 0.54 |
| POL | 55 | 2021-01 | 2025-07 | 0.70 | 1.22 | 0.49 |
| ESP | 55 | 2021-01 | 2025-07 | 0.58 | 1.17 | 0.34 |
| GBR | 54 | 2021-01 | 2025-06 | 0.51 | 0.64 | 0.26 |
| DEU | 55 | 2021-01 | 2025-07 | 0.31 | 0.39 | 0.10 |
| FRA | 55 | 2021-01 | 2025-07 | 0.19 | 0.35 | 0.03 |
| NLD | 55 | 2021-01 | 2025-07 | -0.48 | -0.36 | 0.23 |
