# H1b: PortWatch export volume YoY → IMF monthly exports YoY

*Run 2026-10-07 15:54 UTC by `python -m tracker nowcast H1b` (S23, `tracker/backtest/nowcast.py`). Vault closed: test dates end before 2025-10-08 and forecasts are scored on official values published before then. Research only, not financial advice.*

## Headline (out of sample)

| Out of sample | All test months | Excluding 2020 |
|---|---|---|
| Forecasts (countries, test dates) | 858 (24, 38) | 576 (24, 26) |
| Test dates | 2022-05-15 to 2025-06-15 | 2023-05-15 to 2025-06-15 |
| RMSE, signal model / AR benchmark (pp) | 14.18 / 12.25 | 16.60 / 15.78 |
| **RMSE ratio** (below 1 = signal helps) | **1.16** | **1.05** |
| Ratio in 1st / 2nd half of test dates | 1.13 / 1.25 | 0.95 / 1.38 |
| Directional hit rate (benchmark) | 0.63 (0.61) | 0.61 (0.59) |
| Diebold-Mariano stat (p) | 3.60 (0.000) | 0.50 (0.616) |

**Verdict.** Not supported: forecasts with the signal were 16% worse than the AR benchmark out of sample.

## Method

- Signal: `portwatch_exports`; daily values averaged per calendar month (at least 90% of days), YoY against the same month a year earlier. Target: `imf_exports` YoY. Lunar New Year January-February combined and known breaks respected on both sides.
- Monthly test dates on the 15th from 2020-01. On each, the data as known that day (`Vintages.latest(as_of)`, D39). A country-month is forecast once, on the first test date its signal month is complete and its official value isn't out yet.
- Models are refitted on every test date on months whose official value was already published, with each month's signal as first computed (real-time), per country, needing 24 months. Primary model `arx`: y = a + b·signal + c·(last published y); benchmark `ar`: the same without the signal (both direct forecasts h months ahead, h = months since the last published value).
- Scored on the official YoY as known when the vault starts (the latest vintage then). Most of that history has estimated publication dates (backfilled in September 2026, PLAN §5.1), so results may flatter slightly.

## Every variant tried

| Sample | Model vs benchmark | Forecasts | RMSE ratio | 1st half | 2nd half | Hit rate (benchmark) | DM p |
|---|---|---|---|---|---|---|---|
| all | `arx` vs `ar`: primary: signal + last official value vs the same without the signal | 858 | 1.16 | 1.13 | 1.25 | 0.63 (0.61) | 0.000 |
| all | `x_only` vs `ar`: signal alone vs AR benchmark | 858 | 1.26 | 1.13 | 1.62 | 0.65 (0.61) | 0.000 |
| all | `pooled` vs `ar`: pooled ARX (intercept per country) vs AR benchmark | 858 | 1.02 | 0.97 | 1.16 | 0.63 (0.61) | 0.640 |
| all | `arx` vs `rw`: primary model vs last published value | 858 | 1.20 | 1.25 | 1.07 | 0.63 (–) | 0.002 |
| all | `ar` vs `rw`: AR benchmark vs last published value (benchmark check) | 858 | 1.04 | 1.11 | 0.86 | 0.61 (–) | 0.673 |
| ex-2020 | `arx` vs `ar`: primary: signal + last official value vs the same without the signal | 576 | 1.05 | 0.95 | 1.38 | 0.61 (0.59) | 0.616 |
| ex-2020 | `x_only` vs `ar`: signal alone vs AR benchmark | 576 | 1.22 | 1.05 | 1.74 | 0.62 (0.59) | 0.038 |
| ex-2020 | `pooled` vs `ar`: pooled ARX (intercept per country) vs AR benchmark | 576 | 0.99 | 0.93 | 1.17 | 0.62 (0.59) | 0.586 |
| ex-2020 | `arx` vs `rw`: primary model vs last published value | 576 | 1.66 | 1.85 | 1.42 | 0.61 (–) | 0.004 |
| ex-2020 | `ar` vs `rw`: AR benchmark vs last published value (benchmark check) | 576 | 1.57 | 1.95 | 1.03 | 0.59 (–) | 0.076 |
| sensitivity: likely-estimate countries out | `arx` vs `ar`: primary: signal + last official value vs the same without the signal | 695 | 1.19 | 1.17 | 1.25 | 0.61 (0.60) | 0.000 |
| sensitivity, ex-2020 | `arx` vs `ar`: primary: signal + last official value vs the same without the signal | 470 | 1.04 | 0.94 | 1.39 | 0.61 (0.59) | 0.700 |

## Sensitivity: countries whose recent IMF months are likely estimates

Excluded (as known on 2025-10-08; rule: no monthly Comtrade report, or IMF 6+ months beyond the country's own Comtrade reports):

- CHN: IMF to 2025-05, own Comtrade reports to 2024-08 (9 months behind)
- FRA: IMF to 2025-05, own Comtrade reports to 2024-11 (6 months behind)
- KOR: IMF to 2025-05, own Comtrade reports to 2024-11 (6 months behind)
- TUR: IMF to 2025-05, own Comtrade reports to 2024-11 (6 months behind)
- VNM: no monthly Comtrade data; IMF to 2025-05

The sensitivity rows in the variant table above use the remaining countries.

## By country (primary model, all test months)

| Country | Forecasts | RMSE ratio | Hit rate | RMSE model / benchmark |
|---|---|---|---|---|
| THA | 38 | 0.83 | 0.76 | 6.75 / 8.13 |
| TUR | 38 | 0.86 | 0.66 | 8.81 / 10.26 |
| VNM | 29 | 0.86 | 0.66 | 12.66 / 14.66 |
| KOR | 29 | 0.93 | 0.72 | 8.63 / 9.29 |
| AUS | 38 | 1.00 | 0.63 | 12.58 / 12.54 |
| BRA | 38 | 1.03 | 0.58 | 13.60 / 13.16 |
| NLD | 38 | 1.05 | 0.63 | 8.39 / 7.95 |
| MYS | 29 | 1.08 | 0.59 | 17.16 / 15.90 |
| CHN | 29 | 1.08 | 0.72 | 13.41 / 12.37 |
| GBR | 38 | 1.09 | 0.74 | 13.76 / 12.67 |
| POL | 38 | 1.09 | 0.55 | 8.57 / 7.85 |
| CAN | 38 | 1.10 | 0.63 | 12.60 / 11.44 |
| IDN | 29 | 1.12 | 0.52 | 17.96 / 16.09 |
| ESP | 38 | 1.12 | 0.58 | 9.18 / 8.22 |
| ZAF | 38 | 1.16 | 0.53 | 20.63 / 17.72 |
| FRA | 38 | 1.18 | 0.74 | 7.02 / 5.96 |
| SAU | 38 | 1.19 | 0.50 | 32.02 / 26.85 |
| SGP | 29 | 1.24 | 0.66 | 14.00 / 11.31 |
| IND | 38 | 1.29 | 0.71 | 18.96 / 14.67 |
| USA | 38 | 1.44 | 0.68 | 11.21 / 7.77 |
| DEU | 38 | 1.53 | 0.53 | 10.29 / 6.74 |
| ITA | 38 | 1.58 | 0.55 | 11.91 / 7.52 |
| JPN | 38 | 1.60 | 0.50 | 9.34 / 5.84 |
| MEX | 38 | 1.67 | 0.74 | 15.47 / 9.27 |

## Appendix: in-sample fit (not evidence)

Official YoY on the same month's signal YoY, all months before the vault (latest vintages).

| Country | Months | From | To | Correlation | Slope | R² |
|---|---|---|---|---|---|---|
| VNM | 59 | 2020-02 | 2025-05 | 0.65 | 0.50 | 0.42 |
| JPN | 65 | 2020-01 | 2025-05 | 0.60 | 1.67 | 0.36 |
| THA | 65 | 2020-01 | 2025-05 | 0.59 | 0.54 | 0.35 |
| USA | 65 | 2020-01 | 2025-05 | 0.57 | 1.41 | 0.32 |
| IDN | 59 | 2020-02 | 2025-05 | 0.48 | 0.69 | 0.23 |
| ZAF | 65 | 2020-01 | 2025-05 | 0.47 | 1.31 | 0.22 |
| NLD | 65 | 2020-01 | 2025-05 | 0.47 | 1.19 | 0.22 |
| IND | 65 | 2020-01 | 2025-05 | 0.45 | 0.94 | 0.20 |
| CHN | 59 | 2020-02 | 2025-05 | 0.44 | 0.95 | 0.19 |
| MYS | 59 | 2020-02 | 2025-05 | 0.43 | 0.45 | 0.19 |
| ITA | 65 | 2020-01 | 2025-05 | 0.43 | 0.71 | 0.18 |
| DEU | 65 | 2020-01 | 2025-05 | 0.40 | 0.78 | 0.16 |
| MEX | 65 | 2020-01 | 2025-05 | 0.38 | 0.66 | 0.15 |
| TUR | 65 | 2020-01 | 2025-05 | 0.38 | 0.60 | 0.15 |
| KOR | 59 | 2020-02 | 2025-05 | 0.35 | 0.49 | 0.12 |
| GBR | 65 | 2020-01 | 2025-05 | 0.32 | 0.62 | 0.10 |
| SAU | 65 | 2020-01 | 2025-05 | 0.30 | 0.56 | 0.09 |
| ESP | 65 | 2020-01 | 2025-05 | 0.20 | 0.52 | 0.04 |
| POL | 65 | 2020-01 | 2025-05 | 0.20 | 0.24 | 0.04 |
| AUS | 65 | 2020-01 | 2025-05 | 0.09 | 0.38 | 0.01 |
| SGP | 59 | 2020-02 | 2025-05 | 0.02 | 0.05 | 0.00 |
| FRA | 65 | 2020-01 | 2025-05 | -0.02 | -0.03 | 0.00 |
| BRA | 65 | 2020-01 | 2025-05 | -0.03 | -0.06 | 0.00 |
| CAN | 65 | 2020-01 | 2025-05 | -0.27 | -0.48 | 0.07 |
