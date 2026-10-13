# Why the nowcasts failed: the numbers

*Run 2026-10-07 21:55 UTC by `python -m tracker diagnose why` (`tracker/backtest/why.py`, v2). Exploratory diagnostics logged as `DX2`; vault closed (data as known on 2025-10-08). The plain-English reading is in `WHY_FAILED.md`.*

## 1. In sample versus out of sample

RMSE ratio of the model with the signal (`arx`) to the same model without it (`ar`); below 1 = the signal helps. *In sample*: both fitted on every month before the vault (final vintages). *Oracle coefficients*: the real forecasts' inputs with those whole-sample coefficients (look-ahead on purpose). *Walk-forward*: the real result.

| Hypothesis | Forecasts | In sample | Oracle coefficients | Walk-forward |
|---|---|---|---|---|
| H1 | 820 | 0.94 | 1.00 | 1.11 |
| H1b | 858 | 0.92 | 1.03 | 1.16 |
| H2 | 206 | 0.77 | 1.40 | 1.26 |
| H2b | 208 | 0.82 | 1.40 | 1.26 |

### The benchmark without the COVID years

On the same forecasts (those with at least 12 published months from 2022 to train on): the RMSE (pp) of the engine's AR benchmark (trained on every month since the signal starts, so on 2020-21 too), of the last published value, and of the same AR refitted only on months from 2022 (final vintages). The model with the signal isn't refitted.

| Hypothesis | Forecasts | AR benchmark | Last published value | AR from 2022 |
|---|---|---|---|---|
| H1 | 560 | 9.77 | 10.23 | 12.20 |
| H1b | 594 | 10.20 | 10.05 | 11.87 |
| H2 | 204 | 4.23 | 2.11 | 2.18 |
| H2b | 208 | 4.19 | 2.10 | 2.16 |

### H1 by country

| Country | Forecasts | In sample | Oracle coefficients | Walk-forward |
|---|---|---|---|---|
| TUR | 26 | 0.91 | 0.82 | 0.71 |
| THA | 38 | 0.87 | 0.81 | 0.80 |
| KOR | 20 | 0.89 | 0.89 | 0.91 |
| AUS | 42 | 1.00 | 1.00 | 1.00 |
| CAN | 42 | 1.00 | 0.99 | 1.01 |
| FRA | 26 | 0.97 | 1.09 | 1.01 |
| GBR | 40 | 0.96 | 0.99 | 1.02 |
| NLD | 40 | 0.89 | 0.98 | 1.03 |
| BRA | 44 | 1.00 | 1.00 | 1.03 |
| POL | 40 | 0.99 | 1.03 | 1.06 |
| ZAF | 35 | 0.95 | 1.00 | 1.06 |
| ESP | 38 | 0.96 | 1.03 | 1.06 |
| SAU | 36 | 0.95 | 0.96 | 1.07 |
| CHN | 13 | 0.99 | 1.00 | 1.10 |
| SGP | 26 | 1.00 | 0.98 | 1.19 |
| DEU | 42 | 0.96 | 1.04 | 1.20 |
| MYS | 33 | 0.96 | 1.12 | 1.23 |
| IDN | 33 | 1.00 | 1.00 | 1.30 |
| USA | 42 | 0.84 | 1.01 | 1.35 |
| IND | 42 | 0.88 | 1.05 | 1.36 |
| JPN | 42 | 0.97 | 1.22 | 1.41 |
| MEX | 42 | 0.94 | 1.16 | 1.52 |
| ITA | 38 | 0.92 | 1.31 | 1.60 |

### H1b by country

| Country | Forecasts | In sample | Oracle coefficients | Walk-forward |
|---|---|---|---|---|
| THA | 38 | 0.88 | 0.84 | 0.83 |
| TUR | 38 | 0.95 | 0.93 | 0.86 |
| VNM | 29 | 0.77 | 0.79 | 0.86 |
| KOR | 29 | 0.94 | 0.82 | 0.93 |
| AUS | 38 | 1.00 | 1.00 | 1.00 |
| BRA | 38 | 1.00 | 0.98 | 1.03 |
| NLD | 38 | 0.93 | 1.00 | 1.05 |
| MYS | 29 | 0.94 | 1.07 | 1.08 |
| CHN | 29 | 0.88 | 1.01 | 1.08 |
| GBR | 38 | 0.97 | 1.01 | 1.09 |
| POL | 38 | 0.97 | 1.05 | 1.09 |
| CAN | 38 | 0.98 | 1.03 | 1.10 |
| IDN | 29 | 1.00 | 0.99 | 1.12 |
| ESP | 38 | 0.96 | 1.05 | 1.12 |
| ZAF | 38 | 0.84 | 1.34 | 1.16 |
| FRA | 38 | 0.98 | 1.09 | 1.18 |
| SAU | 38 | 0.98 | 0.95 | 1.19 |
| SGP | 29 | 1.00 | 1.02 | 1.24 |
| IND | 38 | 0.88 | 0.87 | 1.29 |
| USA | 38 | 0.80 | 1.08 | 1.44 |
| DEU | 38 | 0.93 | 1.17 | 1.53 |
| ITA | 38 | 0.92 | 1.30 | 1.58 |
| JPN | 38 | 0.81 | 1.61 | 1.60 |
| MEX | 38 | 0.93 | 1.25 | 1.67 |

### H2 by country

| Country | Forecasts | In sample | Oracle coefficients | Walk-forward |
|---|---|---|---|---|
| POL | 30 | 0.79 | 1.50 | 1.02 |
| ITA | 30 | 0.61 | 1.45 | 1.11 |
| NLD | 30 | 0.91 | 0.86 | 1.26 |
| FRA | 28 | 0.97 | 1.27 | 1.42 |
| DEU | 30 | 0.95 | 1.33 | 1.47 |
| ESP | 30 | 0.75 | 1.51 | 1.63 |
| GBR | 28 | 0.85 | 1.74 | 2.15 |

### H2b by country

| Country | Forecasts | In sample | Oracle coefficients | Walk-forward |
|---|---|---|---|---|
| POL | 30 | 0.80 | 1.45 | 0.96 |
| NLD | 30 | 0.97 | 0.94 | 1.08 |
| ITA | 30 | 0.70 | 1.52 | 1.18 |
| FRA | 30 | 0.98 | 1.34 | 1.44 |
| DEU | 30 | 0.96 | 1.33 | 1.50 |
| ESP | 30 | 0.82 | 1.41 | 1.56 |
| GBR | 28 | 0.90 | 1.71 | 2.80 |

## 2. Correlation of signal and target YoY by period

Median across countries of the same-month correlation (and slope of target on signal) within each period, final vintages.

| Hypothesis | Period | Countries | Median correlation | Median slope | Median R² |
|---|---|---|---|---|---|
| H1 | 2020-21 | 23 | 0.55 | 1.21 | 0.30 |
| H1 | 2022-23 | 23 | 0.34 | 0.41 | 0.14 |
| H1 | 2024-25 | 22 | 0.17 | 0.17 | 0.05 |
| H1b | 2020-21 | 24 | 0.54 | 1.05 | 0.29 |
| H1b | 2022-23 | 24 | 0.29 | 0.37 | 0.13 |
| H1b | 2024-25 | 24 | 0.18 | 0.15 | 0.06 |
| H2 | 2020-21 | 7 | 0.82 | 2.88 | 0.68 |
| H2 | 2022-23 | 7 | 0.11 | 0.04 | 0.20 |
| H2 | 2024-25 | 7 | 0.02 | 0.01 | 0.10 |
| H2b | 2020-21 | 7 | 0.73 | 2.24 | 0.54 |
| H2b | 2022-23 | 7 | 0.06 | 0.06 | 0.20 |
| H2b | 2024-25 | 7 | -0.06 | -0.04 | 0.01 |

### H1: correlation by country and period

| Country | 2020-21 | 2022-23 | 2024-25 |
|---|---|---|---|
| AUS | 0.04 | -0.19 | 0.35 |
| BRA | -0.34 | -0.37 | 0.63 |
| CAN | -0.43 | -0.23 | 0.44 |
| CHN | 0.76 | 0.16 | – |
| DEU | 0.62 | 0.23 | 0.18 |
| ESP | 0.24 | 0.48 | 0.09 |
| FRA | -0.02 | 0.11 | -0.17 |
| GBR | 0.61 | 0.34 | 0.12 |
| IDN | 0.78 | 0.59 | 0.40 |
| IND | 0.38 | 0.77 | 0.27 |
| ITA | 0.59 | 0.18 | -0.39 |
| JPN | 0.74 | 0.15 | -0.29 |
| KOR | 0.22 | 0.62 | 0.50 |
| MEX | 0.56 | 0.02 | 0.02 |
| MYS | 0.55 | 0.51 | 0.16 |
| NLD | 0.64 | 0.37 | -0.20 |
| POL | 0.28 | -0.44 | -0.04 |
| SAU | 0.36 | 0.68 | 0.10 |
| SGP | 0.50 | -0.54 | 0.06 |
| THA | 0.58 | 0.65 | 0.50 |
| TUR | 0.36 | 0.57 | 0.20 |
| USA | 0.71 | 0.71 | 0.30 |
| ZAF | 0.60 | 0.35 | 0.22 |

### H1b: correlation by country and period

| Country | 2020-21 | 2022-23 | 2024-25 |
|---|---|---|---|
| AUS | 0.04 | -0.17 | 0.20 |
| BRA | -0.31 | -0.36 | 0.63 |
| CAN | -0.43 | -0.22 | 0.44 |
| CHN | 0.75 | 0.15 | -0.25 |
| DEU | 0.62 | 0.23 | 0.11 |
| ESP | 0.24 | 0.37 | 0.16 |
| FRA | 0.01 | -0.05 | -0.19 |
| GBR | 0.52 | 0.36 | -0.25 |
| IDN | 0.78 | 0.56 | 0.26 |
| IND | 0.38 | 0.77 | 0.40 |
| ITA | 0.58 | 0.17 | -0.33 |
| JPN | 0.74 | 0.15 | -0.21 |
| KOR | 0.24 | 0.57 | 0.76 |
| MEX | 0.56 | 0.02 | 0.11 |
| MYS | 0.56 | 0.50 | 0.25 |
| NLD | 0.67 | 0.28 | -0.30 |
| POL | 0.29 | -0.35 | 0.16 |
| SAU | 0.23 | 0.65 | 0.22 |
| SGP | 0.48 | -0.56 | 0.10 |
| THA | 0.61 | 0.66 | 0.46 |
| TUR | 0.36 | 0.57 | 0.20 |
| USA | 0.70 | 0.64 | 0.22 |
| VNM | 0.60 | 0.90 | 0.45 |
| ZAF | 0.59 | 0.31 | 0.08 |

### H2: correlation by country and period

| Country | 2020-21 | 2022-23 | 2024-25 |
|---|---|---|---|
| DEU | 0.87 | -0.42 | -0.32 |
| ESP | 0.97 | 0.16 | 0.02 |
| FRA | 0.50 | -0.45 | 0.35 |
| GBR | 0.71 | 0.11 | 0.03 |
| ITA | 0.97 | 0.48 | -0.48 |
| NLD | 0.82 | -0.83 | -0.41 |
| POL | 0.82 | 0.56 | 0.05 |

### H2b: correlation by country and period

| Country | 2020-21 | 2022-23 | 2024-25 |
|---|---|---|---|
| DEU | 0.77 | -0.46 | -0.39 |
| ESP | 0.93 | 0.06 | -0.06 |
| FRA | 0.29 | -0.32 | 0.29 |
| GBR | 0.70 | 0.28 | 0.11 |
| ITA | 0.93 | 0.45 | -0.12 |
| NLD | 0.34 | -0.65 | -0.24 |
| POL | 0.73 | 0.67 | 0.02 |

## 3. Rolling 24-month correlation (median across countries)

Windows ending in June and December, and the last window that still covers at least half the countries (the newest months exist for only a few).

| Window ending | H1 | H1b | H2 | H2b |
|---|---|---|---|---|
| 2021-12 | 0.38 | 0.45 | – | – |
| 2022-06 | 0.47 | 0.44 | – | – |
| 2022-12 | 0.33 | 0.30 | 0.69 | 0.63 |
| 2023-06 | 0.17 | 0.17 | 0.16 | 0.08 |
| 2023-12 | 0.34 | 0.29 | 0.11 | 0.06 |
| 2024-06 | 0.19 | 0.23 | -0.13 | -0.26 |
| 2024-12 | 0.17 | 0.12 | -0.10 | -0.13 |
| 2025-05 | 0.17 | 0.09 | -0.01 | -0.02 |
| 2025-06 | 0.12 | – | -0.02 | -0.05 |
| 2025-07 | – | – | -0.15 | -0.06 |

## 4. Alignment: lead/lag correlations

Median across countries of corr(target_t, signal_(t+k)); k > 0 = the signal k months later. A timing slip would put the peak away from k = 0. Share peaking = share of countries whose highest correlation is at k.

### H1

| k | Median corr, all months | Share peaking | Median corr, 2022 on | Share peaking, 2022 on |
|---|---|---|---|---|
| -3 | 0.08 | 0.13 | 0.15 | 0.13 |
| -2 | 0.14 | 0.00 | 0.04 | 0.04 |
| -1 | 0.29 | 0.04 | 0.16 | 0.09 |
| 0 | 0.39 | 0.30 | 0.22 | 0.43 |
| 1 | 0.35 | 0.17 | 0.15 | 0.04 |
| 2 | 0.28 | 0.17 | 0.13 | 0.04 |
| 3 | 0.18 | 0.17 | 0.16 | 0.22 |

### H1b

| k | Median corr, all months | Share peaking | Median corr, 2022 on | Share peaking, 2022 on |
|---|---|---|---|---|
| -3 | 0.15 | 0.08 | 0.16 | 0.12 |
| -2 | 0.19 | 0.00 | 0.09 | 0.04 |
| -1 | 0.29 | 0.08 | 0.22 | 0.12 |
| 0 | 0.39 | 0.33 | 0.23 | 0.38 |
| 1 | 0.34 | 0.17 | 0.18 | 0.04 |
| 2 | 0.28 | 0.12 | 0.17 | 0.08 |
| 3 | 0.21 | 0.21 | 0.16 | 0.21 |

### H2

| k | Median corr, all months | Share peaking | Median corr, 2022 on | Share peaking, 2022 on |
|---|---|---|---|---|
| -3 | 0.08 | 0.00 | 0.11 | 0.29 |
| -2 | 0.10 | 0.00 | 0.09 | 0.29 |
| -1 | 0.25 | 0.14 | -0.03 | 0.14 |
| 0 | 0.58 | 0.57 | 0.10 | 0.00 |
| 1 | 0.45 | 0.14 | -0.15 | 0.00 |
| 2 | 0.18 | 0.00 | -0.26 | 0.00 |
| 3 | 0.10 | 0.14 | -0.33 | 0.29 |

### H2b

| k | Median corr, all months | Share peaking | Median corr, 2022 on | Share peaking, 2022 on |
|---|---|---|---|---|
| -3 | 0.11 | 0.00 | 0.08 | 0.43 |
| -2 | 0.18 | 0.00 | 0.14 | 0.14 |
| -1 | 0.25 | 0.14 | 0.01 | 0.14 |
| 0 | 0.51 | 0.57 | 0.04 | 0.00 |
| 1 | 0.43 | 0.14 | -0.19 | 0.00 |
| 2 | 0.23 | 0.00 | -0.14 | 0.00 |
| 3 | 0.12 | 0.14 | -0.23 | 0.29 |

## 5. Timing of the nowcasts and stored publication lags

| Hypothesis | Forecasts | Horizon h (share of forecasts) | Test date, days after the signal month | Test date, days before the target month was published |
|---|---|---|---|---|
| H1 | 893 | 1: 0.05, 2: 0.41, 3: 0.15, 4: 0.18, 5: 0.04, 6: 0.03, 7: 0.00, 10: 0.11, 11: 0.00, 13: 0.02, 14: 0.00 | 14 | 76 |
| H1b | 930 | 4: 0.98, 5: 0.02 | 14 | 108 |
| H2 | 214 | 2: 0.98, 3: 0.02 | 14 | 33 |
| H2b | 216 | 2: 0.98, 3: 0.02 | 14 | 33 |

Stored publication lags (first vintage's `available_at` minus the period's end, days):

| Indicator | Vintage type | Values | Median | Min | Max |
|---|---|---|---|---|---|
| comtrade_exports | estimated | 3102 | 90 | 30 | 400 |
| comtrade_exports | live | 28 | 78 | 26 | 269 |
| entsoe_load | estimated | 25664 | 1 | 1 | 1 |
| entsoe_load | live | 52 | 0 | 0 | 0 |
| entsoe_overnight_load | estimated | 18363 | 1 | 1 | 1 |
| entsoe_overnight_load | live | 48 | -1 | -1 | 0 |
| eurostat_ip | estimated | 834 | 47 | 47 | 47 |
| eurostat_ip | live | 1 | 31 | 31 | 31 |
| imf_exports | estimated | 3264 | 122 | 122 | 122 |
| imf_exports | live | 48 | 106 | 92 | 121 |
| neso_load | estimated | 4295 | 1 | 1 | 1 |
| neso_load | live | 1 | 0 | 0 | 0 |
| neso_overnight_load | estimated | 3063 | 1 | 1 | 1 |
| neso_overnight_load | live | 6 | 0 | -1 | 0 |
| oecd_ip | estimated | 1106 | 75 | 55 | 75 |
| oecd_ip | live | 5 | 61 | 61 | 61 |
| portwatch_exports | estimated | 70375 | 11 | 11 | 11 |
| portwatch_exports | live | 425 | 8 | 4 | 10 |

## 6. Value versus volume (CPB World Trade Monitor)

Monthly YoY of official export values in USD (IMF IMTS), CPB export volumes and USD unit values (prices), and PortWatch port exports (monthly mean of daily tonnes; for WLD, the sum over the 25 watchlist countries). R² = squared correlation with the value's YoY.

| Country | Window | Months | R² value~volume | R² value~price | s.d. value | s.d. volume | s.d. price | corr port~volume | corr port~value | corr port~price |
|---|---|---|---|---|---|---|---|---|---|---|
| USA | 2020 on | 65 | 0.74 | 0.69 | 16.5 | 9.8 | 8.7 | 0.66 | 0.57 | 0.35 |
| USA | 2022 on | 41 | 0.24 | 0.79 | 9.7 | 3.3 | 8.1 | 0.58 | 0.47 | 0.40 |
| GBR | 2020 on | 65 | 0.78 | 0.22 | 15.6 | 13.2 | 6.4 | 0.25 | 0.32 | 0.34 |
| GBR | 2022 on | 41 | 0.82 | 0.00 | 13.3 | 13.1 | 4.8 | 0.13 | 0.09 | 0.01 |
| JPN | 2020 on | 65 | 0.91 | 0.40 | 13.6 | 10.8 | 3.7 | 0.61 | 0.60 | 0.28 |
| JPN | 2022 on | 41 | 0.51 | 0.37 | 4.6 | 4.1 | 3.2 | 0.05 | -0.04 | -0.09 |
| CHN | 2020 on | 65 | 0.61 | 0.35 | 15.8 | 13.7 | 8.2 | 0.65 | 0.52 | -0.00 |
| CHN | 2022 on | 41 | 0.16 | 0.31 | 9.6 | 8.7 | 8.0 | 0.34 | 0.11 | -0.23 |
| WLD | 2020 on | 67 | 0.80 | 0.86 | 14.0 | 6.8 | 7.9 | 0.58 | 0.62 | 0.54 |
| WLD | 2022 on | 43 | 0.44 | 0.90 | 7.4 | 2.5 | 5.9 | 0.18 | 0.31 | 0.30 |

## 7. Calendar composition (H2, H2b)

How much each series' YoY moves per percentage point of change in the month's share of weekend days against a year earlier, and the R². The target is calendar-adjusted IP, so it shouldn't move.

### H2

| Country | Series | Months | pp per point | R² |
|---|---|---|---|---|
| DEU | signal | 55 | -0.19 | 0.01 |
| ESP | signal | 55 | -0.26 | 0.04 |
| FRA | signal | 53 | -0.31 | 0.05 |
| GBR | signal | 54 | -0.14 | 0.01 |
| ITA | signal | 55 | -0.30 | 0.03 |
| NLD | signal | 55 | -0.02 | 0.00 |
| POL | signal | 55 | -0.15 | 0.01 |
| DEU | target | 55 | -0.03 | 0.00 |
| ESP | target | 55 | -0.18 | 0.00 |
| FRA | target | 53 | -0.18 | 0.00 |
| GBR | target | 54 | 0.01 | 0.00 |
| ITA | target | 55 | -0.31 | 0.01 |
| NLD | target | 55 | 0.00 | 0.00 |
| POL | target | 55 | -0.21 | 0.00 |

### H2b

| Country | Series | Months | pp per point | R² |
|---|---|---|---|---|
| DEU | signal | 55 | 0.04 | 0.00 |
| ESP | signal | 55 | -0.08 | 0.00 |
| FRA | signal | 55 | -0.03 | 0.00 |
| GBR | signal | 54 | -0.02 | 0.00 |
| ITA | signal | 55 | 0.00 | 0.00 |
| NLD | signal | 55 | 0.04 | 0.00 |
| POL | signal | 55 | 0.02 | 0.00 |
| DEU | target | 55 | -0.03 | 0.00 |
| ESP | target | 55 | -0.18 | 0.00 |
| FRA | target | 55 | -0.15 | 0.00 |
| GBR | target | 54 | 0.01 | 0.00 |
| ITA | target | 55 | -0.31 | 0.01 |
| NLD | target | 55 | 0.00 | 0.00 |
| POL | target | 55 | -0.21 | 0.00 |

