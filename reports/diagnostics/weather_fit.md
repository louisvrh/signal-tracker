# Weather-normalised power demand: fit diagnostics

Generated 2026-10-07 15:07 UTC by `python -m tracker weather` (4 s). Model: `tracker/signals/weather.py` (PLAN §5.4, D45).

Every value below is **out of sample** except the in-sample R²: each month is predicted by a model fitted on the 3 years before it (at least 1), so the errors are one-month-ahead errors. Degree days start in 2019, so outputs start in 2020.

- **MAPE**: mean absolute daily error as % of actual load (all days; and outside 2020, the lockdowns).
- **HDD, CDD**: median coefficient, as % of mean load per degree day.
- **Weather in YoY**: share of the 28-day YoY's variance explained by the YoY change in 28-day mean HDD and CDD (weekly sample). Raw shows how much weather moves the raw YoY; normalised should be near 0.
- **Verdict**: good = MAPE outside 2020 under 4.5% and under 5% weather left; fair = under 6.0% and 12%; poor otherwise.

| Series | Country | From | R² in | R² out | MAPE | MAPE ex-2020 | HDD % | CDD % | Weather in raw YoY | Weather in normalised YoY | SD raw YoY | SD normalised YoY | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eia_us48_load | USA | 2020-01-01 | 0.91 | 0.88 | 3.35 | 3.29 | 1.29 | 4.08 | 47% | 0% | 3.6 | 2.8 | good |
| eia_us48_overnight_load | USA | 2020-01-01 | 0.85 | 0.80 | 3.84 | 3.86 | 1.70 | 3.37 | 54% | 1% | 3.7 | 2.8 | good |
| entsoe_load | DEU | 2020-01-01 | 0.88 | 0.83 | 4.01 | 4.00 | 0.42 | 1.06 | 9% | 2% | 4.7 | 4.5 | good |
| entsoe_load | FRA | 2020-01-01 | 0.94 | 0.92 | 4.27 | 4.09 | 2.48 | 1.70 | 65% | 5% | 6.2 | 3.8 | good |
| entsoe_load | ITA | 2020-01-01 | 0.80 | 0.74 | 5.09 | 4.41 | 0.74 | 2.94 | 43% | 10% | 6.3 | 5.0 | fair |
| entsoe_load | ESP | 2020-01-01 | 0.84 | 0.78 | 3.85 | 3.52 | 1.03 | 1.84 | 46% | 14% | 5.1 | 4.0 | poor |
| entsoe_load | NLD | 2020-01-01 | 0.81 | 0.63 | 6.22 | 6.39 | 0.28 | 1.65 | 1% | 1% | 6.4 | 6.5 | poor |
| entsoe_load | POL | 2020-01-01 | 0.89 | 0.84 | 3.80 | 3.82 | 0.65 | 2.23 | 20% | 3% | 4.8 | 4.5 | good |
| entsoe_overnight_load | DEU | 2020-01-01 | 0.79 | 0.69 | 4.47 | 4.46 | 0.65 | 1.23 | 14% | 2% | 5.3 | 5.0 | good |
| entsoe_overnight_load | FRA | 2020-01-01 | 0.94 | 0.91 | 4.54 | 4.50 | 2.66 | 1.41 | 65% | 5% | 6.3 | 4.0 | fair |
| entsoe_overnight_load | ITA | 2020-01-01 | 0.70 | 0.68 | 4.83 | 4.30 | 0.56 | 3.00 | 49% | 11% | 5.9 | 4.5 | fair |
| entsoe_overnight_load | ESP | 2020-01-01 | 0.75 | 0.71 | 3.43 | 3.11 | 0.68 | 1.48 | 35% | 11% | 4.4 | 3.7 | fair |
| entsoe_overnight_load | NLD | 2020-01-01 | 0.67 | 0.41 | 3.98 | 4.23 | 0.22 | 2.17 | 7% | 4% | 4.3 | 4.2 | good |
| entsoe_overnight_load | POL | 2020-01-01 | 0.76 | 0.70 | 4.31 | 4.42 | 0.69 | 1.89 | 19% | 3% | 5.1 | 4.8 | good |
| neso_load | GBR | 2020-01-01 | 0.90 | 0.85 | 3.69 | 3.31 | 1.00 | 2.22 | 22% | 1% | 4.3 | 3.8 | good |
| neso_overnight_load | GBR | 2020-01-01 | 0.89 | 0.81 | 4.21 | 4.09 | 1.05 | 2.83 | 25% | 3% | 4.5 | 4.0 | good |

Charts per country: [DEU](weather_DEU.html), [ESP](weather_ESP.html), [FRA](weather_FRA.html), [GBR](weather_GBR.html), [ITA](weather_ITA.html), [NLD](weather_NLD.html), [POL](weather_POL.html), [USA](weather_USA.html).
