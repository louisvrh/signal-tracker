# Signal definitions & pre-registered hypotheses (v2)

Rules: write a hypothesis here **before** testing it. Every test run is logged in `hypotheses_tested`. The last 12 months (the vault) are untouched until each hypothesis's final test. The full protocol is in PLAN §5.6.

---

## 1. Transforms

- **Rolling sums:** 7-day and 28-day for daily series.
- **YoY, daily series:** a 28-day sum compared with the same 28-day window 364 days earlier (keeps weekdays aligned).
- **YoY, monthly series:** each month compared with the same month a year earlier. For Lunar New Year countries (CHN, TWN, VNM, KOR, SGP, MYS, IDN), January and February are combined.
- **Moving-holiday flags:** daily windows overlapping Lunar New Year, Ramadan/Eid, Easter, Diwali or Golden Week are flagged. They're shaded on charts and given half weight in composites.
- **Seasonal adjustment:** STL, for monthly series with at least 36 months of history, where YoY alone is inadequate.
- **Robust z-score:** (x − 3-year rolling median) / (1.4826 × 3-year rolling MAD). Requires at least 24 months of history. Charts clip at ±5.

## 2. Weather-normalised power demand

- **Model:** daily load ~ HDD18 + CDD22 + day-of-week + holiday, fitted on a rolling 3-year window. *As built (S18, PLAN D45): also a 24 December-1 January dummy and two annual harmonics, refitted monthly on the 3 years before each month with Huber weights; chain step `wx`.*
- **Anomaly:** actual − expected. The signal is the YoY of weather-normalised demand, then robust-z scored: `wx>yoy>rz(3y)`. Normalised demand = actual − b_hdd·(HDD − normal HDD) − b_cdd·(CDD − normal CDD).
- **Two indicators per country:** total load, and **weekday overnight load** (00:00–05:00 local time), which is robust to rooftop solar.
- **Exclusions:** countries marked `exclude_from_signals: [power_demand]` in `countries.yaml` get no power model. Build the country list with `Registry.signal_countries()`. Currently only ZAF is excluded, because load-shedding means its demand reflects supply limits (PLAN D22).

## 3. Composite country activity score

1. Take the robust z of the YoY change for each alternative indicator, with its sign flipped where `higher_is_stronger` is false.
2. Average within the **trade** category (port exports and imports) and the **energy** category (normalised total and overnight load, Ember demand).
3. The country score is the mean of the category scores. It's reported with coverage and requires at least 2 contributing indicators.
4. Indicators a country is excluded from (`Registry.signal_countries()`, PLAN D22) don't contribute. ZAF therefore has no energy category, and its score is the trade category alone.
5. *As built (S19, PLAN D46):* weekly values ending on the data's latest day; component chains `yoy>rz(3y)` (ports, Ember demand) and `wx>yoy>rz(3y)` (daily load); z(activity score) is the score's own robust z over 3 years, so it starts in January 2024.

## 4. Divergences and flags

- **Market divergence:** D_mkt = z(activity score) − z(country ETF 3-month return minus URTH 3-month return). Strongly negative means real activity is weakening but the market hasn't reacted, which is a short candidate.
- **Official divergence:** D_off = z(activity score) − z(latest official YoY). Strongly negative points to a likely downside surprise in the next official release.
- **Flag on:** D < −1.5 for 3 consecutive weekly observations. **Flag off:** D > −0.5 (hysteresis).
- **Anomaly alert:** |robust z| > 2 on any adjusted series.
- These are starting thresholds. Tune them on training data only.
- *As built (S19, PLAN D46):* the market side is the robust z of the ETF's 91-day return minus URTH's; the official side is industrial production where a country has a monthly series, else its freshest official export series.

---

## 5. Hypotheses

Format: signal → target | lead time | expected sign | test type

- **H1** PortWatch export volume YoY → Comtrade official exports YoY | 1–3 months | positive | nowcast
  Data note (2026-09-30): IMF IMTS monthly trade (`imf_exports`, `imf_imports`; PLAN D42) is a faster official series for China (to May 2026, against Comtrade's December 2024). H1 stays as written; testing against IMF data would be a new, pre-registered variant.
- **H1b** PortWatch export volume YoY → IMF monthly exports YoY | 1–3 months | positive | nowcast
  Written: 2026-10-07, before any test of it. Includes CHN. Sensitivity check: exclude countries whose recent IMF months are likely estimates; record which countries and why.
- **H2** Weather-normalised total power demand (DEU, ITA, FRA, POL, ESP, NLD, GBR) → official industrial production YoY | 1–2 months | positive | nowcast
  GBR added 2026-09-30 from NESO (`neso_load`, `neso_overnight_load`; PLAN D40), built the same way as the ENTSO-E series; its official target is `oecd_ip`.
- **H2b** Weekday overnight load predicts industrial production better than total load | as H2 | overnight has the lower out-of-sample error | nowcast comparison
- **H3** Countries with a D_mkt flag → the country ETF underperforms URTH | 1–3 months | negative | trading (short ETF, long URTH)
- **H4** Chinese port import volume falling while copper prices are stable → copper price declines | 1–3 months | negative | trading
- **H5** Chokepoint transits fall sharply (robust z < −2) → moves in freight (BDRY) and Brent | 2–8 weeks | direction not assumed (two-sided) | event study
- **H6** European gas price spike plus falling EU industrial power demand → European chemicals equities underperform | 1–3 months | negative | trading

### Event-prediction hypotheses

- **E2** From the D_off level, forecast the probability that the next official industrial production or exports print is weaker than the previous one. Scored by Brier score against the base rate.
- **E3** For economic markets on Kalshi or Polymarket (for example, recession or data-release markets), compare the system's probabilities with market prices. Does the system move before the market?
- **E1** *(parked; needs ACLED and WFP data)* Trade collapses in food-importing countries precede unrest.

### Operational definitions for H3-H6 and E2 (written 2026-10-07, before any test of them)

The hypotheses above leave thresholds and instruments open. These fix them before testing (S24, S25); anything else tried is reported as a variant. Common rules: weekly test dates on Mondays 00:00 UTC (monthly on the 15th for E2) with the data as known then (PLAN D39); a trade is entered at the next close and exited at a close; costs 15 bp per side per leg and 2% a year to borrow the short leg (sensitivities 10 bp with 1%, and 20 bp with 3%); nothing on or after the vault start (2025-10-08) is used, including prices.

- **H3:** the D_mkt flag of §4 (on after 3 weekly values below −1.5, off above −0.5), computed from each week's snapshot (real-time history). Trade: short the country ETF, long URTH, equal weight across flagged countries; primary exit when the flag switches off. Variants: fixed 21- and 63-trading-day holds from the flag switching on. All 25 countries (USA trades SPY against URTH).
- **H4:** "Chinese port imports falling" = CHN `portwatch_imports` `yoy>rz(3y)` below −1.0; "copper stable" = HG=F's 91-day return between −5% and +5%. When both hold on a test date: short COPX (copper miners ETF, a borrowable one-leg short) for 63 trading days (primary). Variants: 21 days; HG=F front-month futures instead of COPX (no borrow fee; returns include roll jumps).
- **H5:** event = a chokepoint's `portwatch_transits` `yoy>rz(3y)` falling below −2 after at least 8 weekly test dates at or above −2. Outcome: BDRY and BZ=F returns over 10, 20 and 40 trading days (2, 4, 8 weeks) from the next close, against the same horizons from all other days (Welch t-test, two-sided), events pooled over chokepoints. Variant: cargo volume (`portwatch_transit_volume`) instead of transit counts.
- **H6:** "gas spike" = TTF (`prices_ttf_gas`) up more than 20% over 28 days; "EU industrial power demand falling" = the mean over DEU, FRA, ITA, ESP, NLD and POL of weather-normalised weekday overnight load YoY (`wx>yoy` on `entsoe_overnight_load`, latest value) below 0. When both hold: short EXV7.DE (in USD via EUR=X), long URTH, for 63 trading days (primary). Variants: 21 days; total load instead of overnight.
- **E2:** at each monthly test date, for each country with a D_off value: the event is "the next official print (the first month not yet published) has a lower YoY than the month before it" (IP where a country has monthly IP, else official exports, as in D_off). Probability: logistic regression of the event on D_off, pooled over countries, refitted each date on events already resolved (at least 60). Benchmark: the base rate (share of past resolved events). Scored by Brier score and calibration.

### H5-vault: H5's one look at the vault (pre-registered 2026-10-07, before any vault data was read)

Written and committed on its own before the vault is opened. Your decision of 2026-10-07; the choices you left open are marked *(my choice)* and listed in NOTES "Decisions to review". Nothing below may change after the vault run.

- **Vault:** every test date and price on or after **2025-10-08** (PLAN D49), up to the data stored when the evaluation runs (no fetch first). On 2026-10-07 the store had BDRY and Brent closes to 2026-10-06 and PortWatch chokepoints to 2026-10-04. H5 gets exactly one look, logged in `hypotheses_tested` with `uses_vault = true`; the run refuses to start if H5 already has one.
- **Event (as in `reports/H5.md`):** on weekly test dates (Mondays 00:00 UTC, data as known then, PLAN D39), a chokepoint's `portwatch_transits` `yoy>rz(3y)` falls below −2 after at least 8 weekly test dates at or above −2. Events are pooled over chokepoints. The signal is computed on every Monday from 2020-01-06, so the 8-week calm rule can look back across the vault start. **Vault events are those dated on or after 2025-10-08.** *(my choice)* The Hormuz event of 2025-10-06 (dated before the vault, and dropped from S24 because its horizons reached into the vault) is in neither sample; its returns are shown for information only and never count.
- **Outcome (as in `reports/H5.md`):** BDRY and BZ=F returns over 10, 20 and 40 trading days (2, 4 and 8 weeks), from the first close on or after the event date. An event counts at a horizon only if that horizon's return is complete in the stored prices.
- **Test method (the cluster-respecting test of the diagnostics, `reports/diagnostics/CONTROLS.md` §5; `controls.shifted_event_means` and `randomisation_p`):** the mean return after the vault events is compared with 1,000 placebo sets made by moving all the vault events' dates together by one random number of trading days, wrapping round inside the vault window (which keeps their clustering). Two-sided p = the share of placebo means at least as far from the placebos' median as the real mean. Seed 20261007. *(my choice)* The window for each horizon is the vault's trading days whose return over that horizon is complete (from 2025-10-08 to the last close minus that many trading days), the vault's equivalent of the diagnostics' window, which ended 70 days before the vault so every 40-day return was complete. The Welch t-test of `reports/H5.md` (events against all other vault days) is reported alongside it but doesn't decide anything.
- **Primary test:** BDRY, transit-count events, **20 trading days (4 weeks)**. `reports/H5.md` names no primary horizon, so 4 weeks as you specified.
- **Success:** at the primary test, the vault events' mean return is on the same side of the placebo median as in training, which is **above**: BDRY rose 18.6% on average in the 20 days after H5's 12 training events, against 0.8% on other days. And p < 0.05. Both must hold. One test, so no multiple-testing correction.
- **Not success, whatever they show:** **Brent (BZ=F), at every horizon, is reported but marked "contaminated by public knowledge of the 2026 Hormuz oil spike" and cannot count as a success.** The 10- and 40-day horizons, the cargo-volume variant (`portwatch_transit_volume`) and the Welch t-test are reported for information only.
- **No vault event**, or none with a complete 20-day return: "not testable on the vault", which is **not** a pass.
- **Independent episodes** *(my choice)*: counted events sorted by date, where an event less than 8 weeks after the previous one (any chokepoint) joins its episode. 8 weeks is the event rule's own calm period and the longest horizon, so events in one episode share outcome days. The number of episodes is reported next to the number of events.
- **Report:** `reports/H5_vault.md`; `reports/H5.md` stays unchanged as the pre-vault record.

---

## 6. Graduation criteria (from PLAN §5.6)

A hypothesis graduates only if it beats its benchmark out-of-sample in both halves of the test period, survives costs, doesn't rely on 2020, and passes the vault. It then needs at least 3 months of forward paper testing in `JOURNAL.md` before any real money.

---

## 7. Template for new hypotheses

```
- **Hn** [signal] → [target] | [lead time] | [expected sign] | [test type]
  Written: YYYY-MM-DD. Rationale: [one or two sentences on why this should work].
```
