# Signal Tracker — Master Plan (v2)

*Revised 27 September 2026. Replaces the v1 outline.*

---

## 0. How the project documents fit together

| File | Purpose |
|---|---|
| `docs/PLAN.md` | This file. Goals, architecture, data model, methods, roadmap, risks, decisions. The source of truth. |
| `docs/SIGNALS.md` | Signal formulas and pre-registered hypotheses. |
| `docs/SETUP.md` | Installation, everyday commands, scheduling and backup. |
| `docs/SOURCES.md` | Data sources, access, pitfalls. |
| `docs/NOTES.md` | Running log of API quirks and surprises found while building (not in the public copy). |
| `docs/JOURNAL.md` | Your dated, probability-weighted predictions (starts in Phase 5). |

Progress is tracked with the checkboxes in Section 7. If this plan and another document disagree, this plan wins; fix the other document.

---

## 1. Goal

Build a personal research system that:

1. collects trade, industrial and energy data for about 24 major economies every day;
2. shows on a dashboard what has changed, where, and whether it is unusual;
3. flags **divergences**: real activity weakening (or strengthening) while markets or official statistics have not yet reacted;
4. tests honestly whether those flags predict asset prices, official data releases, or events.

Primary uses: short-selling ideas and event prediction.

### Success criteria, in order of priority

1. **Reliable.** Runs daily without you. Any failure is visible within a day.
2. **Useful.** In under two minutes each morning you can answer: what changed, where, and how unusual is it?
3. **Honest.** Every hypothesis is tested out-of-sample, after costs, and the result is recorded, including failures.
4. **Predictive.** A few signals survive out-of-sample testing and a forward paper-trading period. This is the hoped-for outcome, not a milestone you can schedule. Most hypotheses will fail, and knowing which ones fail is itself valuable.

### Non-goals (for now)

Intraday or real-time data, automated trade execution, a public website, machine-learning models, paid data.

---

## 2. Guiding principles

1. **Thin slice first.** One source working end to end before adding breadth.
2. **Config over code.** A new indicator is a new entry in `indicators.yaml`. A new source is one new fetcher module. Nothing else should need to change.
3. **Never lose history.** Append-only storage, every revision kept, weekly backups. The data vintages you record from your first live run cannot be recreated later; over time they become the project's most valuable asset.
4. **Point-in-time everything.** Any analysis of the past sees only what was knowable at that moment.
5. **Small sessions, frequent commits.** One deliverable per build session.
6. **Boring technology.** Local files and Python. No servers, databases or cloud until there is a clear need.

---

## 3. Architecture

```
  External APIs                          config/*.yaml
       │                                (sources, indicators, countries)
       ▼                                        │
  tracker/fetchers ── raw copy ──▶ data/raw/<source>/<date>/
       │  normalise + validate                  │
       ▼                                        ▼
  store.append_changes() ──▶ data/lake/observations/source=<name>/<run_id>.parquet
       │                        (append-only, every revision kept)
       ▼
  store.latest(as_of=...)  ← DuckDB queries over the Parquet files
       │
       ├──▶ quality checks ──▶ data/meta/quality_issues
       ├──▶ signals (pure functions) ──▶ data/signals/
       │         │
       │         ├──▶ dashboard (Streamlit, read-only)
       │         └──▶ alerts (daily Telegram digest)
       └──▶ backtest (point-in-time, walk-forward) ──▶ reports/
```

### Components

- **Fetchers** download from one source each, save the raw response untouched, and convert it to the standard schema.
- **Store** writes only new or revised values and answers the question "what was the latest known value of X, as of date D?"
- **Quality checks** look for stale, missing or implausible data after each run.
- **Signals** are pure functions (dataframes in, dataframes out, no file or network access) that turn raw series into comparable, seasonally cleaned scores.
- **Dashboard** reads the store and signals. It never writes data; saving chart presets is its one write (D32). Its Country page can replay the data as known on a past date (D43).
- **Alerts** send one Telegram digest per day.
- **Backtests** rebuild the world as it looked on each past date and evaluate hypotheses.

### Folder layout

```
signal-tracker/
  .env                  (your keys; never committed)
  .env.example
  requirements.txt
  README.md
  tracker/              the Python package
    __main__.py         CLI: python -m tracker <command>
    settings.py         loads .env and config files
    registry.py         loads and validates indicators.yaml / countries.yaml
    store.py            lake writes, latest(), Vintages (in-memory as-of reads, D39), meta tables, signal tables, compaction
    quality.py          data quality checks
    precompute.py       STL precompute for the dashboard (`decompose`, D35)
    fetchers/           base.py + one module per source (load.py: local-day load, shared by EIA and ENTSO-E)
    signals/            transforms.py, chain.py, weather.py, composite.py, divergence.py
    alerts/             telegram.py, digest.py
    backtest/           pit.py, nowcast.py, trading.py, events.py
  dashboard/
    app.py              entry point: navigation, sidebar
    data.py             the one shared, cached data loader
    charts.py           display transforms and Plotly figures
    workbench.py        transform-chain builder for the sidebar (D35)
    presets.py          reads and saves config/presets.yaml (D32)
    views/              one script per page (not pages/: see D29)
  config/
    sources.yaml        which sources are enabled + source settings
    indicators.yaml     every indicator, defined once (Section 4.3)
    countries.yaml      country reference data (Section 4.4)
    presets.yaml        named transform chains for the dashboard (D32)
  scripts/              run_daily.ps1, register_task.ps1, backup.ps1, run_dashboard.ps1,
                        dashboard_service.ps1, register_dashboard_task.ps1 (D38)
  tests/
    fixtures/           small saved API responses
  reports/              backtest reports (committed to git)
  docs/                 PLAN, SIGNALS, SOURCES, NOTES, JOURNAL
  data/                 raw/, lake/, signals/, meta/   (git-ignored)
  logs/                 (git-ignored)
```

### Technology

Python 3.12 (3.11+ is fine); pandas, pyarrow, DuckDB; requests + tenacity (retries); pydantic, pyyaml, python-dotenv; statsmodels, scipy; holidays; Streamlit + Plotly; pytest.
Source-specific libraries, added in their own sessions: `entsoe-py`, `comtradeapicall`, `yfinance`.
Anything beyond this list needs your approval.

---

## 4. Data model

### 4.1 Observations (the Parquet "lake")

| Column | Type | Meaning |
|---|---|---|
| `date` | DATE | Start of the period the value describes. Monthly data uses the first of the month. |
| `country` | VARCHAR | ISO3 code. `WLD` = world. `CHK_<NAME>` = shipping chokepoint (e.g. `CHK_SUEZ`, `CHK_PANAMA`, `CHK_HORMUZ`, `CHK_MALACCA`). |
| `indicator` | VARCHAR | Registry id, `<source>_<metric>`, e.g. `portwatch_export_calls`. |
| `value` | DOUBLE | |
| `unit` | VARCHAR | |
| `frequency` | VARCHAR | `D`, `W`, `M` or `Q`. |
| `source` | VARCHAR | |
| `fetched_at` | TIMESTAMP (UTC) | When we downloaded it. |
| `available_at` | TIMESTAMP (UTC) | When the value became public. For live data this equals `fetched_at`. Where the source publishes true vintages (FRED's ALFRED), use those. Otherwise, estimate as period end + the indicator's `publication_lag_days`. |
| `vintage_type` | VARCHAR | `live`, `source_vintage` or `estimated`. |

**Write rule: `append_changes(df)`.** A row is written only if its (date, country, indicator) is new, or its value differs from the latest stored value. Re-downloading unchanged data writes nothing, and revisions are kept as new rows. The result is a complete revision history with no duplicates.

**Read rule: `latest(indicators, countries, start, end, as_of=None)`.** This returns the most recent vintage for each (date, country, indicator). With `as_of`, it only considers rows whose `available_at <= as_of`, which shows what was knowable on that date. The dashboard uses `as_of=None`. Backtests always pass `as_of`.

**Why Parquet files instead of a single DuckDB database.** A DuckDB database file allows only one process to write. On Windows, a dashboard holding the file open would block the daily pipeline. Immutable Parquet files, queried by DuckDB, avoid file locking entirely. A monthly compaction command merges the many small files into fewer large ones, keeping every row.

**Revision capture.** Each indicator has a `refetch_window_days` setting, for example 60 days for daily data and 730 days for official monthly data. Every run re-downloads that recent window so revisions are caught.

### 4.2 Supporting tables (Parquet under `data/meta/` and `data/signals/`)

- **`fetch_log`**: run_id, source, started_at, finished_at, status, rows_fetched, rows_written, error.
- **`quality_issues`**: checked_at, source, indicator, country, check, severity, detail.
- **`signals`**: computed_at, as_of, date, country, signal, value, components (JSON).
- **`hypotheses_tested`**: tested_at, hypothesis_id, variant, parameters, train_period, test_period, metrics, status, uses_vault, notes. Every run is logged, including failed ones (D39).

### 4.3 Indicator registry (`config/indicators.yaml`)

Each indicator has these fields:

- `id`, `source`, `name`, `description`, `unit`, `frequency`
- `category`: one of `trade`, `industry`, `energy`, `market`, `official`, `weather`
- `role`: `alt` (fast alternative signal), `official` (slow "truth" series), `market` (tradable price) or `control` (e.g. weather)
- `higher_is_stronger`: true or false (used to align signs in composites)
- `default_transform`: e.g. `yoy_28d`, `yoy_m`, `level`
- `seasonality`: `weekly`, `annual`, and/or a list of moving holidays
- `publication_lag_days`, `refetch_window_days`
- `countries`: a list, or `watchlist`
- `notes`

### 4.4 Country reference (`config/countries.yaml`)

The current watchlist becomes a set of per-country records:

- `iso3`, `name`, `un_m49` (for Comtrade), `entsoe_code` (European countries only)
- `weather_cities`: 3–5 largest cities, each with name, lat, lon and population weight
- `country_etf`, `fx_pair`, `holiday_calendar`
- `moving_holidays`: any of `lunar_new_year`, `ramadan_eid`, `easter`, `diwali`, `golden_week`

---

## 5. Methodology essentials

These are the parts that decide whether results mean anything.

### 5.1 Point-in-time discipline

Every backtest reads data through `latest(as_of=...)`. Backfilled history has estimated vintages, so results that depend on those estimates are flagged in reports. Real vintages accumulate from your first live run. Re-run key tests every quarter as that record grows.

### 5.2 Seasonality and moving holidays

- **Daily data** is noisy by weekday. Use 7-day and 28-day rolling sums, and compare each 28-day window with the same window 364 days earlier, which keeps weekdays aligned.
- **Moving holidays** shift activity between months and break naive year-on-year comparisons:
  - Lunar New Year: CHN, TWN, VNM, KOR, SGP, MYS, IDN
  - Ramadan and Eid: IDN, MYS, TUR, SAU
  - Easter: European countries, plus ZAF, AUS, CAN, BRA and MEX (Holy Week, Semana Santa, is a major holiday in Brazil and Mexico)
  - Diwali: IND
  - Golden Week: CHN (October), JPN (May)
- For Lunar New Year countries, monthly data is compared as January and February combined. China's customs agency often publishes those two months together anyway. For daily data, windows that overlap a moving holiday are flagged, shaded on charts, and down-weighted in composites.
- Holiday dates come from the `holidays` Python package. The exception is the Easter window, which is computed from the date of Easter itself: the package has no Holy Week days for MEX (not federal holidays) and only Good Friday for BRA.

### 5.3 Robust scaling

Scores are expressed as a **robust z-score**:

z = (x − rolling median) / (1.4826 × rolling MAD)

The window is 3 years, with a minimum of 24 months of history. Unlike a normal mean and standard deviation, this is not distorted by extreme periods like 2020 or the 2022 energy shock.

### 5.4 Power demand as an industry proxy

- **Weather normalisation.** Daily load is modelled on heating degree days (base 18 °C), cooling degree days (base 22 °C), day of week and holidays. The model is fitted on a rolling 3-year window. The anomaly is actual load minus expected load. Temperatures are population-weighted across each country's largest cities.
- **Rooftop solar problem.** Solar panels behind the meter reduce measured *daytime* load. That looks like falling demand when industry hasn't changed, especially in DEU, NLD, ITA, ESP and AUS. The fix is a second indicator, **weekday overnight load** (00:00–05:00 local time). It is barely affected by solar and closer to industrial baseload. Hypothesis H2b tests whether it beats total load.
- **Load-shedding.** Where power is rationed, demand measures supply rather than activity. Such countries are marked `exclude_from_signals: [power_demand]` in `countries.yaml` and get no power-demand signals. Currently this applies only to ZAF (§11 D22).

### 5.5 Composite scores and divergences

- **Country activity score:** the average of direction-adjusted robust z-scores of the alternative indicators, computed within the trade and energy categories and then averaged. The score is reported with its coverage (number of contributing indicators) and requires at least two.
- **Market divergence:** activity score z minus the z-score of the country ETF's 3-month return relative to the global benchmark (URTH). Strongly negative means real activity is weakening but the market hasn't reacted, which makes it a short candidate.
- **Official divergence:** activity score z minus the z-score of the latest official year-on-year figure. Strongly negative points to a likely downside surprise in the next official release.
- **Persistence and hysteresis:** a flag switches on only after the threshold holds for 3 consecutive weekly observations, and switches off only after recovering well past it. This prevents alerts flickering on and off.
- Exact formulas and starting thresholds are in `docs/SIGNALS.md`.

### 5.6 Backtesting protocol

1. **Pre-register.** Write the hypothesis in `SIGNALS.md` before testing it. Log every run in `hypotheses_tested`.
2. **Walk-forward.** Use an expanding window. Parameters and thresholds are chosen only from data before each test point.
3. **The vault.** The most recent 12 months are locked as a final holdout. Each hypothesis gets exactly one look at the vault, at the end.
4. **Benchmarks.** Nowcasts are compared against a naive model (AR(1) or last year's value). Trading rules are compared against buy-and-hold and against random signals with the same turnover.
5. **Nowcast metrics.** Out-of-sample RMSE ratio versus the benchmark (below 1 is good), directional hit rate, and a Diebold–Mariano test.
6. **Trading metrics.** Trades are entered at the next close after the signal becomes available. Costs are 10–20 basis points per side, plus 1–3% per year to borrow shares for shorts. Report annual return, Sharpe ratio, maximum drawdown, hit rate and turnover, plus a *deflated* Sharpe ratio that accounts for how many variants were tried.
7. **Event metrics.** Forecasts are expressed as probabilities, and evaluated by Brier score and calibration against the base rate and, where one exists, against the prediction-market price.
8. **Small samples.** Most alternative series start around 2019, which gives roughly 80 monthly observations dominated by one COVID cycle. Every test is re-run excluding 2020, and all results are treated as tentative.

**A signal "graduates" to your watchlist only if it:** beats its benchmark out-of-sample in *both* halves of the test period, survives costs, doesn't rely on 2020, and passes the vault. After that, it needs at least 3 months of forward paper testing in `JOURNAL.md` before any real money.

### 5.7 Tradability

Before relying on a trading signal, confirm you can actually trade the instrument. For example, retail investors in the UK and EU generally can't buy US-listed ETFs because of PRIIPs rules, so they'd need UCITS equivalents, futures, or CFDs/spread bets where permitted. Short positions carry borrow costs and, in principle, unlimited losses. Tickers live in config, so they can be swapped without code changes.

*This project is research. Nothing it produces is financial advice.*

---

## 6. Build schedule at a glance

| Phase | Sessions | What you'll be able to see | Milestone |
|---|---|---|---|
| 0 Setup | setup only | Tools installed, keys requested | |
| 1 Foundation | S1–S5 | FRED data flowing; `status` command | |
| 2 Core data | S6–S14 | All core sources running daily; backups | |
| 3 Dashboard | S15–S16 | Browse any country or indicator | **A: Baseline** |
| 4 Signals & alerts | S17–S21 (with S17b–c) | Chart workbench, heatmap, divergences, morning digest | **B: Daily signal** |
| 5 Research | S22–S26 | Honest test reports; graduated signals | **C: First findings** |
| 6 Advanced data | S27–S29 (optional) | Satellite and flight signals | |
| 7 Going deeper | ongoing | | |

**Ordering notes**

- ENTSO-E (S8) comes after PortWatch and Ember, so waiting for API approval doesn't block you. If the token hasn't arrived by then, skip ahead and come back.
- **Pause at Milestone A** for about a week of unattended daily runs, and fix whatever breaks before starting Phase 4. Building signals on a flaky pipeline wastes sessions.
- ★ marks sessions with subtle logic, where mistakes silently corrupt results. They get the most careful review.

There are about 28 core sessions.

---

## 7. Roadmap and progress checklist

"Your check" is a quick test *you* do to confirm the session worked.

### Phase 0 — Setup (you)
- [ ] Folder at `C:\signal-tracker`
- [ ] Python and Git installed
- [ ] `.env` created with FRED key
- [ ] ENTSO-E, EIA, Comtrade and Ember keys requested
- [ ] Backup location chosen (external drive or a cloud-synced folder); path added to `.env` as `BACKUP_DIR`
- [ ] Telegram installed on your phone (needed in Phase 4)

### Phase 1 — Foundation
- [x] **S1 Scaffold.** Your check: `python -m tracker --help` lists commands, and `git check-ignore .env` prints `.env`.
- [x] **S2 Store ★.** Your check: `pytest` passes, including the vintage tests.
- [x] **S3 Registry and country reference.** Your check: `python -m tracker status` shows indicator and country counts.
- [x] **S4 FRED fetcher.** Your check: a second fetch reports 0 new rows.
- [x] **S5 Runner and status.** Your check: `status` shows a table with last success and data freshness per source.

### Phase 2 — Core data
- [x] **S6 IMF PortWatch** (ports by country, chokepoints)
- [x] **S7 Ember** (monthly electricity)
- [x] **S8 ENTSO-E** (daily and overnight European load; needs token)
- [x] **S8b UK demand from NESO** (GB daily and overnight load from 2015, from §12; extra session, 2026-09-30, D40)
- [x] **S9 EIA** (US electricity)
- [x] **S10 UN Comtrade** (official trade)
- [x] **S10b IMF monthly trade** (IMF IMTS exports and imports for 24 countries from 2015; China to May 2026 against Comtrade's December 2024; extra session, 2026-09-30, D42)
- [x] *Extra (2026-10-07): copper and EU gas prices read from the IMF (PCPS) instead of FRED, whose copies are retired (D56)*
- [x] **S11 Official industrial production** (Eurostat + OECD)
- [x] **S12 Weather** (Open-Meteo, degree days)
- [x] **S13 Market prices** (ETFs, FX, benchmark)
- [x] **S14 Quality checks, scheduling, backup, compaction.** Your check: the task appears in Windows Task Scheduler, and the backup folder fills.

### Phase 3 — Dashboard
- [x] **S15 Dashboard shell, Data Health and Country pages.** Your check: `scripts\run_dashboard.ps1` opens your browser, and the Germany page shows power and trade charts.
- [x] **S16 Indicator and Shipping/Chokepoint pages**
- [x] **S16b Replay slider** (the Country page as known on a past date, from §12; extra session, 2026-09-30, D43). Your check: Country page, switch Replay on, move the slider back a few years: a yellow banner says what you're seeing, and charts end on that date.
- [x] **★ Milestone A:** one week of unattended daily runs, with any issues fixed. *(Checked 2026-10-07: ten scheduled runs in a row, 28 September to 7 October, all finished OK; every source succeeded every day; the only stale series are source-side. The dashboard service was found stopped; fixed separately, NOTES.)*

### Phase 4 — Signals and alerts
- [x] **S17 Transforms ★** (YoY, moving holidays, robust z)
- [x] **S17b Transform chain engine ★** (display transforms labelled one- or two-sided; chains saved as short strings)
- [x] **S17c Workbench UI** (chain builder, presets, spectrum). Your check: every preset renders on the Germany page, and a chain you save is still there after restarting the dashboard.
- [x] **S17d Known structural breaks** (config list; YoY and robust z never compare across a break; chart markers; extra session, 2026-09-30, D41). Your check: Country page, Netherlands, period All: the load charts show a dashed line on 1 January 2016 with the reason on hover.
- [x] **S18 Weather-normalised power ★** (2026-10-07, D45; 16 daily load series incl. GBR and USA; fit report in `reports/diagnostics/weather_fit.md`)
- [x] **S19 Composite scores and divergences** (2026-10-07, D46; plugged into the S22 engine as panel signals)
- [x] **S20 Overview page** (heatmap, movers, divergences; 2026-10-07, D47)
- [ ] **S21 Telegram morning digest** *(built, tested and in the daily run on 2026-10-07, D48, but not ticked: no real message sent yet, because the bot had no message from you to read the chat id from. Message the bot, run `python -m tracker digest --find-chat`, then `python -m tracker digest`.)*
- [ ] **★ Milestone B:** the digest arrives three mornings in a row.

### Phase 5 — Research
- [x] **S22 Point-in-time backtest engine ★** *(done early, 2026-09-30, before S18-S21: S19's signals plug in later, D39)*. Your check: `pytest tests/test_pit.py` passes, including the leak test.
- [x] **S23 Nowcast tests** (H1, H1b, H2, H2b; 2026-10-07, D49; vault closed. None beat its AR benchmark out of sample: `reports/H1.md`, `H1b.md`, `H2.md`, `H2b.md`)
- [x] **S24 Trading tests** (H3–H6; 2026-10-07, D50; vault closed. None profitable after costs; H5's freight reaction is suggestive but fails the multiple-testing bar: `reports/H3.md` to `H6.md`)
- [x] **S25 Event tests and journal scoring** (E2, E3; 2026-10-07, D51. E2 no skill over the base rate; E3 not testable on history; `docs/JOURNAL.md` and `journal-score`)
- [x] *Extra, exploratory diagnostics (2026-10-07, not a planned session):* engine controls (D52: the engine passed; `reports/diagnostics/CONTROLS.md`); why the nowcasts failed (D53, with the CPB trade-volume source; `reports/diagnostics/WHY_FAILED.md`); the weather model's harmonics predate every H2 result, so H2 and H2b stay confirmatory (D54); FRED's copper and EU gas prices are delayed, not discontinued (D55)
- [x] **S26 Signal Lab page and summary report** (2026-10-08, D57; vault opened for H5 only, as pre-registered: failed. `reports/SUMMARY.md`: nothing graduates; proposed decisions in NOTES)
- [ ] **★ Milestone C:** `reports/SUMMARY.md` written, and graduation decisions recorded in Section 11.
- [x] *Extra (2026-10-13): GitHub preparation: privacy audit (`docs/PRIVACY_AUDIT.md`), README as a project page (setup in `docs/SETUP.md`), public snapshot in `C:\signal-tracker-public`, push steps in `docs/PUBLISHING.md` (D58). Nothing pushed.*

### Phase 6 — Advanced data (optional)
- [ ] **S27** Satellite NO₂ over industrial zones (Google Earth Engine)
- [ ] **S28** VIIRS night lights vs reported GDP
- [ ] **S29** OpenSky flight activity

### Phase 7 — Going deeper (pick from Section 12 as needed)

---

## 8. How to run every session

**Before:** `git status` should say "nothing to commit". If not, commit or discard the leftover changes first.

**During:** one deliverable per session, with a written plan first for multi-file changes. Never delete `data/` or read `.env`.

**After**
1. Run `pytest` (the pre-commit hook runs it too), then do "Your check" from Section 7.
2. Tick the checkbox in Section 7, add any quirks to `NOTES.md` and any new decision to Section 11.
3. Commit with the session's name, e.g. `git commit -m "S4 FRED fetcher"`.

**If a session goes wrong:** `git restore .` undoes uncommitted edits to existing files; `git clean -n` lists new files and `git clean -f` deletes them. This never touches `data/`, because it's git-ignored; don't add the `-x` flag. Then retry with a narrower scope.

---

## 9. Operating rhythm (from Milestone B)

- **Daily, 2 minutes:** read the Telegram digest.
- **Weekly, 30 minutes:** review the dashboard's Overview page and any active divergences. Write 1–3 dated predictions in `docs/JOURNAL.md`.
- **Monthly:** run compaction, check that a recent backup exists, and score resolved journal predictions (Brier score).
- **Quarterly:** re-run backtests with the growing store of live vintages, and revisit hypotheses.

Journal entry format:

```
## 2026-11-02
- Claim: Germany's October industrial production (released December) will be negative year-on-year.
- Probability: 70%
- Basis: overnight load −2.3z for 5 weeks; PortWatch exports −1.1z
- Resolves: 2026-12-08 | Outcome: ___ | Brier: ___
```

A journal of dated predictions made *before* outcomes are known is the most credible evidence you'll have, both for yourself and for any future write-up.

---

## 10. Risk register

| Risk | Mitigation |
|---|---|
| ENTSO-E token delayed | ENTSO-E is scheduled after other sources; skip and return later. |
| API changes or outages | Failures are isolated per source, raw responses are saved, tests use fixtures, and `status` plus the digest show failures. |
| Rate limits (Comtrade free tier: 500 calls/day; ENTSO-E: 400 requests/minute and at most 1 year per request) | Chunked backfills spread across several days; caching. |
| Data licence terms (Open-Meteo free tier is non-commercial; ACLED has restrictions) | Personal research is fine. Check terms before any commercial use or before publishing data. |
| Look-ahead bias | `available_at`, `as_of` reads, the vault, and a leak test in S22. |
| Overfitting and multiple testing | Pre-registration, hypothesis log, deflated Sharpe ratio, vault, forward journal. |
| Laptop asleep at run time | Task Scheduler's "run as soon as possible after a missed start" setting, plus catch-up logic in the runner. |
| Data loss | Daily additive backup of `data/lake`, `data/meta` and `data/raw` at the end of each daily run (D25), shown in `status`. Live vintages are irreplaceable. |
| Leaked API keys | `.env` is git-ignored, secrets are read only through `tracker.settings`, and logged text is redacted. |
| Scope creep | The parking lot (Section 12). Reach Milestone A before adding any new source. |

---

## 11. Decision log

Add new entries with a date. Don't reopen these without a stated reason.

- **D1 (2026-09-27): Parquet files + DuckDB queries, not a single database file.** This avoids Windows single-writer locks between the dashboard and the pipeline.
- **D2: `available_at` and `vintage_type` columns.** These make point-in-time backtests possible from day one, using real vintages where they exist.
- **D3: `append_changes` stores only new or revised values.** This keeps the full revision history without duplicate rows.
- **D4: Streamlit + Plotly for the dashboard.** It's the fastest route to a good-looking local dashboard, and can be swapped later.
- **D5: Telegram instead of email for alerts.** It's simpler to set up, free and immediate. Email would need SMTP app passwords.
- **D6: Robust z-scores (median/MAD).** They resist distortion from the 2020 and 2022 outliers.
- **D7: Official wrapper libraries (`entsoe-py`, `comtradeapicall`) instead of raw API calls.** This means less custom parsing to maintain.
- **D8: Weekday overnight load as a second power indicator.** It's robust to rooftop solar.
- **D9: Local-first; no cloud before Phase 7.**
- **D10 (2026-09-27, S2): Revisions of `estimated`/`live` rows are stamped when seen.** If a re-download changes a stored value, the new row's `available_at` is raised to its `fetched_at` and its `vintage_type` becomes `live`, because we can't know it was public earlier. `source_vintage` rows (ALFRED) keep their true dates and are placed on the key's vintage timeline instead.
- **D11 (2026-09-27, S2): `available_at` is capped at `fetched_at`** on write, since nothing we downloaded can have become public after the download.
- **D12 (2026-09-27, S2): Store dtypes are microsecond resolution** (`datetime64[us]` dates, `datetime64[us, UTC]` timestamps), matching pandas 3 and DuckDB defaults. Values within 1e-9 relative of the stored value count as unchanged.
- **D13 (2026-09-27, S3): `countries.yaml` also holds each country's IANA `timezone`** (needed for local-time day boundaries in S8/S9) and a `chokepoints` list (code, name, source id). Indicators may list watchlist ISO3 codes, `WLD` or a registered chokepoint code. Source-specific fetch parameters (e.g. FRED series ids) live in `sources.yaml` and point to indicator ids; the registry checks every such reference.
- **D14 (2026-09-27, S4): ALFRED vintages count as available from the next midnight UTC after their `realtime_start` date.** FRED gives only a date, so this conservative stamp rules out intraday look-ahead. Market series without vintages use period end + `publication_lag_days`, or the fetch time when that is earlier (`vintage_type='live'`).
- **D15 (2026-09-27, S5): Resume and staleness rules.** A normal run starts at (start time of the source's last successful run − its `refetch_window_days`), never earlier than its backfill start; a source with no success yet starts at the backfill start. An indicator is stale when the period after its latest stored period should be out by now: next period end + `publication_lag_days` + grace (3 days daily/weekly, 7 monthly, 14 quarterly). `fetch` exits with code 1 if any source failed, after running all of them.
- **D16 (2026-09-27, S10): Mirror and budget-limited data.** Taiwan's trade is stored as separate `comtrade_mirror_*` indicators (never mixed with reported data). A fetcher that hits its per-run call budget logs the run as `partial`, which does not count as a success for resuming.
- **D17 (2026-09-27, S12): Quota-limited backfills fill gaps on every run.** A source with `resume_from: backfill_start` always starts at its backfill start; its fetcher skips periods already complete in the store and stops at a per-run call budget (`partial`). Open-Meteo uses this because its free quota can't cover a 2015 backfill in one day.
- **D18 (2026-09-27, S13): Per-source `change_tolerance`.** The write rule treats relative changes below the source's tolerance as unchanged (default 1e-9; prices 1e-6, because Yahoo's adjusted closes jitter at float32 precision). Prices store adjusted closes and re-download full history each run, so dividend re-basings are kept as revisions.
- **D19 (2026-09-27, tidy-up): Accept the tiny-revision rows in the prices lake.** The first prices re-run (before D18) stored 47,159 float-noise "revisions" (~1e-7 relative). They are harmless for any analysis and stay, because the lake is append-only and compaction keeps every row.
- **D20 (2026-09-27, tidy-up): EXV7.DE is the European chemicals instrument for H6** (iShares STOXX Europe 600 Chemicals UCITS ETF (DE), Xetra, EUR; indicator `prices_eu_chemicals`). It is UCITS, so buyable for UK/EU retail investors (§5.7), and has clean Yahoo data; CHM.PA looked stale.
- **D21 (2026-09-27, tidy-up): The Stooq price fallback is broken and parked.** Stooq's CSV download now requires JavaScript verification (a bot check), which we won't work around. The code stays and fails cleanly; yfinance is the only working price provider. A replacement fallback is in the parking lot (§12).
- **D22 (2026-09-27): South Africa (ZAF) joins the watchlist, which becomes 25 countries (ESP stays). ZAF is excluded from power-demand-based signals.** Eskom's load-shedding means South African electricity demand reflects supply limits rather than activity. Its power data is still fetched and stored, but `countries.yaml` marks it `exclude_from_signals: [power_demand]`. Signal code picks countries with `Registry.signal_countries()`, which leaves ZAF out of every indicator tagged `signal_group: power_demand`: Ember demand, EIA load and industrial sales, and future ENTSO-E load. So ZAF gets no weather-normalised power model and no energy category, and its composite score rests on trade. *Extended the same day:* Ember generation (total and by fuel) is just as capped by load-shedding, so it is tagged `signal_group: power_generation` and ZAF is excluded from that group too (`exclude_from_signals: [power_demand, power_generation]`).
- **D23 (2026-09-27, S14): How quality checks report.** `python -m tracker quality` appends its findings to `quality_issues`, plus one `quality_run` row per run. State checks (freshness per series with per-country lags, missing countries, gaps inside series (D27), duplicate rows, unit changes) re-check the whole store each run, and an issue is open while the latest run reports it. Jumps are events: each is reported once, by the first run that sees the new or revised value (a first-ever run checks the full history). `status` shows the latest run's issues plus fetcher-reported issues since the previous run. Values are only flagged, never dropped.
- **D24 (2026-09-27, S14): Jump rule.** Flag a period-on-period change whose robust z exceeds 10. The change is a log change for always-positive series and a plain difference otherwise and for `degC`. The scale is 1.4826 × the MAD of the series' non-zero changes, floored at 10% of its 5–95% range. Log changes must also be at least ×1.5 or ÷1.5. Series need 30+ points. On the full history this gives ~110 flags, mostly real events (typhoon port closures, COVID months) plus a few source errors (Comtrade CAN 2015-09, Taiwan's incomplete latest mirror month).
- **D25 (2026-09-27, S14): Backups are additive and daily.** `run_daily.ps1` ends with `backup.ps1`, which uses robocopy `/E` (never `/MIR` or `/PURGE`) into `<BACKUP_DIR>\signal-tracker\`. An accidental local deletion can then never propagate to the backup. The cost is that files merged by `compact` stay in the backup (restore steps in NOTES). Only new files are copied, so a daily run is cheap.
- **D26 (2026-09-27, S14): The scheduled task runs only while you are logged on.** It uses an interactive logon with limited rights, so no stored password and no administrator step. It runs daily at 07:00, as soon as possible after a missed start, only with a network connection, also on battery, one instance at a time, and is stopped after 3 hours. The action runs `powershell.exe -ExecutionPolicy Bypass` for that process only, so the machine's policy is untouched.
- **D27 (2026-09-27): What counts as missing data is set in config.** Each indicator can declare:
  - `calendar`: `all`, or `weekdays` for trading-day series and weekday-only overnight load.
  - `gap_tolerance`: the number of missing periods in a row that is normal (3 for market series, covering exchange holidays).
  - `known_gaps`: the source's own holes, which will never fill (FRED WTI 2015-01-26 to 29; Comtrade SAU 2017-10 to 12 and ZAF 2020-12).
  - `no_data_expected`: covered countries with no data by nature, such as Ember fuels a country doesn't use.

  The gap check reports missing periods between a series' first and last dates. Countries in `no_data_expected` are still fetched; if data appears for one, an `unexpected_data` info issue says to trim the list. On 2026-09-27 this left one real gap, ZAF's 2026 weather, from the chunk-planning bug fixed in 71e7a1a.
- **D28 (2026-09-27): PortWatch retries and resumes.** ArcGIS's generic "Unable to perform query" error is retried per page, with 5 attempts and 10–80 s backoff; other errors still fail at once. Progress is checkpointed after each page. A failed query is resumed by the next run with the same layer, countries and start date, on any day within 7 days. It finishes the original query, including its original end date; the 60-day refetch window covers the rest. The last saved page is downloaded again first, and the query starts over if it differs, because offsets would have shifted. Checkpoints of completed queries are kept until both layers finish, so a chokepoint failure doesn't mean re-downloading all the port pages.
- **D29 (2026-09-27, S15): Dashboard structure.** `dashboard/app.py` declares the pages with `st.navigation`; page scripts live in `dashboard/views/`, because a folder named `pages/` switches Streamlit into its legacy multipage mode, in which a page URL opened on a freshly started server runs without `app.py`. Every page reads through `dashboard/data.py`: `store.latest()` (DuckDB over the Parquet files, `as_of=None`) and the meta tables, cached with `st.cache_data` for an hour and keyed by the data folder; a sidebar button clears the cache. The server listens on localhost only (`.streamlit/config.toml`). Charts keep one y-axis: the official series is drawn over the matching alternative-data panel only in the YoY view, where both are in %. The pairs follow SIGNALS.md H1/H2 (port exports/imports with Comtrade, mirror data for TWN; power demand and EIA load/sales with official industrial production) and live in `dashboard/charts.py`.
- **D30 (2026-09-27, S15): Moving-holiday windows** (`tracker/signals/moving_holidays.py`, used for chart shading and available to S17). Lunar New Year: the country's public-holiday days from the `holidays` package, merged across gaps of up to 3 days. Ramadan and Eid: from 30 days before the first Eid al-Fitr holiday to the last one. Easter: Maundy Thursday to Easter Monday, computed from Easter itself. Diwali: the public holiday ±2 days. Golden Week: CHN 1–7 October, JPN 29 April–5 May. Monthly charts shade the whole month a window falls in.
- **D31 (2026-09-27, S16): Rankings and seasonal bands on the dashboard.** The Indicator page ranks each country's *latest available* YoY and shows its date, rather than comparing all countries on one common date, because sources publish with different lags (Comtrade, the OECD). Seasonal band charts (chokepoint transits, port activity) smooth the daily series with a trailing 7-day average, put every year on one calendar (29 February dropped), and draw the chosen year against the lowest-to-highest range and average of the 5 years before it (fewer if the series starts later; PortWatch starts in 2019).
- **D32 (2026-09-27): Display transforms are chains of labelled steps (S17b–c).**
  - `tracker/signals/chain.py` holds a registry of display transforms: pure functions with typed parameters, each labelled **one-sided** (uses only past data) or **two-sided** (uses future data, so its last values revise as data arrives). A test sets the label: a step counts as one-sided only if appending data never changes its earlier output.
  - Steps that overlap S17's `transforms.py` (YoY, moving holidays, STL, robust z) wrap those functions, so each calculation has one implementation. The maths uses only statsmodels and scipy; scipy, already installed with statsmodels, becomes a direct dependency (§3).
  - A chain is an ordered list of steps that serialises to a short string. `config/presets.yaml` stores these strings, and a SIGNALS.md hypothesis quotes one to define its transform. Incompatible combinations are rejected when a chain is built.
  - Two-sided values at a past date are not what was knowable then. The dashboard draws their provisional end dashed and has a "real-time only" switch that hides two-sided steps. Backtests (S22) recompute a chain from `store.latest(as_of=...)` at each test date and use only its value at that date.
  - The dashboard still never writes data. Saving presets to `config/presets.yaml` is its one write.
- **D33 (2026-09-27, S17): Transform conventions** (`tracker/signals/transforms.py`).
  - Daily YoY compares 28-day windows 364 days apart. A complete window is compared by its sum; a window with a few days missing (at least 75% present) is compared by its average, which equals the sum ratio when both windows are complete. Fewer days than that gives no value.
  - Monthly YoY for Lunar New Year countries puts the YoY of the January-February average on February and leaves January empty, because January can't be judged before February is known.
  - Daily moving-holiday handling: a date is flagged when its 28-day window or its comparison window a year earlier overlaps a holiday window. The level adjustment replaces holiday days with a straight line between the nearest ordinary days *of the same weekday*, which keeps the weekly pattern.
  - STL needs 3 years of data. Monthly series use a robust STL; daily series use MSTL (weekly and annual cycles; 5 and 261 for weekday-only series) without the robust option, which would be several times slower.
  - Robust z: a trailing window of 36 months, or 1,095 calendar days for daily series, including the current value; a score needs 24 months since the series' first value; a zero MAD gives no score.
- **D34 (2026-09-27, S17b): Chain steps, labels and provisional ends** (`tracker/signals/chain.py`).
  - 22 steps: `sma`, `csma`, `ema`; `yoy`, `stl`, `hol`, `wx`; `d1`, `dlog`, `lin`, `hp`, `ham`; `bk`, `cf`, `bwz`, `bwc`; `level`, `log`, `idx`, `rz`, `pr` (trailing window), `prf` (full sample). Parameters left out, or written `auto`, are chosen from the series' frequency.
  - The append test gives these two-sided labels: `csma`, `stl`, `hol`, `lin`, `hp`, `ham`, `bk`, `cf`, `bwz`, `prf`. `hol` is two-sided because a daily holiday is interpolated towards the next ordinary day of the same weekday, and a monthly January waits for February. `idx` is one-sided once its base date is in the data. `wx` is labelled two-sided until S18 runs the append test on it. *(S18: the append test labels it one-sided, D45.)*
  - Provisional end points: `csma` half its window, `stl` the last year, `hol` an open holiday window, `bk` its k (no values there yet), `hp` 1.2 × λ^¼ periods, `cf` 0.75 × the longest cycle, `bwz` 0.5 × the longest cycle. These are the points that typically move by more than 20% of the filtered series' spread when data is added (random-walk simulations). Full-sample fits (`lin`, `ham`, `prf`) revise the whole history slightly and have no dashed end.
  - Lengths can't be finer than the series' frequency (no days or weeks on monthly data); months and years on daily data are converted (1m = 30.44 days). Band-passes need at least one full longest cycle of data, so the 1.5-8 year band doesn't run on PortWatch (from 2019).
  - `wx` calls `tracker.signals.weather.adjust(values, context)` once that exists (S18) and refuses with a message until then. Degree days reach it through `context.extra`. It only accepts indicators with `signal_group: power_demand`.
- **D35 (2026-09-27, S17c): The workbench.**
  - The Country and Indicator pages get a third view, **Chain**, next to Level and YoY; the sidebar chain builder appears in that view. Level and YoY (with the official overlays, D29) are unchanged.
  - The chain is kept in session state as its string and shared by both pages. The real-time switch and the spectrum view are sidebar toggles.
  - Each panel applies the chain to its own series. Where the chain doesn't fit (days on monthly data, too little history for STL or a band-pass), the panel shows the raw series, its subtitle says "chain not applied" and an expander lists the reasons. *Days on monthly data: changed by D37 (converted instead).*
  - A panel draws the raw series faintly behind the result when every step keeps the units, and dashes the provisional end. What each step removed goes in a strip below with its own axis (not a second y-axis). On the Indicator page, daily panels leave out the raw series.
  - Default presets: Raw `level`, Smoothed 28d `sma(28d)`, Seasonally adjusted `hol>stl`, Business-cycle band `cf(1.5y,8y)`, and Short cycle (2-18 months) `cf(2m,18m)` for series that start in 2019. Defaults can't be overwritten or deleted.
  - STL is precomputed. `python -m tracker decompose`, a `run_daily.ps1` step after quality, fits every series with 3+ years of data, raw and holiday-adjusted, and writes the seasonal parts to `data/signals/stl.parquet` (8 processes, ~1 minute). The `stl` step uses a precomputed fit only if it was made on exactly the series it receives; otherwise it fits live.
- **D36 (2026-09-29, S8): ENTSO-E load.**
  - Six countries: DEU, FRA, ITA, ESP, NLD, POL, each by its country area code (`DE` is Germany alone from 2015; the bidding zones `DE_AT_LU` and `DE_LU` include Austria or Luxembourg). sources.yaml `areas` can override a code or piece a country together from `{area, from, to}` periods if a published area changes.
  - GBR and TUR are not fetched. GB's load on the platform ends on 2021-06-14 (after Brexit), and TR has none. UK demand from NESO is in the parking lot (§12). *GBR now comes from NESO (D40).*
  - Requests go through entsoe-py's raw client, one per calendar year (the API's limit), 1 s apart, with entsoe-py's parser (D7). Its pandas client would split load queries into months.
  - Values are weighted by the time they cover (15, 30 or 60 minutes; most TSOs moved to 15 minutes in 2024-25), in the shared `tracker/fetchers/load.py`, which EIA now uses too. A day that isn't fully covered isn't stored and is reported as a `missing_hours` quality issue. Its night is still stored when 00:00-05:00 is complete.
  - Raw XML is saved gzipped (a 15-minute year is ~4.4 MB, ~0.2 MB gzipped).
- **D37 (2026-09-29): The dashboard converts chain lengths to each series' frequency.**
  - Before applying a chain to a panel, the dashboard rewrites every length whose unit isn't the series' own unit to the nearest whole number of that unit, at least one: 28d → 1m and 2w → 1m on monthly data, 90d → 3m, 1q → 3m; 1m → 30d, 3m → 91d and 4w → 28d on daily data; 28d → 4w on weekly data. Years are kept at every frequency. So one preset string works on every panel: "Smoothed 28d" runs as `sma(1m)` on monthly panels instead of leaving them raw.
  - Each converted panel's subtitle says "lengths converted", and an expander under the charts lists each change with the charts it applied to, e.g. `` `sma(28d)` → `sma(1m)` (monthly data) ``. Charts name the steps as they ran (`sma(1m)`).
  - The engine stays strict: `chain.apply` converts only when called with `convert=True` (`chain.adapt`), which only the dashboard does. Backtests and SIGNALS.md hypotheses (D32) use chains exactly as written, so a hypothesis can't silently mean different windows on different series.
  - "Chain not applied" remains for chains that can't run at all (too little history for STL or a band-pass, a band too short for the frequency).
- **D38 (2026-09-29): The dashboard runs as a service on loopback only; private phone access goes through a VPN proxy.**
  - The dashboard runs all the time: the scheduled task "SignalTracker Dashboard" starts `scripts/dashboard_service.ps1` at logon in a hidden window. It runs only while you are logged on, with limited rights and no stored password (like D26), has no time limit, and Task Scheduler restarts it every minute if it fails *(only if it fails to start; a 5-minute watchdog trigger and a daily check were added in D44)*. The script starts Streamlit and restarts it whenever it exits (after 10 s, backing off to 5 minutes if it keeps failing), logging to `logs/dashboard.log`.
  - Streamlit listens on 127.0.0.1:8551 only. Streamlit's default 8501 is left free for other apps. The service never stops another program; it only clears a Streamlit left over from its own earlier run.
  - A private mesh VPN's HTTPS proxy, in the background, forwards port 8443 to `http://127.0.0.1:8551`. It is reachable only from the owner's devices on that VPN; nothing is public. (The setup details are left out of the public copy.)
  - No Streamlit setting was loosened. The proxy passes the original Host header and websockets, so Streamlit's default origin check (CORS) and XSRF protection stay on. Websockets from the dashboard's own origin connect, and other origins get HTTP 403.
  - This covers the "phone access" half of the §12 cloud-server idea without a server. D9 (local-first) is unchanged: data and compute stay on this PC.
- **D39 (2026-09-30, S22, done early): The point-in-time backtest engine** (`tracker/backtest/pit.py`). *Accepted 2026-10-07: the project's read rule now allows backtests to read through `store.Vintages.latest(as_of=...)` as long as the leak test passes.*
  - Backtests read through `store.Vintages`: every stored vintage of the series they need, read once, whose `latest(as_of=...)` applies the store's read rule in memory (`available_at <= as_of`; the newest `available_at`, then `fetched_at`, wins). Tests check it returns exactly what `store.latest(as_of=...)` returns on every date, on synthetic lakes with revisions and ALFRED-style vintages; on the real lake too. Reason for not calling `store.latest()` per date (the read rule, PLAN §4.1): a backtest asks for hundreds of dates, one DuckDB query re-reads the Parquet files each time (0.1 s against 0.005 s per snapshot on the real lake), and backtest functions must stay pure, so the caller reads the lake once.
  - On each test date the engine takes the snapshot and recomputes every signal from it alone, keeping only its end value (D32). Chains run exactly as written (D37) and never get the daily run's precomputed STL (fitted on today's data). `wx` gets degree days from the same snapshot. A series whose data hasn't changed since an earlier test date reuses that result (keyed by a fingerprint of its dates and values, so a revision is always recomputed). Holiday windows are memoised: they come from fixed calendars known in advance.
  - Each result row records the end value's date, the latest period known, whether a two-sided step would still revise it, the `vintage_type` of the latest raw value (so results resting on `estimated` backfill vintages can be flagged, §5.1) and why a chain couldn't run (e.g. STL before 3 years of data).
  - Models train on `real_time_history`: for each period, the end value recorded on the last test date while that period was the end. Targets come from the snapshot at the test date, so a label published later doesn't exist yet.
  - **S19 hook.** Composite scores and divergences are *panel signals*: functions `(snapshot, as_of, registry) -> rows (date, country, value[, components])` that see every indicator and country as known on the test date; the engine keeps each country's end value. `composite.py` and `divergence.py` should expose them in a `PANEL_SIGNALS` dict, which `pit.available_panel_signals()` picks up, and the `signals` command should call the same functions with the latest snapshot, so live and backtest signals share one implementation.
  - **Walk-forward and the vault.** `walk_forward` makes one fold per test date: an expanding (or rolling) training window of earlier test dates, a minimum length and an optional embargo. Test dates on or after `vault_start(last_date)` (the most recent 12 months) are neither tested nor trained on unless `include_vault=True`, the hypothesis's one look. Fix the vault start once per hypothesis and pass it to every run, so the vault doesn't slide as data arrives. `split_halves` gives the two halves the graduation rule needs.
  - **`hypotheses_tested`** lives in `data/meta/` with the §4.2 columns plus `status` (ok or failed, so failed runs stay visible) and `uses_vault` (so `vault_looks()` can enforce the single look). `parameters` and `metrics` are JSON; periods are `first/last` dates.
  - **The leak test** replaces everything published after the test date with huge random values, adds a poisoned revision of every known value one second later and poisoned future periods, then requires identical results at three test dates: the snapshot, 11 chain signals (one- and two-sided, daily and monthly, alternative and official series), a stand-in panel signal, real-time histories and a walk-forward nowcast. It was checked against a deliberately leaky engine (filtering by date instead of by publication time), which fails it.
- **D40 (2026-09-30, S8b): UK demand from NESO** (`tracker/fetchers/neso.py`).
  - Source: the NESO data portal (CKAN API at api.neso.energy, no account or key, NESO Open Data Licence). History comes from the yearly **Historic Demand Data** CSVs (settled, updated daily ~21 days in arrears); the days after the yearly file ends come from the **Demand Data Update** CSV (actuals up to the current half hour; its forecast rows are dropped). On the 40 days both covered, every value was identical; where both cover a day the yearly file wins, and later corrections arrive as revisions.
  - Measure: national demand (`ND`) plus NESO's estimates of embedded wind and solar generation, i.e. demand including distribution-connected generation. That matches ENTSO-E's Actual Total Load definition (all generation, minus exports, plus imports, minus storage) better than `ND` alone, which falls as rooftop solar grows. It covers Great Britain only: Northern Ireland (SONI, a separate system) is not in it. Stored as country `GBR`.
  - Built like the ENTSO-E series (D36): settlement periods are converted to UTC starts from local midnight (46 periods on the spring clock-change day, 50 in autumn), then `tracker/fetchers/load.py` makes local-day totals (`neso_load`, MWh) and weekday 00:00-05:00 means (`neso_overnight_load`, MW); incomplete days aren't stored.
  - Refetch window 365 days (NESO corrects embedded solar months later); publication lag 1 day. The portal allows 1 API request a second and 2 datastore requests a minute; CSV downloads are spaced 31 s apart to be safe, so a daily run takes ~1-2 minutes (2 metadata requests, 1-2 yearly files and the update file). The 2015 backfill is 12 yearly files plus the update file, ~6 minutes.
  - Both indicators are in `signal_group: power_demand`, and GBR joins H2 (SIGNALS.md) with `oecd_ip` as its target: the construction (local days, weekday overnight window, total load including embedded generation) is the ENTSO-E one, and H2 compares each country's load with its own IP, so levels needn't match across sources.
- **D41 (2026-09-30, S17d): Known structural breaks.**
  - A break is a date from which a series is measured differently (a definition or reporting change), so values before and after aren't comparable. Breaks are listed per indicator in `indicators.yaml` as `breaks: {country: [{date, reason}]}`, next to `known_gaps` (D27); the registry checks each country is covered and each break has a reason. First entry: NLD's `entsoe_load` and `entsoe_overnight_load` on 2016-01-01 (TenneT's reporting change, +17% and +23%).
  - **YoY** gives no value where the data it compares doesn't lie between the same two breaks: daily, from the start of the comparison window a year earlier (date − 391 days) to the date; monthly and quarterly, from the start of the period a year earlier to the end of the current period; Lunar New Year Februaries from the January a year earlier. So a break on 1 January 2016 blanks 2016's YoY and nothing else; a break inside a month also blanks that month's comparisons.
  - **Robust z** scores each stretch between breaks on its own, so no window mixes both sides and the 24-month history rule starts again after a break.
  - This lives in `transforms.py` (`segment_ids`, `comparable`) and reaches every user through `chain.Context.breaks`, which `context_for` fills from the registry: the dashboard's chains, the backtest engine (D39) and S19's signals. The dashboard's simple YoY view and Indicator ranking (D29, D31) apply the same rule. Other steps (percentile ranks, differences, filters) don't split at breaks; a chain that needs that should start with `yoy`.
  - Charts draw each break in the displayed range as a dashed vertical line with "Break <date>: <reason>" on hover (a line trace, since Plotly shapes have no hover text), on the Level, YoY and Chain views of the Country and Indicator pages.
  - Breaks are treated as known from the start, also in backtests: they are definition changes, normally announced, and a break only matters once data after it exists.
- **D42 (2026-09-30, S10b): IMF monthly trade (IMTS) as an additional official series** (`tracker/fetchers/imf_trade.py`, source `imf`).
  - The IMF's new data portal serves SDMX 2.1 and 3.0 at api.imf.org/external/sdmx; structure and data queries worked with no key, account or header (checked 2026-09-30; the portal's sign-in is only for its Swagger explorer). No rate limit is documented; the fetcher makes one request per run.
  - Dataset: IMTS (International Trade in Goods by partner country, the former Direction of Trade Statistics), monthly, US dollars, partner = world (`G001`). `imf_exports` = exports FOB (`XG_FOB_USD`), `imf_imports` = imports CIF (`MG_CIF_USD`, the only import measure every country has). Every watchlist country but Taiwan is covered, from 2015.
  - **Faster than Comtrade where H1 needs it:** on 2026-09-30 every country ran to May 2026, while Comtrade's China stops at December 2024 (FRA, KOR and TUR at December 2025). For most other countries Comtrade is a month or two ahead, so Comtrade stays the primary target and IMF is a second official series; the dashboard's H1 overlay uses Comtrade first, then IMF, then the Taiwan mirror.
  - **Caveat:** every IMTS series is flagged "mixed" (reported data combined with IMF staff calculations or estimates) with no per-month flag, so recent months of slow reporters (e.g. Vietnam, which reports nothing monthly to Comtrade) are probably IMF estimates, and China's combined January-February figures are split into months by the IMF. Where both exist, IMTS matches Comtrade closely (median ratio 1.00 for most countries; NLD exports +22%, CAN and MEX imports +6%).
  - Publication lag 122 days: on 2026-09-30 May was out and June wasn't, so the true lag is between 92 and 122 days; the conservative end avoids look-ahead in backfilled `available_at` until live vintages show the real release dates. Refetch window 730 days (estimates are replaced as reports arrive).
  - H1 itself is unchanged (pre-registered against Comtrade). Testing it against IMF data would be a new, separately registered variant.
- **D43 (2026-09-30, S16b): Replay on the Country page.**
  - A "Replay" switch under the page's controls; when it's on, an "As known on" date slider (from the country's first data date to today, starting a year back) sets the day. The page then reads `store.latest(countries=[country], as_of=end of that day, UTC)`, cached per country and day (up to 64 entries), so values published later and later revisions of earlier values are hidden. This is the dashboard's only `as_of` read; everything else stays `as_of=None` (§4.1).
  - Transforms are one-sided only. The Level view (daily values with a trailing 28-day average) and the YoY view (a trailing window against a year earlier, with official overlays from the same as-of data) use only past data already. The Chain view keeps only the chain's one-sided steps: the sidebar's "Real-time only" switch is shown on and locked while replay is on (without changing the user's own setting), and the caption lists the hidden steps. The precomputed STL is only used by a two-sided step, so it never reaches a replay. *(Weather adjustment `wx` turned out one-sided in S18, D45, so replay runs it, with degree days as known on the replay day.)*
  - It is labelled: a yellow banner ("Replay: <country> as known at the end of <day> (UTC)") says what is hidden, that most history before September 2026 rests on estimated publication dates, and that revisions made before the first download can't be replayed; chart subtitles add "as known <day>"; the x-axis ends on the replay day. A day before anything was published says so.
  - Holiday shading and known breaks (D41) are calendar and config facts, shown as usual.
- **D44 (2026-10-07): Dashboard watchdog.** The dashboard task stopped on 2026-09-30 with 0xC000013A (its console got a Ctrl+C or close event) and stayed down for a week: Task Scheduler's restart-on-failure only covers a task that fails to *start*, not a running task whose process is killed. Two checks now bring it back:
  - **Task level:** "SignalTracker Dashboard" has a second trigger that fires every 5 minutes with no end date. With "one instance at a time" (IgnoreNew), a start while the service runs does nothing, so the trigger only restarts a killed service. Restart-on-failure stays on for launch failures. `register_dashboard_task.ps1 -Stop` now also disables the task, so a deliberate stop holds.
  - **Daily run:** `run_daily.ps1` begins with `scripts\check_dashboard.ps1`: if nothing listens on 8551 it starts the task (stopping it first if Task Scheduler still shows it running) and waits up to 60 s. Results: up, restarted, off (task disabled; left alone) or DOWN (a failed step). The digest reports it (S21).
- **D45 (2026-10-07, S18): Weather-normalised power demand** (`tracker/signals/weather.py`, chain step `wx`).
  - **Model:** daily load ~ HDD18 + CDD22 + day of week (Tuesday-Friday only for weekday-overnight series) + weekday public holidays (the country's national `holidays` calendar) + a 24 December-1 January dummy + two annual harmonics. Refitted at the start of each month on the 3 years before it (at least a full year, 80% of its days present), with Huber weights (D6), and applied to that month. No fit window crosses a known break (D41).
  - **The annual harmonics are an addition to §5.4's formula.** Without them the degree-day coefficients also absorb seasonal effects that aren't weather (daylight, summer holidays, rooftop solar), and normalisation then leaves weather in the YoY: tested on load data alone (never on the IP targets), the median share of normalised YoY variance explained by the YoY change in degree days fell from 9.9% to 3.1% (USA 39% to 0.4%, GBR 13% to 1.5%) and the out-of-sample daily error from 4.4% to 4.1%. Variants tried: §5.4 as written, + lagged HDD, + 2 harmonics, + 2 harmonics and lagged HDD, + 3 harmonics; 2 harmonics without the lag was chosen. *(Checked 2026-10-07: chosen before any H2 result existed, D54.)*
  - **Outputs:** expected load and anomaly (actual − expected); weather effect = b_hdd·(HDD − normal) + b_cdd·(CDD − normal), with "normal" the fit window's mean degree days within ±15 days of the day of year; **normalised demand** = actual − weather effect (demand in normal weather). The signal of SIGNALS.md §2 is `wx>yoy>rz(3y)`.
  - **`wx` is one-sided** (the append test, tests/test_weather.py): a day's value uses only data up to its month's start plus that day's own load and degree days. Its values are final when published, so backtests need no provisional flag and the replay view runs it with degree days as known then. It needs daily load (monthly power series such as Ember demand are refused) and a positive input (so `yoy>wx` is refused).
  - Degree days start in 2019 for these countries, so outputs start in January 2020 and normalised YoY in January 2021. As the Open-Meteo backfill reaches 2015 (about 9 more daily runs from 2026-10-07), rerunning gives five more years.
  - `python -m tracker weather` writes the daily outputs to `data/signals/weather.parquet` and the fit report to `reports/diagnostics/` (a table and one chart page per country). It is not part of the daily run; S19's `signals` command computes what it needs through the same functions.
- **D46 (2026-10-07, S19): Composites, divergences and flags** (`tracker/signals/composite.py`, `divergence.py`; `python -m tracker signals`, a `run_daily.ps1` step).
  - **Weekly grid ending on the snapshot day.** Every signal is computed on dates 7 days apart ending on the day of the data snapshot, so the newest value always reflects the newest data and "this week's change" is against 7 days earlier. At each grid date a component uses its latest value on or before that date if it is recent enough (45 days for daily series, 130 days after the period's end for monthly). Monthly values count from the end of their month.
  - **Components** (SIGNALS.md §3): trade = PortWatch exports and imports, `yoy>rz(3y)`; energy = weather-normalised total and overnight load (ENTSO-E, NESO, EIA), `wx>yoy>rz(3y)` (D45), and Ember demand, `yoy>rz(3y)`. Daily components whose YoY window overlaps a moving holiday get half weight. Category score = weighted mean; country score = mean of the categories present; at least 2 components. Countries come from `Registry.signal_countries()` (ZAF: trade only).
  - **z(activity score)** = the robust z of the weekly score over its trailing 3 years (at least 2 years of scores, §5.3). With PortWatch from 2019, component z-scores start in 2022 and score z-scores in January 2024, so divergences exist from 2024 only.
  - **Market side:** robust z (3 years of trading days) of the country ETF's 91-day return minus URTH's. **Official side:** robust z of the latest official YoY: monthly industrial production where a country has it (Eurostat, OECD, FRED; first fresh one), otherwise the freshest official export series (Comtrade, IMF IMTS, Taiwan's mirror). Taiwan's mirror months count only once 80% of the usual reporters have filed. The components JSON names the series used.
  - **Flags:** D < −1.5 for 3 consecutive weekly values switches a flag on; D > −0.5 switches it off; a missing week breaks a run but leaves an active flag on. Live flags are computed on the weekly history as known today (revised); backtests apply the same rule to real-time histories.
  - **One implementation for live and backtest (D39):** `PANEL_SIGNALS` in both modules (activity, activity_trade, activity_energy, activity_z; d_mkt, d_off, market_z, official_z) wrap the same functions the `signals` command calls. Component scores are memoised by their exact input, and a snapshot's frame is computed once for all panel signals. About 12 s per snapshot on the real store.
  - Tables (replaced on each run): `data/signals/signals.parquet` (§4.2 columns; weekly history of every signal plus each component's z as `z:<indicator>`, used for anomaly alerts, |z| > 2) and `data/signals/flags.parquet` (flag state and on/off events per country and divergence).
- **D47 (2026-10-07, S20): The Overview page is the dashboard's landing page** (`dashboard/views/overview.py`).
  - It reads only the `signals` step's tables (`data/signals/signals.parquet`, `flags.parquet`, D46) through `tracker/daily_signals.py`'s summary functions (`latest_scores`, `movers`, `active_flags`, `anomalies`), which the Telegram digest uses too, so page, digest and command show the same numbers. Before the first `signals` run it says so.
  - Contents: counts (countries scored, market and official flags, anomalies); a country × column heatmap of the latest week (Trade, Energy, Score, Score z; robust z units, diverging red-blue, clipped at ±3, weakest first; hover shows coverage and each component's z, holiday weight and date); the 5 largest score changes against 7 days earlier; one card per active flag (value, weeks on, the two sides) linking to the Country page; anomalies (|z| > 2) in an expander.
  - The Country page's address is now `/country?country=XXX`. An old `/?country=XXX` link opened on a session's first run goes on to the Country page.
  - On the real store it loads in ~1.9 s from cold and reruns in ~0.1 s (AppTest, 2026-10-07).
- **D48 (2026-10-07, S21): The Telegram digest** (`tracker/alerts/`: `telegram.py` client, `digest.py` message, `run.py` I/O; `python -m tracker digest`).
  - **Always one message a day**, the last step of `run_daily.ps1` (after backup): a two-line **all quiet** message when nothing is new; otherwise the news (data failures, newly stale or recovered series, divergence flags switching on or off, new anomalies with |z| > 2, the top 3 active divergences with a one-line reason each). If any step failed, the message opens with "daily run FAILED at: <steps>". If the digest itself can't be built, a short failure message naming the steps goes instead. Every message states the dashboard check's result (up, restarted, off, DOWN). A flag that stays on is not news; the Overview page shows all of them.
  - **"New" means new since the last digest sent**: each sent digest's state (active flags, anomalies, stale series) is saved in `data/meta/digests`. The first digest counts last week's flag events as new.
  - **Plain text** (no Markdown or HTML parse mode, so nothing needs escaping), at most 4,096 characters, each list of crossings capped at 6 lines. The bot token is in every request URL, so the client never puts a URL or a `requests` error message into an error or log line (errors carry the method, HTTP status and Telegram's description). Connection errors, HTTP 5xx and 429 are retried 3 times; other 4xx aren't.
  - **Not set up = not a failure.** Without `TELEGRAM_CHAT_ID` the digest is written to the run's log and the step succeeds, so Task Scheduler still shows real failures. A send that fails for any other reason fails the step.
  - **Chat id:** `python -m tracker digest --find-chat` reads the bot's pending messages (Telegram keeps them 24 hours) and, if exactly one private chat wrote, appends its id to `.env` (never reading the file) and sends a test message. It refuses to guess between several chats.
- **D49 (2026-10-07, S23): The nowcast protocol** (`tracker/backtest/nowcast.py`, `tracker/research.py`; `python -m tracker nowcast H1 H1b H2 H2b`).
  - **Vault fixed at 2025-10-08** (12 months before the first test run, `pit.vault_start`) for every hypothesis, now and in later reruns. Test dates end before it, and forecasts are scored on the official values as known that day, so nothing published on or after it is read.
  - **Monthly test dates on the 15th** from January 2020. Each country-month is forecast once: on the first test date its signal month is complete (90% of days) and its official value isn't out yet. That is the earliest, longest-lead nowcast.
  - **Signal:** daily alternative data averaged per calendar month (weather-normalised first for load), monthly YoY with the same rules as the official target (Lunar New Year, breaks). Training uses each month's signal as first computed (real-time).
  - **Models:** primary `arx`, y = a + b·signal + c·(last published y), per country, direct h-step; benchmark `ar` = the same without the signal (§5.6's AR(1), estimated directly at the horizon). Also reported: signal alone, pooled across countries, and the last published value. Expanding window, at least 24 months. Primary metric: RMSE ratio `arx`/`ar`, with both halves, ex-2020, directional hit rate (forecasts that call a direction), and a Diebold-Mariano test on squared errors averaged over countries per test date (Newey-West, 3 lags).
  - Every variant is logged to `hypotheses_tested`, with a run version in its notes (v1-v3: two bugs found and fixed; NOTES).
- **D50 (2026-10-07, S24): The trading and event-study protocol** (`tracker/backtest/trading.py`, `tracker/research_trading.py`; `python -m tracker trading H3 H4 H5 H6`).
  - **Rules fixed before testing:** SIGNALS.md's "Operational definitions" (written 2026-10-07, before any S24/S25 run) give each hypothesis its thresholds, instruments and holding periods. Nothing is fitted, so the whole window counts as out of sample; the window starts when the signal first exists and ends before the vault (2025-10-08), prices included.
  - **Execution:** weekly snapshots on Mondays 00:00 UTC (D39); a position changes at that day's close and earns from the next day. Costs 15 bp per side per leg and 2% a year borrow on the short leg (sensitivities 10 bp/1% and 20 bp/3%). Several trades at once (H3's countries) are equally weighted. Prices are adjusted closes as known at the vault start; EXV7.DE is converted to USD with EUR=X.
  - **Reported:** annual return, Sharpe (252 days), maximum drawdown, hit rate per trade, time in the market, turnover; the same trade held all the time ("always on"); random entries with the same number and lengths of trades (share doing at least as well, 300 draws); the Sharpe in each half; and the deflated Sharpe ratio (Bailey and López de Prado) over the hypothesis's variants. H5 is an event study: returns after events against all other days (Welch t-test), with a Bonferroni bar for the number of tests.
  - H3's weekly real-time D_mkt and D_off values are stored in `data/signals/pit_divergences.parquet` for E2.
- **D51 (2026-10-07, S25): Event tests and the journal.**
  - **E2** (`tracker/backtest/events.py`, `python -m tracker events`): monthly test dates; D_off as known that week; event = the next official print's YoY below the last one's (in the series D_off used); logistic regression on D_off, pooled, refitted on events resolved by each date (at least 60) against the base rate; Brier score, skill score, calibration bins, halves.
  - **E3** can't be tested on history: no prediction market priced these prints before the vault (Kalshi's matching markets start in 2026 with no volume; Polymarket has none). `reports/E3.md` describes the forward test.
  - **Journal:** `docs/JOURNAL.md` (template from §9, no predictions: they are yours) and `python -m tracker journal-score [--write]`, which scores entries with an outcome by Brier score and can fill in their blank `Brier: ___` fields.
- **D52 (2026-10-07): Engine controls, and exploratory runs in `hypotheses_tested`.** `python -m tracker diagnose controls` (`tracker/backtest/controls.py`, `tracker/research_controls.py`) checks the backtests on the real store with the vault closed; results in `reports/diagnostics/CONTROLS.md`. **The engine's code passed every check**, so S23-S25's results stand and nothing was rerun.
  - **Positive controls**, through the unchanged nowcast engine. Comtrade exports nowcasting IMF exports for the same country (13 countries where Comtrade comes out first): RMSE ratio 0.33, both halves below 0.5. EIA industrial power sales → INDPRO is not a usable control: EIA publishes ~40 days after INDPRO, so no honest nowcast exists, and with INDPRO artificially delayed the signal makes forecasts worse (2.13), because outside 2020-21 the two don't move together even in sample (signal t = 0.8 from 2022). A synthetic signal (the target's final values plus noise of known size, published at month end, in each hypothesis's own sample) measures each design's power: YoY noise of 1.4, 4.2 and 14 pp gives ratios of 0.12, 0.36 and 0.81 in H1's setting and 0.32, 0.76 and 1.19 in H2's.
  - **Placebos:** H1's and H2's own signals shifted in time (8 random shifts of 14-34 months, wrapping round so the sample keeps its length) and taken from another country (5 random draws). H1: none beat the benchmark (ratios 1.03-1.16; the real signal's 1.11 sits among them). H2: wrong-country signals 1.60-1.95, shifts 0.93-1.23; one shift (15 months, 0.93, DM p 0.02) beat the benchmark significantly. No bug behind it: its timing audit is clean, and its gain comes from Germany and Spain in 2023-24 against a benchmark weakened by training on the 2021 rebound (NOTES). Pure-noise signals published at month end, white or persistent, never beat the benchmark significantly (0 of 100 draws in the two settings, `CONTROLS.md` §2), so this placebo is unusual, but it comes from the data, not the code.
  - **Replication and timing:** `replicate_nowcast` (separate code, statsmodels) reproduces every primary and benchmark forecast for the countries without Lunar New Year or break rules (H1 742, H2 183) to 1e-13; `audit_timing` checks from the raw publication times that no forecast used a signal month not yet out or forecast a target month already out (0 of 893 for H1, 0 of 214 for H2). The leak test passes.
  - **Trading, event and E2 code:** perfect-foresight positions built with the real position code win on every day or trade they're on (H3's pairs and H4's short, daily and in 63-day holds) and the reverse loses on every one. E2 with D_off replaced by the answer plus noise scores a Brier skill of 0.56; shuffled, −0.01. H5's Welch t-test calls 8-10% of random 12-day event sets significant at 5%; for H5's own events, randomisation tests that keep the events' clustering give p 0.010, 0.003 and 0.005 at 10, 20 and 40 days (exploratory, chosen after H5's result, mostly one episode: H5 stays a lead to watch).
  - **New nowcast engine options** serve only the controls; their defaults keep the hypotheses' results unchanged. A monthly signal is used as it is; `NowcastSpec.signal_from` takes another country's signal; `NowcastSpec.shift_months` gives each month the value from k months earlier, wrapping round.
  - **Exploratory runs** are logged in `hypotheses_tested` with status `exploratory` (`exploratory-failed` for a run that crashed) and `"exploratory": true` in `parameters`, under ids `DX1` (engine controls), `DX2` (why the nowcasts failed), `DX3` (weather-model check) and `DX4` (stale FRED prices). They aren't hypothesis tests, never read the vault and never count as a hypothesis's look at it.
- **D53 (2026-10-07): The CPB World Trade Monitor is the trade-volume source** (`tracker/fetchers/cpb.py`, source `cpb`).
  - Both candidates were checked. The IMF's volume and price data (`IMF.STA/ITG` on the keyless IMF SDMX API, the former IFS trade indicators) can't serve: the export volume index (`XG_VI`) has no monthly value for any watchlist country, export price deflators stop in 2017-2021, and only the USA and China have current export price indices. CPB's monitor is public with no key or registration: the latest release page (`https://www.cpb.nl/en/worldtrademonitor/latest`) links one Excel file holding every series, published monthly about 8 weeks after the month (July 2026 on 25 September 2026).
  - Stored: merchandise export and import volumes (seasonally adjusted, 2021=100) and US-dollar unit values (prices, 2021=100) for WLD, USA, GBR, JPN and CHN: `cpb_export_volume`, `cpb_export_price`, `cpb_import_volume`, `cpb_import_price`. Volume × price = value, in index form. Regional aggregates (euro area, emerging Asia, ...) aren't stored, because the lake holds ISO3 countries and WLD (PLAN §4.1).
  - The .xlsx is read with the standard library (zipfile and XML); no spreadsheet dependency is added (§3).
  - Releases revise recent months, so every run reads the whole file (`resume_from: backfill_start`, from 2015; two requests). Publication lag 57 days. The source is enabled, so the daily run fetches it.
  - Backfill on 2026-10-07: 2,780 rows (20 series, 2015-01 to 2026-07); a rerun wrote 0.
  - **Why the nowcasts failed** (`python -m tracker diagnose why`, `tracker/backtest/why.py`, `DX2`): descriptive checks only (no new hypothesis tested out of sample), written up in plain English in `reports/diagnostics/WHY_FAILED.md`. Ranked: H2/H2b's link was the COVID swing and broke in 2022; H1/H1b's link is weak, fading and costs more to estimate than it gives; dollar export values have been driven by prices since 2022 (CPB); COVID-dominated training samples (for H2 they doubled even the benchmark's error); coverage and definitions; timing and bugs ruled out. Three refined hypotheses are proposed there and not pre-registered.
- **D54 (2026-10-07): The weather model's two annual harmonics were chosen on load-fit diagnostics before any H2 result existed, so H2 and H2b stay confirmatory tests.** Checked from git and the S18-S25 session transcript (times UTC, 2026-10-07):
  - 14:47-15:08: the five model variants of D45 compared on load and degree days only (daily out-of-sample error, weather left in the normalised YoY); the harmonics code first written at 14:59.
  - 15:11: the two-harmonic model committed (`210054f`, with D45's rationale). `tracker/signals/weather.py` hasn't changed since.
  - No step before then read an IP series with load: the first tool calls naming any IP indicator came at 15:15 and 15:18 (S19's divergence code and a synthetic test fixture), after the commit.
  - 15:44: the nowcast driver (`tracker/research.py`) first written; 15:50: the first H2 result logged in `hypotheses_tested`.
  - Caveat, as for every test: since S15 the dashboard had shown raw (not weather-normalised) power demand YoY next to IP, so the data weren't unseen; the harmonics only change the normalisation, which no chart showed against IP before H2 ran.
- **D55 (2026-10-07): FRED's IMF commodity prices are delayed, not discontinued; FRED stays their source for now.** `PCOPPUSDM` (copper) and `PNGASEUUSDM` (EU gas), like the rest of FRED release 365 ("Primary Commodity Prices", e.g. `PALLFNFINDEXM`), were last updated on 2026-08-17, with July 2026 as their latest month, and neither is marked discontinued. FRED copies the IMF irregularly: its release dates include a six-month gap (2025-07-14 to 2026-01-22). The IMF's own dataset (`IMF.RES/PCPS` on the keyless IMF SDMX API) already had August and September 2026 on 2026-10-07 and matches FRED exactly through July (copper 13,542.82 USD/t, EU gas 17.93 USD/MMBtu). The run asked for a replacement only if they were discontinued, so none was built.
  - **Proposed, if FRED is still behind at the next check:** read the two series from IMF PCPS directly (key `G001.PCOPP+PNGASEU.USD.M`, monthly averages about a week after the month, no key, no new dependency; the same API and pattern as the IMTS fetcher), keeping the indicator ids or adding `imf_copper` and `imf_eu_gas`. The World Bank Pink Sheet would also work (monthly, an .xlsx; the CPB fetcher's standard-library reader could read it) but is a second step removed from the same IMF data.
  - Neither series feeds a pre-registered test (H4 uses COMEX copper `HG=F` and H6 Dutch TTF gas `TTF=F`, both daily), so the delay only affects the dashboard and the stale list in `status` and the digest.
  - The stored availability of these values (month + 20 days) is earlier than they reached FRED in late batches, though not earlier than the IMF published them.
- **D56 (2026-10-07): Copper and EU gas prices come from the IMF directly; FRED's copies are retired.** Done as D55 proposed, because FRED was still at July 2026. Revisits D55's "FRED stays their source for now" (reason: you asked for it on 2026-10-07).
  - New indicators `imf_copper` (PCPS `PCOPP`: copper, grade A cathode, LME spot, USD per metric ton) and `imf_eu_gas` (`PNGASEU`: natural gas, Netherlands TTF day-ahead, USD per MMBtu), country `WLD`. They are fetched by the existing `imf` source as a second request per run (`commodity_prices` in sources.yaml; `IMF.RES/PCPS`, key `G001.PCOPP+PNGASEU.USD.M`, same keyless SDMX 3.0 API, CSV and parser as IMTS). New ids rather than reusing `fred_copper` and `fred_eu_gas`, because ids are `<source>_<metric>` (PLAN §4.1) and the FRED history stays in the store under its own ids.
  - Publication lag 10 days (September 2026 was out by 7 October; 10 allows for later months), so backfilled `available_at` is month end + 10 days. Backfill on 2026-10-07: 282 rows, 2015-01 to 2026-09; identical to FRED's values on all 139 shared months.
  - **Retiring an indicator:** a new optional `retired: {date, reason, replaced_by}` field in indicators.yaml. A retired indicator stays defined, so its stored history stays readable (dashboard, replay, backtests); its fetcher skips it (FRED skips retired series); `status`, the digest and the quality freshness and missing-country checks never report it stale or missing; Data Health shows a Retired column. `replaced_by` must name a defined indicator. The two FRED series stay listed in sources.yaml with a comment, so the switch lives in one place.
  - Neither series feeds a pre-registered test (H4 and H6 use daily futures), so no result changes.
- **D57 (2026-10-08, S26): The one-time vault run and the Signal Lab page.**
  - `python -m tracker vault H5` (`tracker/research_vault.py`; pure parts in `tracker/backtest/vault.py`) runs exactly the test pre-registered in SIGNALS.md "H5-vault". The pre-registration was committed on its own (`d954e4a`) before the run. The command refuses to start if `pit.vault_looks` finds a look for the hypothesis. It logs every test with `uses_vault = true` and one shared `tested_at`, and a crash after the guard is logged as a failed look, because the vault may have been read. Weekly dates go through `pit.walk_forward(include_vault=True)`; data are read with `store.latest(as_of=<run time>)` and `store.Vintages` (D39). Ran once, on 2026-10-07 at 23:12 UTC: `reports/H5_vault.md`.
  - The test reuses the diagnostics' clustered randomisation (`controls.shifted_event_means`, `randomisation_p`, D52) inside the vault. There is one window per horizon (trading days with a complete return), a fresh generator per test (seed 20261007, 1,000 draws), and independent episodes (events less than 8 weeks apart count once).
  - `--rehearse` runs the same code on pre-vault data only (window from 2022, prices as known at the vault start), logged as exploratory `DX5`. It reproduced S24's H5 events and mean returns exactly, and CONTROLS.md §5's clustered p-values closely (0.015, < 0.001 and 0.008, against 0.010, 0.003 and 0.005).
  - The **Signal Lab** page (`dashboard/views/signal_lab.py`, helpers in `dashboard/lab.py`) has one row per hypothesis. Statuses come from the first table of `reports/SUMMARY.md`, verdicts from each report's Headline section, and vault state and run counts from `hypotheses_tested`. Each report's headline and full text can be opened (the vault report too), and the full run log is shown with exploratory diagnostics hidden by default. Read-only, cached like every page.
- **D58 (2026-10-13): GitHub: a private repo with the full history, and a public snapshot built from it.**
  - The private repo is this one, with its full history. The audit (`docs/PRIVACY_AUDIT.md`) found no secrets in any commit, so the contents of history aren't rewritten; the personal details it holds (a remote-access address, the git e-mail, names from this PC) stay private. Commit messages were tidied on 2026-10-13, which changed every commit id: trees, authors and dates are unchanged, the ids cited in the docs were updated, and NOTES has the old-to-new map.
  - The public repo is a separate folder with one fresh commit, built by `scripts/make_public_snapshot.py`: every tracked file except NOTES, PRIVACY_AUDIT, PUBLISHING, the privacy scan, the snapshot script itself, the Plotly HTML reports and the development-tooling files (session prompts and tool configuration), with fixed rewrites of the remote-access, other-project and tooling wording. Each rewrite must match exactly, and the build stops if any of a list of private terms is left in the copy. It is checked with `scripts/privacy_scan.py --fail-on medium` before any push.
  - No downloaded data is redistributed (`data/` stays ignored); the repo holds code, config, tests (with small API fixtures), docs and reports. MIT licence, covering the repository only.
  - README charts are drawn from committed report tables (`scripts/make_figures.py`), never from the store, so they need no data and publish only summary statistics.
  - How to push both and refresh the public copy: `docs/PUBLISHING.md`.
- **Graduation decisions:** *(record Milestone C outcomes here)*

---

## 12. Parking lot (good ideas, not now)

- Trade by product code (HS codes): semiconductors 8542, vehicles 8703, iron ore 2601, LNG 271111
- GDELT news tone by country
- Mapping listed companies to their country and port exposure
- Paper-trading prediction markets
- A small cloud server for always-on runs and phone access to the dashboard *(phone access done without one: a private VPN proxy, D38)*
- More countries, or regional detail (US states, Chinese provinces)
- Nowcasting official GDP
- Food prices and unrest (E1; needs ACLED and WFP data)
- A working second price provider to replace the Stooq fallback (D21); needs an official API or key, and your approval for any new dependency
- A "replay" date slider on the dashboard: the data and one-sided transforms exactly as they looked on a past date, read with `store.latest(as_of=...)` *(done on the Country page: S16b, D43)*
- UK demand from NESO (the GB system operator's historic and daily demand data), to replace ENTSO-E's GB load, which ends in June 2021 (D36) *(done: S8b, D40)*

---

## 13. Glossary

- **Vintage:** a version of a data point as published at a particular time. Official data is often revised, so one month can have several vintages.
- **Point-in-time / look-ahead bias:** testing a strategy using data that wasn't available yet at the time makes results look better than they could really have been.
- **Robust z-score:** how unusual a value is compared with its recent history, measured in a way that ignores extreme outliers.
- **Divergence:** real-activity data and market prices (or official data) telling different stories.
- **Walk-forward:** repeatedly training on the past and testing on the next period, so the test is always "the future" from the model's point of view.
- **Vault:** the most recent 12 months of data, kept untouched until each hypothesis's final test.
- **Brier score:** the average squared error of probability forecasts. Lower is better, 0 is perfect, and always saying 50% scores 0.25.
- **Basis point (bp):** 0.01%.
