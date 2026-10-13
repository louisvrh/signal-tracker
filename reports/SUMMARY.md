# Summary of findings (S26, Milestone C)

*Written 2026-10-08 (UTC) after H5's one look at the vault. Covers every pre-registered hypothesis in `docs/SIGNALS.md`. Each line links to the report with the numbers. Research only, not financial advice.*

## In one paragraph

Ten hypotheses were pre-registered and tested; none passed. Eight failed out of sample before the vault was opened. E3 can't be tested on history. **H5** was the one lead: freight rallied after chokepoint traffic collapsed in 2022-2025. It then **failed its one look at the vault**: in the 12 months the tests had never seen, freight did no better after the 7 new events than at random times. Under the graduation rule in PLAN §5.6, **nothing graduates** and nothing goes to paper trading. The tests were small and short, so "failed" means "no evidence it works", not "proven useless". But no signal has earned any trust yet.

## Every hypothesis

| Id | Status | The claim | What happened | Why, in plain English |
|---|---|---|---|---|
| H1 | failed | Port export volumes help nowcast official (Comtrade) exports | Forecasts with the port signal were **11% worse** than the benchmark built from past official numbers (820 forecasts, 23 countries, 2022-2025). 4% worse without 2020, and worse in both halves. ([H1](H1.md)) | Official exports are dollar values. Since 2022 they have moved mostly with prices and the dollar, not with tonnes shipped, and the port signal tracked them less each year (correlation 0.55, then 0.34, then 0.17). ([why](diagnostics/WHY_FAILED.md)) |
| H1b | failed | The same signal nowcasts IMF exports, China included | **16% worse** (858 forecasts, 24 countries). 19% worse without the five countries whose recent IMF months are probably estimates. ([H1b](H1b.md)) | As H1. Estimating the signal's weight in real time cost more than it added. |
| H2 | failed | Weather-normalised power demand nowcasts industrial production (7 European countries) | **26% worse** (206 forecasts, 2023-2025), worse in both halves. Only Poland came close to breaking even. ([H2](H2.md)) | Power and production moved together only during COVID's swing (correlation 0.82 in 2020-21, 0.11 in 2022-23, 0.02 in 2024-25). After the 2022 energy shock, energy-heavy industry cut its power use far more than its output. |
| H2b | failed | Weekday overnight load beats total load | No difference: the same error on the same 206 months (ratio 1.00). ([H2b](H2b.md)) | Both versions failed for the same reason, so there was nothing to tell them apart. |
| H3 | failed | Countries flagged by the market divergence underperform the world index (short the country ETF against URTH) | **Lost 6.4% a year** after costs (Sharpe −0.51, 38 trades). Random entry dates did as well 40% of the time. Deflated Sharpe 0.12. ([H3](H3.md)) | Some country was flagged 83% of the time, so the rule was nearly always in the market and showed no timing skill. A 63-day-hold variant made money in the first half and lost in the second. |
| H4 | failed | Chinese port imports falling while copper is calm leads to copper falling (short copper miners) | Only **3 trades**, net −6.3% a year. Random timing did as well 62% of the time. ([H4](H4.md)) | The condition rarely held, and the three trades it gave lost. Too little to learn from either way. |
| H5 | failed (the one lead, failed the vault) | A sharp fall in chokepoint transits moves freight (BDRY) and Brent | **Before the vault (2022-2025):** BDRY rose 18.6% on average in the 4 weeks after 12 events, against 0.8% normally. That missed the multiple-testing bar (p 0.04 against 0.004), but looked stronger under a test that respects clustering (p 0.003, exploratory). **The vault (Oct 2025-Oct 2026), pre-registered:** 7 events in 3 episodes. BDRY rose 3.0% in 4 weeks after them, below the 6.2% typical of random dates that year. p 0.46, the opposite side to training. **Failed.** ([H5](H5.md), [vault](H5_vault.md)) | The training result rested mostly on one episode, the 2023-24 Red Sea crisis, when shipping was diverted and dry-bulk freight surged. In the new year, traffic dips at Taiwan, Hormuz, Korea, Dover, Malacca and the Bosporus came with no freight rally. |
| H6 | failed | A European gas spike plus falling EU industrial power demand leads to European chemicals underperforming | **6 trades, all losing** (−7.2% a year), worse than every random schedule. ([H6](H6.md)) | In all six holds, European chemicals did better than the world index: the opposite of the claim. Six trades is also far too few. |
| E2 | failed | The official divergence (D_off) predicts whether the next official print is weaker than the last | Brier score 0.250 against 0.252 for always using the base rate (410 forecasts, 25 countries, 2024-2025). A skill of 0.005 (p 0.73), and poorly calibrated. ([E2](E2.md)) | D_off carries no usable information about the next print's direction. |
| E3 | not testable | The system's probabilities move before prediction-market prices | No market priced these prints before the vault. The few that exist opened in 2026 and haven't traded. ([E3](E3.md)) | It can only be run forward, in the journal. |
| E1 | parked | Trade collapses in food importers come before unrest | Not built (it needs ACLED and WFP data). | n/a |

## The vault: how many independent episodes, and what that means

Only H5 opened the vault. The test was pre-registered in `docs/SIGNALS.md` ("H5-vault") and committed on its own (`d954e4a`) before any vault data was read. It ran once, on 2026-10-07 at 23:12 UTC, and a second run is refused.

- **The vault held 3 independent episodes, not 7 independent events.** Events less than 8 weeks apart share outcome days, so they count once:
  1. late November-December 2025: Taiwan Strait and Hormuz;
  2. February-April 2026: Korea Strait, Dover and Malacca;
  3. June-August 2026: Dover and the Bosporus.

  The cargo-volume variant also gave 3 episodes (9 events). For comparison, the training period's 12 events were 5 episodes, and the largest, October 2023 to February 2024 around the Red Sea crisis, held 7 of the 12.
- **What 3 episodes means for confidence.** Three episodes are three data points. A pass would have been weak support at best. The test also had limited room: in a one-year window, about one placebo date-set in six lands within 4 weeks of the real events and shares their outcome, so even a real, steady effect would struggle to reach p < 0.05. A failure on its own therefore can't prove H5 wrong. But this one wasn't a near miss: freight did *worse* after the events than after random dates, on the side opposite to training. So the vault gives no support at all. Together with the training data, the most likely reading is that H5's training result was the story of one crisis, not a general rule.
- **Brent** was reported but is *contaminated by public knowledge of the 2026 Hormuz oil spike*, so it could never count. For the record, Brent rose more after the vault's events than at random (12.4% against 5.1% at 4 weeks, p 0.19; 22.8% against 10.6% at 8 weeks, p 0.014). That is what you would expect from a spike everyone already knew about.
- **The rest of the vault is unread by any test.** Every other hypothesis had already failed out of sample, and PLAN §5.6 needs a hypothesis to pass both out of sample and in the vault, so opening it for them could change nothing. Their vault data (ports, power, official statistics from 2025-10-08) is still unused by any test. PortWatch chokepoint data and BDRY and Brent prices from the vault year *have* now been used: a new chokepoint or freight hypothesis can't be tested blind on that year.

## Against the graduation rule (PLAN §5.6)

A signal graduates only if it beats its benchmark out of sample in both halves, survives costs, doesn't rely on 2020, and passes the vault.

| Hypothesis | First rule it fails | Graduates? |
|---|---|---|
| H1, H1b, H2, H2b | Out of sample: worse than the benchmark (in both halves) | no |
| H3, H4, H6 | Costs: lost money after costs, and no better than random timing | no |
| H5 | Out of sample: missed its pre-registered multiple-testing bar. The vault: failed | no |
| E2 | Out of sample: no skill over the base rate | no |
| E3, E1 | Not tested | no |

## Caveats that apply to everything

- **Short samples.** Most alternative series start around 2019, so the out-of-sample periods are 2-4 years with one COVID cycle and one energy shock. Effects smaller than the noise can't be seen.
- **Estimated publication dates.** Almost all history was downloaded in September 2026, so when each value became public is estimated (period end plus a typical lag). Real vintages only exist from 2026-09-27 (PLAN §5.1).
- **Not perfectly blind.** Thresholds were fixed before each test, but the data had been looked at on the dashboard while the system was built.
- **The engine itself was checked** (`reports/diagnostics/CONTROLS.md`): positive controls, placebos, an independent re-implementation and a timing audit all passed. The failures are the signals', not the code's.

## What next

- Nothing to paper-trade. The dashboard, the daily digest and your journal (`docs/JOURNAL.md`) remain the way to use the system: as context and for dated predictions.
- Rerun the tests each quarter as real vintages build up (PLAN §9). A hypothesis that has had its vault look can't use the same vault again.
- New ideas need new pre-registration. The refined H2c, H1c and H1d in `reports/diagnostics/WHY_FAILED.md` were shaped by looking at 2020-2025. Their own data's vault year is still unread, so each could get one honest look there, or be tested forward.
- Graduation decisions for you to record in PLAN §11 are proposed in `docs/NOTES.md`.
