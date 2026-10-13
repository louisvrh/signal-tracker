# Setup and operations

How to install, run, schedule and back up the tracker on Windows. The project overview is in the [README](../README.md); the master plan is [`docs/PLAN.md`](PLAN.md).

## Setup (Windows PowerShell)

```powershell
cd signal-tracker   # the cloned folder
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env   # then open .env in an editor and fill in your keys
git config core.hooksPath scripts/hooks   # pre-commit hook: runs pytest, blocks failing commits
```

If PowerShell refuses to run scripts in `scripts\`, allow local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### API keys

Free keys, set in `.env` (see `.env.example`); every other source needs none.

| `.env` name | Source |
|---|---|
| `FRED_API_KEY` | FRED (St. Louis Fed) |
| `EIA_API_KEY` | US Energy Information Administration, API v2 |
| `ENTSOE_API_TOKEN` | ENTSO-E Transparency Platform (security token) |
| `COMTRADE_SUBSCRIPTION_KEY` | UN Comtrade (free tier, 500 calls a day) |
| `EMBER_API_KEY` | Ember |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | the morning digest (below) |
| `BACKUP_DIR` | not a key: the folder backups are copied to |

## Everyday commands

```powershell
.venv\Scripts\python -m tracker --help
.venv\Scripts\python -m tracker fetch                       # all enabled sources
.venv\Scripts\python -m tracker fetch --source fred          # one source
.venv\Scripts\python -m tracker fetch --source fred --backfill --since 2015-01-01
.venv\Scripts\python -m tracker status                      # freshness, quality issues, last backup
.venv\Scripts\python -m tracker quality                     # run the data quality checks
.venv\Scripts\python -m tracker quality --list              # show open issues without re-checking
.venv\Scripts\python -m tracker decompose                   # precompute STL for the dashboard (~1 minute)
.venv\Scripts\python -m tracker weather                     # weather-normalised load + fit report (reports\diagnostics)
.venv\Scripts\python -m tracker signals                     # composites, divergences, flags (data\signals)
.venv\Scripts\python -m tracker digest --dry-run           # print today's Telegram digest without sending it
.venv\Scripts\python -m tracker digest --find-chat         # one-time: save your chat id (message the bot first)
.venv\Scripts\python -m tracker nowcast H1 H1b H2 H2b      # nowcast tests, vault closed (reports\H1.md ...)
.venv\Scripts\python -m tracker trading H3 H4 H5 H6        # trading and event tests, vault closed (H3 takes ~12 minutes)
.venv\Scripts\python -m tracker events                     # E2 event test (needs the H3 run's stored D_off values)
.venv\Scripts\python -m tracker diagnose controls          # engine controls, exploratory (~30 minutes; reports\diagnostics\CONTROLS.md)
.venv\Scripts\python -m tracker diagnose why               # why the nowcasts failed, exploratory (~3 minutes; reports\diagnostics)
.venv\Scripts\python -m tracker vault H5 --rehearse         # the vault test's code on pre-vault data only, exploratory (~4 minutes)
.venv\Scripts\python -m tracker vault H5                    # H5's one look at the vault (done 2026-10-07; a second look is refused)
.venv\Scripts\python -m tracker journal-score              # Brier scores of resolved docs\JOURNAL.md predictions
.venv\Scripts\python -m tracker compact                     # monthly: merge lake files (not during a fetch)
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\make_figures.py                # redraw reports\figures from the committed report tables
```

Logs go to `logs\tracker.log` (rotating). Data lives in `data\` (git-ignored).

## Daily run, scheduling and backup

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_daily.ps1      # dashboard check, fetch, quality, decompose, signals, backup, digest
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\backup.ps1         # backup only
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_task.ps1  # schedule run_daily at 07:00
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_task.ps1 -Unregister
```

`-ExecutionPolicy Bypass` applies to that one PowerShell process only, so no
policy change is needed. (With `RemoteSigned` set for your user, as above,
`.\scripts\run_daily.ps1` also works.) The daily run logs to `logs\run_daily.log`.
The backup copies `data\lake`, `data\meta` and `data\raw` to
`<BACKUP_DIR>\signal-tracker\` (set `BACKUP_DIR` in `.env`) and never deletes
anything there.

## Telegram digest

Every daily run ends with one Telegram message: "all quiet", the day's news
(failures, new divergence flags and anomalies, the top divergences), or
"daily run FAILED at: <steps>". To set it up once:

1. Create a bot with BotFather and put its token in `.env` as `TELEGRAM_BOT_TOKEN`.
2. Open your bot in Telegram and send it any message.
3. Within 24 hours, run `.venv\Scripts\python -m tracker digest --find-chat`. It saves
   `TELEGRAM_CHAT_ID` to `.env` and sends you a test message.
4. Run `.venv\Scripts\python -m tracker digest` to send today's digest.

Until then the digest is written to `logs\run_daily.log` instead.

## Dashboard

The scheduled task "SignalTracker Dashboard" keeps the dashboard running at
http://localhost:8551. It starts at logon in a hidden window and logs to
`logs\dashboard.log`. If Streamlit stops, the service script restarts it; if the
service itself is killed, the task's 5-minute watchdog trigger starts it again,
and the daily run (`scripts\check_dashboard.ps1`, its first step) restarts the
task if nothing is listening on 8551 (not Streamlit's default 8501, so it
doesn't clash with other Streamlit apps).

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_dashboard.ps1                        # open it in the browser
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_dashboard_task.ps1 -Start       # register and (re)start the task
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_dashboard_task.ps1 -Stop        # stop and disable it until -Start
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_dashboard_task.ps1 -Unregister  # stop and remove the task
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_dashboard.ps1                      # restart it if nothing listens on 8551
```

`run_dashboard.ps1` opens the running dashboard, or, if nothing serves port 8551,
starts one in that window (Ctrl+C stops it). After editing dashboard code, run
`register_dashboard_task.ps1 -Start` to restart it. The dashboard only reads
`data\`. Data is cached for an hour; use "Reload data" in the sidebar after a
fetch. The landing page is the **Overview** (composite scores, this week's movers and active divergence flags, from the daily `signals` step). Open a country directly with `http://localhost:8551/country?country=DEU`; the old `http://localhost:8551/?country=DEU` still works.

On the Country and Indicator pages, **Show: Chain** opens the transform-chain
workbench in the sidebar: pick a preset or build a chain of steps (smoothing,
seasonal adjustment, detrending, band-passes, scaling), switch on "Real-time
only" to hide steps that use later data, or the spectrum view. Chains are short
strings such as `stl>sma(28d)>rz(3y)`; "Copy as hypothesis" gives the line for
`docs\SIGNALS.md`. Lengths are converted to each chart's frequency (`sma(28d)`
runs as `sma(1m)` on monthly charts, `1m` as `30d` on daily ones); the
"Lengths converted" note under the charts lists every change. Saved presets go to `config\presets.yaml`, the dashboard's
only write.

## Rules

- Secrets live only in `.env`, which is git-ignored. Never commit it or paste keys anywhere.
- `data\` is append-only; never edit or delete files in `data\lake\` by hand.
