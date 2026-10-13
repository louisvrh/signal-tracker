"""Trading and event tests H3-H6 (S24): signals as known each week, trades at the next close, costs.

`python -m tracker trading H3 H4 H5 H6` follows the operational definitions
written in SIGNALS.md on 2026-10-07 before any test, logs every variant to
`hypotheses_tested` and writes `reports/<id>.md`. Signals come from weekly
snapshots (Mondays 00:00 UTC, PLAN D39); prices are the adjusted closes as
known when the vault starts, and no price on or after the vault start
(2025-10-08) is used, so the vault stays closed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from tracker import settings, store
from tracker.backtest import pit
from tracker.backtest import trading as tr
from tracker.registry import Registry, load_registry
from tracker.research import REPORTS_DIR, VAULT_FROM, _fmt
from tracker.signals import divergence as dv
from tracker.signals import weather as wx

# v1 (2026-10-07): H4 and H5; H4's window started in 2020 although its signal starts in 2022.
# v2: every trading window starts when the signal first exists; ex-2020 only where 2020 is in it.
RUN_VERSION = "v2"
COSTS = {"central": tr.Costs(15, 2.0), "low": tr.Costs(10, 1.0), "high": tr.Costs(20, 3.0)}
HOLD_1M, HOLD_3M = 21, 63
EU6 = ("DEU", "FRA", "ITA", "ESP", "NLD", "POL")
PRICE_IDS = ("prices_country_etf", "prices_benchmark", "prices_copper_miners", "prices_copper_future",
             "prices_freight", "prices_brent_future", "prices_ttf_gas", "prices_eu_chemicals", "prices_fx")
TITLES = {
    "H3": "Countries with a market-divergence flag: the country ETF underperforms URTH",
    "H4": "Chinese port imports falling while copper is stable: copper declines",
    "H5": "Chokepoint transits fall sharply: freight (BDRY) and Brent move",
    "H6": "European gas spike plus falling EU industrial power demand: European chemicals underperform",
}
PIT_TABLE = "pit_divergences"  # H3's weekly real-time D_mkt and D_off, reused by E2 (S25)


def mondays(start: str) -> pd.DatetimeIndex:
    """Weekly test dates (Mondays 00:00 UTC) from `start` to the last before the vault."""
    dates = pd.date_range(pd.Timestamp(start, tz="UTC"), VAULT_FROM, freq="W-MON")
    return pd.DatetimeIndex([f.test for f in pit.walk_forward(dates, VAULT_FROM, min_train=0)])


def prices(data_dir: Path | None = None) -> pd.DataFrame:
    """Every price series as known when the vault starts, dates before it only."""
    df = store.latest(indicators=list(PRICE_IDS), as_of=VAULT_FROM, data_dir=data_dir)
    return df[df["date"] < VAULT_FROM.tz_localize(None)]


def price(df: pd.DataFrame, indicator: str, country: str) -> pd.Series:
    part = df[(df["indicator"] == indicator) & (df["country"] == country)]
    return pit.as_series(part) if not part.empty else pd.Series(dtype="float64")


def before(series: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """Values dated before the test date's day (a Monday 00:00 snapshot knows Friday's close)."""
    return series[series.index < as_of.tz_localize(None).normalize()]


def change_over(series: pd.Series, as_of: pd.Timestamp, days: int) -> float:
    """% change of a price from the last close `days` calendar days before the test date to the last close before it."""
    s = before(series, as_of).dropna()
    if s.empty:
        return float("nan")
    base = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    return float((s.iloc[-1] / base.iloc[-1] - 1) * 100) if len(base) else float("nan")


def end_values(rows: pd.DataFrame, signal: str, country: str | None = None) -> pd.Series:
    """A signal's end value on each test date (pit.compute rows), indexed by test date."""
    part = rows[(rows["signal"] == signal) & rows["value"].notna()]
    if country is not None:
        part = part[part["country"] == country]
    return pd.Series(part["value"].to_numpy(), index=pd.DatetimeIndex(part["as_of"])).sort_index()


# ---------------------------------------------------------------- evaluation


def evaluate(name: str, positions: pd.DataFrame, returns: pd.DataFrame, costs: tr.Costs, note: str,
             always_on: pd.DataFrame | None = None, random_n: int = 300) -> dict[str, Any]:
    """Metrics of one variant: the strategy net of costs, the always-on version of the same trade,
    random trades with the same number and lengths, and the two halves of the test period."""
    net, on = tr.portfolio(positions, returns, costs)
    m = tr.metrics(net, on)
    base = always_on if always_on is not None else pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    base_net, base_on = tr.portfolio(base, returns, costs)
    b = tr.metrics(base_net, base_on)
    rnd = tr.random_portfolio(positions, returns, costs, n=random_n)
    mid = net.index[len(net) // 2] if len(net) else None
    halves = []
    for part in (net.index < mid, net.index >= mid) if mid is not None else ():
        s = tr.metrics(net[part], on[part]).get("sharpe", float("nan"))
        sb = tr.metrics(base_net[part], base_on[part]).get("sharpe", float("nan"))
        halves.append((s, sb))
    per_trade = []
    for col in positions.columns:
        col_net = tr.backtest(positions[col].reindex(returns.index).fillna(0.0), returns[col], costs)
        per_trade += tr.trades(positions[col].reindex(returns.index).fillna(0.0), col_net)
    return {"variant": name, "note": note, **m,
            "hit_rate": float(np.mean([t > 0 for t in per_trade])) if per_trade else float("nan"),
            "trades": len(per_trade),
            "always_on_annual_return": b.get("annual_return"), "always_on_sharpe": b.get("sharpe"),
            **rnd,
            "sharpe_first_half": halves[0][0] if halves else float("nan"),
            "sharpe_second_half": halves[1][0] if halves else float("nan"),
            "always_on_sharpe_first_half": halves[0][1] if halves else float("nan"),
            "always_on_sharpe_second_half": halves[1][1] if halves else float("nan"),
            "beats_always_on_both_halves": bool(halves and all(s > sb for s, sb in halves)),
            "costs": f"{costs.per_side_bp:g} bp/side, borrow {costs.borrow_pct_year:g}%/yr" if costs.short_leg
            else f"{costs.per_side_bp:g} bp/side, no borrow fee"}


def add_deflated(rows: list[dict[str, Any]]) -> None:
    """Deflated Sharpe of each variant, against all the hypothesis's variants (§5.6 item 6)."""
    sharpes = [r.get("sharpe") for r in rows]
    for r in rows:
        r["deflated_sharpe"] = tr.deflated_sharpe(r.get("sharpe", float("nan")), sharpes, r.get("days", 0),
                                                  r.get("skew", 0.0), r.get("kurtosis", 3.0))


def without_2020(positions: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = returns.index.year != 2020
    return positions.reindex(returns.index).fillna(0.0)[keep], returns[keep]


# --------------------------------------------------------------------- H3


def h3(registry: Registry, px: pd.DataFrame, data_dir: Path | None, progress: Callable[[str], None]) -> dict[str, Any]:
    vint = store.Vintages.load(indicators=dv.needed_indicators(), data_dir=data_dir)
    dates = mondays("2024-01-01")
    panel = {"d_mkt": dv.PANEL_SIGNALS["d_mkt"], "d_off": dv.PANEL_SIGNALS["d_off"]}
    started = time.time()

    def tick(n: int, total: int, as_of: pd.Timestamp) -> None:
        if n % 10 == 0 or n == total:
            progress(f"  H3 snapshots {n}/{total} ({as_of:%Y-%m-%d}, {time.time() - started:.0f} s)")

    rows = pit.compute(vint, registry, dates, panel=panel, progress=tick)
    store.write_signal_table(PIT_TABLE, rows, data_dir)
    urth = price(px, "prices_benchmark", "WLD")
    first = pd.DatetimeIndex(rows.loc[(rows["signal"] == "d_mkt") & rows["value"].notna(), "as_of"]).min()
    days = urth.index[urth.index >= first.tz_localize(None)]  # from the first D_mkt value
    pos, rets, entries = {}, {}, {}
    for country in sorted(rows["country"].dropna().unique()):
        d = end_values(rows, "d_mkt", country)
        etf = price(px, "prices_country_etf", country)
        if d.empty or etf.empty:
            continue
        flag = dv.flag_states(d)
        pos[country] = tr.positions_from_signal(flag.astype(float), days)
        rets[country] = tr.pair_returns(long=urth, short=etf).reindex(days).fillna(0.0)
        switched = flag & ~flag.shift(1, fill_value=False)
        entries[country] = flag.index[switched]
    returns = pd.DataFrame(rets).iloc[1:]
    primary = pd.DataFrame(pos).reindex(returns.index)
    hold = {n: pd.DataFrame({c: tr.fixed_holds(e, days, n) for c, e in entries.items()}).reindex(returns.index)
            for n in (HOLD_1M, HOLD_3M)}
    variants = [
        evaluate("exit when the flag switches off (primary)", primary, returns, COSTS["central"],
                 "short country ETF, long URTH, equal weight across flagged countries"),
        evaluate(f"hold {HOLD_1M} trading days", hold[HOLD_1M], returns, COSTS["central"], "fixed 1-month hold"),
        evaluate(f"hold {HOLD_3M} trading days", hold[HOLD_3M], returns, COSTS["central"], "fixed 3-month hold"),
        evaluate("primary, low costs", primary, returns, COSTS["low"], "cost sensitivity"),
        evaluate("primary, high costs", primary, returns, COSTS["high"], "cost sensitivity"),
    ]
    flags_seen = sum(len(e) for e in entries.values())
    extra = [f"Flags switching on in the test period: {flags_seen} in {sum(1 for e in entries.values() if len(e))} "
             f"countries ({', '.join(f'{c} {len(e)}' for c, e in entries.items() if len(e))})."]
    return {"variants": variants, "extra": extra, "test_dates": dates}


# --------------------------------------------------------------------- H4


def h4(registry: Registry, px: pd.DataFrame, data_dir: Path | None, progress: Callable[[str], None]) -> dict[str, Any]:
    vint = store.Vintages.load(indicators=["portwatch_imports"], countries=["CHN"], data_dir=data_dir)
    dates = mondays("2020-01-06")
    rows = pit.compute(vint, registry, dates, series=[pit.ChainSignal("imports_z", "yoy>rz(3y)",
                                                                     ("portwatch_imports",), ("CHN",))])
    z = end_values(rows, "imports_z", "CHN")
    copper = price(px, "prices_copper_future", "WLD")
    ret91 = pd.Series({d: change_over(copper, d, 91) for d in z.index})
    condition = (z < -1.0) & (ret91.abs() < 5.0)
    entries = condition.index[condition.to_numpy()]
    copx, hg = price(px, "prices_copper_miners", "WLD"), copper
    out = []
    first = z.index.min().tz_localize(None)  # the window starts when the signal first exists
    for label, series, costs_key in (("COPX", copx, "central"), ("HG=F", hg, None)):
        days = series.index[series.index >= first]
        rets = pd.DataFrame({label: tr.pair_returns(long=None, short=series).reindex(days).fillna(0.0)}).iloc[1:]
        one_leg = tr.Costs(COSTS["central"].per_side_bp, COSTS["central"].borrow_pct_year, legs=1,
                           short_leg=costs_key is not None)
        for hold in (HOLD_3M, HOLD_1M):
            pos = pd.DataFrame({label: tr.fixed_holds(entries, days, hold)}).reindex(rets.index)
            primary = label == "COPX" and hold == HOLD_3M
            out.append(evaluate(f"short {label}, hold {hold} days" + (" (primary)" if primary else ""), pos, rets,
                                one_leg, "one-leg short" + ("" if costs_key else "; futures: no borrow fee, roll jumps in returns")))
            if primary:
                for key in ("low", "high"):
                    c = tr.Costs(COSTS[key].per_side_bp, COSTS[key].borrow_pct_year, legs=1)
                    out.append(evaluate(f"short COPX, hold 63 days, {key} costs", pos, rets, c, "cost sensitivity"))
                if (rets.index.year == 2020).any():
                    p2, r2 = without_2020(pos, rets)
                    out.append(evaluate("short COPX, hold 63 days, ex-2020", p2, r2, one_leg, "2020 dropped"))
    extra = [f"Signal available from {z.index.min():%Y-%m-%d} (robust z needs 2 years of YoY); condition true on "
             f"{int(condition.sum())} of {len(condition)} weekly test dates "
             f"({', '.join(f'{d:%Y-%m-%d}' for d in entries[:12])}{' ...' if len(entries) > 12 else ''})."]
    return {"variants": out, "extra": extra, "test_dates": dates}


# --------------------------------------------------------------------- H5


def chokepoint_events(rows: pd.DataFrame, signal: str, gap_weeks: int = 8) -> pd.DataFrame:
    """Test dates on which a chokepoint's z falls below −2 after `gap_weeks` dates at or above it."""
    events = []
    for country, part in rows[(rows["signal"] == signal) & rows["value"].notna()].groupby("country"):
        z = pd.Series(part["value"].to_numpy(), index=pd.DatetimeIndex(part["as_of"])).sort_index()
        below = z < -2
        calm = (~below).astype(int).rolling(gap_weeks, min_periods=gap_weeks).sum().shift(1) == gap_weeks
        for d in z.index[below & calm]:
            events.append({"as_of": d, "chokepoint": country, "z": float(z[d])})
    return pd.DataFrame(events, columns=["as_of", "chokepoint", "z"])


def h5(registry: Registry, px: pd.DataFrame, data_dir: Path | None, progress: Callable[[str], None]) -> dict[str, Any]:
    chokepoints = registry.countries_for("portwatch_transits")
    vint = store.Vintages.load(indicators=["portwatch_transits", "portwatch_transit_volume"], countries=chokepoints,
                               data_dir=data_dir)
    dates = mondays("2020-01-06")
    rows = pit.compute(vint, registry, dates, series=[
        pit.ChainSignal("transits_z", "yoy>rz(3y)", ("portwatch_transits",), tuple(chokepoints)),
        pit.ChainSignal("volume_z", "yoy>rz(3y)", ("portwatch_transit_volume",), tuple(chokepoints))])
    tables, out = {}, []
    for signal, label in (("transits_z", "transit counts (primary)"), ("volume_z", "cargo volume")):
        ev = chokepoint_events(rows, signal)
        for instrument, ind in (("BDRY", "prices_freight"), ("Brent BZ=F", "prices_brent_future")):
            study = tr.event_study(pd.DatetimeIndex(ev["as_of"]), price(px, ind, "WLD"), (10, 20, 40))
            tables[(label, instrument)] = (ev, study)
            for r in study.itertuples():
                out.append({"variant": f"{label}: {instrument}, {r.horizon_days} days", "note": "event study",
                            "events": r.events, "mean": r.mean, "unconditional_mean": r.unconditional_mean,
                            "difference": r.difference, "t": r.t, "p": r.p, "share_up": r.share_up,
                            "share_up_unconditional": r.share_up_unconditional})
    return {"variants": out, "tables": tables, "test_dates": dates}


# --------------------------------------------------------------------- H6


def h6(registry: Registry, px: pd.DataFrame, data_dir: Path | None, progress: Callable[[str], None]) -> dict[str, Any]:
    vint = store.Vintages.load(indicators=["entsoe_overnight_load", "entsoe_load", wx.HDD, wx.CDD], countries=list(EU6),
                               data_dir=data_dir)
    dates = mondays("2021-01-04")
    rows = pit.compute(vint, registry, dates, series=[
        pit.ChainSignal("overnight", "wx>yoy", ("entsoe_overnight_load",), EU6),
        pit.ChainSignal("total", "wx>yoy", ("entsoe_load",), EU6)])
    ttf = price(px, "prices_ttf_gas", "WLD")
    spike = pd.Series({d: change_over(ttf, d, 28) > 20.0 for d in dates})
    chem_eur, fx = price(px, "prices_eu_chemicals", "WLD"), price(px, "prices_fx", "DEU")
    chem_usd = (chem_eur / fx.reindex(chem_eur.index).ffill()).dropna()
    urth = price(px, "prices_benchmark", "WLD")
    have = rows[rows["value"].notna()].groupby("as_of")["value"].count()
    first = have[have >= 4].index.min().tz_localize(None)  # the window starts when the signal first exists
    days = urth.index[urth.index >= first]
    rets = pd.DataFrame({"pair": tr.pair_returns(long=urth, short=chem_usd).reindex(days).fillna(0.0)}).iloc[1:]
    out, extra = [], [f"Signal (4+ countries' weather-normalised load YoY) first available {first:%Y-%m-%d}; "
                      "the evaluation window starts there."]
    for signal, label in (("overnight", "overnight load"), ("total", "total load")):
        demand = rows[rows["signal"] == signal].groupby("as_of")["value"].agg(["mean", "count"])
        falling = (demand["mean"] < 0) & (demand["count"] >= 4)
        both = spike.reindex(falling.index).fillna(False) & falling
        entries = both.index[both.to_numpy()]
        extra.append(f"{label}: conditions both true on {len(entries)} weekly test dates"
                     + (f" ({', '.join(f'{d:%Y-%m-%d}' for d in entries[:12])}{' ...' if len(entries) > 12 else ''})"
                        if len(entries) else ""))
        for hold in (HOLD_3M, HOLD_1M):
            if signal == "total" and hold == HOLD_1M:
                continue
            pos = pd.DataFrame({"pair": tr.fixed_holds(entries, days, hold)}).reindex(rets.index)
            primary = signal == "overnight" and hold == HOLD_3M
            out.append(evaluate(f"{label}, hold {hold} days" + (" (primary)" if primary else ""), pos, rets,
                                COSTS["central"], "short EXV7.DE in USD, long URTH"))
            if primary:
                for key in ("low", "high"):
                    out.append(evaluate(f"{label}, hold 63 days, {key} costs", pos, rets, COSTS[key], "cost sensitivity"))
    extra.append(f"TTF up more than 20% over 28 days on {int(spike.sum())} of {len(spike)} weekly test dates.")
    return {"variants": out, "extra": extra, "test_dates": dates}


# ------------------------------------------------------------------- reports


def _trading_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Variant | Days | Trades | Time on | Annual return | Sharpe | Max drawdown | Hit rate | Turnover/yr "
             "| Always-on Sharpe | Random p | Deflated Sharpe | Sharpe 1st / 2nd half | Costs |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r.get('days', 0)} | {r.get('trades', 0)} | {_fmt(r.get('share_on'))} "
            f"| {_fmt(100 * r['annual_return'], 1) + '%' if r.get('annual_return') == r.get('annual_return') and r.get('annual_return') is not None else '–'} "
            f"| {_fmt(r.get('sharpe'))} | {_fmt(100 * r['max_drawdown'], 1) + '%' if r.get('max_drawdown') is not None else '–'} "
            f"| {_fmt(r.get('hit_rate'))} | {_fmt(r.get('turnover_per_year'), 1)} | {_fmt(r.get('always_on_sharpe'))} "
            f"| {_fmt(r.get('random_p'))} | {_fmt(r.get('deflated_sharpe'))} "
            f"| {_fmt(r.get('sharpe_first_half'))} / {_fmt(r.get('sharpe_second_half'))} | {r.get('costs', '')} |")
    return lines


def trading_verdict(r: dict[str, Any]) -> str:
    if not r.get("trades"):
        return "Not testable yet: the signal never switched on in the test period."
    sharpe, rp, ds = r.get("sharpe", float("nan")), r.get("random_p", float("nan")), r.get("deflated_sharpe", float("nan"))
    if sharpe > 0.5 and rp < 0.05 and r.get("beats_always_on_both_halves") and ds > 0.95:
        return "Promising: positive after costs, better than random timing and the always-on trade in both halves, and the deflated Sharpe holds up. Still needs the vault and paper trading."
    if sharpe > 0 and rp < 0.1:
        return (f"Weak: positive after costs (Sharpe {sharpe:.2f}) and timing better than most random entries "
                f"(p {rp:.2f}), but {'not in both halves' if not r.get('beats_always_on_both_halves') else 'the deflated Sharpe is low'}"
                f" with only {r['trades']} trades. Treat as unproven.")
    if sharpe > 0:
        return (f"Not supported: positive after costs (Sharpe {sharpe:.2f}) but no better than random timing "
                f"(p {rp:.2f}) with {r['trades']} trades.")
    return f"Not supported: lost money after costs (Sharpe {sharpe:.2f}, {r['trades']} trades)."


def write_report(hypothesis: str, result: dict[str, Any], out_dir: Path = REPORTS_DIR) -> Path:
    rows = result["variants"]
    dates = result["test_dates"]
    lines = [f"# {hypothesis}: {TITLES[hypothesis]}", "",
             f"*Run {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `python -m tracker trading {hypothesis}` (S24, "
             f"{RUN_VERSION}). Definitions: SIGNALS.md, operational definitions written 2026-10-07 before testing. "
             f"Weekly test dates {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}; vault closed (no data or price on or "
             f"after {VAULT_FROM:%Y-%m-%d}). Research only, not financial advice.*", ""]
    if hypothesis == "H5":
        lines += ["## Headline (out of sample: fixed rules, nothing fitted)", "",
                  "| Variant | Events | Mean return | All other days | Difference | t | p | Share up (other days) |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['variant']} | {r['events']} | {_fmt(100 * r['mean'], 2)}% | {_fmt(100 * r['unconditional_mean'], 2)}% "
                         f"| {_fmt(100 * r['difference'], 2)} pp | {_fmt(r['t'])} | {_fmt(r['p'], 3)} "
                         f"| {_fmt(r['share_up'])} ({_fmt(r['share_up_unconditional'])}) |")
        primary = [r for r in rows if r["variant"].startswith("transit counts")]
        best = min(primary, key=lambda r: r["p"] if r["p"] == r["p"] else 1.0) if primary else None
        n_events = primary[0]["events"] if primary else 0
        lines += ["", f"**Verdict.** {n_events} events (transit counts). "
                  + (f"The smallest p-value among the {len(primary)} primary tests is {best['p']:.3f} "
                     f"({best['variant']}); with {len(rows)} tests in all, a p-value would need to be below "
                     f"{0.05 / max(len(rows), 1):.4f} to survive a Bonferroni correction. "
                     if best else "")
                  + ("Not supported." if not best or best["p"] > 0.05 / max(len(rows), 1) else "Promising but tentative."), ""]
        lines += ["## Events", ""]
        for (label, instrument), (ev, _) in result["tables"].items():
            if instrument.startswith("BDRY"):
                lines.append(f"- {label}: " + (", ".join(f"{e.chokepoint} {pd.Timestamp(e.as_of):%Y-%m-%d} (z {e.z:.1f})"
                                                         for e in ev.itertuples()) or "none"))
    else:
        primary = next(r for r in rows if "(primary)" in r["variant"])
        lines += ["## Headline (out of sample: fixed rules, nothing fitted)", "",
                  f"Primary variant: **{primary['variant']}**, net of {primary['costs']}.", "",
                  "| | Strategy | Always on |", "|---|---|---|",
                  f"| Annual return | {_fmt(100 * primary['annual_return'], 1) if primary.get('annual_return') is not None else '–'}% "
                  f"| {_fmt(100 * primary['always_on_annual_return'], 1) if primary.get('always_on_annual_return') is not None else '–'}% |",
                  f"| Sharpe ratio | {_fmt(primary.get('sharpe'))} | {_fmt(primary.get('always_on_sharpe'))} |",
                  f"| Sharpe, 1st / 2nd half | {_fmt(primary.get('sharpe_first_half'))} / {_fmt(primary.get('sharpe_second_half'))} "
                  f"| {_fmt(primary.get('always_on_sharpe_first_half'))} / {_fmt(primary.get('always_on_sharpe_second_half'))} |",
                  f"| Maximum drawdown | {_fmt(100 * primary['max_drawdown'], 1) if primary.get('max_drawdown') is not None else '–'}% | |",
                  f"| Trades, hit rate | {primary.get('trades', 0)}, {_fmt(primary.get('hit_rate'))} | |",
                  f"| Time in the market | {_fmt(primary.get('share_on'))} | 1.00 |",
                  f"| Random entries with the same trades: share doing at least as well | {_fmt(primary.get('random_p'))} | |",
                  f"| Deflated Sharpe (probability, {len(rows)} variants) | {_fmt(primary.get('deflated_sharpe'))} | |",
                  "", f"**Verdict.** {trading_verdict(primary)}", "",
                  "## Every variant tried", "", *_trading_table(rows)]
    if result.get("extra"):
        lines += ["", "## Notes", "", *[f"- {e}" for e in result["extra"]]]
    if hypothesis == "H5":
        lines += ["", "## Caveats specific to H5", "",
                  "- Events cluster: several fall in the same weeks (the Red Sea crisis of late 2023 and early 2024 "
                  "shows up at Bab el-Mandeb, Suez, Gibraltar and Dover), so they are far from independent, and the "
                  "t-tests (which treat them as independent, and compare with overlapping windows on all other days) "
                  "overstate the evidence.",
                  "- An event's horizon that would reach past the vault start is dropped (Hormuz, 2025-10-06)."]
    lines += ["", "## Caveats", "",
              "- Short samples: few independent trades, so every number here is fragile. Most history before September "
              "2026 has estimated publication dates (PLAN §5.1).",
              "- Prices are Yahoo adjusted closes (total return for ETFs). Futures are continuous front-month series "
              "whose returns include roll jumps. Trades enter at the next close after the weekly snapshot.",
              "- The thresholds were fixed before testing, but the data had been looked at while building the "
              "dashboard and signals, so this is not a perfectly blind test."]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{hypothesis}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _log(hypothesis: str, rows: list[dict[str, Any]], dates: pd.DatetimeIndex) -> None:
    for r in rows:
        metrics = {k: v for k, v in r.items() if isinstance(v, (int, float, bool, str)) and k not in ("variant", "note")}
        store.append_hypotheses_tested(pit.hypothesis_record(
            hypothesis, r["variant"], {"note": r.get("note", ""), "vault_from": str(VAULT_FROM), "definitions": "SIGNALS.md 2026-10-07"},
            [dates.min(), dates.max()], [dates.min(), dates.max()], metrics, status="ok", uses_vault=False,
            notes=f"S24 {RUN_VERSION}; fixed pre-registered rules, nothing fitted"))


RUNNERS = {"H3": h3, "H4": h4, "H5": h5, "H6": h6}


def run_trading(hypotheses: list[str], data_dir: Path | None = None,
                progress: Callable[[str], None] = print) -> dict[str, Any]:
    registry = load_registry()
    px = prices(data_dir)
    summary = {}
    for hypothesis in hypotheses:
        started = time.time()
        try:
            result = RUNNERS[hypothesis](registry, px, data_dir, progress)
        except Exception as exc:  # every run is logged, failed ones too (PLAN §4.2)
            store.append_hypotheses_tested(pit.hypothesis_record(
                hypothesis, "all variants (crashed)", {"vault_from": str(VAULT_FROM)}, None, None, {},
                status="failed", notes=f"S24 {RUN_VERSION}: {type(exc).__name__}: {settings.redact(str(exc))[:300]}"))
            progress(f"{hypothesis}: FAILED ({type(exc).__name__}: {exc}); logged")
            continue
        if hypothesis != "H5":
            add_deflated(result["variants"])
        _log(hypothesis, result["variants"], result["test_dates"])
        path = write_report(hypothesis, result)
        summary[hypothesis] = {"result": result, "report": path}
        progress(f"{hypothesis}: {len(result['variants'])} variants in {time.time() - started:.0f} s -> "
                 f"{path.relative_to(settings.ROOT)}")
    return summary
