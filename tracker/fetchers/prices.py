"""Market prices: country ETFs, FX pairs, sector tickers and the benchmark.

Primary provider: yfinance (adjusted closes, `auto_adjust=True`). Fallback:
Stooq's CSV download, per ticker, for anything yfinance returns empty.
Config tickers are provider-neutral (US ETFs as `EWG`, FX as `USDEUR`,
futures as `HG=F`, other listings with an exchange suffix such as
`EXV7.DE`) and are mapped to each provider's symbol here.

Every close is stamped `available_at` = trade date 22:00 UTC (after the US
close). Bars whose 22:00 UTC hasn't happened yet (today's intraday bar) are
dropped. Weekend rows (Yahoo FX sometimes has them) are dropped too.

Adjusted closes change for the whole history after each dividend or split,
so the source re-downloads everything on each run (large refetch window);
the store then records those revisions like any other.
"""

from __future__ import annotations

import io
import re
from datetime import date, timedelta
from typing import Any

import pandas as pd

from tracker import store
from tracker.fetchers.base import BaseFetcher, FetchError

FX_RE = re.compile(r"^USD([A-Z]{3})$")
CLOSE_HOUR_UTC = 22


def yahoo_symbol(ticker: str, overrides: dict[str, str] | None = None) -> str:
    """Config ticker -> Yahoo symbol (USDEUR -> EUR=X; others unchanged)."""
    if overrides and ticker in overrides:
        return overrides[ticker]
    fx = FX_RE.match(ticker)
    return f"{fx.group(1)}=X" if fx else ticker


def stooq_symbol(ticker: str, overrides: dict[str, str | None] | None = None) -> str | None:
    """Config ticker -> Stooq symbol (EWG -> ewg.us, USDEUR -> usdeur,
    EXV7.DE -> exv7.de); None where Stooq has no equivalent."""
    if overrides and ticker in overrides:
        return overrides[ticker]
    if FX_RE.match(ticker) or "." in ticker:
        return ticker.lower()
    if "=" in ticker:  # futures need an explicit override
        return None
    return f"{ticker.lower()}.us"


def closes_to_long(closes: pd.DataFrame) -> pd.DataFrame:
    """Wide closes (index = date, one column per symbol) -> (date, symbol, value)."""
    if closes.empty:
        return pd.DataFrame(columns=["date", "symbol", "value"])
    long = closes.rename_axis(index="date", columns="symbol").stack().rename("value").reset_index()
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long.dropna(subset=["value"])


def completed_bars(closes: pd.DataFrame, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """Keep weekday bars whose close time (trade date 22:00 UTC) has passed,
    and add that time as `available_at`."""
    closes = closes[closes["date"].dt.dayofweek < 5]
    close_time = (closes["date"] + pd.Timedelta(hours=CLOSE_HOUR_UTC)).dt.tz_localize("UTC")
    keep = close_time <= fetched_at
    return closes[keep].assign(available_at=close_time[keep])


def parse_stooq_csv(text: str) -> pd.DataFrame:
    """Stooq daily CSV -> (date, value) of the Close column.

    Stooq now often answers with an HTML page that needs JavaScript (a bot
    check); that is reported as an error rather than worked around.
    """
    head = text.lstrip()[:100].lower()
    if head.startswith("<!doctype") or head.startswith("<html"):
        raise FetchError("stooq returned an HTML page (JavaScript verification), not CSV")
    if not head.startswith("date,"):
        raise FetchError(f"stooq: unexpected response: {text[:80]!r}")
    df = pd.read_csv(io.StringIO(text))
    return pd.DataFrame({"date": pd.to_datetime(df["Date"]), "value": pd.to_numeric(df["Close"], errors="coerce")}).dropna()


class PricesFetcher(BaseFetcher):
    """Daily adjusted closes for the instruments in sources.yaml / countries.yaml."""

    source = "prices"
    min_interval_s = 1.0

    def instruments(self) -> list[dict[str, str]]:
        """Every (ticker, indicator, country) to fetch; a ticker may serve several countries."""
        cfg = self.config
        out = []
        etf_ind = cfg.get("country_etf_indicator")
        fx_ind = cfg.get("fx_indicator")
        for iso3 in self.registry.watchlist():
            ref = self.registry.countries[iso3]
            if etf_ind and iso3 in self.registry.countries_for(etf_ind):
                out.append({"ticker": ref.country_etf, "indicator": etf_ind, "country": iso3})
            if fx_ind and ref.fx_pair and iso3 in self.registry.countries_for(fx_ind):
                out.append({"ticker": ref.fx_pair, "indicator": fx_ind, "country": iso3})
        for spec in [cfg["benchmark"], *cfg.get("sector", [])] if cfg.get("benchmark") else cfg.get("sector", []):
            out.append({"ticker": spec["ticker"], "indicator": spec["indicator"], "country": spec.get("country", "WLD")})
        return out

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        instruments = self.instruments()
        tickers = sorted({i["ticker"] for i in instruments})
        closes = self._from_yfinance(tickers, start, end)
        missing = [t for t in tickers if t not in set(closes["ticker"])]
        if missing:
            closes = pd.concat([closes, self._from_stooq(missing, start, end)], ignore_index=True)
        still_missing = [t for t in tickers if t not in set(closes["ticker"])]
        for t in still_missing:
            for inst in (i for i in instruments if i["ticker"] == t):
                self.report_issue(inst["indicator"], inst["country"], "no_price_data", "warning",
                                  f"{t}: no data from yfinance or stooq")
        if len(still_missing) == len(tickers):
            raise FetchError("prices: no provider returned any data")

        closes = closes.assign(date=pd.to_datetime(closes["date"]), value=pd.to_numeric(closes["value"]))
        closes = completed_bars(closes, self.fetched_at)
        frames = []
        for inst in instruments:
            rows = closes[closes["ticker"] == inst["ticker"]]
            if rows.empty:
                continue
            data = pd.DataFrame({"date": rows["date"], "country": inst["country"], "value": rows["value"]})
            frames.append(self.frame(inst["indicator"], data, available_at=rows["available_at"], vintage_type="estimated"))
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    # ------------------------------------------------------------- providers

    def _from_yfinance(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        overrides = (self.config.get("symbols") or {}).get("yfinance")
        to_symbol = {t: yahoo_symbol(t, overrides) for t in tickers}
        try:
            raw = yf.download(
                list(to_symbol.values()), start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True, progress=False, threads=False,
            )
        except Exception as exc:  # yfinance raises assorted errors (rate limits, parsing)
            self.log.warning("prices: yfinance failed: %s", exc)
            return pd.DataFrame(columns=["date", "ticker", "value"])
        if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame(columns=["date", "ticker", "value"])
        closes = raw["Close"]
        if isinstance(closes, pd.Series):  # a single ticker comes back without the ticker level
            closes = closes.to_frame(next(iter(to_symbol.values())))
        self.save_raw("yfinance_close", closes.to_csv(), "csv")
        long = closes_to_long(closes)
        back = {v: k for k, v in to_symbol.items()}
        long["ticker"] = long["symbol"].map(back)
        self.log.info("prices: yfinance returned %d tickers", long["ticker"].nunique())
        return long[["date", "ticker", "value"]]

    def _from_stooq(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        overrides = (self.config.get("symbols") or {}).get("stooq")
        frames = []
        for t in tickers:
            symbol = stooq_symbol(t, overrides)
            if not symbol:
                continue
            params: dict[str, Any] = {"s": symbol, "i": "d", "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d")}
            try:
                text = self.get(self.config.get("stooq_url", "https://stooq.com/q/d/l/"), params,
                                raw_name=f"stooq_{symbol}", raw_ext="csv").text
                data = parse_stooq_csv(text)
            except FetchError as exc:
                self.log.warning("prices: stooq %s failed: %s", symbol, exc)
                continue
            frames.append(data.assign(ticker=t))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", "value"])
