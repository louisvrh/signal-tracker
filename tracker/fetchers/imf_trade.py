"""IMF monthly merchandise trade (IMTS) and commodity prices (PCPS) from the IMF
data portal's SDMX 3.0 API (no key).

The IMF's International Trade in Goods by partner country dataset (IMTS,
formerly Direction of Trade Statistics) has monthly exports (FOB) and imports
(CIF) in US dollars for every watchlist country except Taiwan, with the world
(`G001`) as partner. All countries and both flows come in one request:

    {base_url}/data/dataflow/IMF.STA/IMTS/+/{countries}.{indicators}.G001.M
        ?c[TIME_PERIOD]=ge:YYYY-MM+le:YYYY-MM

Values are in US dollars (the `SCALE` attribute is only a display hint).
Every series is flagged `DERIVATION_TYPE = M` ("mixed": reported data
combined with IMF staff calculations or estimates) with no flag per month,
so recent months of slow reporters may be IMF estimates. This is an
additional official series next to Comtrade, much faster for China (PLAN D42).

Copper and EU gas prices come from the IMF's Primary Commodity Prices
(PCPS) in a second request, when `commodity_prices` is configured (PLAN D56):

    {base_url}/data/dataflow/IMF.RES/PCPS/+/G001.{indicators}.USD.M

Monthly averages in US dollars; the area `G001` (world prices) is stored as
`WLD`. FRED's copies of the same series arrive irregularly, months late (D55).

Responses are SDMX-CSV 2.0. A query that matches nothing still returns HTTP
200, with rows whose country and period are blank; they are dropped.
"""

from __future__ import annotations

import io
from datetime import date

import pandas as pd

from tracker import store
from tracker.fetchers.base import BaseFetcher, FetchError
from tracker.registry import WORLD

CSV_ACCEPT = "application/vnd.sdmx.data+csv;version=2.0.0"


def parse_imts_csv(text: str, codes: dict[str, str], countries: set[str]) -> pd.DataFrame:
    """SDMX-CSV 2.0 text -> DataFrame(indicator, date, country, value).

    `codes` maps IMF indicator codes (e.g. XG_FOB_USD) to registry ids; rows
    for other codes, other countries or monthly periods that don't parse are dropped.
    """
    if not text.strip():
        return pd.DataFrame(columns=["indicator", "date", "country", "value"])
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str)
    except pd.errors.ParserError as exc:
        raise FetchError(f"imf: SDMX-CSV could not be parsed: {exc}") from exc
    missing = [c for c in ("COUNTRY", "INDICATOR", "FREQUENCY", "TIME_PERIOD", "OBS_VALUE") if c not in df.columns]
    if missing:
        raise FetchError(f"imf: SDMX-CSV lacks columns {missing}: {text[:200]!r}")
    df = df.dropna(subset=["COUNTRY", "TIME_PERIOD", "OBS_VALUE"])
    df = df[(df["FREQUENCY"] == "M") & df["INDICATOR"].isin(codes) & df["COUNTRY"].isin(countries)]
    out = pd.DataFrame({
        "indicator": df["INDICATOR"].map(codes),
        "date": pd.to_datetime(df["TIME_PERIOD"].str.replace("-M", "-", regex=False), format="%Y-%m",
                               errors="coerce"),
        "country": df["COUNTRY"],
        "value": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
    }).dropna()
    if out.duplicated(["indicator", "date", "country"]).any():
        raise FetchError("imf: more than one value per country, indicator and month; is the key specific enough?")
    return out.reset_index(drop=True)


class IMFTradeFetcher(BaseFetcher):
    """Monthly exports and imports of goods for the watchlist (IMF IMTS), plus
    monthly commodity prices (IMF PCPS) when `commodity_prices` is configured."""

    source = "imf"
    min_interval_s = 1.0  # two requests per run (trade, prices); the old IMF service allowed 10 per 5 s
    timeout_s = 120.0

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        series = self.config["series"]
        codes = {s["code"]: s["indicator"] for s in series}
        countries = sorted({c for s in series for c in self.registry.countries_for(s["indicator"])})
        key = ".".join(["+".join(countries), "+".join(codes), self.config["counterpart"], "M"])
        url = f"{self.config['base_url']}/data/dataflow/{self.config['dataflow']}/+/{key}"
        params = {"c[TIME_PERIOD]": f"ge:{start:%Y-%m}+le:{end:%Y-%m}"}
        response = self.get(url, params, raw_name="imts_monthly", headers={"Accept": CSV_ACCEPT}, raw_ext="csv")
        data = parse_imts_csv(response.text, codes, set(countries))
        self.log.info("imf: %d monthly values for %d countries", len(data), data["country"].nunique())
        if self.config.get("commodity_prices"):
            data = pd.concat([data, self._fetch_prices(params)], ignore_index=True)
        frames = [self.frame(ind, part[["date", "country", "value"]]) for ind, part in data.groupby("indicator")]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    def _fetch_prices(self, params: dict[str, str]) -> pd.DataFrame:
        """Commodity prices (PCPS) for the same months, one request; area G001 -> WLD."""
        cfg = self.config["commodity_prices"]
        codes = {s["code"]: s["indicator"] for s in cfg["series"]}
        key = ".".join([cfg["area"], "+".join(codes), "USD", "M"])
        url = f"{self.config['base_url']}/data/dataflow/{cfg['dataflow']}/+/{key}"
        response = self.get(url, params, raw_name="pcps_monthly", headers={"Accept": CSV_ACCEPT}, raw_ext="csv")
        data = parse_imts_csv(response.text, codes, {cfg["area"]}).assign(country=WORLD)
        self.log.info("imf: %d monthly commodity prices for %d series", len(data), data["indicator"].nunique())
        return data
