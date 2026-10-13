"""Helpers shared by the SDMX fetchers (Eurostat, OECD): SDMX-CSV parsing."""

from __future__ import annotations

import io

import pandas as pd

from tracker.fetchers.base import FetchError


def parse_sdmx_csv(text: str, area_col: str, area_to_iso3: dict[str, str]) -> pd.DataFrame:
    """SDMX-CSV text -> DataFrame(date, country, value) for monthly data.

    `area_col` is the column holding the country code (e.g. `geo`, `REF_AREA`)
    and `area_to_iso3` maps those codes to ISO3; other areas are dropped.
    Empty responses return an empty frame.
    """
    if not text.strip():
        return pd.DataFrame(columns=["date", "country", "value"])
    try:
        df = pd.read_csv(io.StringIO(text), dtype={area_col: str, "TIME_PERIOD": str})
    except pd.errors.ParserError as exc:
        raise FetchError(f"SDMX-CSV could not be parsed: {exc}") from exc
    missing = [c for c in (area_col, "TIME_PERIOD", "OBS_VALUE") if c not in df.columns]
    if missing:
        raise FetchError(f"SDMX-CSV lacks columns {missing}: {text[:200]}")
    df = df[df[area_col].isin(area_to_iso3)]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m"),
        "country": df[area_col].map(area_to_iso3),
        "value": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
    }).dropna(subset=["value"])
    if out.duplicated(["date", "country"]).any():
        raise FetchError("SDMX-CSV has more than one series per country; the key is not specific enough")
    return out.reset_index(drop=True)
