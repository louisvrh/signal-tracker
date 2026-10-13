"""OECD fetcher (SDMX REST API at sdmx.oecd.org, no key).

Each configured series is a dataflow plus a key whose `{area}` placeholder
is replaced by the ISO3 codes joined with `+`, e.g. the short-term statistics
dataflow `OECD.SDD.STES,DSD_STES@DF_INDSERV,` with key
`CAN+JPN.M.PRVM.IX.BTE.Y._Z._Z.N` (monthly production volume index,
industry excl. construction, seasonally adjusted). All countries go in one
request because the OECD allows only about 60 downloads per hour.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from tracker import store
from tracker.fetchers.base import BaseFetcher
from tracker.fetchers.eurostat import month
from tracker.fetchers.sdmx import parse_sdmx_csv


class OECDFetcher(BaseFetcher):
    """Monthly series from the OECD Data Explorer API."""

    source = "oecd"
    min_interval_s = 5.0  # ~60 requests/hour allowed; we need 2 per run
    backoff_min_s = 30.0  # a 429 means waiting for the hourly window
    backoff_max_s = 300.0
    max_attempts = 3

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        frames = [self._fetch_series(spec, start, end) for spec in self.config.get("series", [])]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    def _fetch_series(self, spec: dict[str, Any], start: date, end: date) -> pd.DataFrame:
        areas = sorted(spec["areas"])
        key = spec["key"].format(area="+".join(areas))
        url = f"{self.config['base_url']}/{spec['dataflow']}/{key}"
        params = {"startPeriod": month(start), "endPeriod": month(end), "format": "csvfile"}
        text = self.get(url, params, raw_name=spec["indicator"], raw_ext="csv").text
        data = parse_sdmx_csv(text, "REF_AREA", {a: a for a in areas})
        self.log.info("oecd: %s: %d observations", spec["indicator"], len(data))
        return self.frame(spec["indicator"], data, lag_by_country=self.config.get("publication_lag_days_by_country"))
