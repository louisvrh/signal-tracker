"""Eurostat fetcher (SDMX 2.1 dissemination API, no key).

Each configured series is a dataset plus a dot-separated key whose `{geo}`
placeholder is replaced by the requested countries joined with `+`, e.g.
`sts_inpr_m` / `M.PRD.B-D.SCA.I21.DE+FR` (monthly industrial production,
industry excl. construction, seasonally and calendar adjusted, 2021=100).
Responses are SDMX-CSV.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from tracker import store
from tracker.fetchers.base import BaseFetcher
from tracker.fetchers.sdmx import parse_sdmx_csv


def month(d: date) -> str:
    """YYYY-MM for SDMX startPeriod/endPeriod."""
    return f"{d.year:04d}-{d.month:02d}"


class EurostatFetcher(BaseFetcher):
    """Monthly series from Eurostat's SDMX API."""

    source = "eurostat"
    min_interval_s = 1.0

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        frames = [self._fetch_series(spec, start, end) for spec in self.config.get("series", [])]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    def _fetch_series(self, spec: dict[str, Any], start: date, end: date) -> pd.DataFrame:
        geo = spec["geo"]  # ISO3 -> Eurostat geo code
        key = spec["key"].format(geo="+".join(geo[c] for c in sorted(geo)))
        url = f"{self.config['base_url']}/data/{spec['dataset']}/{key}"
        params = {"format": "SDMX-CSV", "startPeriod": month(start), "endPeriod": month(end)}
        text = self.get(url, params, raw_name=f"{spec['dataset']}_{spec['indicator']}", raw_ext="csv").text
        data = parse_sdmx_csv(text, "geo", {v: k for k, v in geo.items()})
        self.log.info("eurostat: %s: %d observations", spec["indicator"], len(data))
        return self.frame(spec["indicator"], data, lag_by_country=self.config.get("publication_lag_days_by_country"))
