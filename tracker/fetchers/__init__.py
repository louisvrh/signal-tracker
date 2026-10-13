"""Fetchers, one module per source. `get_fetcher_class(name)` imports lazily so
a broken or missing optional dependency only affects its own source."""

from __future__ import annotations

import importlib

# source name -> "module:ClassName"
FETCHERS: dict[str, str] = {
    "fred": "tracker.fetchers.fred:FredFetcher",
    "portwatch": "tracker.fetchers.portwatch:PortWatchFetcher",
    "ember": "tracker.fetchers.ember:EmberFetcher",
    "eia": "tracker.fetchers.eia:EIAFetcher",
    "entsoe": "tracker.fetchers.entsoe:EntsoeFetcher",
    "neso": "tracker.fetchers.neso:NesoFetcher",
    "comtrade": "tracker.fetchers.comtrade:ComtradeFetcher",
    "imf": "tracker.fetchers.imf_trade:IMFTradeFetcher",
    "cpb": "tracker.fetchers.cpb:CPBFetcher",
    "eurostat": "tracker.fetchers.eurostat:EurostatFetcher",
    "oecd": "tracker.fetchers.oecd:OECDFetcher",
    "openmeteo": "tracker.fetchers.openmeteo:OpenMeteoFetcher",
    "prices": "tracker.fetchers.prices:PricesFetcher",
}


def get_fetcher_class(source: str) -> type:
    """Return the fetcher class for a source name."""
    try:
        target = FETCHERS[source]
    except KeyError:
        raise KeyError(f"no fetcher registered for source '{source}'") from None
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(module_name), class_name)
