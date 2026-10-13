"""IMF PortWatch fetcher (ArcGIS feature services, no key).

* Ports (`Daily_Ports_Data`): port calls and estimated import/export tonnes
  per port per day. We ask the server to sum them by (ISO3, date), so only
  country-level rows are downloaded.
* Chokepoints (`Daily_Chokepoints_Data`): daily transits per chokepoint,
  stored with `CHK_<NAME>` country codes from countries.yaml.

Both layers are updated weekly (Tuesdays); dates are UTC days.

Large queries are paged with `resultOffset` (1,000 rows a page; the ArcGIS
docs guarantee a stable order across pages for a constant where clause).
ArcGIS sometimes answers a heavy aggregated page with its generic "Unable to
perform query" error although the same query worked a page earlier, so that
error is retried with backoff. Progress is checkpointed page by page under
`data/raw/portwatch/resume/`, so a query that still fails can be resumed by
the next run with the same layer, countries and start date: it finishes the
original query (same end date) from the failed page, after re-downloading
the last saved page to check the data hasn't shifted in between.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from tracker import store
from tracker.fetchers.base import BaseFetcher, FetchError, RetryableHTTPError

PAGE_SIZE = 1000  # the layers' maxRecordCount
# ArcGIS error text that is worth retrying (intermittent on big aggregated queries).
RETRYABLE_ERRORS = ("unable to perform query",)


def _date_filter(start: date, end: date) -> str:
    return f"date >= DATE '{start.isoformat()}' AND date <= DATE '{end.isoformat()}'"


def _in_list(values: list[str]) -> str:
    return ",".join("'" + v.replace("'", "''") + "'" for v in values)


def _is_last_page(payload: dict[str, Any]) -> bool:
    features = payload.get("features", [])
    return not features or (len(features) < PAGE_SIZE and not payload.get("exceededTransferLimit"))


def to_long(attrs: pd.DataFrame, key_col: str, fields: list[dict[str, str]]) -> pd.DataFrame:
    """Wide ArcGIS attributes -> long rows (date, country, indicator, value).

    `key_col` holds the country code; `fields` maps PortWatch field -> indicator.
    Missing values are dropped.
    """
    if attrs.empty:
        return pd.DataFrame(columns=["date", "country", "indicator", "value"])
    frames = []
    for spec in fields:
        if spec["field"] not in attrs.columns:
            raise FetchError(f"portwatch: field '{spec['field']}' missing from response")
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(attrs["date"]),
            "country": attrs[key_col].astype(str),
            "indicator": spec["indicator"],
            "value": pd.to_numeric(attrs[spec["field"]], errors="coerce"),
        }))
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["value"]).reset_index(drop=True)


class PortWatchFetcher(BaseFetcher):
    """Daily country port activity and chokepoint transits."""

    source = "portwatch"
    min_interval_s = 0.5
    query_attempts = 5            # per page, for RETRYABLE_ERRORS
    query_backoff_min_s = 10.0    # waits 10, 20, 40, 80 s between attempts
    query_backoff_max_s = 120.0
    resume_max_age_days = 7       # older checkpoints are ignored (well inside the 60-day refetch window)

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        # Checkpoints of completed queries are kept until every query has
        # finished, so a failure in the chokepoint query doesn't mean
        # downloading all the port pages again.
        self._finished_checkpoints: list[Path] = []
        frames = []
        if self.config.get("ports"):
            frames.append(self._fetch_ports(start, end))
        if self.config.get("chokepoints"):
            frames.append(self._fetch_chokepoints(start, end))
        for checkpoint in self._finished_checkpoints:
            checkpoint.unlink(missing_ok=True)
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()

    # ---------------------------------------------------------------- queries

    def _query_all(
        self, layer: str, params: dict[str, Any], label: str, resume_key: dict[str, Any] | None = None,
        keep_checkpoint: bool = False,
    ) -> pd.DataFrame:
        """Run an ArcGIS query, following pagination; returns the attributes.

        `resume_key` identifies the query for resuming (e.g. countries and
        start date, but not the end date); by default the params themselves.
        The checkpoint is deleted once the query completes, unless
        `keep_checkpoint` (then `_fetch` deletes it after all queries).
        """
        url = f"{self.config['base_url']}/{layer}/FeatureServer/0/query"
        checkpoint = self._checkpoint_path(label, layer, resume_key if resume_key is not None else params)
        state = self._resume(checkpoint, url, label)
        if state is None:
            state = {"layer": layer, "params": params, "started_at": self.fetched_at.isoformat(), "pages": []}
            offset = 0
        else:
            params = state["params"]
            offset = state["pages"][-1]["offset"] + state["pages"][-1]["rows"]
        done = bool(state["pages"]) and state["pages"][-1]["last"]
        while not done:
            payload, path = self._query_page(url, params, offset, label)
            rows = len(payload.get("features", []))
            done = _is_last_page(payload)
            state["pages"].append({"offset": offset, "rows": rows, "last": done,
                                   "path": path.relative_to(self.raw_root).as_posix()})
            self._write_checkpoint(checkpoint, state)
            offset += rows
        attrs = [f["attributes"] for page in state["pages"] for f in self._read_page(page)]
        if keep_checkpoint:
            self._finished_checkpoints.append(checkpoint)
        else:
            checkpoint.unlink(missing_ok=True)  # complete: the next identical query downloads afresh
        return pd.DataFrame(attrs)

    def _query_page(self, url: str, params: dict[str, Any], offset: int, label: str) -> tuple[dict[str, Any], Path]:
        """One page, retrying ArcGIS's intermittent query errors; saves every
        response (errors too) as a raw copy and returns (payload, raw path)."""
        page = dict(params, f="json", resultOffset=offset, resultRecordCount=PAGE_SIZE)
        attempt = 0

        def once() -> tuple[dict[str, Any], Path]:
            nonlocal attempt
            attempt += 1
            response = self.get(url, page)
            payload = response.json()
            error = payload.get("error")
            name = f"{label}_{offset}" if not error else f"{label}_{offset}_error{attempt}"
            path = self.save_raw(name, response.content)
            if error:
                message = f"portwatch: {url.split('/')[-4]} query failed: {error}"
                if any(m in json.dumps(error).lower() for m in RETRYABLE_ERRORS):
                    self.log.warning("%s (page at offset %d, attempt %d of %d)", message, offset, attempt, self.query_attempts)
                    raise RetryableHTTPError(message)
                raise FetchError(message)
            return payload, path

        retrying = Retrying(
            stop=stop_after_attempt(self.query_attempts),
            wait=wait_exponential(multiplier=self.query_backoff_min_s, min=self.query_backoff_min_s,
                                  max=self.query_backoff_max_s),
            retry=retry_if_exception_type(RetryableHTTPError),
            reraise=True,
        )
        return retrying(once)

    # ------------------------------------------------------------ checkpoints

    def _checkpoint_path(self, label: str, layer: str, key: dict[str, Any]) -> Path:
        digest = hashlib.sha1(json.dumps([layer, key], sort_keys=True, default=str).encode()).hexdigest()[:16]
        return self.raw_root / "resume" / f"{label}_{digest}.json"

    def _write_checkpoint(self, path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(path)

    def _read_page(self, page: dict[str, Any]) -> list[dict[str, Any]]:
        return json.loads((self.raw_root / page["path"]).read_text(encoding="utf-8")).get("features", [])

    def _resume(self, checkpoint: Path, url: str, label: str) -> dict[str, Any] | None:
        """Load an interrupted query's checkpoint if it is recent, complete on
        disk and still matches the server; otherwise discard it (None)."""
        if not checkpoint.exists():
            return None
        try:
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            age = self.fetched_at - pd.Timestamp(state["started_at"])
            stored = [self._read_page(p) for p in state["pages"]]
        except (OSError, ValueError, KeyError) as exc:
            self.log.warning("portwatch: unreadable checkpoint %s (%s); starting the query over", checkpoint.name, exc)
            checkpoint.unlink(missing_ok=True)
            return None
        if not state["pages"] or age > pd.Timedelta(days=self.resume_max_age_days):
            checkpoint.unlink(missing_ok=True)
            return None
        last = state["pages"][-1]
        fresh, path = self._query_page(url, state["params"], last["offset"], label)
        if fresh.get("features", []) != stored[-1]:
            self.log.warning("portwatch: data changed since the interrupted %s query; starting it over", label)
            checkpoint.unlink(missing_ok=True)
            return None
        last["path"] = path.relative_to(self.raw_root).as_posix()
        self.log.warning("portwatch: resuming the interrupted %s query (started %s) after %d saved pages",
                         label, state["started_at"][:10], len(state["pages"]))
        return state

    # --------------------------------------------------------------- layers

    def _fetch_ports(self, start: date, end: date) -> pd.DataFrame:
        cfg = self.config["ports"]
        countries = sorted(set().union(*(self.registry.countries_for(f["indicator"]) for f in cfg["fields"])))
        stats = [
            {"statisticType": "sum", "onStatisticField": f["field"], "outStatisticFieldName": f["field"]}
            for f in cfg["fields"]
        ]
        params = {
            "where": f"ISO3 IN ({_in_list(countries)}) AND {_date_filter(start, end)}",
            "groupByFieldsForStatistics": "ISO3,date",
            "outStatistics": json.dumps(stats),
            "orderByFields": "ISO3,date",
        }
        resume_key = {"countries": countries, "start": start.isoformat(), "stats": stats}
        attrs = self._query_all(cfg["layer"], params, "ports", resume_key, keep_checkpoint=True)
        self.log.info("portwatch: %d country-days of port data", len(attrs))
        return self._to_frames(to_long(attrs, "ISO3", cfg["fields"]))

    def _fetch_chokepoints(self, start: date, end: date) -> pd.DataFrame:
        cfg = self.config["chokepoints"]
        id_to_code = {c.portwatch_id: c.code for c in self.registry.chokepoints.values() if c.portwatch_id}
        if not id_to_code:
            raise FetchError("portwatch: no chokepoints with portwatch_id in countries.yaml")
        out_fields = ["date", "portid"] + [f["field"] for f in cfg["fields"]]
        params = {
            "where": f"portid IN ({_in_list(sorted(id_to_code))}) AND {_date_filter(start, end)}",
            "outFields": ",".join(out_fields),
            "orderByFields": "portid,date",
            "returnGeometry": "false",
        }
        resume_key = {"ids": sorted(id_to_code), "start": start.isoformat(), "fields": out_fields}
        attrs = self._query_all(cfg["layer"], params, "chokepoints", resume_key, keep_checkpoint=True)
        self.log.info("portwatch: %d chokepoint-days", len(attrs))
        if not attrs.empty:
            attrs["code"] = attrs["portid"].map(id_to_code)
            attrs = attrs.dropna(subset=["code"])
        return self._to_frames(to_long(attrs, "code", cfg["fields"]))

    def _to_frames(self, long: pd.DataFrame) -> pd.DataFrame:
        frames = [self.frame(ind, part) for ind, part in long.groupby("indicator", sort=True)]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()
