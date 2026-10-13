"""Append-only observation store (PLAN §4.1–4.2).

Observations live in immutable Parquet files under
`data/lake/observations/source=<source>/<run_id>.parquet` and are queried with
DuckDB. Two rules make the store point-in-time correct:

* **Write rule** (`append_changes`): a row is written only if its
  (date, country, indicator) is new or its value differs from what is already
  stored. Revisions become new rows, so the full revision history is kept.
* **Read rule** (`latest`): returns the newest vintage per key; with `as_of`,
  only rows whose `available_at <= as_of` are considered.

Supporting tables (fetch_log, quality_issues, ...) are Parquet files under
`data/meta/<table>/`, one file per write.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tracker import settings

log = logging.getLogger(__name__)

SCHEMA = pa.schema(
    [
        ("date", pa.date32()),
        ("country", pa.string()),
        ("indicator", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        ("frequency", pa.string()),
        ("source", pa.string()),
        ("fetched_at", pa.timestamp("us", tz="UTC")),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("vintage_type", pa.string()),
    ]
)
COLUMNS: list[str] = SCHEMA.names
KEY: list[str] = ["date", "country", "indicator"]
KEY_ORDER: list[str] = ["indicator", "country", "date"]  # the order `latest` returns rows in
STRING_COLUMNS = ["country", "indicator", "unit", "frequency", "source", "vintage_type"]
TIMESTAMP_COLUMNS = ["fetched_at", "available_at"]
FREQUENCIES = {"D", "W", "M", "Q"}
VINTAGE_TYPES = {"live", "source_vintage", "estimated"}

SOURCE_RE = re.compile(r"^[a-z0-9]+$")
COUNTRY_RE = re.compile(r"^([A-Z]{3}|CHK_[A-Z0-9_]+)$")
INDICATOR_RE = re.compile(r"^[a-z0-9]+_[a-z0-9_]+$")

# Values closer than this (relative) are treated as unchanged, so float noise
# from recomputation never creates a fake revision.
REL_TOL = 1e-9
ABS_TOL = 1e-12

_TS = pa.timestamp("us", tz="UTC")
META_SCHEMAS: dict[str, pa.Schema] = {
    "fetch_log": pa.schema([
        ("run_id", pa.string()), ("source", pa.string()), ("started_at", _TS), ("finished_at", _TS),
        ("status", pa.string()), ("rows_fetched", pa.int64()), ("rows_written", pa.int64()),
        ("error", pa.string()),
    ]),
    "quality_issues": pa.schema([
        ("checked_at", _TS), ("source", pa.string()), ("indicator", pa.string()), ("country", pa.string()),
        ("check", pa.string()), ("severity", pa.string()), ("detail", pa.string()),
    ]),
    # Every backtest run, failed ones included (PLAN §4.2, §5.6). parameters and
    # metrics are JSON; uses_vault marks the one allowed look at the vault.
    "hypotheses_tested": pa.schema([
        ("tested_at", _TS), ("hypothesis_id", pa.string()), ("variant", pa.string()),
        ("parameters", pa.string()), ("train_period", pa.string()), ("test_period", pa.string()),
        ("metrics", pa.string()), ("status", pa.string()), ("uses_vault", pa.bool_()), ("notes", pa.string()),
    ]),
    # Every Telegram digest sent (S21): what it reported, so the next one can say what is new.
    "digests": pa.schema([
        ("sent_at", _TS), ("kind", pa.string()), ("failed_steps", pa.string()), ("dashboard", pa.string()),
        ("state", pa.string()), ("chars", pa.int64()), ("message_id", pa.int64()),
    ]),
}
META_TABLES: dict[str, list[str]] = {name: schema.names for name, schema in META_SCHEMAS.items()}


class SchemaError(ValueError):
    """Raised when a DataFrame does not match the observation schema."""


# --------------------------------------------------------------------------- paths


def _data_dir(data_dir: Path | None) -> Path:
    return Path(data_dir) if data_dir is not None else settings.DATA_DIR


def observations_dir(data_dir: Path | None = None) -> Path:
    """Directory holding one `source=<name>` folder per source."""
    return _data_dir(data_dir) / "lake" / "observations"


def source_dir(source: str, data_dir: Path | None = None) -> Path:
    """Directory holding the Parquet files of one source."""
    return observations_dir(data_dir) / f"source={source}"


def meta_dir(table: str, data_dir: Path | None = None) -> Path:
    """Directory holding the Parquet files of one supporting table."""
    return _data_dir(data_dir) / "meta" / table


def _parquet_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.parquet")) if folder.exists() else []


def _all_source_files(data_dir: Path | None, sources: Iterable[str] | None = None) -> list[Path]:
    root = observations_dir(data_dir)
    if not root.exists():
        return []
    if sources is None:
        folders = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("source="))
    else:
        folders = [source_dir(s, data_dir) for s in sorted(set(sources))]
    return [f for folder in folders for f in _parquet_files(folder)]


def new_run_id(now: datetime | None = None) -> str:
    """Return a sortable, unique run id such as `20260927T070000Z_1a2b3c`."""
    stamp = _to_utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    return f"{stamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------- validation


def empty_frame() -> pd.DataFrame:
    """An empty observations DataFrame with the standard columns and dtypes."""
    return _normalise(pd.DataFrame({c: [] for c in COLUMNS}))


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to the canonical pandas dtypes (no validation)."""
    out = df[COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[us]")
    out["value"] = out["value"].astype("float64")
    for col in STRING_COLUMNS:
        out[col] = out[col].astype("str")
    for col in TIMESTAMP_COLUMNS:
        out[col] = pd.to_datetime(out[col], utc=True).astype("datetime64[us, UTC]")
    return out.reset_index(drop=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check `df` against the observation schema and return a normalised copy.

    Raises SchemaError with a message naming the offending column and a
    sample of the bad values.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLUMNS]
    if missing or extra:
        raise SchemaError(f"columns do not match schema: missing={missing}, unexpected={extra}")
    if df.empty:
        return empty_frame()

    for col in COLUMNS:
        if df[col].isna().any():
            raise SchemaError(f"column '{col}' has {int(df[col].isna().sum())} missing values")

    out = df[COLUMNS].copy()
    try:
        out["date"] = pd.to_datetime(out["date"])
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"column 'date' could not be parsed as dates: {exc}") from exc
    if getattr(out["date"].dt, "tz", None) is not None:
        raise SchemaError("column 'date' must be a plain date, not a timezone-aware timestamp")
    has_time = out["date"] != out["date"].dt.normalize()
    if has_time.any():
        raise SchemaError(f"column 'date' must not have a time part, e.g. {out.loc[has_time, 'date'].iloc[0]}")
    out["date"] = out["date"].astype("datetime64[us]")

    try:
        out["value"] = pd.to_numeric(out["value"]).astype("float64")
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"column 'value' must be numeric: {exc}") from exc
    if not np.isfinite(out["value"]).all():
        raise SchemaError("column 'value' contains NaN or infinite values")

    for col in TIMESTAMP_COLUMNS:
        ts = pd.to_datetime(out[col])
        if getattr(ts.dt, "tz", None) is None:
            raise SchemaError(f"column '{col}' must be timezone-aware (UTC)")
        out[col] = ts.dt.tz_convert("UTC").dt.floor("us").astype("datetime64[us, UTC]")
    # A value we downloaded was public no later than the download itself.
    out["available_at"] = out[["available_at", "fetched_at"]].min(axis=1).astype("datetime64[us, UTC]")

    for col in STRING_COLUMNS:
        out[col] = out[col].astype("str")
        if (out[col].str.len() == 0).any():
            raise SchemaError(f"column '{col}' has empty strings")

    _check_values(out, "frequency", out["frequency"].isin(FREQUENCIES), f"must be one of {sorted(FREQUENCIES)}")
    _check_values(out, "vintage_type", out["vintage_type"].isin(VINTAGE_TYPES), f"must be one of {sorted(VINTAGE_TYPES)}")
    _check_values(out, "source", out["source"].str.fullmatch(SOURCE_RE), "must be lowercase letters/digits")
    _check_values(out, "country", out["country"].str.fullmatch(COUNTRY_RE), "must be ISO3, WLD or CHK_<NAME>")
    _check_values(out, "indicator", out["indicator"].str.fullmatch(INDICATOR_RE), "must look like <source>_<metric>")
    prefix_ok = pd.Series(
        [ind.startswith(f"{src}_") for ind, src in zip(out["indicator"], out["source"])], index=out.index
    )
    _check_values(out, "indicator", prefix_ok, "must start with '<source>_'")

    sv = out["vintage_type"] == "source_vintage"
    dup_other = out[~sv].duplicated(KEY, keep=False)
    if dup_other.any():
        sample = out[~sv][dup_other].head(3)[KEY].to_dict("records")
        raise SchemaError(f"duplicate (date, country, indicator) rows in one batch, e.g. {sample}")
    dup_sv = out[sv].duplicated(KEY + ["available_at"], keep=False)
    if dup_sv.any():
        sample = out[sv][dup_sv].head(3)[KEY + ["available_at"]].to_dict("records")
        raise SchemaError(f"duplicate source vintages (same key and available_at), e.g. {sample}")
    return out.reset_index(drop=True)


def _check_values(df: pd.DataFrame, col: str, ok: pd.Series, rule: str) -> None:
    ok = ok.fillna(False).astype(bool)
    if not ok.all():
        bad = sorted(set(df.loc[~ok, col].astype(str)))[:5]
        raise SchemaError(f"column '{col}' {rule}; bad values: {bad}")


# --------------------------------------------------------------------------- write


def _to_table(df: pd.DataFrame) -> pa.Table:
    arrays = {
        "date": pa.array(df["date"].to_numpy().astype("datetime64[D]"), type=pa.date32()),
        "value": pa.array(df["value"].to_numpy(), type=pa.float64()),
    }
    for col in STRING_COLUMNS:
        arrays[col] = pa.array(df[col].tolist(), type=pa.string())
    for col in TIMESTAMP_COLUMNS:
        arrays[col] = pa.array(df[col]).cast(pa.timestamp("us", tz="UTC"))
    return pa.table([arrays[c] for c in COLUMNS], schema=SCHEMA)


def _values_differ(a: pd.Series, b: pd.Series, rel_tol: float = REL_TOL) -> pd.Series:
    """True where values differ beyond float noise (NaN in `b` counts as differ)."""
    a_arr = a.to_numpy(dtype="float64")
    b_arr = b.to_numpy(dtype="float64")
    same = np.isclose(a_arr, b_arr, rtol=rel_tol, atol=ABS_TOL)
    return pd.Series(~same | np.isnan(b_arr), index=a.index)


def _select_changes(incoming: pd.DataFrame, existing: pd.DataFrame, rel_tol: float = REL_TOL) -> pd.DataFrame:
    """Return the incoming rows that are new or revised relative to `existing`.

    * `source_vintage` rows (true vintages, e.g. ALFRED) are placed on the
      key's vintage timeline; a row is kept if it is not already stored and its
      value differs from the vintage just before it.
    * `live` / `estimated` rows are compared with the latest stored value.
      A revision of an existing key cannot have been public before we saw it,
      so its `available_at` is raised to `fetched_at` and it becomes `live`.
    """
    sv_in = incoming[incoming["vintage_type"] == "source_vintage"]
    other_in = incoming[incoming["vintage_type"] != "source_vintage"]
    accepted: list[pd.DataFrame] = []

    if not sv_in.empty:
        ex = existing.merge(sv_in[KEY].drop_duplicates(), on=KEY, how="inner")
        timeline = pd.concat(
            [ex.assign(_origin=0), sv_in.assign(_origin=1)], ignore_index=True
        ).sort_values(KEY + ["available_at", "_origin", "fetched_at"], kind="mergesort")
        prev = timeline.groupby(KEY, sort=False)["value"].shift(1)
        changed = _values_differ(timeline["value"], prev, rel_tol)
        keep = (timeline["_origin"] == 1) & changed
        accepted.append(timeline.loc[keep, COLUMNS])

    if not other_in.empty:
        pool = pd.concat([existing] + accepted, ignore_index=True) if accepted else existing
        latest_rows = _latest_per_key(pool)[KEY + ["value"]].rename(columns={"value": "_prev"})
        merged = other_in.merge(latest_rows, on=KEY, how="left")
        changed = _values_differ(merged["value"], merged["_prev"], rel_tol)
        new_rows = merged.loc[changed].copy()
        revision = new_rows["_prev"].notna()
        bump = revision & (new_rows["available_at"] < new_rows["fetched_at"])
        new_rows.loc[bump, "available_at"] = new_rows.loc[bump, "fetched_at"]
        new_rows.loc[revision & (new_rows["vintage_type"] == "estimated"), "vintage_type"] = "live"
        accepted.append(new_rows[COLUMNS])

    if not accepted:
        return incoming.iloc[0:0]
    return pd.concat(accepted, ignore_index=True)


def _latest_per_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ordered = df.sort_values(KEY + ["available_at", "fetched_at"], kind="mergesort")
    return ordered.drop_duplicates(KEY, keep="last")


def _unique_path(folder: Path, stem: str) -> Path:
    path = folder / f"{stem}.parquet"
    n = 1
    while path.exists():
        path = folder / f"{stem}_{n}.parquet"
        n += 1
    return path


def append_changes(
    df: pd.DataFrame, run_id: str, data_dir: Path | None = None, rel_tol: float | None = None
) -> int:
    """Validate `df` and write only new or changed rows; return rows written.

    Rows are written to `data/lake/observations/source=<source>/<run_id>.parquet`
    (one file per source in `df`). Re-sending unchanged data writes nothing.
    `rel_tol` is the relative change below which a value counts as unchanged
    (default 1e-9); sources with noisy re-downloads can set it higher.
    """
    incoming = validate(df)
    if incoming.empty:
        return 0
    tol = REL_TOL if rel_tol is None else float(rel_tol)
    written = 0
    for source, part in incoming.groupby("source", sort=True):
        existing = _read_source_rows(str(source), sorted(part["indicator"].unique()), data_dir)
        changes = _select_changes(part, existing, tol)
        if changes.empty:
            log.info("store: %s: 0 of %d rows new or changed", source, len(part))
            continue
        folder = source_dir(str(source), data_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = _unique_path(folder, run_id)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(_to_table(changes), tmp)
        tmp.replace(path)
        written += len(changes)
        log.info("store: %s: wrote %d of %d rows to %s", source, len(changes), len(part), path.name)
    return written


# ---------------------------------------------------------------------------- read


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    return con


def _sql_file_list(files: list[Path]) -> str:
    quoted = ", ".join("'" + f.as_posix().replace("'", "''") + "'" for f in files)
    return f"read_parquet([{quoted}], union_by_name=true, hive_partitioning=false)"


def _to_utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _read_source_rows(source: str, indicators: list[str], data_dir: Path | None) -> pd.DataFrame:
    """All stored vintages for the given indicators of one source."""
    files = _parquet_files(source_dir(source, data_dir))
    if not files:
        return empty_frame()
    con = _connect()
    try:
        df = con.execute(
            f"SELECT {', '.join(COLUMNS)} FROM {_sql_file_list(files)} WHERE list_contains(?, indicator)",
            [indicators],
        ).df()
    finally:
        con.close()
    return _normalise(df)


def latest(
    indicators: Iterable[str] | None = None,
    countries: Iterable[str] | None = None,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    as_of: str | date | datetime | pd.Timestamp | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Return the newest vintage of each (date, country, indicator).

    Args:
        indicators: indicator ids to include (all if None).
        countries: country codes to include (all if None).
        start, end: inclusive bounds on the observation `date`.
        as_of: only consider rows with `available_at <= as_of`, i.e. what was
            knowable at that instant. A naive value is read as UTC, so
            "2020-03-01" means 2020-03-01 00:00 UTC.
        data_dir: data root (defaults to `settings.DATA_DIR`).

    Returns:
        DataFrame with the standard columns, sorted by indicator, country, date.
        Empty (with the right columns) if nothing matches.
    """
    indicators = sorted(set(indicators)) if indicators is not None else None
    sources = {i.split("_", 1)[0] for i in indicators} if indicators is not None else None
    files = _all_source_files(data_dir, sources)
    if not files or indicators == []:
        return empty_frame()

    where, params = ["TRUE"], []
    if indicators is not None:
        where.append("list_contains(?, indicator)")
        params.append(indicators)
    if countries is not None:
        where.append("list_contains(?, country)")
        params.append(sorted(set(countries)))
    if start is not None:
        where.append("date >= ?::DATE")
        params.append(pd.Timestamp(start).date())
    if end is not None:
        where.append("date <= ?::DATE")
        params.append(pd.Timestamp(end).date())
    if as_of is not None:
        where.append("available_at <= ?::TIMESTAMPTZ")
        params.append(_to_utc(as_of).to_pydatetime())

    sql = f"""
        SELECT {', '.join(COLUMNS)} FROM (
            SELECT *, row_number() OVER (
                PARTITION BY date, country, indicator
                ORDER BY available_at DESC, fetched_at DESC
            ) AS _rn
            FROM {_sql_file_list(files)}
            WHERE {' AND '.join(where)}
        ) WHERE _rn = 1
        ORDER BY indicator, country, date
    """
    con = _connect()
    try:
        df = con.execute(sql, params).df()
    finally:
        con.close()
    return _normalise(df)


def history(
    indicators: Iterable[str] | None = None,
    countries: Iterable[str] | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Return every stored vintage (all revisions) for the given filters."""
    indicators = sorted(set(indicators)) if indicators is not None else None
    sources = {i.split("_", 1)[0] for i in indicators} if indicators is not None else None
    files = _all_source_files(data_dir, sources)
    if not files:
        return empty_frame()
    where, params = ["TRUE"], []
    if indicators is not None:
        where.append("list_contains(?, indicator)")
        params.append(indicators)
    if countries is not None:
        where.append("list_contains(?, country)")
        params.append(sorted(set(countries)))
    con = _connect()
    try:
        df = con.execute(
            f"SELECT {', '.join(COLUMNS)} FROM {_sql_file_list(files)} WHERE {' AND '.join(where)} "
            "ORDER BY indicator, country, date, available_at, fetched_at",
            params,
        ).df()
    finally:
        con.close()
    return _normalise(df)


def _microseconds(values: np.ndarray) -> np.ndarray:
    """Naive datetime64 values as int64 microseconds since the epoch."""
    return values.astype("datetime64[us]").view("int64")


class Vintages:
    """Every stored vintage of some series, held in memory, answering `latest(as_of=...)`.

    The in-memory twin of `store.latest()` for backtests, which ask for the
    data as known on hundreds of dates: one DuckDB query per date would read
    the Parquet files hundreds of times. The read rule is the same (only rows
    with `available_at <= as_of`; the newest `available_at`, then `fetched_at`,
    wins per key), and tests check the two agree on every date. Build it with
    `Vintages.load()` (reads the lake once) or from a frame of vintages.
    """

    def __init__(self, rows: pd.DataFrame) -> None:
        rows = _normalise(rows) if len(rows) else empty_frame()
        self.rows = rows.sort_values(KEY_ORDER + ["available_at", "fetched_at"], kind="mergesort").reset_index(drop=True)
        keys = self.rows[KEY_ORDER]
        self._key_id = (keys.ne(keys.shift()).any(axis=1)).cumsum().to_numpy()
        self._available = _microseconds(self.rows["available_at"].dt.tz_localize(None).to_numpy())

    @classmethod
    def load(
        cls,
        indicators: Iterable[str] | None = None,
        countries: Iterable[str] | None = None,
        data_dir: Path | None = None,
    ) -> "Vintages":
        """Read every stored vintage of the given indicators and countries (see `history`)."""
        return cls(history(indicators, countries, data_dir))

    def __len__(self) -> int:
        return len(self.rows)

    def latest(
        self,
        as_of: str | date | datetime | pd.Timestamp | None = None,
        indicators: Iterable[str] | None = None,
        countries: Iterable[str] | None = None,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        """The newest vintage of each (date, country, indicator) known at `as_of`.

        Same arguments and result as `store.latest()`: only rows with
        `available_at <= as_of` count (a naive `as_of` is UTC), the result
        is sorted by indicator, country, date and has the standard columns.
        """
        keep = np.ones(len(self.rows), dtype=bool)
        if as_of is not None:
            limit = _microseconds(np.array([_to_utc(as_of).tz_localize(None).to_datetime64()]))[0]
            keep &= self._available <= limit
        if indicators is not None:
            keep &= self.rows["indicator"].isin(set(indicators)).to_numpy()
        if countries is not None:
            keep &= self.rows["country"].isin(set(countries)).to_numpy()
        if start is not None:
            keep &= (self.rows["date"] >= pd.Timestamp(start).normalize()).to_numpy()
        if end is not None:
            keep &= (self.rows["date"] <= pd.Timestamp(end).normalize()).to_numpy()
        index = np.flatnonzero(keep)
        if len(index) == 0:
            return empty_frame()
        ids = self._key_id[index]
        last = np.append(ids[1:] != ids[:-1], True)  # rows are in key, then vintage, order
        return self.rows.iloc[index[last]].reset_index(drop=True)



def source_summary(data_dir: Path | None = None) -> pd.DataFrame:
    """Per source and indicator: stored rows, keys and the latest data date."""
    files = _all_source_files(data_dir)
    cols = ["source", "indicator", "frequency", "rows", "series", "first_date", "last_date"]
    if not files:
        return pd.DataFrame(columns=cols)
    con = _connect()
    try:
        df = con.execute(
            f"""SELECT source, indicator, any_value(frequency) AS frequency, count(*) AS rows,
                       count(DISTINCT country) AS series, min(date) AS first_date, max(date) AS last_date
                FROM {_sql_file_list(files)} GROUP BY source, indicator ORDER BY source, indicator"""
        ).df()
    finally:
        con.close()
    return df[cols]


# --------------------------------------------------------------------- meta tables


_last_meta_stamp = pd.Timestamp(0, tz="UTC")


def _meta_stamp() -> str:
    """A file-name time stamp that increases with every call in this process.

    Meta files are read back in name order, and two writes can fall within the
    clock's resolution; without this their order would be left to the random suffix.
    """
    global _last_meta_stamp
    now = pd.Timestamp.now(tz="UTC").floor("us")
    _last_meta_stamp = max(now, _last_meta_stamp + pd.Timedelta(microseconds=1))
    return _last_meta_stamp.strftime("%Y%m%dT%H%M%S%fZ")


def append_meta(table: str, df: pd.DataFrame, data_dir: Path | None = None) -> Path | None:
    """Append rows to a supporting table under data/meta/<table>/ (one file per call)."""
    if df.empty:
        return None
    schema = META_SCHEMAS.get(table)
    df = df.reset_index(drop=True)
    if schema is not None:
        missing = [c for c in schema.names if c not in df.columns]
        if missing:
            raise SchemaError(f"{table}: missing columns {missing}")
        df = df[schema.names].copy()
        for field in schema:
            if pa.types.is_timestamp(field.type):
                df[field.name] = pd.to_datetime(df[field.name], utc=True)
            elif pa.types.is_string(field.type):
                df[field.name] = df[field.name].astype(object).where(df[field.name].notna(), None)
        arrow = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)
    else:
        arrow = pa.Table.from_pandas(df, preserve_index=False)
    folder = meta_dir(table, data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = _unique_path(folder, f"{_meta_stamp()}_{uuid.uuid4().hex[:6]}")
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(arrow, tmp)
    tmp.replace(path)
    return path


def read_meta(table: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read every row of a supporting table (empty DataFrame if none)."""
    files = _parquet_files(meta_dir(table, data_dir))
    if not files:
        return pd.DataFrame(columns=META_TABLES.get(table, []))
    con = _connect()
    try:
        return con.execute(f"SELECT * FROM {_sql_file_list(files)}").df()
    finally:
        con.close()


def append_fetch_log(df: pd.DataFrame, data_dir: Path | None = None) -> Path | None:
    """Append fetch_log rows (run_id, source, started_at, finished_at, status, ...)."""
    return append_meta("fetch_log", df, data_dir)


def read_fetch_log(data_dir: Path | None = None) -> pd.DataFrame:
    """Return the full fetch_log, oldest first."""
    df = read_meta("fetch_log", data_dir)
    return df.sort_values("started_at", kind="mergesort").reset_index(drop=True) if not df.empty else df


def append_quality_issues(df: pd.DataFrame, data_dir: Path | None = None) -> Path | None:
    """Append quality_issues rows (checked_at, source, indicator, country, check, ...)."""
    return append_meta("quality_issues", df, data_dir)


def read_quality_issues(data_dir: Path | None = None) -> pd.DataFrame:
    """Return every recorded quality issue."""
    return read_meta("quality_issues", data_dir)


def append_hypotheses_tested(df: pd.DataFrame, data_dir: Path | None = None) -> Path | None:
    """Log backtest runs (rows from `tracker.backtest.pit.test_record`), failed ones included."""
    return append_meta("hypotheses_tested", df, data_dir)


def read_hypotheses_tested(data_dir: Path | None = None) -> pd.DataFrame:
    """Every logged backtest run, oldest first."""
    df = read_meta("hypotheses_tested", data_dir)
    return df.sort_values("tested_at", kind="mergesort").reset_index(drop=True) if not df.empty else df


def append_digest(df: pd.DataFrame, data_dir: Path | None = None) -> Path | None:
    """Log a sent digest (sent_at, kind, failed_steps, dashboard, state JSON, chars, message_id)."""
    return append_meta("digests", df, data_dir)


def read_digests(data_dir: Path | None = None) -> pd.DataFrame:
    """Every digest sent, oldest first."""
    df = read_meta("digests", data_dir)
    return df.sort_values("sent_at", kind="mergesort").reset_index(drop=True) if not df.empty else df


# ------------------------------------------------------------------ signal tables


def signal_table_path(name: str, data_dir: Path | None = None) -> Path:
    """Path of a derived table under data/signals/ (e.g. `stl` -> data/signals/stl.parquet)."""
    return _data_dir(data_dir) / "signals" / f"{name}.parquet"


def write_signal_table(name: str, df: pd.DataFrame, data_dir: Path | None = None) -> Path:
    """Replace a derived table under data/signals/.

    Signal tables are recomputed from the lake, so unlike the lake they are
    overwritten, not appended to. The file is written to `*.tmp` and renamed,
    so readers never see a half-written table.
    """
    path = signal_table_path(name, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False), tmp)
    tmp.replace(path)
    return path


def read_signal_table(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read a derived table from data/signals/ (empty DataFrame if it doesn't exist yet)."""
    path = signal_table_path(name, data_dir)
    return pq.read_table(path).to_pandas() if path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------- compaction


def _compact_folder(folder: Path, sort_by: list[str] | None) -> tuple[int, int]:
    files = _parquet_files(folder)
    if len(files) < 2:
        return len(files), 0
    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables, promote_options="permissive")
    if sort_by:
        table = table.sort_by([(c, "ascending") for c in sort_by if c in table.column_names])
    expected = sum(t.num_rows for t in tables)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    target = _unique_path(folder, f"compacted_{stamp}")
    tmp = target.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    if pq.read_metadata(tmp).num_rows != expected:
        tmp.unlink()
        raise RuntimeError(f"compaction of {folder} lost rows; nothing was changed")
    tmp.replace(target)  # new file first, so a crash can only leave duplicates
    for f in files:
        f.unlink()
    return len(files), expected


def compact(data_dir: Path | None = None, include_meta: bool = True) -> pd.DataFrame:
    """Merge each source's Parquet files into one file, keeping every row.

    Returns one row per folder with the number of files merged and rows kept.
    Query results are unchanged because no row is added, altered or removed.
    """
    report = []
    root = observations_dir(data_dir)
    folders = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    for folder in folders:
        n_files, rows = _compact_folder(folder, ["indicator", "country", "date", "available_at"])
        report.append({"folder": folder.name, "files_merged": n_files if rows else 0, "rows": rows})
    if include_meta:
        for table in META_TABLES:
            n_files, rows = _compact_folder(meta_dir(table, data_dir), None)
            report.append({"folder": f"meta/{table}", "files_merged": n_files if rows else 0, "rows": rows})
    return pd.DataFrame(report, columns=["folder", "files_merged", "rows"])
