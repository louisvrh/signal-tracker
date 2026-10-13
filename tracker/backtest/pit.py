"""Point-in-time backtest engine (S22; PLAN §5.1, §5.6, D32).

Rebuilds what was knowable on each test date and computes signals from that
alone. Four rules keep look-ahead out:

1. **Data as known then.** A test date's snapshot is
   `store.Vintages.latest(as_of=date)`: only values whose `available_at` is on
   or before that date, each at its newest vintage by then. Later periods and
   later revisions don't exist for that date.
2. **Recomputed, not sliced.** Every signal is recomputed from the snapshot on
   each test date and only its end value (the latest period known then) is
   kept (D32). A two-sided chain (STL, band-passes, full-sample ranks) is
   judged on the provisional end value it really had, never on the value it
   has now. Chains run exactly as written (no length conversion, D37), and
   never with the daily run's precomputed STL, which was fitted on today's data.
3. **Real-time history.** `real_time_history` rebuilds a signal's past from
   the end values recorded on earlier test dates, so a model fitted on a date
   sees each earlier value as it was then.
4. **Walk-forward with a vault.** `walk_forward` fits only on dates before each
   test date and keeps the vault (the most recent 12 months) out of every fold
   until the one final look, which `hypotheses_tested` records.

Everything here is pure: vintages, the registry and signal definitions in,
frames out. Reading the lake (`store.Vintages.load`) and writing the
`hypotheses_tested` log (`store.append_hypotheses_tested`) are the caller's job.

S19's composite scores and divergences plug in as **panel signals**: functions
`(snapshot, as_of, registry) -> rows` that see the whole snapshot (every
indicator and country). `available_panel_signals()` picks them up from
`tracker/signals/composite.py` and `divergence.py` once those define
`PANEL_SIGNALS`; until then there are none.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Protocol

import numpy as np
import pandas as pd

from tracker import store
from tracker.signals import chain as chains
from tracker.signals.moving_holidays import holiday_windows

if TYPE_CHECKING:
    from tracker.registry import Registry

VAULT_MONTHS = 12
DEGREE_DAYS = ("openmeteo_hdd18", "openmeteo_cdd22")  # the weather step's extra series (D34)
PANEL_MODULES = ("tracker.signals.composite", "tracker.signals.divergence")  # S19
PANEL_ATTRIBUTE = "PANEL_SIGNALS"
PIT_COLUMNS = ["as_of", "signal", "indicator", "country", "date", "value", "last_date", "provisional",
               "vintage_type", "components", "problem"]


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ----------------------------------------------------------------- test dates


def as_of_dates(start: Any, end: Any, freq: str = "W-MON") -> pd.DatetimeIndex:
    """Test dates from `start` to `end` at 00:00 UTC, e.g. every Monday (`W-MON`) or month start (`MS`)."""
    return pd.date_range(_utc(start).normalize(), _utc(end).normalize(), freq=freq)


def snapshot(vintages: store.Vintages, as_of: Any, indicators: Iterable[str] | None = None,
             countries: Iterable[str] | None = None) -> pd.DataFrame:
    """The data as known at `as_of`: the newest vintage of each value with `available_at <= as_of`."""
    return vintages.latest(as_of=_utc(as_of), indicators=indicators, countries=countries)


# -------------------------------------------------------------- chain signals


@dataclass(frozen=True)
class ChainSignal:
    """A signal defined by a transform chain, computed on each series of some indicators.

    Attributes:
        name: the signal's name in results (a hypothesis id, a preset name, ...).
        chain: a chain string (tracker/signals/chain.py), used exactly as written (D37).
        indicators: indicator ids the chain runs on.
        countries: country codes; None means every country the registry lets
            signals use for each indicator (`Registry.signal_countries`, D22).
    """

    name: str
    chain: str
    indicators: tuple[str, ...]
    countries: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        chains.parse(self.chain)  # a typo fails here, not on the 300th test date
        object.__setattr__(self, "indicators", tuple(self.indicators))
        if self.countries is not None:
            object.__setattr__(self, "countries", tuple(self.countries))


def preset_signals(presets: Mapping[str, str], indicators: Iterable[str],
                   countries: Iterable[str] | None = None) -> list[ChainSignal]:
    """One `ChainSignal` per saved chain ({name: chain string}, e.g. config/presets.yaml)."""
    indicators = tuple(indicators)
    countries = tuple(countries) if countries is not None else None
    return [ChainSignal(name, text, indicators, countries) for name, text in presets.items()]


def _fingerprint(*series: pd.Series) -> str:
    digest = hashlib.sha1()
    for s in series:
        digest.update(np.asarray(s.index.asi8).tobytes())
        digest.update(s.to_numpy(dtype="float64").tobytes())
    return digest.hexdigest()


def as_series(rows: pd.DataFrame) -> pd.Series:
    """Observation rows of one series (date, value) as a series indexed by date."""
    return pd.Series(rows["value"].to_numpy(), index=pd.DatetimeIndex(rows["date"])).sort_index()


@lru_cache(maxsize=4096)
def _windows(calendar: str, holidays: tuple[str, ...], first_year: int, last_year: int) -> pd.DataFrame:
    return holiday_windows(calendar, list(holidays), first_year, last_year)


def _context(registry: "Registry", indicator: str, country: str, values: pd.Series,
             extra: Mapping[str, Any]) -> chains.Context:
    """`chain.series_context` with the holiday windows memoised (they come from fixed calendars,
    known in advance, so reusing them across test dates reveals nothing)."""
    ref = registry.countries.get(country)
    windows = None
    if ref is not None and ref.moving_holidays and len(values):
        windows = _windows(ref.holiday_calendar, tuple(ref.moving_holidays), values.index.min().year,
                           values.index.max().year)
    return chains.context_for(registry, indicator, country, windows, extra)


def chain_end(chain: chains.Chain, values: pd.Series, ctx: chains.Context) -> tuple[pd.Timestamp | None, float, bool]:
    """A chain's end value on one series: (date, value, provisional).

    The date is that of the chain's last output value (usually the series'
    last date); `provisional` says a two-sided step would still revise it.
    Raises ChainError if the chain can't run on the series (e.g. too little
    history for STL).
    """
    result = chains.apply(chain, values, ctx)
    if result.values.empty:
        return None, float("nan"), False
    date = result.values.index[-1]
    provisional = result.provisional_from is not None and date >= result.provisional_from
    return date, float(result.values.iloc[-1]), provisional


def series_signals(
    snap: pd.DataFrame,
    as_of: Any,
    signals: Iterable[ChainSignal],
    registry: "Registry",
    cache: dict[tuple[str, ...], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Every chain signal's end value on each series of one snapshot (PIT_COLUMNS rows).

    `snap` must be the snapshot at `as_of` (`snapshot`). A series whose chain
    can't run gets a row with no value and the reason in `problem`. `cache`
    (a dict kept across test dates) reuses results for series that haven't
    changed since an earlier date, e.g. monthly data tested weekly.
    """
    as_of = _utc(as_of)
    groups = {key: part for key, part in snap.groupby(["indicator", "country"], sort=False)}
    rows = []
    for sig in signals:
        chain = chains.parse(sig.chain)
        needs_weather = any(step.key == "wx" for step in chain.steps)
        for indicator in sig.indicators:
            countries = sig.countries if sig.countries is not None else registry.signal_countries(indicator)
            for country in countries:
                part = groups.get((indicator, country))
                if part is None or part.empty:
                    continue
                values = as_series(part)
                extra: dict[str, pd.Series] = {}
                if needs_weather:  # degree days as known at the same date
                    for dd in DEGREE_DAYS:
                        if (dd, country) in groups:
                            extra[dd] = as_series(groups[(dd, country)])
                key = (sig.chain, indicator, country, _fingerprint(values, *extra.values()))
                found = cache.get(key) if cache is not None else None
                if found is None:
                    found = {"date": pd.NaT, "value": float("nan"), "provisional": False, "problem": ""}
                    try:
                        ctx = _context(registry, indicator, country, values, extra)
                        date, value, provisional = chain_end(chain, values, ctx)
                        found.update(date=date if date is not None else pd.NaT, value=value, provisional=provisional)
                    except chains.ChainError as exc:
                        found["problem"] = str(exc)
                    if cache is not None:
                        cache[key] = found
                last = part.loc[part["date"].idxmax()]
                rows.append({"as_of": as_of, "signal": sig.name, "indicator": indicator, "country": country,
                             "last_date": last["date"], "vintage_type": last["vintage_type"],
                             "components": None, **found})
    return _frame(rows)


# -------------------------------------------------------------- panel signals


class PanelSignal(Protocol):
    """A signal computed from the whole snapshot: S19's composites and divergences plug in here.

    Called with the snapshot at `as_of` (every loaded indicator and country,
    newest vintage known then), the test date and the registry. Returns rows
    with columns `date`, `country`, `value` and optionally `components` (a
    JSON string). It may return the signal's whole history; the engine keeps
    each country's latest date only (its end value, D32). The same function
    serves the daily `signals` command, called with the latest snapshot.
    """

    def __call__(self, snapshot: pd.DataFrame, as_of: pd.Timestamp, registry: "Registry") -> pd.DataFrame: ...


def available_panel_signals() -> dict[str, PanelSignal]:
    """S19's panel signals: the `PANEL_SIGNALS` dicts ({name: function}) of
    tracker/signals/composite.py and divergence.py. Empty until they exist."""
    found: dict[str, PanelSignal] = {}
    for name in PANEL_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        found.update(getattr(module, PANEL_ATTRIBUTE, {}) or {})
    return found


def panel_end_values(snap: pd.DataFrame, as_of: Any, panel: Mapping[str, PanelSignal],
                     registry: "Registry") -> pd.DataFrame:
    """Each panel signal's end value per country on one snapshot (PIT_COLUMNS rows)."""
    as_of = _utc(as_of)
    rows = []
    for name, fn in panel.items():
        out = fn(snap, as_of, registry)
        if out is None or out.empty:
            continue
        out = out.dropna(subset=["value"]).sort_values("date")
        for country, part in out.groupby("country", sort=True):
            end = part.iloc[-1]
            rows.append({"as_of": as_of, "signal": name, "indicator": None, "country": country,
                         "date": pd.Timestamp(end["date"]), "value": float(end["value"]),
                         "last_date": pd.Timestamp(end["date"]), "provisional": False, "vintage_type": None,
                         "components": end.get("components"), "problem": ""})
    return _frame(rows)


# --------------------------------------------------------------------- engine


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=PIT_COLUMNS)
    df["as_of"] = pd.to_datetime(df["as_of"], utc=True)
    for col in ("date", "last_date"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[us]")
    df["value"] = df["value"].astype("float64")
    df["provisional"] = df["provisional"].astype(bool)
    return df


def compute(
    vintages: store.Vintages,
    registry: "Registry",
    dates: Iterable[Any],
    series: Iterable[ChainSignal] = (),
    panel: Mapping[str, PanelSignal] | None = None,
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
) -> pd.DataFrame:
    """Signals as known on each test date: one row per (test date, signal, series).

    For each date: the snapshot at that date, then every chain signal's end
    value on each series and every panel signal's end value per country.
    Nothing computed for one date is carried into another, except results
    for series whose data hasn't changed (the same input gives the same
    output).

    Returns:
        PIT_COLUMNS: as_of (the test date), signal, indicator (None for
        panel signals), country, date (of the end value), value, last_date
        (the latest period known), provisional (a two-sided step would still
        revise it), vintage_type (of the latest raw value; `estimated` flags
        backfilled history, PLAN §5.1), components (panel signals), problem
        (why a chain couldn't run).
    """
    series = list(series)
    panel = dict(panel or {})
    dates = [_utc(d) for d in dates]
    cache: dict[tuple[str, ...], dict[str, Any]] = {}
    parts = []
    for n, as_of in enumerate(dates, start=1):
        snap = snapshot(vintages, as_of)
        if series:
            parts.append(series_signals(snap, as_of, series, registry, cache))
        if panel:
            parts.append(panel_end_values(snap, as_of, panel, registry))
        if progress is not None:
            progress(n, len(dates), as_of)
    parts = [p for p in parts if not p.empty]
    if not parts:
        return _frame([])
    return pd.concat(parts, ignore_index=True)


def real_time_history(pit: pd.DataFrame, as_of: Any, signal: str, country: str,
                      indicator: str | None = None) -> pd.Series:
    """A signal's past as it was known in real time, up to `as_of`.

    For each period (`date`), the end value recorded on the last test date on
    or before `as_of` at which it was the signal's end. Indexed by period;
    this is what a model fitted at `as_of` may train on (D32).
    """
    rows = pit[(pit["signal"] == signal) & (pit["country"] == country) & (pit["as_of"] <= _utc(as_of))
               & pit["value"].notna()]
    if indicator is not None:
        rows = rows[rows["indicator"] == indicator]
    rows = rows.sort_values("as_of", kind="mergesort").drop_duplicates("date", keep="last")
    return pd.Series(rows["value"].to_numpy(), index=pd.DatetimeIndex(rows["date"]), name=signal).sort_index()


# ------------------------------------------------------------ walk-forward


@dataclass(frozen=True)
class Fold:
    """One walk-forward step: fit on `train`, predict at `test` (PLAN §5.6).

    Attributes:
        test: the test date; the model predicts with what is known then.
        train: earlier test dates whose data may fit the model's parameters and thresholds.
        in_vault: the test date is in the vault (only in the one final look).
    """

    test: pd.Timestamp
    train: pd.DatetimeIndex
    in_vault: bool = False


def vault_start(last_date: Any, months: int = VAULT_MONTHS) -> pd.Timestamp:
    """First day of the vault, the most recent `months` months before `last_date` (PLAN §5.6).

    Fix it once per hypothesis (e.g. from the data's last date when the test
    is first run) and pass the same date to every run, so the vault doesn't
    slide as data arrives.
    """
    last = pd.Timestamp(last_date)
    start = (last - pd.DateOffset(months=months)).normalize() + pd.Timedelta(days=1)
    return _utc(start)


def walk_forward(
    dates: Iterable[Any],
    vault_from: Any,
    min_train: int = 24,
    step: int = 1,
    embargo: int = 0,
    window: int | None = None,
    include_vault: bool = False,
) -> list[Fold]:
    """Walk-forward folds over test dates, respecting the vault.

    Each fold tests one date. Its training dates are the earlier dates (all
    of them, or the last `window` for a rolling window), minus the last
    `embargo` before the test date (e.g. a target's publication lag in
    periods, if training labels aren't taken from the test date's snapshot),
    and there must be at least `min_train`. Every `step`-th eligible date is
    tested.

    Dates on or after `vault_from` are neither tested nor trained on, unless
    `include_vault` (the hypothesis's one look at the vault, logged with
    `uses_vault=True`): then vault dates are tested too, each trained on the
    dates before it.
    """
    all_dates = pd.DatetimeIndex(sorted({_utc(d) for d in dates}))
    vault_from = _utc(vault_from)
    usable = all_dates if include_vault else all_dates[all_dates < vault_from]
    folds = []
    tested = 0
    for i, test in enumerate(usable):
        train = usable[:max(i - embargo, 0)]
        if window is not None:
            train = train[-window:]
        if len(train) < min_train:
            continue
        if tested % step == 0:
            folds.append(Fold(test=test, train=train, in_vault=bool(test >= vault_from)))
        tested += 1
    return folds


def split_halves(folds: list[Fold]) -> tuple[list[Fold], list[Fold]]:
    """The out-of-sample folds (vault excluded) in two halves by test date, for the
    graduation rule "beats its benchmark in both halves" (PLAN §5.6)."""
    oos = [f for f in folds if not f.in_vault]
    middle = len(oos) // 2
    return oos[:middle], oos[middle:]


# ------------------------------------------------------- hypotheses_tested


def _period(dates: Iterable[Any] | tuple[Any, Any] | None) -> str:
    stamps = [pd.Timestamp(d) for d in (dates if dates is not None else [])]
    if not stamps:
        return ""
    return f"{min(stamps):%Y-%m-%d}/{max(stamps):%Y-%m-%d}"


def hypothesis_record(
    hypothesis_id: str,
    variant: str,
    parameters: Mapping[str, Any],
    train_period: Iterable[Any] | None,
    test_period: Iterable[Any] | None,
    metrics: Mapping[str, Any],
    status: str = "ok",
    uses_vault: bool = False,
    notes: str = "",
    tested_at: Any = None,
) -> pd.DataFrame:
    """One `hypotheses_tested` row for `store.append_hypotheses_tested` (PLAN §4.2).

    Log every run, failed ones too (`status="failed"`, the error in `notes`).
    `parameters` and `metrics` are stored as JSON; periods as `first/last`
    dates of the dates given (e.g. the folds' training and test dates).
    """
    return pd.DataFrame([{
        "tested_at": _utc(tested_at) if tested_at is not None else pd.Timestamp.now(tz="UTC"),
        "hypothesis_id": hypothesis_id,
        "variant": variant,
        "parameters": json.dumps(dict(parameters), sort_keys=True, default=str),
        "train_period": _period(train_period),
        "test_period": _period(test_period),
        "metrics": json.dumps(dict(metrics), sort_keys=True, default=str),
        "status": status,
        "uses_vault": bool(uses_vault),
        "notes": notes,
    }])


def vault_looks(log: pd.DataFrame, hypothesis_id: str) -> pd.DataFrame:
    """Logged runs of a hypothesis that used the vault. Each hypothesis gets exactly one
    (PLAN §5.6), so a vault run should check this is empty first."""
    if log.empty:
        return log
    return log[(log["hypothesis_id"] == hypothesis_id) & log["uses_vault"].astype(bool)]
