"""Indicator registry and country reference (PLAN §4.3–4.4).

`load_registry()` reads `config/indicators.yaml`, `config/countries.yaml` and
`config/sources.yaml`, validates them and cross-checks them against each other.
Every problem is reported in a single RegistryError that names the file, the
entry and the field.
"""

from __future__ import annotations

import re
import zoneinfo
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import holidays
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from tracker import settings

Frequency = Literal["D", "W", "M", "Q"]
Category = Literal["trade", "industry", "energy", "market", "official", "weather"]
Role = Literal["alt", "official", "market", "control"]
MovingHoliday = Literal["lunar_new_year", "ramadan_eid", "easter", "diwali", "golden_week"]
Transform = Literal["yoy_28d", "yoy_m", "yoy_q", "level", "return_3m"]
# Groups of indicators a country can be excluded from in signals (PLAN D22).
SignalGroup = Literal["power_demand", "power_generation"]
# Which periods a series should have: every period, or Monday-Friday only.
Calendar = Literal["all", "weekdays"]
SEASONALITY_VALUES = {"weekly", "annual", "lunar_new_year", "ramadan_eid", "easter", "diwali", "golden_week"}

WORLD = "WLD"
CHOKEPOINT_RE = re.compile(r"^CHK_[A-Z0-9_]+$")


class RegistryError(ValueError):
    """Raised when a config file is invalid; the message lists every problem."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Break(_Model):
    """A known structural break: from `date` on, the series is measured differently
    (a definition or reporting change), so values before and after aren't comparable."""

    date: date
    reason: str = Field(min_length=1)


class Retired(_Model):
    """An indicator no longer fetched from `date` on. Its stored history stays readable,
    but it is never reported stale or missing, and its fetcher skips it (PLAN D56)."""

    date: date
    reason: str = Field(min_length=1)
    replaced_by: str | None = None


class Indicator(_Model):
    """One indicator, defined once in config/indicators.yaml."""

    id: str = Field(pattern=r"^[a-z0-9]+_[a-z0-9_]+$")
    source: str = Field(pattern=r"^[a-z0-9]+$")
    name: str
    description: str = ""
    unit: str
    frequency: Frequency
    category: Category
    role: Role
    higher_is_stronger: bool
    default_transform: Transform
    seasonality: list[str] = []
    publication_lag_days: int = Field(ge=0)
    refetch_window_days: int = Field(ge=0)
    countries: list[str] | Literal["watchlist"]
    signal_group: SignalGroup | None = None
    # Covered countries with no data by nature (e.g. no nuclear plants): not
    # reported as missing, but still stored if data ever appears.
    no_data_expected: list[str] = []
    calendar: Calendar = "all"
    # Runs of up to this many missing periods are normal (e.g. exchange holidays).
    gap_tolerance: int = Field(default=0, ge=0)
    # Gaps the source itself has and will never fill: country -> [[first, last]]
    # missing period (dates); the gap check doesn't report gaps inside them.
    known_gaps: dict[str, list[tuple[date, date]]] = {}
    # Known structural breaks: country -> [{date, reason}]. YoY and robust
    # z-scores never compare values across one; charts mark them (PLAN D41).
    breaks: dict[str, list[Break]] = {}
    # No longer fetched; history kept (PLAN D56).
    retired: Retired | None = None
    notes: str = ""

    @field_validator("seasonality")
    @classmethod
    def _known_seasonality(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - SEASONALITY_VALUES)
        if unknown:
            raise ValueError(f"unknown seasonality {unknown}; allowed: {sorted(SEASONALITY_VALUES)}")
        return value

    @model_validator(mode="after")
    def _id_matches_source(self) -> "Indicator":
        if not self.id.startswith(f"{self.source}_"):
            raise ValueError(f"id '{self.id}' must start with its source '{self.source}_'")
        return self

    @model_validator(mode="after")
    def _known_gaps_ordered(self) -> "Indicator":
        for country, spans in self.known_gaps.items():
            for first, last in spans:
                if first > last:
                    raise ValueError(f"known_gaps for {country}: {first} is after {last}")
        return self


class City(_Model):
    """A weather city with its population weight within the country."""

    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    weight: float = Field(gt=0, le=1)


class Country(_Model):
    """One watchlist country (PLAN §4.4)."""

    iso3: str = Field(pattern=r"^[A-Z]{3}$")
    name: str
    un_m49: int = Field(ge=1, le=999)
    comtrade_code: int | None = Field(default=None, ge=1, le=999)
    entsoe_code: str | None = None
    timezone: str
    weather_cities: list[City] = Field(min_length=1, max_length=5)
    country_etf: str
    fx_pair: str | None = None
    holiday_calendar: str
    moving_holidays: list[MovingHoliday] = []
    exclude_from_signals: list[SignalGroup] = []

    @property
    def comtrade_reporter(self) -> int:
        """Comtrade reporter code (differs from M49 for a few countries)."""
        return self.comtrade_code or self.un_m49

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            zoneinfo.ZoneInfo(value)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA time zone '{value}'") from exc
        return value

    @field_validator("holiday_calendar")
    @classmethod
    def _valid_calendar(cls, value: str) -> str:
        if value not in holidays.list_supported_countries():
            raise ValueError(f"'{value}' is not a country code supported by the holidays package")
        return value

    @field_validator("fx_pair")
    @classmethod
    def _fx_format(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Z]{6}", value):
            raise ValueError(f"fx_pair '{value}' must look like USDEUR (six capital letters)")
        return value

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "Country":
        total = sum(c.weight for c in self.weather_cities)
        if abs(total - 1.0) > 0.011:
            raise ValueError(f"weather_cities weights sum to {total:.3f}, expected 1.0")
        return self


class Chokepoint(_Model):
    """A shipping chokepoint, stored with a `CHK_<NAME>` country code."""

    code: str = Field(pattern=r"^CHK_[A-Z0-9_]+$")
    name: str
    portwatch_id: str | None = None


@dataclass(frozen=True)
class Registry:
    """Validated indicators, countries and chokepoints."""

    indicators: dict[str, Indicator]
    countries: dict[str, Country]
    chokepoints: dict[str, Chokepoint] = field(default_factory=dict)

    def watchlist(self) -> list[str]:
        """ISO3 codes of every watchlist country, in file order."""
        return list(self.countries)

    def indicator(self, indicator_id: str) -> Indicator:
        """Return one indicator, with a clear error if it isn't registered."""
        try:
            return self.indicators[indicator_id]
        except KeyError:
            raise KeyError(f"indicator '{indicator_id}' is not defined in config/indicators.yaml") from None

    def by_source(self, source: str) -> list[Indicator]:
        """All indicators belonging to a source."""
        return [i for i in self.indicators.values() if i.source == source]

    def countries_for(self, indicator: Indicator | str) -> list[str]:
        """Country codes an indicator covers (`watchlist` expanded)."""
        ind = self.indicator(indicator) if isinstance(indicator, str) else indicator
        return self.watchlist() if ind.countries == "watchlist" else list(ind.countries)

    def is_excluded(self, indicator: Indicator | str, country: str) -> bool:
        """True if signals for `country` must not use `indicator`, because the
        country is excluded from the indicator's `signal_group` (PLAN D22)."""
        ind = self.indicator(indicator) if isinstance(indicator, str) else indicator
        ref = self.countries.get(country)
        return ind.signal_group is not None and ref is not None and ind.signal_group in ref.exclude_from_signals

    def signal_countries(self, indicator: Indicator | str) -> list[str]:
        """Countries whose signals may use `indicator`: `countries_for` minus
        exclusions. Fetchers use `countries_for`; signal code uses this."""
        ind = self.indicator(indicator) if isinstance(indicator, str) else indicator
        return [c for c in self.countries_for(ind) if not self.is_excluded(ind, c)]

    def breaks_for(self, indicator: Indicator | str, country: str) -> list[Break]:
        """Known structural breaks of one series (indicators.yaml `breaks`), oldest first."""
        ind = self.indicator(indicator) if isinstance(indicator, str) else indicator
        return sorted(ind.breaks.get(country, []), key=lambda b: b.date)


def _format_errors(file: str, label: str, exc: ValidationError) -> list[str]:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(entry)"
        got = err.get("input")
        got_text = "" if isinstance(got, dict) else f" (got {got!r})"
        lines.append(f"{file}: {label}: field '{loc}': {err['msg']}{got_text}")
    return lines


def _entries(data: dict[str, Any], key: str, file: str, errors: list[str]) -> list[Any]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{file}: top-level '{key}' must be a list")
        return []
    return value


def _parse(model: type[_Model], entries: list[Any], id_field: str, file: str, kind: str, errors: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for n, entry in enumerate(entries, start=1):
        ident = entry.get(id_field, "?") if isinstance(entry, dict) else "?"
        label = f"{kind} #{n} ('{ident}')"
        if not isinstance(entry, dict):
            errors.append(f"{file}: {label}: must be a mapping of fields")
            continue
        try:
            obj = model.model_validate(entry)
        except ValidationError as exc:
            errors.extend(_format_errors(file, label, exc))
            continue
        key = getattr(obj, id_field)
        if key in parsed:
            errors.append(f"{file}: {label}: duplicate {id_field} '{key}'")
            continue
        parsed[key] = obj
    return parsed


def _referenced_indicators(node: Any) -> list[str]:
    """Every value stored under an `indicator` or `*_indicator` key anywhere
    in sources.yaml."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if (key == "indicator" or str(key).endswith("_indicator")) and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_referenced_indicators(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_referenced_indicators(item))
    return found


def load_registry(config_dir: Path | None = None) -> Registry:
    """Load, validate and cross-check the registry config files."""
    config_dir = config_dir or settings.CONFIG_DIR
    errors: list[str] = []

    countries_data = settings.load_config("countries", config_dir)
    countries = _parse(Country, _entries(countries_data, "countries", "countries.yaml", errors),
                       "iso3", "countries.yaml", "country", errors)
    chokepoints = _parse(Chokepoint, _entries(countries_data, "chokepoints", "countries.yaml", errors),
                         "code", "countries.yaml", "chokepoint", errors)

    ind_data = settings.load_config("indicators", config_dir)
    indicators = _parse(Indicator, _entries(ind_data, "indicators", "indicators.yaml", errors),
                        "id", "indicators.yaml", "indicator", errors)

    sources = settings.load_sources(config_dir)
    valid_codes = set(countries) | set(chokepoints) | {WORLD}
    for ind in indicators.values():
        if sources and ind.source not in sources:
            errors.append(f"indicators.yaml: indicator '{ind.id}': source '{ind.source}' is not in sources.yaml")
        if ind.countries != "watchlist":
            unknown = [c for c in ind.countries if c not in valid_codes]
            if unknown:
                errors.append(
                    f"indicators.yaml: indicator '{ind.id}': countries {unknown} are not in countries.yaml "
                    "(use ISO3 codes from the watchlist, WLD, or a chokepoint code)"
                )
        covered = list(countries) if ind.countries == "watchlist" else ind.countries
        for field_name, codes in (("no_data_expected", ind.no_data_expected), ("known_gaps", list(ind.known_gaps)),
                                  ("breaks", list(ind.breaks))):
            stray = [c for c in codes if c not in covered]
            if stray:
                errors.append(f"indicators.yaml: indicator '{ind.id}': {field_name} {stray} "
                              "are not among the indicator's countries")
        if ind.retired and ind.retired.replaced_by and ind.retired.replaced_by not in indicators:
            errors.append(f"indicators.yaml: indicator '{ind.id}': retired.replaced_by "
                          f"'{ind.retired.replaced_by}' is not in indicators.yaml")

    for name, cfg in sources.items():
        for ref in _referenced_indicators(cfg):
            if ref not in indicators:
                errors.append(f"sources.yaml: source '{name}' refers to indicator '{ref}', which is not in indicators.yaml")

    if errors:
        raise RegistryError("invalid registry config:\n  " + "\n  ".join(errors))
    return Registry(indicators=indicators, countries=countries, chokepoints=chokepoints)


def registry_counts(reg: Registry) -> dict[str, Any]:
    """Summary counts used by `python -m tracker status`."""
    by_source: dict[str, int] = {}
    by_role: dict[str, int] = {}
    for ind in reg.indicators.values():
        by_source[ind.source] = by_source.get(ind.source, 0) + 1
        by_role[ind.role] = by_role.get(ind.role, 0) + 1
    return {
        "indicators": len(reg.indicators),
        "countries": len(reg.countries),
        "chokepoints": len(reg.chokepoints),
        "by_source": dict(sorted(by_source.items())),
        "by_role": dict(sorted(by_role.items())),
    }
