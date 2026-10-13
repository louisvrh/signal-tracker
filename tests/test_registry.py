"""Tests for the indicator registry and country reference."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from tracker import settings
from tracker.registry import RegistryError, load_registry, registry_counts


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """A writable copy of the real config folder."""
    target = tmp_path / "config"
    shutil.copytree(settings.CONFIG_DIR, target)
    return target


def edit_yaml(path: Path, change) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def first_indicator(data: dict) -> dict:
    return data["indicators"][0]


def test_real_config_is_valid() -> None:
    reg = load_registry()
    counts = registry_counts(reg)
    assert counts["countries"] == len(reg.watchlist()) >= 20
    assert "fred_indpro" in reg.indicators
    assert counts["by_source"]["fred"] >= 1


def test_every_watchlist_country_is_complete() -> None:
    reg = load_registry()
    for country in reg.countries.values():
        assert 1 <= len(country.weather_cities) <= 5
        assert abs(sum(c.weight for c in country.weather_cities) - 1) < 0.011
        assert country.country_etf
    lny = {c for c, v in reg.countries.items() if "lunar_new_year" in v.moving_holidays}
    assert {"CHN", "TWN", "VNM", "KOR", "SGP", "MYS", "IDN"} <= lny


def test_every_fred_series_in_sources_has_an_indicator() -> None:
    reg = load_registry()
    for series in settings.load_sources()["fred"]["series"]:
        assert reg.indicator(series["indicator"]).source == "fred"


def test_countries_for_expands_watchlist(config_dir: Path) -> None:
    edit_yaml(config_dir / "indicators.yaml", lambda d: first_indicator(d).update(countries="watchlist"))
    reg = load_registry(config_dir)
    first = next(iter(reg.indicators.values()))
    assert reg.countries_for(first) == reg.watchlist()


def test_zaf_is_fetched_but_excluded_from_power_demand_signals() -> None:
    reg = load_registry()
    assert "ZAF" in reg.watchlist() and "ESP" in reg.watchlist()
    assert "ZAF" in reg.countries_for("ember_demand")  # still fetched and stored
    assert "ZAF" not in reg.signal_countries("ember_demand")
    assert reg.is_excluded("ember_demand", "ZAF") and not reg.is_excluded("ember_demand", "DEU")
    assert "ZAF" in reg.signal_countries("portwatch_exports")  # untagged indicators unaffected
    power = [i.id for i in reg.indicators.values() if i.signal_group == "power_demand"]
    assert set(power) == {"ember_demand", "eia_us48_load", "eia_us48_overnight_load", "eia_industrial_sales",
                          "entsoe_load", "entsoe_overnight_load", "neso_load", "neso_overnight_load"}
    generation = {i.id for i in reg.indicators.values() if i.signal_group == "power_generation"}
    assert generation == {"ember_generation"} | {f"ember_gen_{f}" for f in ("coal", "gas", "nuclear", "hydro", "wind", "solar")}
    assert all(reg.is_excluded(i, "ZAF") and not reg.is_excluded(i, "DEU") for i in generation)


def test_easter_countries_match_plan() -> None:
    reg = load_registry()
    easter = {c for c, v in reg.countries.items() if "easter" in v.moving_holidays}
    assert {"ZAF", "AUS", "CAN", "BRA", "MEX", "DEU", "GBR", "ESP"} <= easter
    assert not easter & {"USA", "CHN", "JPN", "IND"}


def test_no_data_expected_must_be_covered_countries(config_dir: Path) -> None:
    reg = load_registry()
    assert "ITA" in reg.indicator("ember_gen_nuclear").no_data_expected
    assert "ITA" in reg.countries_for("ember_gen_nuclear")  # still fetched, in case data appears
    edit_yaml(config_dir / "indicators.yaml", lambda d: first_indicator(d).update(no_data_expected=["DEU"]))
    with pytest.raises(RegistryError, match=r"no_data_expected \['DEU'\] are not among the indicator's countries"):
        load_registry(config_dir)


def test_exclusion_applies_to_any_country_in_the_group(config_dir: Path) -> None:
    edit_yaml(config_dir / "countries.yaml",
              lambda d: next(c for c in d["countries"] if c["iso3"] == "USA").update(exclude_from_signals=["power_demand"]))
    reg = load_registry(config_dir)
    assert reg.signal_countries("eia_us48_load") == []
    assert reg.countries_for("eia_us48_load") == ["USA"]
    assert "USA" in reg.signal_countries("fred_indpro")


def test_unknown_indicator_lookup_is_clear() -> None:
    with pytest.raises(KeyError, match="not defined in config/indicators.yaml"):
        load_registry().indicator("fred_nope")


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda d: first_indicator(d).update(category="industy"),
         "indicators.yaml: indicator #1 ('fred_indpro'): field 'category'"),
        (lambda d: first_indicator(d).update(frequency="Y"), "field 'frequency'"),
        (lambda d: first_indicator(d).update(publication_lag_days=-1), "field 'publication_lag_days'"),
        (lambda d: first_indicator(d).pop("unit"), "field 'unit': Field required"),
        (lambda d: first_indicator(d).update(colour="red"), "field 'colour': Extra inputs are not permitted"),
        (lambda d: first_indicator(d).update(id="ember_indpro"), "must start with its source 'fred_'"),
        (lambda d: first_indicator(d).update(countries=["XXX"]), "countries ['XXX'] are not in countries.yaml"),
        (lambda d: first_indicator(d).update(seasonality=["monthly"]), "unknown seasonality ['monthly']"),
        (lambda d: first_indicator(d).update(source="nosuch", id="nosuch_x"), "source 'nosuch' is not in sources.yaml"),
        (lambda d: first_indicator(d).update(signal_group="power"), "field 'signal_group'"),
        (lambda d: first_indicator(d).update(calendar="business"), "field 'calendar'"),
        (lambda d: first_indicator(d).update(gap_tolerance=-1), "field 'gap_tolerance'"),
        (lambda d: first_indicator(d).update(known_gaps={"USA": [["2020-02-01", "2020-01-01"]]}), "2020-02-01 is after 2020-01-01"),
        (lambda d: first_indicator(d).update(known_gaps={"DEU": [["2020-01-01", "2020-01-01"]]}),
         "known_gaps ['DEU'] are not among the indicator's countries"),
        (lambda d: d["indicators"].append(dict(first_indicator(d))), "duplicate id 'fred_indpro'"),
    ],
)
def test_indicator_errors_say_what_is_wrong(config_dir: Path, change, expected: str) -> None:
    edit_yaml(config_dir / "indicators.yaml", change)
    with pytest.raises(RegistryError) as exc:
        load_registry(config_dir)
    assert expected in str(exc.value)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda d: d["countries"][0].update(timezone="Mars/Olympus"), "unknown IANA time zone 'Mars/Olympus'"),
        (lambda d: d["countries"][0].update(holiday_calendar="XX"), "not a country code supported by the holidays"),
        (lambda d: d["countries"][0]["weather_cities"][0].update(weight=0.9), "weights sum to"),
        (lambda d: d["countries"][0]["weather_cities"][0].update(lat=123), "field 'weather_cities.0.lat'"),
        (lambda d: d["countries"][0].update(iso3="us"), "field 'iso3'"),
        (lambda d: d["countries"][0].update(fx_pair="EUR/USD"), "must look like USDEUR"),
        (lambda d: d["countries"][0].update(moving_holidays=["christmas"]), "field 'moving_holidays.0'"),
        (lambda d: d["countries"][0].update(exclude_from_signals=["weather"]), "field 'exclude_from_signals.0'"),
        (lambda d: d["countries"].append(dict(d["countries"][0])), "duplicate iso3 'USA'"),
    ],
)
def test_country_errors_say_what_is_wrong(config_dir: Path, change, expected: str) -> None:
    edit_yaml(config_dir / "countries.yaml", change)
    with pytest.raises(RegistryError) as exc:
        load_registry(config_dir)
    assert expected in str(exc.value)


def test_sources_must_reference_defined_indicators(config_dir: Path) -> None:
    edit_yaml(config_dir / "sources.yaml",
              lambda d: d["sources"]["fred"]["series"].append({"series_id": "X", "indicator": "fred_x"}))
    with pytest.raises(RegistryError, match="refers to indicator 'fred_x'"):
        load_registry(config_dir)


def test_all_errors_reported_together(config_dir: Path) -> None:
    edit_yaml(config_dir / "indicators.yaml", lambda d: first_indicator(d).update(category="x", role="y"))
    with pytest.raises(RegistryError) as exc:
        load_registry(config_dir)
    assert "field 'category'" in str(exc.value) and "field 'role'" in str(exc.value)


def test_chokepoint_codes_accepted(config_dir: Path) -> None:
    edit_yaml(config_dir / "countries.yaml",
              lambda d: d["chokepoints"].append({"code": "CHK_TEST", "name": "Test Strait"}))
    edit_yaml(config_dir / "indicators.yaml", lambda d: first_indicator(d).update(countries=["CHK_TEST", "WLD"]))
    reg = load_registry(config_dir)
    assert "CHK_TEST" in reg.chokepoints
    assert reg.countries_for("fred_indpro") == ["CHK_TEST", "WLD"]


def test_unregistered_chokepoint_rejected(config_dir: Path) -> None:
    edit_yaml(config_dir / "indicators.yaml", lambda d: first_indicator(d).update(countries=["CHK_NOWHERE"]))
    with pytest.raises(RegistryError, match="CHK_NOWHERE"):
        load_registry(config_dir)


def test_status_prints_registry_counts(capsys: pytest.CaptureFixture[str]) -> None:
    from tracker.__main__ import main

    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "indicators" in out and "countries" in out


def test_known_breaks_are_read_and_must_be_covered_countries(config_dir: Path) -> None:
    reg = load_registry()
    brk, unexplained = reg.breaks_for("entsoe_load", "NLD")  # the second found on 2026-10-07 (NOTES)
    assert str(brk.date) == "2016-01-01" and "TenneT" in brk.reason
    assert str(unexplained.date) == "2026-09-10" and "revises" in unexplained.reason
    assert reg.breaks_for("entsoe_overnight_load", "NLD") and reg.breaks_for("entsoe_load", "DEU") == []
    edit_yaml(config_dir / "indicators.yaml",
              lambda d: first_indicator(d).update(breaks={"DEU": [{"date": "2020-01-01", "reason": "test"}]}))
    with pytest.raises(RegistryError, match=r"breaks \['DEU'\] are not among the indicator's countries"):
        load_registry(config_dir)
    edit_yaml(config_dir / "indicators.yaml",
              lambda d: first_indicator(d).update(breaks={"USA": [{"date": "2020-01-01"}]}))
    with pytest.raises(RegistryError, match="reason"):
        load_registry(config_dir)


def test_retired_indicators_name_a_defined_replacement(config_dir: Path) -> None:
    reg = load_registry()
    assert {i.id for i in reg.indicators.values() if i.retired} == {"fred_copper", "fred_eu_gas"}
    assert reg.indicator("fred_eu_gas").retired.replaced_by == "imf_eu_gas"
    assert reg.indicator("imf_eu_gas").retired is None
    edit_yaml(config_dir / "indicators.yaml", lambda d: first_indicator(d).update(
        retired={"date": "2026-10-07", "reason": "test", "replaced_by": "imf_nothing"}))
    with pytest.raises(RegistryError, match="retired.replaced_by 'imf_nothing' is not in indicators.yaml"):
        load_registry(config_dir)

