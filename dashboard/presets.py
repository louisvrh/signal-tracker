"""Named transform chains for the workbench, kept in config/presets.yaml (PLAN D32).

Saving or deleting a preset rewrites that file: the dashboard's one write.
The five default presets always exist and can't be overwritten or deleted.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tracker import settings
from tracker.signals.chain import ChainError, parse

DEFAULT_PRESETS = {
    "Raw": "level",
    "Smoothed 28d": "sma(28d)",
    "Seasonally adjusted": "hol>stl",
    "Business-cycle band": "cf(1.5y,8y)",  # cycles of 1.5-8 years
    "Short cycle (2-18 months)": "cf(2m,18m)",  # for series starting in 2019 (PortWatch, EIA)
}
HEADER = """\
# Named transform chains for the dashboard workbench (PLAN D32, D34).
# Each chain is a short string of steps from tracker/signals/chain.py, e.g. stl>sma(28d)>rz(3y).
# The first five are the defaults. Presets saved from the dashboard are added below them;
# saving rewrites this file (the dashboard's one write).

"""
MAX_NAME = 60


def _path(path: Path | None) -> Path:
    return path or settings.PRESETS_FILE


def _entries(file: Path) -> list:
    if not file.exists():
        return []
    return (yaml.safe_load(file.read_text(encoding="utf-8")) or {}).get("presets") or []


def load_presets(path: Path | None = None) -> tuple[dict[str, str], list[str]]:
    """All presets as {name: chain string}, defaults first, and problems with the file.

    Entries whose chain doesn't parse are left out and reported, so a typo
    in the file never stops the dashboard.
    """
    presets = dict(DEFAULT_PRESETS)
    problems: list[str] = []
    file = _path(path)
    try:
        entries = _entries(file)
    except yaml.YAMLError as exc:
        return presets, [f"{file.name} can't be read: {exc}"]
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name") or "chain" not in entry:
            problems.append(f"{file.name}: skipped an entry without a name and chain: {entry!r}")
            continue
        name = str(entry["name"])
        if name in DEFAULT_PRESETS:
            continue
        try:
            presets[name] = str(parse(str(entry["chain"] or "")))
        except ChainError as exc:
            problems.append(f"{file.name}: preset '{name}' skipped: {exc}")
    return presets, problems


def _rewrite(path: Path | None, drop: str, add: tuple[str, str] | None = None) -> None:
    """Rewrite the file: the defaults, then every other entry as it was (unreadable ones
    included, so a typo isn't lost) except `drop`, then `add`."""
    file = _path(path)
    kept = [e for e in _entries(file)
            if not (isinstance(e, dict) and str(e.get("name")) in set(DEFAULT_PRESETS) | {drop})]
    entries = [{"name": n, "chain": c} for n, c in DEFAULT_PRESETS.items()] + kept
    if add:
        entries.append({"name": add[0], "chain": add[1]})
    tmp = file.with_suffix(".yaml.tmp")
    tmp.write_text(HEADER + yaml.safe_dump({"presets": entries}, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    tmp.replace(file)


def save_preset(name: str, chain_text: str, path: Path | None = None) -> str:
    """Save (or replace) a custom preset and return its canonical chain string.

    Raises ValueError for an empty or default name, or a chain that doesn't parse.
    """
    name = " ".join(str(name).split())
    if not name:
        raise ValueError("Give the preset a name.")
    if len(name) > MAX_NAME:
        raise ValueError(f"Preset names are at most {MAX_NAME} characters.")
    if name in DEFAULT_PRESETS:
        raise ValueError(f"'{name}' is a default preset; choose another name.")
    canonical = str(parse(chain_text))
    if not canonical:
        raise ValueError("The chain is empty; add a step first.")
    _rewrite(path, drop=name, add=(name, canonical))
    return canonical


def delete_preset(name: str, path: Path | None = None) -> None:
    """Remove a custom preset (defaults can't be deleted)."""
    if name in DEFAULT_PRESETS:
        raise ValueError(f"'{name}' is a default preset and can't be deleted.")
    _rewrite(path, drop=name)
