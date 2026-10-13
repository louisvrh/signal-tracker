"""Draw the README's figure from a committed report: no data store, no network.

The power-industry decoupling chart reads the per-country correlations from the
"H2: correlation by country and period" table in
`reports/diagnostics/why_failed_tables.md` (written by `python -m tracker diagnose why`)
and writes a light and a dark SVG to `reports/figures/`. Standard library only.

Usage (PowerShell, from the repo root):

    .venv\\Scripts\\python scripts\\make_figures.py
"""

from __future__ import annotations

import re
import statistics
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "reports" / "diagnostics" / "why_failed_tables.md"
OUT_DIR = ROOT / "reports" / "figures"

# Chart tokens per theme (dataviz reference palette: series-1 blue, muted context gray).
THEMES = {
    "light": {"surface": "#fcfcfb", "border": "rgba(11,11,11,0.10)", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#6f6e69", "grid": "#e1e0d9", "base": "#c3c2b7", "series": "#2a78d6", "context": "#898781"},
    "dark": {"surface": "#1a1a19", "border": "rgba(255,255,255,0.10)", "ink": "#ffffff", "ink2": "#c3c2b7",
             "muted": "#a3a29b", "grid": "#2c2c2a", "base": "#4a4a46", "series": "#3987e5", "context": "#898781"},
}
PERIOD_NOTES = {"2020-21": "COVID collapse and rebound", "2022-23": "energy-price shock", "2024-25": "after the shock"}


def _num(v: float, digits: int = 2, signed: bool = False) -> str:
    """A number with a typographic minus sign."""
    return f"{v:{'+' if signed else ''}.{digits}f}".replace("-", "−")


def read_table(path: Path, heading: str) -> tuple[list[str], dict[str, list[float]]]:
    """Return the period columns and {country: correlations} of the markdown table under `heading`."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(heading)
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("#"):
            break
        if line.startswith("|") and not re.match(r"^\|[-| ]+\|$", line):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    periods = rows[0][1:]
    return periods, {r[0]: [float(v.replace("−", "-")) for v in r[1:]] for r in rows[1:]}


def spread_labels(targets: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Move label positions apart by at least `gap`, keeping order and staying within [lo, hi]."""
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    placed = [0.0] * len(targets)
    prev = lo - gap
    for i in order:
        placed[i] = max(targets[i], prev + gap)
        prev = placed[i]
    overflow = prev - hi
    if overflow > 0:
        for i in order:
            placed[i] -= overflow
    return placed


def decoupling_svg(periods: list[str], data: dict[str, list[float]], theme: dict[str, str]) -> str:
    """The slope chart: each country's load-IP correlation per period, and the median."""
    W, H = 760, 486
    x0, x1, y_top, y_bot = 136, 576, 132, 362
    xs = [x0 + (x1 - x0) * i / (len(periods) - 1) for i in range(len(periods))]

    def y(v: float) -> float:
        return y_bot - (v + 1) / 2 * (y_bot - y_top)

    medians = [statistics.median(vals[i] for vals in data.values()) for i in range(len(periods))]
    t = theme
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-labelledby="title desc" font-family="system-ui, -apple-system, \'Segoe UI\', sans-serif">',
        '<title id="title">Power demand stopped tracking industrial production after 2022</title>',
        '<desc id="desc">Median correlation of weather-normalised electricity demand with industrial production, '
        f'year on year, across {len(data)} countries: '
        + ", ".join(f"{p} {m:.2f}" for p, m in zip(periods, medians)) + ".</desc>",
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="{t["surface"]}" stroke="{t["border"]}"/>',
        f'<text x="28" y="40" font-size="19" font-weight="600" fill="{t["ink"]}">Power demand stopped tracking industrial production after 2022</text>',
        f'<text x="28" y="64" font-size="13" fill="{t["ink2"]}">Correlation of weather-normalised electricity demand with industrial production (monthly, year on year),</text>',
        f'<text x="28" y="82" font-size="13" fill="{t["ink2"]}">within each two-year period. Hypothesis H2; final data as of the vault start, 2025-10-08.</text>',
    ]
    # Legend (two series: the median and the countries).
    out += [
        f'<line x1="28" y1="106" x2="52" y2="106" stroke="{t["series"]}" stroke-width="3" stroke-linecap="round"/>',
        f'<text x="60" y="110" font-size="12.5" fill="{t["ink2"]}">Median of {len(data)} countries</text>',
        f'<line x1="214" y1="106" x2="238" y2="106" stroke="{t["context"]}" stroke-width="2" stroke-linecap="round" opacity="0.7"/>',
        f'<text x="246" y="110" font-size="12.5" fill="{t["ink2"]}">Each country</text>',
    ]
    # Grid, zero baseline and y ticks.
    for v in (-1, -0.5, 0, 0.5, 1):
        colour, label = (t["base"], "0") if v == 0 else (t["grid"], _num(v, 1))
        out.append(f'<line x1="64" y1="{y(v):.1f}" x2="{x1 + 20}" y2="{y(v):.1f}" stroke="{colour}" stroke-width="1"/>')
        out.append(f'<text x="54" y="{y(v) + 4:.1f}" font-size="12" text-anchor="end" fill="{t["muted"]}" '
                   f'style="font-variant-numeric: tabular-nums">{label}</text>')
    # X labels.
    for x, p in zip(xs, periods):
        out.append(f'<text x="{x:.1f}" y="{y_bot + 26}" font-size="13" font-weight="600" text-anchor="middle" fill="{t["ink"]}">{escape(p)}</text>')
        out.append(f'<text x="{x:.1f}" y="{y_bot + 43}" font-size="12" text-anchor="middle" fill="{t["muted"]}">{escape(PERIOD_NOTES.get(p, ""))}</text>')
    # Country lines and dots (context), with leader-line labels at the right.
    for country, vals in data.items():
        pts = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{t["context"]}" stroke-width="2" stroke-linejoin="round" '
                   f'stroke-linecap="round" opacity="0.55"><title>{country}: ' + ", ".join(_num(v) for v in vals) + "</title></polyline>")
        for x, v in zip(xs, vals):
            out.append(f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="3.5" fill="{t["context"]}" stroke="{t["surface"]}" stroke-width="2"/>')
    ends = [y(vals[-1]) for vals in data.values()]
    label_y = spread_labels(ends, 14, y_top, y_bot)
    for (country, vals), yt, yl in zip(data.items(), ends, label_y):
        out.append(f'<polyline points="{xs[-1] + 7:.1f},{yt:.1f} {xs[-1] + 30:.1f},{yl:.1f} {xs[-1] + 36:.1f},{yl:.1f}" '
                   f'fill="none" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{xs[-1] + 40:.1f}" y="{yl + 4:.1f}" font-size="12" fill="{t["muted"]}">{country} '
                   f'<tspan style="font-variant-numeric: tabular-nums">{_num(vals[-1], signed=True)}</tspan></text>')
    # The median: the story.
    pts = " ".join(f"{x:.1f},{y(m):.1f}" for x, m in zip(xs, medians))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{t["series"]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
    for i, (x, m) in enumerate(zip(xs, medians)):
        out.append(f'<circle cx="{x:.1f}" cy="{y(m):.1f}" r="6" fill="{t["series"]}" stroke="{t["surface"]}" stroke-width="2"/>')
        # The first label sits left of its dot, clear of the countries clustered to its right.
        lx, ly, anchor = (x - 14, y(m) + 5, "end") if i == 0 else (x + 12, y(m) - 10, "start")
        out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="14" font-weight="600" text-anchor="{anchor}" '
                   f'fill="{t["ink"]}">{_num(m)}</text>')
    out.append(f'<text x="28" y="{H - 38}" font-size="11.5" fill="{t["muted"]}">Source: reports/diagnostics/why_failed_tables.md '
               '(Pearson correlation per country and period).</text>')
    out.append(f'<text x="28" y="{H - 20}" font-size="11.5" fill="{t["muted"]}">Data: ENTSO-E and NESO (load), Open-Meteo (weather), '
               'Eurostat and OECD (industrial production).</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    periods, data = read_table(TABLES, "### H2: correlation by country and period")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        suffix = "" if name == "light" else "_dark"
        path = OUT_DIR / f"power_ip_decoupling{suffix}.svg"
        path.write_text(decoupling_svg(periods, data, theme), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
