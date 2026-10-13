"""CPB World Trade Monitor (no key): merchandise trade volumes and unit values.

CPB (the Netherlands Bureau for Economic Policy Analysis) publishes the monitor
monthly, about eight weeks after the month (July 2026 on 25 September 2026),
as an Excel file linked from the latest release page
(`https://www.cpb.nl/en/worldtrademonitor/latest`, which redirects to it;
checked 2026-10-07). Sheet `trade_out` has one row per series, with a code
such as `xgz_us_qnmi_sn` in column C, and one column per month (`2000m01`,
`2000m02`, ... in the header row):

* `..._qnmi_sn`: volumes, seasonally adjusted, fixed base 2021=100;
* `..._pdmi_sn`: prices (unit values) in US dollars, seasonally adjusted, 2021=100.

Codes start with `xgz` (exports) or `mgz` (imports), then an area code:
`w1` world, `us`, `gb`, `jp`, `cn` single countries; the others are regional
aggregates (euro area, emerging Asia, ...), which aren't stored because the
lake holds ISO3 countries and WLD (PLAN §4.1). Volume is value
deflated by price, consistently at every level, so volume × price is the
value series in index form.

Every release revises recent months (late reports, seasonal factors), so the
whole history is read on each run (`resume_from: backfill_start`) and the
store keeps the changes as revisions. The .xlsx is read with the standard
library (zipfile and XML): no spreadsheet dependency.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import date
from urllib.parse import urljoin
from xml.etree import ElementTree

import pandas as pd

from tracker import store
from tracker.fetchers.base import BaseFetcher, FetchError

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
_CELL_RE = re.compile(r"^([A-Z]+)(\d+)$")
_MONTH_RE = re.compile(r"^(\d{4})m(\d{2})$")
_XLSX_LINK_RE = re.compile(r'href="([^"]*world-trade-monitor[^"]*\.xlsx)"', re.IGNORECASE)


def read_xlsx_sheet(content: bytes, sheet: str) -> dict[int, dict[str, str]]:
    """The cells of one worksheet as {row number: {column letters: text}} (shared strings resolved,
    numbers as written). Raises FetchError if the file or sheet can't be read."""
    try:
        book = zipfile.ZipFile(io.BytesIO(content))
        names = set(book.namelist())
        shared = []
        if "xl/sharedStrings.xml" in names:
            for item in ElementTree.fromstring(book.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in item.iter(f"{{{_NS['m']}}}t")))
        workbook = ElementTree.fromstring(book.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise FetchError(f"cpb: not a readable .xlsx file ({exc})") from exc
    targets = {r.get("Id"): r.get("Target") for r in rels}
    path = None
    for item in workbook.find("m:sheets", _NS):
        if item.get("name") == sheet:
            target = targets[item.get(f"{{{_NS['r']}}}id")]
            path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
    if path is None or path not in names:
        raise FetchError(f"cpb: the workbook has no sheet {sheet!r}")
    cells: dict[int, dict[str, str]] = {}
    for row in ElementTree.fromstring(book.read(path)).iter(f"{{{_NS['m']}}}row"):
        for cell in row.findall("m:c", _NS):
            match = _CELL_RE.match(cell.get("r", ""))
            if not match:
                continue
            kind, value = cell.get("t"), cell.find("m:v", _NS)
            if kind == "inlineStr":
                text = "".join(t.text or "" for t in cell.iter(f"{{{_NS['m']}}}t"))
            elif value is None or value.text is None:
                continue
            else:
                text = shared[int(value.text)] if kind == "s" else value.text
            cells.setdefault(int(match.group(2)), {})[match.group(1)] = text
    return cells


def parse_trade_sheet(content: bytes, series: dict[str, tuple[str, str]], sheet: str = "trade_out",
                      code_column: str = "C") -> pd.DataFrame:
    """The monitor's trade sheet -> DataFrame(indicator, date, country, value).

    `series` maps CPB series codes (e.g. `xgz_us_qnmi_sn`) to (indicator id, country). Month
    columns are found from their `YYYYmMM` header; rows for other codes are ignored.
    """
    cells = read_xlsx_sheet(content, sheet)
    months: dict[str, pd.Timestamp] = {}
    for row in cells.values():  # the header is the row with the most month labels
        found = {col: _MONTH_RE.match(text.strip()) for col, text in row.items()}
        found = {col: m for col, m in found.items() if m}
        if len(found) > len(months):
            months = {col: pd.Timestamp(int(m.group(1)), int(m.group(2)), 1) for col, m in found.items()}
    if not months:
        raise FetchError(f"cpb: no month header (YYYYmMM) found in sheet {sheet!r}")
    rows = []
    seen = set()
    for row in cells.values():
        code = row.get(code_column, "").strip()
        if code not in series:
            continue
        seen.add(code)
        indicator, country = series[code]
        for col, month in months.items():
            text = row.get(col)
            try:
                value = float(text) if text not in (None, "") else None
            except ValueError:
                value = None
            if value is not None:
                rows.append({"indicator": indicator, "date": month, "country": country, "value": value})
    missing = sorted(set(series) - seen)
    if missing:
        raise FetchError(f"cpb: series {missing} not found in sheet {sheet!r}; has the layout changed?")
    return pd.DataFrame(rows, columns=["indicator", "date", "country", "value"])


def latest_xlsx_url(page_html: str, page_url: str) -> str:
    """The Excel download linked from a release page."""
    links = _XLSX_LINK_RE.findall(page_html)
    if not links:
        raise FetchError("cpb: no World Trade Monitor .xlsx link on the latest release page")
    return urljoin(page_url, links[0])


class CPBFetcher(BaseFetcher):
    """Monthly merchandise trade volumes and unit values (CPB World Trade Monitor)."""

    source = "cpb"
    min_interval_s = 2.0  # two requests per run (the release page, then its Excel file)
    timeout_s = 120.0

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        series = {s["code"]: (s["indicator"], s["country"]) for s in self.config["series"]}
        page = self.get(self.config["latest_url"], raw_name="latest_release", raw_ext="html")
        url = latest_xlsx_url(page.text, getattr(page, "url", None) or self.config["latest_url"])
        self.log.info("cpb: latest release file %s", url)
        book = self.get(url, raw_name=url.rsplit("/", 1)[-1].removesuffix(".xlsx"), raw_ext="xlsx")
        data = parse_trade_sheet(book.content, series)
        data = data[(data["date"] >= pd.Timestamp(start)) & (data["date"] <= pd.Timestamp(end))]
        self.log.info("cpb: %d monthly values for %d series", len(data), data.groupby(["indicator", "country"]).ngroups)
        frames = [self.frame(ind, part[["date", "country", "value"]]) for ind, part in data.groupby("indicator")]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()
