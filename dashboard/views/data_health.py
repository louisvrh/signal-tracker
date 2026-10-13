"""Data Health page: freshness per source, the fetch log and open quality issues."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import data

FETCH_LOG_ROWS = 200


def utc(ts) -> str:
    """Format a timestamp as 'YYYY-MM-DD HH:MM UTC' ('-' if missing)."""
    return "-" if ts is None or pd.isna(ts) else f"{pd.Timestamp(ts).tz_convert('UTC'):%Y-%m-%d %H:%M} UTC"


def short(ts) -> str:
    """Format a timestamp as '27 Sep 07:04' in UTC ('-' if missing), for metric tiles."""
    return "-" if ts is None or pd.isna(ts) else f"{pd.Timestamp(ts).tz_convert('UTC'):%d %b %H:%M}"


st.title("Data Health")

sources = data.source_status()
issues = data.open_quality_issues()
log = data.fetch_log()

enabled = sources[sources["enabled"]]
n_fresh = int((enabled["freshness"] == "fresh").sum())
severity = issues["severity"].value_counts() if not issues.empty else pd.Series(dtype=int)
backup = data.last_backup()  # "<UTC time> <folder>", written by scripts/backup.ps1
backup_time = pd.to_datetime(backup.split(" ", 1)[0], utc=True, errors="coerce") if backup else None
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Sources fresh", f"{n_fresh} / {len(enabled)}", border=True)
m2.metric("Open errors", int(severity.get("error", 0)), border=True)
m3.metric("Open warnings", int(severity.get("warning", 0)), border=True)
m4.metric("Last fetch run (UTC)", short(log["started_at"].max()) if not log.empty else "-", border=True)
m5.metric("Last backup (UTC)", short(backup_time) if backup else "none", border=True,
          help=backup or r"No backup recorded yet (scripts\backup.ps1).")

# ------------------------------------------------------------------ freshness
st.subheader("Freshness per source")
st.caption("A source is stale when a period that should be published by now (period end + publication lag + "
           "grace) isn't in the store (PLAN D15).")
st.dataframe(
    sources,
    hide_index=True,
    column_order=["source", "freshness", "enabled", "last_run", "last_success", "rows_written_last",
                  "rows_in_lake", "latest_data", "last_error"],
    column_config={
        "source": "Source",
        "freshness": "Freshness",
        "enabled": st.column_config.CheckboxColumn("Enabled"),
        "last_run": "Last run",
        "last_success": st.column_config.DatetimeColumn("Last success (UTC)", format="YYYY-MM-DD HH:mm"),
        "rows_written_last": st.column_config.NumberColumn("Rows written (last)", format="%d"),
        "rows_in_lake": st.column_config.NumberColumn("Rows in lake", format="%d"),
        "latest_data": st.column_config.DateColumn("Latest data"),
        "last_error": "Last error",
    },
)

with st.expander("Freshness per indicator"):
    fresh = data.indicator_freshness().sort_values(["stale", "days_late", "source", "indicator"],
                                                   ascending=[False, False, True, True])
    st.dataframe(
        fresh,
        hide_index=True,
        column_config={
            "source": "Source", "indicator": "Indicator", "frequency": "Freq",
            "last_date": st.column_config.DateColumn("Latest data"),
            "next_due": st.column_config.DatetimeColumn("Next period due (UTC)", format="YYYY-MM-DD"),
            "stale": st.column_config.CheckboxColumn("Stale"),
            "days_late": st.column_config.NumberColumn("Days late", format="%d"),
            "retired": st.column_config.CheckboxColumn("Retired", help="No longer fetched; history kept (PLAN D56)"),
        },
    )

# ------------------------------------------------------------------ fetch log
st.subheader("Fetch log")
if log.empty:
    st.info("No fetch runs recorded yet.")
else:
    f1, f2 = st.columns(2)
    pick_sources = f1.multiselect("Source", sorted(log["source"].unique()), placeholder="All sources", key="log_source")
    pick_status = f2.multiselect("Status", sorted(log["status"].unique()), placeholder="All statuses", key="log_status")
    shown = log
    if pick_sources:
        shown = shown[shown["source"].isin(pick_sources)]
    if pick_status:
        shown = shown[shown["status"].isin(pick_status)]
    shown = shown.sort_values("started_at", ascending=False).head(FETCH_LOG_ROWS)
    shown = shown.assign(duration_s=(shown["finished_at"] - shown["started_at"]).dt.total_seconds())
    st.dataframe(
        shown,
        hide_index=True,
        column_order=["started_at", "source", "status", "rows_fetched", "rows_written", "duration_s", "error",
                      "run_id"],
        column_config={
            "started_at": st.column_config.DatetimeColumn("Started (UTC)", format="YYYY-MM-DD HH:mm"),
            "source": "Source", "status": "Status",
            "rows_fetched": st.column_config.NumberColumn("Rows fetched", format="%d"),
            "rows_written": st.column_config.NumberColumn("Rows written", format="%d"),
            "duration_s": st.column_config.NumberColumn("Seconds", format="%.0f"),
            "error": "Error", "run_id": "Run id",
        },
    )
    st.caption(f"Newest first; at most {FETCH_LOG_ROWS} rows.")

# ------------------------------------------------------------------ quality issues
st.subheader("Open quality issues")
st.caption(f"Found by the latest quality run ({utc(data.last_quality_run())}), plus issues reported by "
           "fetchers since the run before it (PLAN D23).")
if issues.empty:
    st.success("No open issues.")
else:
    q1, q2, q3 = st.columns(3)
    pick_sev = q1.multiselect("Severity", ["error", "warning", "info"], placeholder="All severities")
    pick_check = q2.multiselect("Check", sorted(issues["check"].unique()), placeholder="All checks")
    pick_src = q3.multiselect("Source", sorted(issues["source"].dropna().unique()), placeholder="All sources",
                             key="issue_source")
    shown = issues
    if pick_sev:
        shown = shown[shown["severity"].isin(pick_sev)]
    if pick_check:
        shown = shown[shown["check"].isin(pick_check)]
    if pick_src:
        shown = shown[shown["source"].isin(pick_src)]
    counts = issues.groupby(["check", "severity"]).size().rename("issues").reset_index()
    left, right = st.columns([1, 3])
    left.dataframe(counts, hide_index=True)
    right.dataframe(
        shown,
        hide_index=True,
        column_order=["severity", "check", "source", "indicator", "country", "detail"],
        column_config={"severity": "Severity", "check": "Check", "source": "Source", "indicator": "Indicator",
                       "country": "Country", "detail": st.column_config.TextColumn("Detail", width="large")},
    )
