"""Signal Lab page (S26): every hypothesis with its status, report headline and vault state,
and the `hypotheses_tested` log of every run."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import data, lab

st.title("Signal Lab")
st.caption("Pre-registered hypotheses (docs/SIGNALS.md) and every logged test run. Statuses come from "
           "reports/SUMMARY.md; verdicts and headlines from each hypothesis's report. The vault (data from "
           "2025-10-08) is opened once per hypothesis. Research only, not financial advice.")

log = data.hypotheses_tested()
reports = data.reports()
table = lab.hypothesis_table(log, reports, reports.get("SUMMARY", ""))

if table.empty:
    st.info("No hypothesis tested yet.")
    st.stop()

tests = log[~log["status"].isin(lab.EXPLORATORY)] if not log.empty else log
statuses = table["status"].str.lower()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Hypotheses", len(table), border=True)
m2.metric("Test runs logged", len(tests), border=True, help="Exploratory diagnostics (DX...) not counted")
m3.metric("Vault looks used", int((table["vault"] != "closed").sum()), border=True,
          help="Each hypothesis gets exactly one look at the vault (PLAN §5.6)")
m4.metric("Passed", int(statuses.str.startswith("passed").sum()), border=True,
          help="Graduation also needs 3 months of forward paper testing in JOURNAL.md")

# ---------------------------------------------------------------- hypotheses
st.subheader("Hypotheses")
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "hypothesis": "Id",
        "title": st.column_config.TextColumn("Hypothesis", width="medium"),
        "status": "Status",
        "verdict": st.column_config.TextColumn("Verdict (report headline)", width="large"),
        "vault": "Vault",
        "runs": st.column_config.NumberColumn("Runs", format="%d"),
        "last_run": st.column_config.DatetimeColumn("Last run (UTC)", format="YYYY-MM-DD HH:mm"),
    },
)

# -------------------------------------------------------------------- report
st.subheader("Report")
titles = dict(zip(table["hypothesis"], table["title"]))
chosen = st.selectbox("Hypothesis", list(titles), format_func=lambda h: f"{h}: {titles[h]}" if titles[h] else h,
                      key="lab_hypothesis")
if chosen in reports:
    parts = lab.split_report(reports[chosen])
    with st.container(border=True):
        st.markdown(parts["headline"] or parts["verdict"] or "No headline section in this report.")
    vault_text = reports.get(chosen + lab.VAULT_SUFFIX)
    if vault_text:
        st.markdown("**Vault test** (the one look)")
        with st.container(border=True):
            st.markdown(lab.split_report(vault_text)["headline"])
        with st.expander(f"Full vault report (reports/{chosen}{lab.VAULT_SUFFIX}.md)"):
            st.markdown(vault_text)
    with st.expander(f"Full report (reports/{chosen}.md)"):
        st.markdown(reports[chosen])
else:
    st.info(f"No report for {chosen} in reports/.")

# ----------------------------------------------------------------------- log
st.subheader("Run log")
st.caption("data/meta/hypotheses_tested: one row per variant per run, failed runs included. "
           "Metrics and parameters are JSON.")
show_exploratory = st.toggle("Include exploratory diagnostics", value=False, key="lab_exploratory",
                             help="Engine controls and checks (DX1-DX5): not hypothesis tests, never the vault")
shown = log if show_exploratory else tests
picked = st.multiselect("Hypotheses", sorted(shown["hypothesis_id"].unique(), key=lab.sort_key) if not shown.empty
                        else [], placeholder="All", key="lab_log_ids")
if picked:
    shown = shown[shown["hypothesis_id"].isin(picked)]
if shown.empty:
    st.info("No runs logged.")
else:
    st.dataframe(
        shown.sort_values("tested_at", ascending=False),
        hide_index=True,
        column_order=["tested_at", "hypothesis_id", "variant", "status", "uses_vault", "test_period", "metrics",
                      "notes", "parameters", "train_period"],
        column_config={
            "tested_at": st.column_config.DatetimeColumn("Tested (UTC)", format="YYYY-MM-DD HH:mm"),
            "hypothesis_id": "Id",
            "variant": st.column_config.TextColumn("Variant", width="medium"),
            "status": "Status",
            "uses_vault": st.column_config.CheckboxColumn("Vault"),
            "test_period": "Test period",
            "metrics": st.column_config.TextColumn("Metrics", width="large"),
            "notes": "Notes",
            "parameters": "Parameters",
            "train_period": "Train period",
        },
    )
    st.caption(f"{len(shown)} rows; newest first. Latest run {pd.Timestamp(shown['tested_at'].max()):%Y-%m-%d %H:%M} UTC.")
