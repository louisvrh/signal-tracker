"""Controls shared by the dashboard pages."""

from __future__ import annotations

from datetime import date
from typing import Callable, Sequence

import pandas as pd
import streamlit as st

from dashboard import charts


def url_selectbox(
    container,
    label: str,
    options: Sequence[str],
    key: str,
    default: str,
    format_func: Callable[[str], str] = str,
) -> str:
    """A selectbox whose choice is mirrored in the URL (`?<key>=...`) and kept across pages.

    The choice lives in session state under `key`, seeded once from the URL,
    else from the last choice made on any page, else `default`. Passing no
    `index` keeps the widget's identity stable across reruns (an `index` that
    changes would reset the widget to it).
    """
    options = list(options)
    remembered = f"last_{key}"
    if key not in st.session_state:
        requested = st.query_params.get(key)
        st.session_state[key] = requested if requested in options else st.session_state.get(remembered, default)
    if st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]
    value = container.selectbox(label, options, key=key, format_func=format_func)
    st.session_state[remembered] = value  # widget state is dropped while another page is shown
    st.query_params[key] = value
    return value


def mode_and_period(mode_col, period_col) -> tuple[str, int | None]:
    """The Level/YoY/Chain toggle and the period selector; returns (mode, years or None for all).

    "Chain" draws the transform chain built in the sidebar (dashboard/workbench.py).
    """
    mode = mode_col.radio("Show", [charts.LEVEL, charts.YOY, charts.CHAIN], horizontal=True,
                          help="Chain: build a transform chain in the sidebar (presets, STL, band-passes, ...).")
    periods = list(charts.PERIODS)
    period = period_col.selectbox("Period", periods, index=periods.index(charts.DEFAULT_PERIOD))
    return mode, charts.PERIODS[period]


def period_start(first: pd.Timestamp, end: pd.Timestamp, years: int | None) -> pd.Timestamp:
    """Start of the displayed window: `years` before `end`, or the first date for all."""
    return end - pd.DateOffset(years=years) if years else first


REPLAY_KEY = "replay_on"
REPLAY_DATE_KEY = "replay_date"


def replay_date(first: date, last: date) -> date | None:
    """The replay switch and, when it's on, a date slider from `first` to `last`.

    Returns the chosen day, or None when replay is off. The page then shows
    the data as known at the end of that day (PLAN D43).
    """
    with st.container(horizontal=True, vertical_alignment="center"):
        on = st.toggle("Replay", key=REPLAY_KEY,
                       help="Show the data exactly as it was known on a past date: values published later and "
                            "later revisions are hidden, and chains use one-sided steps only.")
        if not on:
            return None
        default = max(first, (pd.Timestamp(last) - pd.DateOffset(years=1)).date())  # a year back
        return st.slider("As known on", min_value=first, max_value=last, value=default, format="YYYY-MM-DD",
                         key=REPLAY_DATE_KEY)
