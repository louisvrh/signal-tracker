"""Signal Tracker dashboard (Streamlit multipage app, read-only).

Start it with scripts\\run_dashboard.ps1, or:
    .venv\\Scripts\\python -m streamlit run dashboard\\app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run` only puts dashboard/ on the path
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard import data  # noqa: E402

st.set_page_config(page_title="Signal Tracker", page_icon=":material/monitoring:", layout="wide")

# Page scripts live in views/, not pages/: Streamlit treats a pages/ folder as
# a legacy multipage app and would then run a page URL opened on a freshly
# started server without this file (no layout, no sidebar).
PAGES = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/country.py", title="Country", icon=":material/flag:"),
    st.Page("views/indicator.py", title="Indicator", icon=":material/stacked_line_chart:"),
    st.Page("views/shipping.py", title="Shipping & Chokepoints", icon=":material/directions_boat:"),
    st.Page("views/signal_lab.py", title="Signal Lab", icon=":material/science:"),
    st.Page("views/data_health.py", title="Data Health", icon=":material/health_and_safety:"),
]

page = st.navigation(PAGES)
# The session's first run: the Overview sends an old /?country=XXX link on to the Country page.
st.session_state["first_run"] = "started" not in st.session_state
st.session_state["started"] = True

with st.sidebar:
    if st.button("Reload data", icon=":material/refresh:", help="Data is cached for an hour; reload after a fetch."):
        data.clear()
        st.rerun()
    st.caption("Read-only view of the lake (latest vintages).")

page.run()
