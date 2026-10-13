"""The transform-chain workbench (S17c, PLAN D32): sidebar chain builder and chain panels.

Used by the Country and Indicator pages in their "Chain" view. The chain
lives in session state as its string, so both pages share it. Every change
(a preset, a typed chain, a parameter, adding, removing or moving a step)
goes through a callback that stores the new chain and bumps a version
number. The builder's widgets carry that number in their keys, so on the next
run they are rebuilt from the chain instead of keeping stale values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import charts, data, presets
from tracker.signals import chain as chains
from tracker.signals.chain import Chain, ChainError, Step

CHAIN_KEY = "wb_chain"
VERSION_KEY = "wb_version"
MESSAGE_KEY = "wb_message"
REALTIME_KEY = "wb_realtime"
SPECTRUM_KEY = "wb_spectrum"
SAVE_NAME_KEY = "wb_save_name"
DEFAULT_PRESET = "Seasonally adjusted"
CUSTOM = "Custom"
HYPOTHESIS = """- **Hn** {signal}, transform `{chain}` → [target] | [lead time] | [expected sign] | [test type]
  Written: {today}. Rationale: [one or two sentences on why this should work]."""


@dataclass
class Workbench:
    """The sidebar's choices.

    Attributes:
        chain: the chain as built.
        applied: what the charts use (two-sided steps dropped in real-time mode).
        dropped: the two-sided steps hidden by real-time mode.
        spectrum: draw periodograms instead of time series.
    """

    chain: Chain
    applied: Chain
    dropped: list[Step] = field(default_factory=list)
    spectrum: bool = False


def hypothesis_text(signal: str, chain: Chain, today: date | None = None) -> str:
    """A SIGNALS.md §7 hypothesis line quoting the chain string (PLAN D32)."""
    return HYPOTHESIS.format(signal=signal, chain=chain, today=(today or date.today()).isoformat())


# ------------------------------------------------------------------ callbacks


def current_chain() -> Chain:
    """The chain being built (the default preset until something is chosen)."""
    return chains.parse(st.session_state.get(CHAIN_KEY, presets.DEFAULT_PRESETS[DEFAULT_PRESET]))


def _store(chain: Chain | str, message: tuple[str, str] | None = None) -> None:
    st.session_state[CHAIN_KEY] = str(chain)
    st.session_state[VERSION_KEY] = st.session_state.get(VERSION_KEY, 0) + 1
    st.session_state[MESSAGE_KEY] = message


def _error(text: str) -> None:
    st.session_state[MESSAGE_KEY] = ("error", text)


def _on_preset(key: str, known: dict[str, str]) -> None:
    name = st.session_state[key]
    if name in known:
        _store(known[name])


def _on_text(key: str) -> None:
    try:
        _store(chains.parse(st.session_state[key] or ""))
    except ChainError as exc:
        _error(str(exc))  # keep the typed text so it can be corrected


def _on_param(key: str, position: int, name: str) -> None:
    chain = current_chain()
    step = chain.steps[position]
    param = next(p for p in step.spec.params if p.name == name)
    try:
        value = param.parse(st.session_state[key])
    except ChainError as exc:
        _error(f"{step.key}: {exc}")
        return
    args = tuple(value if p.name == name else v for p, v in zip(step.spec.params, step.args))
    steps = list(chain.steps)
    steps[position] = Step(step.key, args)
    _store(Chain(tuple(steps)))


def _on_move(position: int, delta: int) -> None:
    steps = list(current_chain().steps)
    steps[position], steps[position + delta] = steps[position + delta], steps[position]
    _store(Chain(tuple(steps)))


def _on_remove(position: int) -> None:
    steps = list(current_chain().steps)
    del steps[position]
    _store(Chain(tuple(steps)))


def _on_add(key: str) -> None:
    spec = chains.STEPS[st.session_state[key]]
    reason = spec.unavailable() if spec.unavailable else None
    if reason:
        _error(reason)
        return
    _store(Chain(current_chain().steps + (Step(spec.key),)))


def _on_save() -> None:
    try:
        name = st.session_state.get(SAVE_NAME_KEY, "")
        saved = presets.save_preset(name, str(current_chain()))
    except (ValueError, OSError) as exc:
        _error(str(exc))
        return
    st.session_state[SAVE_NAME_KEY] = ""
    st.session_state[MESSAGE_KEY] = ("success", f"Saved preset '{' '.join(name.split())}': {saved}")


def _on_delete(name: str) -> None:
    try:
        presets.delete_preset(name)
    except (ValueError, OSError) as exc:
        _error(str(exc))
        return
    st.session_state[MESSAGE_KEY] = ("success", f"Deleted preset '{name}'.")


# -------------------------------------------------------------------- sidebar


def _step_label(key: str) -> str:
    spec = chains.STEPS[key]
    reason = spec.unavailable() if spec.unavailable else None
    return f"{spec.group} · {spec.name} ({key})" + (" · needs S18" if reason else "")


def _step_box(position: int, step: Step, count: int, version: int, realtime: bool) -> None:
    spec = step.spec
    with st.container(border=True, key=f"wb_step_{version}_{position}"):
        st.markdown(f"**{position + 1}. {spec.name}** `{step}`")
        two_sided = spec.sided == chains.TWO_SIDED
        st.badge(spec.sided, color="orange" if two_sided else "blue",
                 help="Uses later data: its latest values revise" if two_sided else "Uses only past data")
        if realtime and two_sided:
            st.caption("Hidden: real-time only is on.")
        for param, value in zip(spec.params, step.args):
            key = f"wb_param_{version}_{position}_{param.name}"
            if param.kind == "choice":
                options = ["auto", *param.choices]
                st.selectbox(param.name, options, index=options.index(param.format(value)), key=key,
                             help=param.help, on_change=_on_param, args=(key, position, param.name))
            else:
                st.text_input(param.name, value="" if value is None else param.format(value), placeholder="auto",
                              key=key, help=param.help, on_change=_on_param, args=(key, position, param.name))
        with st.container(horizontal=True, gap="small"):
            st.button("Up", icon=":material/arrow_upward:", type="tertiary", key=f"wb_up_{version}_{position}",
                      disabled=position == 0, on_click=_on_move, args=(position, -1))
            st.button("Down", icon=":material/arrow_downward:", type="tertiary", key=f"wb_down_{version}_{position}",
                      disabled=position == count - 1, on_click=_on_move, args=(position, 1))
            st.button("Remove", icon=":material/close:", type="tertiary", key=f"wb_remove_{version}_{position}",
                      on_click=_on_remove, args=(position,))


def sidebar(signal: str, realtime_forced: str | None = None) -> Workbench:
    """Draw the chain builder in the sidebar and return the choices.

    Args:
        signal: how the "copy as hypothesis" line names the signal, e.g.
            "[indicator] DEU" on the Country page.
        realtime_forced: when set (the reason, e.g. replay), "Real-time only" is
            on and locked, so two-sided steps are hidden whatever the user's setting.
    """
    known, problems = presets.load_presets()
    st.session_state.setdefault(VERSION_KEY, 0)
    st.session_state.setdefault(CHAIN_KEY, known[DEFAULT_PRESET])
    chain = current_chain()
    version = st.session_state[VERSION_KEY]

    with st.sidebar:
        st.subheader("Transform chain")
        current = next((name for name, text in known.items() if text == str(chain)), CUSTOM)
        options = list(known) + ([CUSTOM] if current == CUSTOM else [])
        key = f"wb_preset_{version}"
        st.selectbox("Preset", options, index=options.index(current), key=key,
                     on_change=_on_preset, args=(key, known))
        key = f"wb_text_{version}"
        st.text_input("Chain", value=str(chain), key=key, on_change=_on_text, args=(key,),
                      placeholder="e.g. stl>sma(28d)>rz(3y)",
                      help="Steps separated by >, parameters in brackets; see tracker/signals/chain.py.")
        message = st.session_state.pop(MESSAGE_KEY, None)
        if message:
            (st.error if message[0] == "error" else st.success)(message[1])
        if realtime_forced:
            realtime = st.toggle("Real-time only", value=True, disabled=True, key=f"{REALTIME_KEY}_forced",
                                 help=f"On while {realtime_forced}: two-sided steps use later data (PLAN D32).")
        else:
            realtime = st.toggle("Real-time only", key=REALTIME_KEY, persist_state="session",
                                 help="Hide two-sided steps: they use later data, so their past values weren't "
                                      "knowable at the time (PLAN D32).")
        spectrum = st.toggle("Spectrum view", key=SPECTRUM_KEY, persist_state="session",
                             help="Periodogram of the raw series (faint) and the chain's output (solid).")

        for position, step in enumerate(chain.steps):
            _step_box(position, step, len(chain.steps), version, realtime)

        addable = [k for k, s in chains.STEPS.items() if not (realtime and s.sided == chains.TWO_SIDED)]
        key = f"wb_add_choice_{version}"
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.selectbox("Add a step", addable, key=key, format_func=_step_label)
            st.button("Add", icon=":material/add:", key=f"wb_add_{version}", on_click=_on_add, args=(key,))

        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.text_input("Save as preset", key=SAVE_NAME_KEY, placeholder="Preset name")
            st.button("Save", icon=":material/save:", key="wb_save", on_click=_on_save,
                      disabled=not chain.steps)
        if current not in presets.DEFAULT_PRESETS and current != CUSTOM:
            st.button(f"Delete preset '{current}'", icon=":material/delete:", type="tertiary",
                      key=f"wb_delete_{version}", on_click=_on_delete, args=(current,))
        with st.popover("Copy as hypothesis", icon=":material/content_copy:"):
            st.code(hypothesis_text(signal, chain), language=None, wrap_lines=True)
            st.caption("Paste into docs/SIGNALS.md §5 and fill in the brackets.")
        for problem in problems:
            st.warning(problem)

    applied, dropped = chain.one_sided_only() if realtime else (chain, [])
    return Workbench(chain=chain, applied=applied, dropped=dropped, spectrum=spectrum)


def caption(wb: Workbench) -> str:
    """What the chain panels show, for the page caption."""
    if wb.spectrum:
        text = (f"Periodograms of the raw series (faint) and `{wb.applied}` (solid): each cycle length's share of the "
                "variance, trend removed. A band-pass's band is shaded.")
    elif wb.applied.steps:
        text = (f"Chain `{wb.applied}`: faint line = raw series (when the chain keeps its units); dashed = values that "
                "may still revise (two-sided steps); lower strip = what each step removed.")
    else:
        text = "No chain steps to apply, so the raw series is shown."
    if wb.dropped:
        text += f" Real-time only: hidden {', '.join(f'`{s}`' for s in wb.dropped)}."
    return text


# --------------------------------------------------------------------- panels


def panel(
    wb: Workbench,
    values: pd.Series,
    indicator: str,
    country: str,
    title: str,
    subtitle: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    windows: pd.DataFrame | None,
    with_raw: bool = True,
    height: int = charts.PANEL_HEIGHT,
    replay_day=None,
) -> tuple[go.Figure, str | None, list[chains.Conversion]]:
    """The chain chart of one series, why the chain wasn't applied (None if it
    was), and the steps whose lengths were converted to the series' frequency.

    Lengths in another unit are converted (28d runs as 1m on monthly data,
    PLAN D37). A chain that still doesn't fit the series (e.g. too little
    history for STL) leaves the raw series on the chart. `replay_day`: the
    Country page's replay day, so weather adjustment uses degree days as known then.
    """
    ind = data.registry().indicator(indicator)
    result = data.chain_result(str(wb.applied), indicator, country, values, replay_day)
    problem, conversions = None, []
    if isinstance(result, str):
        problem = result
        result = chains.apply(Chain(), values, chains.Context(frequency=ind.frequency, unit=ind.unit))
        subtitle = f"{subtitle} · chain not applied"
    elif result.conversions:
        conversions = result.conversions
        subtitle = f"{subtitle} · lengths converted"
    if wb.spectrum:
        raw = charts.periodogram(values, ind.frequency, ind.calendar)
        out = None if problem or not wb.applied.steps else charts.periodogram(result.values, ind.frequency, ind.calendar)
        figure = charts.spectrum_figure(raw, out, title, subtitle, band=charts.band_years(wb.applied), height=height)
        return figure, problem, conversions
    y_title = ind.unit if problem else charts.chain_axis_label(wb.applied, ind.unit)
    fig = charts.chain_figure(
        result, title, y_title, subtitle=subtitle,
        windows=charts.period_windows(windows, ind.frequency) if windows is not None else None,
        start=start, with_raw=with_raw, height=height, breaks=data.breaks(indicator, country),
    )
    fig.update_xaxes(range=[start, end])
    return fig, problem, conversions


def problems_note(problems: dict[str, str]) -> None:
    """List the panels the chain couldn't be applied to, with the reasons."""
    if problems:
        with st.expander(f"Chain not applied to {len(problems)} chart(s)"):
            for name, reason in problems.items():
                st.markdown(f"- **{name}**: {reason}")


def conversions_note(converted: dict[str, list[chains.Conversion]]) -> None:
    """List the panels whose chain lengths were converted to the series' frequency
    (PLAN D37), one line per distinct change with the charts it applied to."""
    if not converted:
        return
    groups: dict[str, list[str]] = {}
    for name, items in converted.items():
        groups.setdefault("; ".join(str(c) for c in items), []).append(name)
    with st.expander(f"Lengths converted on {len(converted)} chart(s)"):
        for change, names in groups.items():
            st.markdown(f"- {change}: {', '.join(f'**{n}**' for n in names)}")
