"""`python -m tracker digest`: gather the digest's inputs, send it, remember what it said (S21).

The digest is always sent, whatever happened: if building it fails, a short
failure message (`digest.failure_text`) goes instead. Nothing is recorded for
a dry run. `find_chat` is the one-time setup that saves TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tracker import settings, store
from tracker.alerts import digest, telegram
from tracker.daily_signals import FLAGS_TABLE, SIGNALS_TABLE

log = logging.getLogger(__name__)

DASHBOARD_PORT = 8551  # scripts/dashboard_service.ps1


@dataclass
class DigestResult:
    """What `run_digest` did: the kind and text of the message, and its Telegram id if sent."""

    kind: str
    text: str
    sent: bool
    message_id: int = 0


def dashboard_listening(port: int = DASHBOARD_PORT) -> str:
    """'up' if something accepts connections on 127.0.0.1:<port>, else 'DOWN'."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return "up"
    except OSError:
        return "DOWN"


def latest_fetches(data_dir: Path | None = None) -> pd.DataFrame:
    """Each enabled source's most recent fetch_log row (a source never fetched gets an empty row)."""
    enabled = [name for name, cfg in settings.load_sources().items() if (cfg or {}).get("enabled", True)]
    log_rows = store.read_fetch_log(data_dir)
    last = log_rows.sort_values("started_at").groupby("source").tail(1).set_index("source") \
        if not log_rows.empty else pd.DataFrame(columns=["started_at", "status", "error"])
    rows = [{"source": s, "started_at": last["started_at"].get(s, pd.NaT), "status": last["status"].get(s),
             "error": last["error"].get(s)} for s in enabled]
    return pd.DataFrame(rows, columns=["source", "started_at", "status", "error"])


def collect(failed_steps: list[str], dashboard: str | None, daily_run: bool,
            data_dir: Path | None = None) -> digest.DigestInputs:
    """Read everything the digest needs from the store, signal tables and meta tables."""
    from tracker import quality
    from tracker.registry import load_registry
    from tracker.status import indicator_freshness

    registry = load_registry()
    fresh = indicator_freshness(registry)
    has_data = set(store.source_summary(data_dir)["indicator"])
    stale = fresh[fresh["stale"] & fresh["indicator"].isin(has_data)][["indicator", "last_date"]]
    issues = quality.open_issues(store.read_quality_issues(data_dir))
    errors = issues[issues["severity"] == "error"] if not issues.empty else issues
    sent = store.read_digests(data_dir)
    last_state = json.loads(sent["state"].iloc[-1]) if not sent.empty else None
    return digest.DigestInputs(
        now=pd.Timestamp.now(tz="UTC"), failed_steps=failed_steps,
        dashboard=dashboard or dashboard_listening(), sources=latest_fetches(data_dir), stale=stale,
        quality_errors=errors, signals=store.read_signal_table(SIGNALS_TABLE, data_dir),
        flags=store.read_signal_table(FLAGS_TABLE, data_dir), last_state=last_state, daily_run=daily_run,
    )


def run_digest(dry_run: bool = False, failed_steps: list[str] | None = None, dashboard: str | None = None,
               daily_run: bool = False, data_dir: Path | None = None, session=None) -> DigestResult:
    """Build the digest (or, if that fails, the short failure message), then send it or, for a
    dry run, only return it. A sent digest's state is saved for the next one."""
    failed_steps = list(failed_steps or [])
    dashboard = dashboard or dashboard_listening()
    state = None
    try:
        inputs = collect(failed_steps, dashboard, daily_run, data_dir)
        kind, text, state = digest.build(inputs)
    except Exception as exc:  # the message must go out regardless
        log.exception("digest could not be built")
        kind = "failure"
        text = digest.failure_text(pd.Timestamp.now(tz="UTC"), failed_steps or ["digest"], dashboard,
                                   type(exc).__name__)
    if dry_run:
        return DigestResult(kind, text, sent=False)
    try:
        message_id = telegram.send_message(text, session=session)
    except telegram.NotConfigured as exc:  # setup not finished: keep the text in the log instead
        log.warning("digest not sent: %s", exc)
        return DigestResult(kind, text, sent=False)
    if state is not None:
        store.append_digest(pd.DataFrame([{
            "sent_at": pd.Timestamp.now(tz="UTC"), "kind": kind, "failed_steps": ",".join(failed_steps),
            "dashboard": dashboard, "state": json.dumps(state), "chars": len(text), "message_id": message_id,
        }]), data_dir)
    return DigestResult(kind, text, sent=True, message_id=message_id)


def find_chat(session=None) -> str:
    """One-time setup: save the id of the private chat that last messaged the bot as TELEGRAM_CHAT_ID
    (appended to .env, which is never read here), then send a test message. Returns what happened;
    never the token or the id."""
    if settings.has_secret("TELEGRAM_CHAT_ID"):
        return "TELEGRAM_CHAT_ID is already set in .env; nothing changed."
    chats = telegram.private_chats(telegram.get_updates(session=session))
    if not chats:
        return ("No message to the bot found. Telegram keeps messages for 24 hours: send your bot any message "
                "in Telegram, then run this again.")
    if len(chats) > 1:
        return (f"{len(chats)} different private chats have messaged the bot; not guessing which is yours. "
                "Add TELEGRAM_CHAT_ID to .env yourself.")
    with open(settings.ENV_FILE, "a", encoding="utf-8") as fh:
        fh.write(f"\nTELEGRAM_CHAT_ID={chats[0]}\n")
    telegram.send_message("Signal Tracker: this chat will receive the morning digest.", chat_id=str(chats[0]),
                          session=session)
    return "Saved TELEGRAM_CHAT_ID to .env and sent a test message to that chat."
