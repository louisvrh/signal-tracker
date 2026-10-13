"""Minimal Telegram Bot API client (S21): send a text message, read updates.

Docs: https://core.telegram.org/bots/api. Requests go to
`https://api.telegram.org/bot<token>/<method>` and answer
`{"ok": true, "result": ...}` or `{"ok": false, "error_code": ..., "description": ...}`,
with `parameters.retry_after` on HTTP 429. A message is 1-4096 characters.

The token is part of every request URL, so nothing here ever puts a URL or a
`requests` exception message into an error, a log line or output: errors carry
the method, HTTP status and Telegram's own description only. The token and
chat id come from `.env` through `tracker.settings`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from tracker import settings

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
MAX_LENGTH = 4096
ATTEMPTS = 3
TIMEOUT_S = 30


class TelegramError(RuntimeError):
    """A failed Bot API call; the message never contains the token or the request URL."""


class NotConfigured(TelegramError):
    """TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing from .env."""


def _call(method: str, token: str, payload: dict[str, Any], session: requests.Session | Any | None = None) -> Any:
    """POST one Bot API method and return its `result`; retries connection errors, 5xx and 429."""
    http = session or requests
    url = f"{API}/bot{token}/{method}"
    problem = "no attempt made"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = http.post(url, json=payload, timeout=TIMEOUT_S)
        except requests.RequestException as exc:  # its message holds the URL: keep only the type
            problem = f"{type(exc).__name__} (connection problem)"
        else:
            try:
                body = response.json()
            except ValueError:
                body = {}
            if response.status_code == 200 and body.get("ok"):
                return body.get("result")
            description = settings.redact(str(body.get("description", "no description")))
            problem = f"HTTP {response.status_code}: {description}"
            if response.status_code == 429:
                wait = float((body.get("parameters") or {}).get("retry_after", 5))
                time.sleep(min(wait, 60))
                continue
            if response.status_code < 500:
                break  # a bad request won't succeed on retry
        if attempt < ATTEMPTS:
            time.sleep(2 * attempt)
    raise TelegramError(f"Telegram {method} failed: {problem}")


def _credentials(token: str | None, chat_id: str | None = None, need_chat: bool = True) -> tuple[str, str | None]:
    token = token or settings.get_secret("TELEGRAM_BOT_TOKEN")
    if not token:
        raise NotConfigured("TELEGRAM_BOT_TOKEN is not set in .env")
    chat_id = chat_id or settings.get_secret("TELEGRAM_CHAT_ID")
    if need_chat and not chat_id:
        raise NotConfigured("TELEGRAM_CHAT_ID is not set in .env (python -m tracker digest --find-chat)")
    return token, chat_id


def fit(text: str, limit: int = MAX_LENGTH) -> str:
    """The text cut to Telegram's limit, ending with a note if anything was cut."""
    if len(text) <= limit:
        return text
    note = "\n[cut: message too long]"
    return text[: limit - len(note)].rstrip() + note


def send_message(text: str, token: str | None = None, chat_id: str | None = None, session: Any = None) -> int:
    """Send plain text (no Markdown or HTML parsing, so nothing needs escaping); returns the message id."""
    token, chat_id = _credentials(token, chat_id)
    result = _call("sendMessage", token, {"chat_id": chat_id, "text": fit(text),
                                          "link_preview_options": {"is_disabled": True}}, session)
    return int(result.get("message_id", 0)) if isinstance(result, dict) else 0


def get_updates(token: str | None = None, session: Any = None) -> list[dict[str, Any]]:
    """Pending updates (Telegram keeps them up to 24 hours, and not at all while a webhook is set)."""
    token, _ = _credentials(token, need_chat=False)
    result = _call("getUpdates", token, {"limit": 100, "timeout": 0}, session)
    return list(result or [])


def private_chats(updates: list[dict[str, Any]]) -> list[int]:
    """Ids of the private chats that wrote to the bot, most recent first, each once."""
    seen: list[int] = []
    for update in sorted(updates, key=lambda u: u.get("update_id", 0), reverse=True):
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        if chat.get("type") == "private" and isinstance(chat.get("id"), int) and chat["id"] not in seen:
            seen.append(chat["id"])
    return seen
