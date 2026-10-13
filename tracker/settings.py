"""Project settings: paths, secrets from .env, config files and logging.

Secrets are read from `.env` into the process environment and must only be
accessed through `get_secret()` / `has_secret()`. Never print or log them;
use `redact()` on any text (such as an exception message containing a URL)
before it is logged or stored.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
PRESETS_FILE = CONFIG_DIR / "presets.yaml"  # workbench presets: the dashboard's one write (PLAN D32)
DATA_DIR = Path(os.environ.get("TRACKER_DATA_DIR", str(ROOT / "data")))
RAW_DIR = DATA_DIR / "raw"
LAKE_DIR = DATA_DIR / "lake"
META_DIR = DATA_DIR / "meta"
SIGNALS_DIR = DATA_DIR / "signals"
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"
FIXTURES_DIR = ROOT / "tests" / "fixtures"

ENV_FILE = ROOT / ".env"

# Names of every secret the project uses (values live only in .env).
SECRET_NAMES = (
    "FRED_API_KEY",
    "EIA_API_KEY",
    "ENTSOE_API_TOKEN",
    "COMTRADE_SUBSCRIPTION_KEY",
    "EMBER_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
)

load_dotenv(ENV_FILE, override=False)


def get_secret(name: str) -> str | None:
    """Return a secret from the environment, or None if unset or empty."""
    value = os.environ.get(name, "").strip()
    return value or None


def has_secret(name: str) -> bool:
    """Return True if the named secret is set (never reveals the value)."""
    return get_secret(name) is not None


def get_setting(name: str, default: str | None = None) -> str | None:
    """Return a non-secret setting from the environment (e.g. BACKUP_DIR)."""
    value = os.environ.get(name, "").strip()
    return value or default


def redact(text: str) -> str:
    """Replace any secret value found in `text` with '***'."""
    for name in SECRET_NAMES:
        value = get_secret(name)
        if value and len(value) >= 4:
            text = text.replace(value, "***")
    return text


def load_config(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load `config/<name>.yaml` and return its contents (empty dict if absent)."""
    path = (config_dir or CONFIG_DIR) / f"{name}.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_sources(config_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the `sources` mapping from config/sources.yaml."""
    return load_config("sources", config_dir).get("sources", {}) or {}


class _RedactingFormatter(logging.Formatter):
    """Formatter that strips secret values from the whole formatted record,
    including tracebacks (which can contain request URLs with keys)."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(verbose: bool = False, log_dir: Path | None = None) -> None:
    """Log to a rotating file (logs/tracker.log) and to the console.

    The file always receives INFO and above; the console receives WARNING and
    above, or INFO with `verbose=True`.
    """
    log_dir = log_dir or LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(
        log_dir / "tracker.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(_RedactingFormatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    # Third-party HTTP libraries log full URLs (which may contain keys) at DEBUG;
    # entsoe-py logs its request parameters, including the security token.
    for noisy in ("urllib3", "requests", "yfinance", "peewee", "entsoe"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
