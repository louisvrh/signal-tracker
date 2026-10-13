"""Smoke tests for the S1 scaffold: CLI, settings and secret handling."""

from __future__ import annotations

import logging

import pytest

from tracker import settings
from tracker.__main__ import build_parser


def test_cli_lists_all_commands() -> None:
    help_text = build_parser().format_help()
    for command in ("fetch", "status", "quality", "signals", "digest", "compact"):
        assert command in help_text


def test_cli_parses_fetch_options() -> None:
    args = build_parser().parse_args(["fetch", "--source", "fred", "--backfill", "--since", "2015-01-01"])
    assert args.source == "fred" and args.backfill and args.since == "2015-01-01"


def test_paths_are_under_project_root() -> None:
    import subprocess
    import sys

    assert settings.CONFIG_DIR == settings.ROOT / "config"
    # conftest redirects LOGS_DIR/DATA_DIR for isolation, so check the real
    # defaults in a fresh interpreter.
    code = "from tracker import settings as s; print(s.LOGS_DIR == s.ROOT / 'logs', s.DATA_DIR == s.ROOT / 'data')"
    env = {k: v for k, v in __import__("os").environ.items() if k != "TRACKER_DATA_DIR"}
    out = subprocess.run([sys.executable, "-c", code], cwd=settings.ROOT, capture_output=True, text=True, env=env)
    assert out.stdout.split() == ["True", "True"], out.stderr


def test_load_config_reads_sources() -> None:
    sources = settings.load_sources()
    assert "fred" in sources


def test_load_config_missing_file_is_empty(tmp_path) -> None:
    assert settings.load_config("nope", config_dir=tmp_path) == {}


def test_redact_removes_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "abcd1234secret")
    text = "GET https://x/y?api_key=abcd1234secret&file_type=json"
    assert "abcd1234secret" not in settings.redact(text)


def test_log_output_is_redacted(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("EIA_API_KEY", "zzzz9999secret")
    settings.setup_logging(log_dir=tmp_path)
    logging.getLogger("tracker.test").warning("url=%s", "https://x?api_key=zzzz9999secret")
    for handler in logging.getLogger().handlers:
        handler.flush()
    content = (tmp_path / "tracker.log").read_text(encoding="utf-8")
    assert "zzzz9999secret" not in content
    assert "api_key=***" in content
    for handler in list(logging.getLogger().handlers):
        handler.close()
        logging.getLogger().removeHandler(handler)
