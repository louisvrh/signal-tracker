"""Telegram digest (S21): tracker/alerts/digest.py, run.py, telegram.py. No network: a fake session."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import requests

from tracker import settings, store
from tracker.alerts import digest, run, telegram

NOW = pd.Timestamp("2026-10-08 06:20", tz="UTC")
TOKEN = "123456:SECRET-TOKEN-VALUE"
WEEK = pd.Timestamp("2026-10-08")


class FakeResponse:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code, self._body = status, body

    def json(self) -> dict:
        return self._body


class FakeTelegram:
    """Records every Bot API call; answers from a queue (default: ok)."""

    def __init__(self, answers: list | None = None, updates: list | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.answers = list(answers or [])
        self.updates = updates or []

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        method = url.rsplit("/", 1)[-1]
        self.calls.append((method, json))
        if self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        result = self.updates if method == "getUpdates" else {"message_id": len(self.calls)}
        return FakeResponse(200, {"ok": True, "result": result})


@pytest.fixture
def secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setattr(telegram.time, "sleep", lambda s: None)


def signal_rows() -> pd.DataFrame:
    rows = []
    for country, d, z in (("CHN", -6.2, -3.0), ("DEU", -0.4, 0.3)):
        info = json.dumps({"activity_z": z, "official_z": 3.2, "official": "imf_exports"})
        rows.append({"date": WEEK, "country": country, "signal": "d_off", "value": d, "components": info})
    rows.append({"date": WEEK, "country": "JPN", "signal": "z:portwatch_imports", "value": -3.7, "components": "{}"})
    rows.append({"date": WEEK, "country": "DEU", "signal": "z:portwatch_exports", "value": 0.5, "components": "{}"})
    return pd.DataFrame(rows)


def flag_rows(on: bool = True) -> pd.DataFrame:
    dates = [WEEK - pd.Timedelta(days=14), WEEK - pd.Timedelta(days=7), WEEK]
    return pd.DataFrame({"date": dates, "country": "CHN", "signal": "d_off", "value": [-2.0, -3.0, -6.2],
                         "flag": [False, on, on], "event": [None, "on" if on else None, None]})


def inputs(**kw) -> digest.DigestInputs:
    sources = pd.DataFrame({"source": ["fred", "entsoe"], "started_at": [NOW - pd.Timedelta(hours=1)] * 2,
                            "status": ["success", "success"], "error": [None, None]})
    base = dict(now=NOW, dashboard="up", sources=sources, signals=signal_rows(), flags=flag_rows(), daily_run=True)
    return digest.DigestInputs(**{**base, **kw})


# ----------------------------------------------------------------- building


def test_nothing_new_gives_the_short_all_quiet_message() -> None:
    first = inputs()
    state = digest.state_of(first)
    kind, text, _ = digest.build(inputs(last_state=state))
    assert kind == "quiet" and "all quiet" in text and "dashboard up" in text
    assert text.count("\n") == 1 and "1 divergence flag(s) on, unchanged" in text


def test_a_failed_step_is_named_first_and_the_rest_still_follows() -> None:
    kind, text, _ = digest.build(inputs(failed_steps=["fetch", "backup"], last_state=digest.state_of(inputs())))
    first_line = text.splitlines()[0]
    assert kind == "failure" and "FAILED at: fetch, backup" in first_line
    assert "Dashboard: up" in text and "Top divergences" in text and "CHN official -6.20" in text


def test_new_since_the_last_digest() -> None:
    last = {"flags": ["DEU:d_off"], "anomalies": [], "stale": ["fred_copper"]}
    stale = pd.DataFrame({"indicator": ["fred_eu_gas"], "last_date": ["2026-07-01"]})
    kind, text, state = digest.build(inputs(last_state=last, stale=stale))
    assert kind == "full" and state["flags"] == ["CHN:d_off"] and state["anomalies"] == ["JPN:portwatch_imports"]
    assert "+ CHN official divergence flag ON (-6.20)" in text
    assert "- DEU official divergence flag off" in text
    assert "! anomaly: JPN portwatch_imports z -3.70" in text
    assert "Newly stale: fred_eu_gas" in text and "Fresh again: fred_copper" in text
    assert "1. CHN official -6.20 (2 wk): activity z -3.00 vs imf_exports z +3.20" in text


def test_the_first_digest_counts_last_weeks_flag_events_as_new() -> None:
    _, text, _ = digest.build(inputs())
    assert "+ CHN official divergence flag ON" in text


def test_dashboard_trouble_and_source_problems_are_news() -> None:
    state = digest.state_of(inputs())
    sources = pd.DataFrame({"source": ["fred", "neso", "imf"],
                            "started_at": [NOW - pd.Timedelta(hours=1), NOW - pd.Timedelta(hours=30), pd.NaT],
                            "status": ["failed", "success", None],
                            "error": ["HTTP 502 from the API\nmore detail", None, None]})
    kind, text, _ = digest.build(inputs(sources=sources, dashboard="restarted", last_state=state))
    assert kind == "full" and "Dashboard: restarted" in text
    assert "- fred: FAILED: HTTP 502 from the API" in text and "more detail" not in text
    assert "- neso: no fetch since 07 Oct 00:20 UTC" in text and "- imf: no fetch since never" in text
    assert digest.build(inputs(dashboard="off", last_state=state))[0] == "quiet"  # a deliberate stop isn't news


def test_long_lists_are_capped_for_a_phone() -> None:
    rows = [{"date": WEEK, "country": f"C{n:02d}", "signal": "z:portwatch_exports", "value": -2.0 - n / 10,
             "components": "{}"} for n in range(10)]
    _, text, _ = digest.build(inputs(signals=pd.DataFrame(rows), flags=pd.DataFrame()))
    lines = [line for line in text.splitlines() if line.startswith("! anomaly")]
    assert len(lines) == digest.MAX_LINES and "C09" in lines[0]  # largest |z| first
    assert "... and 3 more anomalies" in text  # 9 above |z| 2 (C00's -2.0 isn't)
    assert len(telegram.fit("x" * 5000)) == telegram.MAX_LENGTH


# ------------------------------------------------------ sending and recording


def test_run_digest_sends_and_records_and_a_dry_run_does_neither(secrets, monkeypatch) -> None:
    fake = FakeTelegram()
    monkeypatch.setattr(run, "collect", lambda failed, dash, daily, data_dir=None: inputs(failed_steps=failed))
    dry = run.run_digest(dry_run=True, dashboard="up", session=fake)
    assert not dry.sent and fake.calls == [] and store.read_digests().empty
    sent = run.run_digest(dashboard="up", daily_run=True, session=fake)
    method, payload = fake.calls[0]
    assert sent.sent and method == "sendMessage" and payload["chat_id"] == "42" and payload["text"] == sent.text
    assert "parse_mode" not in payload  # plain text: nothing to escape
    log = store.read_digests()
    assert len(log) == 1 and json.loads(log.loc[0, "state"])["flags"] == ["CHN:d_off"]
    # The next digest compares against that state: nothing new, so all quiet.
    monkeypatch.setattr(run, "collect", lambda failed, dash, daily, data_dir=None: inputs(
        last_state=json.loads(store.read_digests()["state"].iloc[-1])))
    assert run.run_digest(dashboard="up", session=fake).kind == "quiet"


def test_a_digest_that_cannot_be_built_still_sends_a_failure_message(secrets, monkeypatch) -> None:
    fake = FakeTelegram()

    def broken(*args, **kwargs):
        raise KeyError("no such column")

    monkeypatch.setattr(run, "collect", broken)
    result = run.run_digest(failed_steps=["signals"], dashboard="up", daily_run=True, session=fake)
    assert result.sent and result.kind == "failure"
    assert "FAILED at: signals" in result.text and "digest itself failed too (KeyError)" in result.text
    assert store.read_digests().empty  # nothing to compare against next time


def test_telegram_errors_never_contain_the_token(secrets) -> None:
    url_error = requests.ConnectionError(f"Max retries for url: /bot{TOKEN}/sendMessage")
    fake = FakeTelegram([url_error, url_error, url_error])
    with pytest.raises(telegram.TelegramError) as caught:
        telegram.send_message("hi", session=fake)
    assert TOKEN not in str(caught.value) and "ConnectionError" in str(caught.value) and len(fake.calls) == 3
    fake = FakeTelegram([FakeResponse(400, {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"})])
    with pytest.raises(telegram.TelegramError, match="HTTP 400: Bad Request: chat not found"):
        telegram.send_message("hi", session=fake)
    assert len(fake.calls) == 1  # a bad request isn't retried
    fake = FakeTelegram([FakeResponse(429, {"ok": False, "parameters": {"retry_after": 1}})])
    assert telegram.send_message("hi", session=fake) == 2  # retried after the wait


def test_missing_settings_are_explained(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    with pytest.raises(telegram.TelegramError, match="TELEGRAM_CHAT_ID is not set"):
        telegram.send_message("hi", session=FakeTelegram())


def test_find_chat_saves_the_only_private_chat_and_never_reads_env(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("OTHER=1", encoding="utf-8")  # no trailing newline
    monkeypatch.setattr(settings, "ENV_FILE", env)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert "No message to the bot found" in run.find_chat(session=FakeTelegram(updates=[]))
    two = [{"update_id": 1, "message": {"chat": {"id": 7, "type": "private"}}},
           {"update_id": 2, "message": {"chat": {"id": 8, "type": "private"}}}]
    assert "2 different private chats" in run.find_chat(session=FakeTelegram(updates=two))
    one = [{"update_id": 5, "message": {"chat": {"id": 99, "type": "private"}}},
           {"update_id": 6, "message": {"chat": {"id": -100, "type": "group"}}},
           {"update_id": 7, "edited_message": {"chat": {"id": 99, "type": "private"}}}]
    fake = FakeTelegram(updates=one)
    assert "Saved TELEGRAM_CHAT_ID" in run.find_chat(session=fake)
    assert env.read_text(encoding="utf-8").splitlines() == ["OTHER=1", "TELEGRAM_CHAT_ID=99"]
    assert fake.calls[-1][0] == "sendMessage" and fake.calls[-1][1]["chat_id"] == "99"


def test_digest_command_exit_codes(secrets, monkeypatch, capsys) -> None:
    from tracker.__main__ import main

    monkeypatch.setattr(run, "collect", lambda failed, dash, daily, data_dir=None: inputs(failed_steps=failed))
    assert main(["digest", "--dry-run", "--failed-steps", "fetch,quality", "--dashboard", "up"]) == 0
    out = capsys.readouterr().out
    assert "[dry run: failure digest" in out and "FAILED at: fetch, quality" in out
    monkeypatch.delenv("TELEGRAM_CHAT_ID")
    assert main(["digest", "--dashboard", "up"]) == 0  # not set up yet: the text goes to the log
    out = capsys.readouterr()
    assert "NOT SENT" in out.out and "FAILED" not in out.out.splitlines()[0] and TOKEN not in out.out + out.err
    assert store.read_digests().empty
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    broken = FakeTelegram([FakeResponse(401, {"ok": False, "description": "Unauthorized"})])
    monkeypatch.setattr(telegram, "requests", type("R", (), {"post": broken.post, "RequestException": Exception}))
    assert main(["digest", "--dashboard", "up"]) == 1  # a real send failure fails the step
    assert TOKEN not in capsys.readouterr().err
