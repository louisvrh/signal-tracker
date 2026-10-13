"""Test-wide safety: never write to the real logs/, data/ or config/presets.yaml."""

from __future__ import annotations

import logging
import shutil

import pytest

from tracker import settings


@pytest.fixture(autouse=True)
def _isolate_logs_and_data(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path_factory.mktemp("isolated")
    monkeypatch.setattr(settings, "LOGS_DIR", root / "logs")
    monkeypatch.setattr(settings, "DATA_DIR", root / "data")
    monkeypatch.setattr(settings, "RAW_DIR", root / "data" / "raw")
    # Saving a preset writes config/presets.yaml: tests get their own copy.
    presets = root / "presets.yaml"
    if settings.PRESETS_FILE.exists():
        shutil.copyfile(settings.PRESETS_FILE, presets)
    monkeypatch.setattr(settings, "PRESETS_FILE", presets)
    yield
    for handler in list(logging.getLogger().handlers):
        handler.close()
        logging.getLogger().removeHandler(handler)
