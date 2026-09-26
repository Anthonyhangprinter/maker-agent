"""Guards against a repeat of the 2026-09-26 23:58 incident: a test run overwrote the
owner's real ~/.openclaw/cad-web/sessions.json (~200 creations) with `[]`, no backup.

Covers the two independent fixes in webui/app.py:
  1. SESSIONS_FILE is env-overridable (CAD_WEB_SESSIONS_FILE) — conftest.py sets that env
     var before any test module can `import app`, so no webui test touches the real file.
     This file asserts the override actually took effect.
  2. `_persist()` refuses to replace a non-empty on-disk file with an empty row list
     unless the caller says a deletion just happened (`allow_empty=True`), and rotates one
     dated backup per day, pruning anything past the last 7.

No subprocess, no real build, no GPU — pure file-state assertions against `_persist`/
`_rotate_backup` called directly.

Run: python3 -m pytest tests/test_webui_history_persist.py -q
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webui"))

import app  # noqa: E402


def setup_function(_fn):
    app._jobs.clear()


def test_sessions_file_env_override_is_in_effect():
    # conftest.py sets CAD_WEB_SESSIONS_FILE before this module (or any other webui test
    # module) can import app.py — if this ever drifts back to the real path, every other
    # assertion in this file would be exercising the owner's actual history.
    assert str(app.SESSIONS_FILE) == os.environ["CAD_WEB_SESSIONS_FILE"]
    assert ".openclaw/cad-web/sessions.json" not in str(app.SESSIONS_FILE) \
        or "cad-web-test-sessions-" in str(app.SESSIONS_FILE)


def test_persist_refuses_to_overwrite_nonempty_file_with_empty_store(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    existing = [{"id": "real-job-1", "spec": "a real creation", "created_at": 1.0}]
    sessions_file.write_text(json.dumps(existing))
    monkeypatch.setattr(app, "SESSIONS_FILE", sessions_file)

    app._jobs.clear()          # simulates the race: in-memory store is empty
    app._persist()             # allow_empty defaults to False — no delete happened

    assert json.loads(sessions_file.read_text()) == existing


def test_persist_allows_empty_after_an_actual_delete(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    existing = [{"id": "real-job-1", "spec": "a real creation", "created_at": 1.0}]
    sessions_file.write_text(json.dumps(existing))
    monkeypatch.setattr(app, "SESSIONS_FILE", sessions_file)

    app._jobs.clear()
    app._persist(allow_empty=True)          # api_delete_job's own call, last job removed

    assert json.loads(sessions_file.read_text()) == []


def test_persist_writes_normally_when_jobs_are_present(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    monkeypatch.setattr(app, "SESSIONS_FILE", sessions_file)

    app._jobs.clear()
    app._jobs["job1"] = {"id": "job1", "spec": "a bracket", "status": "done",
                          "created_at": time.time(), "updated_at": time.time(),
                          "coder": "auto", "image": None, "result": None, "error": None}
    app._persist()

    rows = json.loads(sessions_file.read_text())
    assert [r["id"] for r in rows] == ["job1"]


def test_persist_overwrites_an_already_empty_existing_file(tmp_path, monkeypatch):
    # An existing file that is ALREADY `[]` is not "history to protect" — the guard must
    # only fire when it would actually destroy something.
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text("[]")
    monkeypatch.setattr(app, "SESSIONS_FILE", sessions_file)

    app._jobs.clear()
    app._jobs["job1"] = {"id": "job1", "spec": "x", "status": "done",
                          "created_at": time.time(), "updated_at": time.time(),
                          "coder": "auto", "image": None, "result": None, "error": None}
    app._persist()

    assert [r["id"] for r in json.loads(sessions_file.read_text())] == ["job1"]


def test_rotate_backup_creates_one_dated_copy_and_prunes_to_seven(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text(json.dumps([{"id": "x", "created_at": 1.0}]))
    monkeypatch.setattr(app, "SESSIONS_FILE", sessions_file)

    # Seed 8 fake older backups (day-stamped names sort chronologically as strings).
    old_days = [f"202601{n:02d}" for n in range(1, 9)]
    for day in old_days:
        (tmp_path / f"sessions.json.bak-{day}").write_text("[]")

    app._rotate_backup()

    baks = sorted(p.name for p in tmp_path.glob("sessions.json.bak-*"))
    assert len(baks) == app.SESSIONS_BACKUPS_KEPT
    # the two oldest fake backups were pruned to make room for today's new one
    assert "sessions.json.bak-20260101" not in baks
    assert "sessions.json.bak-20260102" not in baks
    today = time.strftime("%Y%m%d")
    assert f"sessions.json.bak-{today}" in baks
    assert json.loads((tmp_path / f"sessions.json.bak-{today}").read_text()) == \
        [{"id": "x", "created_at": 1.0}]


def test_rotate_backup_is_a_noop_when_todays_backup_already_exists(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text(json.dumps([{"id": "x", "created_at": 1.0}]))
    monkeypatch.setattr(app, "SESSIONS_FILE", sessions_file)
    today = time.strftime("%Y%m%d")
    bak = tmp_path / f"sessions.json.bak-{today}"
    bak.write_text("sentinel — must not be touched again today")

    app._rotate_backup()

    assert bak.read_text() == "sentinel — must not be touched again today"


def test_rotate_backup_does_nothing_when_no_sessions_file_exists_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SESSIONS_FILE", tmp_path / "sessions.json")
    app._rotate_backup()          # must not raise
    assert list(tmp_path.glob("*.bak-*")) == []
