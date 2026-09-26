"""Session-wide guard for the CAD web UI tests.

`webui/app.py` is a singleton module: several test files `import app` directly (rather
than only going through FastAPI's TestClient) to reach its internals, and Python caches
that import — every test file in one pytest run shares the SAME `_jobs` dict, the SAME
background worker threads, and (before this fix) the SAME real SESSIONS_FILE, which is the
owner's actual creation history at ~/.openclaw/cad-web/sessions.json.

That sharing is exactly what caused the 2026-09-26 23:58 incident: a test module cleared
`_jobs` in a per-test setup fixture while a real background worker thread's `finally:`
block was concurrently calling the real `_persist()` — the race landed with `_jobs` empty
at that instant, and the owner's ~200-row history got atomically overwritten with `[]`.

app.py now honours CAD_WEB_SESSIONS_FILE to redirect where it persists/loads history (see
its SESSIONS_FILE definition). Setting it here, at module scope in conftest.py, is what
guarantees it happens before ANY test module's `import app` — conftest.py is always
collected first, whichever test file pytest happens to run first. No webui test should
ever be able to touch the real file again, no matter what future test adds a `_jobs.clear()`
or a new persist-triggering endpoint call.
"""
import os
import tempfile
from pathlib import Path

_tmp_dir = tempfile.mkdtemp(prefix="cad-web-test-sessions-")
os.environ.setdefault("CAD_WEB_SESSIONS_FILE", str(Path(_tmp_dir) / "sessions.json"))
