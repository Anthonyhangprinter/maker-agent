# Phase 0 "Instruments" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command reproduces a benchmark card of every local candidate model on the internal suites plus two public text-to-CAD suites, served through a new swappable "maker" server, so the strong-rung base model can be chosen by a written rule.

**Architecture:** A second llama-server user unit (`maker-server`, port 8088) is started by the CAD engine's existing eviction hooks in place of the resident when `cad.json` has an enabled `maker` block. `benchmarks/arms.json` declares candidate models; `scripts/arms.py` downloads them and points the maker server at one. `scripts/run_card.py` iterates arms and suites (internal suites plus new `benchmarks/cadprompt` and `benchmarks/cad-arena` suites built by `scripts/fetch_external.py`), runs one-shot builds through `scripts/fluid_gen.py`, scores validity, gate results, acceptance and unit-normalised Chamfer bands, and writes `benchmarks/results/card/<stamp>/card.{json,md}`. The web UI gets a read-only Lab view of the latest card.

**Tech Stack:** Python 3.12 system interpreter (build123d installed), llama.cpp CUDA build at `~/llama.cpp-cuda-src/build/bin`, systemd user units, `hf` CLI, trimesh 4.12 + open3d 0.19 (already installed) via `~/repos/cadqueryeval`, FastAPI web UI in `webui/.venv` (Python 3.11), pytest (`python3 -m pytest tests/`).

**Spec:** `docs/MAKER-1.0-CAMPAIGN.md` sections 4.1, 4.1a, 4.2, 4.5, 5 (Phase 0 row), 6.

## Global Constraints

- All local at runtime: no cloud calls in any script, timer or button this plan creates.
- Every card spec must be excluded from training data: the guard in `scripts/harvest_census.py:suite_slugs()` must cover every suite directory the card reads.
- One GPU: any server start goes through the engine's stop/start hooks so the resident is restored in a `finally:`.
- Model files live under `/mnt/nvme-apps/LinuxModels/`; nothing model-sized on the root SSD.
- No em dashes in any user-facing copy (README, web UI strings, card.md).
- Commit after every task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Never edit `run_agent.py`, `cli.py`, `gateway/run.py` (Hermes core, not in this repo anyway).
- Tests run from the skill dir: `cd ~/.openclaw/skills/cad-builder && python3 -m pytest tests/ -q`.

---

## File map

| File | Responsibility |
|---|---|
| `cad_v5/config.py` (modify) | `maker_config()`, `LOCAL_CODER_PORT/URL/HEALTH` derived from the maker block, `CAD_CONFIG_FILE` env override |
| `cad_engine.py` (modify :206-284, :461-506) | swap hooks aware of the maker unit; `_LAST_USAGE` capture; stale comment |
| `~/.local/bin/maker-server` (create, outside repo, copy kept at `deploy/maker-server`) | launcher reading `~/.openclaw/maker.env` |
| `~/.config/systemd/user/maker-server.service` (create; copy at `deploy/maker-server.service`) | user unit, `Conflicts=qwen38-server.service` |
| `benchmarks/arms.json` (create) | candidate model declarations |
| `scripts/arms.py` (create) | `list / download / use / restore` |
| `scripts/fetch_external.py` (create) | builds `benchmarks/cadprompt/`, `benchmarks/cad-arena/`, optional `benchmarks/text2cadquery/` |
| `scripts/geom_bands.py` (modify) | `normalize=True` path: scale both meshes to diagonal 100 before scoring |
| `scripts/harvest_census.py` (modify :158-168) | `suite_slugs()` covers the new suite dirs |
| `scripts/run_benchcad.py` (create) + `benchmarks/external/benchcad/local_adapter.py` | official BenchCAD harness per arm |
| `scripts/run_card.py` (create) | the card runner and writer |
| `scripts/card_report.py` (create) | pure functions: summarise rows, render markdown |
| `webui/app.py` (modify) + `webui/static/index.html` (modify) | `/api/lab/card`, Lab view, stale label |
| `README.md` (modify :3,:7,:59) | hardware truth lines |
| `tests/test_maker_config.py`, `tests/test_arms.py`, `tests/test_external_suites.py`, `tests/test_geom_normalize.py`, `tests/test_card_report.py` (create) | offline tests |

---

### Task 1: maker block in config

**Files:**
- Modify: `cad_v5/config.py:60-68` (LOCAL_CODER_URL block) and `:194-212` (`load_config`)
- Test: `tests/test_maker_config.py`

**Interfaces:**
- Produces: `maker_config() -> dict` returning `{"enabled": bool, "port": int, "alias": str, "unit": str}` with defaults `{"enabled": False, "port": 8086, "alias": "qwen3.8-27b", "unit": "qwen38-server"}`; module constants `LOCAL_CODER_PORT: int`, `LOCAL_CODER_URL: str`, `LOCAL_CODER_HEALTH: str`, `CODE_MODEL_STRONG: str` computed from it at import; `CAD_CONFIG_FILE` honours env `CAD_CONFIG_FILE`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_maker_config.py
import importlib, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def _reload_with(tmp_path, cad_json: dict):
    p = tmp_path / "cad.json"
    p.write_text(json.dumps(cad_json))
    os.environ["CAD_CONFIG_FILE"] = str(p)
    import cad_v5.config as cfg
    return importlib.reload(cfg)


def test_defaults_point_at_resident(tmp_path):
    cfg = _reload_with(tmp_path, {})
    m = cfg.maker_config()
    assert m == {"enabled": False, "port": 8086, "alias": "qwen3.8-27b", "unit": "qwen38-server"}
    assert cfg.LOCAL_CODER_URL == "http://127.0.0.1:8086/v1/chat/completions"
    assert cfg.CODE_MODEL_STRONG == "local:qwen3.8-27b"


def test_enabled_maker_rewrites_port_and_alias(tmp_path):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088, "alias": "gemma-4-31b"}})
    m = cfg.maker_config()
    assert m["enabled"] and m["port"] == 8088 and m["unit"] == "maker-server"
    assert cfg.LOCAL_CODER_URL == "http://127.0.0.1:8088/v1/chat/completions"
    assert cfg.LOCAL_CODER_HEALTH == "http://127.0.0.1:8088/health"
    assert cfg.CODE_MODEL_STRONG == "local:gemma-4-31b"
    assert cfg.CODE_MODEL_LADDER[1] == "local:gemma-4-31b"


def teardown_module(module):
    os.environ.pop("CAD_CONFIG_FILE", None)
    import cad_v5.config as cfg
    importlib.reload(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_maker_config.py -q`
Expected: FAIL with `AttributeError: module 'cad_v5.config' has no attribute 'maker_config'`

- [ ] **Step 3: Implement**

In `cad_v5/config.py`, make the config path env-overridable (find the existing `CAD_CONFIG_FILE = ...` assignment near the top, around the `_OPENCLAW` definitions) and replace it with:

```python
CAD_CONFIG_FILE = Path(os.environ.get("CAD_CONFIG_FILE", str(_OPENCLAW / "cad.json")))
```

Move `load_config()` (currently :194-212) ABOVE the model-constant block so it is defined before line 60, then replace the block at :64-68 with:

```python
def maker_config() -> dict:
    """The optional swappable CAD coder server ("maker" block in cad.json).

    Disabled (default): the strong rung is the resident on :8086.
    Enabled: the strong rung is `maker-server` on `port`, serving `alias`.
    """
    m = load_config().get("cad", {}).get("maker") or {}
    enabled = bool(m.get("enabled", False))
    return {
        "enabled": enabled,
        "port": int(m.get("port", 8088)) if enabled else 8086,
        "alias": str(m.get("alias", "qwen3.8-27b")),
        "unit": "maker-server" if enabled else "qwen38-server",
    }


_MAKER = maker_config()
CODE_MODEL_STRONG  = "local:" + _MAKER["alias"]
LOCAL_CODER_PORT   = _MAKER["port"]
LOCAL_CODER_URL    = f"http://127.0.0.1:{LOCAL_CODER_PORT}/v1/chat/completions"
LOCAL_CODER_HEALTH = f"http://127.0.0.1:{LOCAL_CODER_PORT}/health"
```

Keep `CODE_MODEL_LADDER = [CODE_MODEL_FAST, CODE_MODEL_STRONG]` after it (already :74). Update the comment above (was "port 8085 elsewhere") to say "resident :8086, or the maker server when cad.json maker.enabled".

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_maker_config.py tests/ -q`
Expected: new tests PASS; the pre-existing tests unchanged (note any that were already failing before this task in the commit message, do not fix them here).

- [ ] **Step 5: Commit**

```bash
git add cad_v5/config.py tests/test_maker_config.py
git commit -m "config: maker block (swappable CAD coder server) drives the strong rung URL and alias"
```

---

### Task 2: maker-server launcher and unit

**Files:**
- Create: `deploy/maker-server` (bash), `deploy/maker-server.service`, `deploy/README.md` (three lines: what the two files are, the install command)
- Install: copy to `~/.local/bin/maker-server` (chmod +x) and `~/.config/systemd/user/maker-server.service`, then `systemctl --user daemon-reload`

**Interfaces:**
- Consumes: `~/.openclaw/maker.env` written by Task 4 with keys `MODEL`, `MMPROJ` (may be empty), `CTX`, `PORT`, `ALIAS`, `EXTRA_ARGS`.
- Produces: `maker-server.service` user unit; health at `http://127.0.0.1:$PORT/health`.

- [ ] **Step 1: Write the launcher**

```bash
#!/usr/bin/env bash
# maker-server: the swappable CAD coder server (Maker Agent 1.0 campaign, docs/MAKER-1.0-CAMPAIGN.md 4.2).
# Reads ~/.openclaw/maker.env (written by scripts/arms.py use <arm>). Never runs beside qwen38-server.
set -euo pipefail
ENV_FILE="${MAKER_ENV:-$HOME/.openclaw/maker.env}"
[ -f "$ENV_FILE" ] || { echo "maker-server: $ENV_FILE missing; run scripts/arms.py use <arm>" >&2; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"
DIR="${DIR:-$HOME/llama.cpp-cuda-src/build/bin}"
PORT="${PORT:-8088}"
CTX="${CTX:-16384}"
: "${MODEL:?maker.env must set MODEL}"
: "${ALIAS:?maker.env must set ALIAS}"

# Evict lingering Ollama guests (same reason as qwen38-server: CUDA alloc fails beside a keepalive'd guest).
if curl -sf --max-time 2 http://127.0.0.1:11434/api/ps >/dev/null 2>&1; then
  for m in $(curl -s http://127.0.0.1:11434/api/ps | python3 -c 'import json,sys;[print(x["name"]) for x in json.load(sys.stdin).get("models",[])]'); do
    curl -s -X POST http://127.0.0.1:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null || true
  done
fi

MMPROJ_FLAG=()
[ -n "${MMPROJ:-}" ] && MMPROJ_FLAG=(--mmproj "$MMPROJ")
# shellcheck disable=SC2086
exec env LD_LIBRARY_PATH="$DIR" "$DIR/llama-server" \
  -m "$MODEL" "${MMPROJ_FLAG[@]}" \
  --alias "$ALIAS" -ngl 999 \
  -c "$CTX" -np 1 --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 -t 6 -fa on --jinja \
  --host 127.0.0.1 --port "$PORT" ${EXTRA_ARGS:-} "$@"
```

- [ ] **Step 2: Write the unit**

```ini
# deploy/maker-server.service  ->  ~/.config/systemd/user/maker-server.service
[Unit]
Description=Maker Agent CAD coder server (swappable llama.cpp arm, RTX 3090)
After=network.target
Conflicts=qwen38-server.service

[Service]
ExecStart=%h/.local/bin/maker-server
Restart=no
MemoryHigh=8G

[Install]
WantedBy=default.target
```

`Restart=no` on purpose: a crashed arm must not crash-loop against the resident; the engine's hook restores the resident.

- [ ] **Step 3: Install and smoke with the stock 27B**

```bash
cd ~/.openclaw/skills/cad-builder
install -m 755 deploy/maker-server ~/.local/bin/maker-server
install -m 644 deploy/maker-server.service ~/.config/systemd/user/maker-server.service
systemctl --user daemon-reload
cat > ~/.openclaw/maker.env <<'EOF'
MODEL=/mnt/nvme-apps/LinuxModels/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMPROJ=/mnt/nvme-apps/LinuxModels/models/mmproj-Qwen3.8-27B-F16.gguf
CTX=16384
PORT=8088
ALIAS=qwen3.8-27b
EXTRA_ARGS=--spec-type draft-mtp
EOF
systemctl --user start maker-server            # Conflicts= stops qwen38-server
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8088/health && break; sleep 2; done
curl -s http://127.0.0.1:8088/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
nvidia-smi --query-gpu=memory.used --format=csv
systemctl --user stop maker-server && systemctl --user start qwen38-server
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8086/health && break; sleep 2; done
systemctl --user is-active qwen38-server maker-server
```

Expected: health JSON on :8088, "OK" reply, VRAM well under 24GB at 16k ctx, then `active` / `inactive`. Paste the output into the task report.

- [ ] **Step 4: Commit**

```bash
git add deploy/
git commit -m "deploy: maker-server launcher + user unit (swappable CAD coder arm, Conflicts= the resident)"
```

---

### Task 3: engine swap hooks honour the maker unit

**Files:**
- Modify: `cad_engine.py:206-284` (`_QWEN36_UNIT`, `_pause_default_server_for`, `_resume_default_server`, `_ensure_default_server`), `:461-506` (`local:` branch), `:204` stale comment
- Test: `tests/test_maker_config.py` (extend with a hook test using a stubbed `subprocess.run`)

**Interfaces:**
- Consumes: `cad_v5.config.maker_config()`, `LOCAL_CODER_HEALTH` (Task 1).
- Produces: `_ensure_default_server(timeout=180)` starts the maker unit (stopping the resident) when maker is enabled, else the resident as today; `_resume_default_server()` stops the maker unit and restarts the resident when the maker was started; module global `_LAST_USAGE: dict | None` set from the last `local:` response's `usage`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_maker_config.py`:

```python
def test_ensure_hook_starts_maker_and_resume_restores(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088, "alias": "arm-x"}})
    import cad_engine
    importlib.reload(cad_engine)
    calls = []
    monkeypatch.setattr(cad_engine.subprocess, "run", lambda argv, **kw: calls.append(list(argv)))
    monkeypatch.setattr(cad_engine, "_unload_ollama_guests", lambda *_a, **_k: None)
    monkeypatch.setattr(cad_engine, "_wait_health", lambda url, timeout: None)
    cad_engine._ensure_default_server(timeout=1)
    assert ["systemctl", "--user", "stop", "qwen38-server"] in calls
    assert ["systemctl", "--user", "start", "maker-server"] in calls
    calls.clear()
    cad_engine._resume_default_server()
    assert calls == [["systemctl", "--user", "stop", "maker-server"],
                     ["systemctl", "--user", "start", "qwen38-server"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_maker_config.py::test_ensure_hook_starts_maker_and_resume_restores -q`
Expected: FAIL (`_wait_health` missing, or `maker-server` never started).

- [ ] **Step 3: Implement**

In `cad_engine.py` at the block starting :206:

```python
from cad_v5.config import maker_config  # add to the existing config import at :93

_QWEN36_UNIT = "qwen38-server"          # the resident (name kept for grep-ability)
_MAKER_UNIT  = "maker-server"           # the swappable CAD coder arm (docs/MAKER-1.0-CAMPAIGN.md 4.2)
_MAKER_STARTED = False


def _wait_health(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                json.loads(r.read()); return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"{url} did not become healthy within {timeout}s")
```

Rewrite `_ensure_default_server` (:264-284) as:

```python
def _ensure_default_server(timeout: int = 180) -> None:
    """Make the strong-rung server answer on LOCAL_CODER_HEALTH.

    Maker disabled: start the resident (as before).
    Maker enabled: stop the resident, start maker-server, remember to restore.
    """
    global _MAKER_STARTED
    _unload_ollama_guests(0.0)
    m = maker_config()
    if m["enabled"]:
        subprocess.run(["systemctl", "--user", "stop", _QWEN36_UNIT], check=False)
        subprocess.run(["systemctl", "--user", "start", _MAKER_UNIT], check=False)
        _MAKER_STARTED = True
    else:
        subprocess.run(["systemctl", "--user", "start", _QWEN36_UNIT], check=False)
    _wait_health(LOCAL_CODER_HEALTH, timeout)
```

Rewrite `_resume_default_server` (:232-242) as:

```python
def _resume_default_server() -> None:
    global _PAUSED_DEFAULT_SERVER, _MAKER_STARTED
    if not (_PAUSED_DEFAULT_SERVER or _MAKER_STARTED):
        return
    _unload_ollama_guests(0.0)
    if _MAKER_STARTED:
        subprocess.run(["systemctl", "--user", "stop", _MAKER_UNIT], check=False)
        _MAKER_STARTED = False
    subprocess.run(["systemctl", "--user", "start", _QWEN36_UNIT], check=False)
    _PAUSED_DEFAULT_SERVER = False
```

In the `local:` branch (:461-506), after `resp = json.loads(...)` add:

```python
    global _LAST_USAGE
    _LAST_USAGE = resp.get("usage")
```

and declare `_LAST_USAGE: Optional[dict] = None` next to `_ACTIVE_CODE_MODEL` (:194). Fix the stale comment at :204 to read `# resident :8086, or maker-server when cad.json maker.enabled (see cad_v5.config.maker_config)`.

Also in `scripts/fluid_gen.py:_result()` (:155-166) add `"usage": getattr(engine, "_LAST_USAGE", None)` to the dict so the card can count tokens.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/ -q`
Expected: PASS (plus any pre-existing failures noted in Task 1).

- [ ] **Step 5: Live check, maker disabled (no behaviour change)**

```bash
cd ~/.openclaw/skills/cad-builder
python3 -c "import cad_engine; print(cad_engine.maker_config(), cad_engine.LOCAL_CODER_HEALTH)"
```
Expected: `{'enabled': False, 'port': 8086, ...} http://127.0.0.1:8086/health`.

- [ ] **Step 6: Commit**

```bash
git add cad_engine.py scripts/fluid_gen.py tests/test_maker_config.py
git commit -m "engine: swap hooks start/stop maker-server when cad.json maker.enabled; capture token usage"
```

---

### Task 4: arms.json and scripts/arms.py

**Files:**
- Create: `benchmarks/arms.json`, `scripts/arms.py`
- Test: `tests/test_arms.py`

**Interfaces:**
- Produces: `arms.py list`, `arms.py download <name>` (hf download into `/mnt/nvme-apps/LinuxModels/<dir>/`), `arms.py use <name>` (writes `~/.openclaw/maker.env` and sets `cad.json` `maker: {enabled: true, port: 8088, alias: <alias>}`), `arms.py restore` (sets `maker.enabled=false`, stops maker-server, starts the resident). Library functions: `load_arms(path=ARMS_FILE) -> dict[str, dict]`, `render_env(arm: dict) -> str`, `apply_arm(arm: dict, cad_json: Path, env_path: Path) -> None`.

- [ ] **Step 1: Write arms.json**

```json
{
  "store": "/mnt/nvme-apps/LinuxModels",
  "arms": [
    {"name": "qwen3.8-27b-nothink", "alias": "qwen3.8-27b", "role": "control",
     "gguf": "models/Qwen3.8-27B-UD-Q4_K_XL.gguf", "mmproj": "models/mmproj-Qwen3.8-27B-F16.gguf",
     "ctx": 16384, "extra_args": "--spec-type draft-mtp --chat-template-kwargs {\"enable_thinking\":false}",
     "hf": null, "notes": "resident model, thinking off"},
    {"name": "qwen3.8-27b-think", "alias": "qwen3.8-27b", "role": "control",
     "gguf": "models/Qwen3.8-27B-UD-Q4_K_XL.gguf", "mmproj": "models/mmproj-Qwen3.8-27B-F16.gguf",
     "ctx": 16384, "extra_args": "--spec-type draft-mtp --chat-template-kwargs {\"reasoning_effort\":\"medium\"}",
     "hf": null, "notes": "resident model, thinking medium"},
    {"name": "gemma-4-31b", "alias": "gemma-4-31b", "role": "candidate",
     "gguf": "gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf", "mmproj": "gemma-4-31B-it-GGUF/mmproj-F16.gguf",
     "ctx": 16384, "extra_args": "", "hf": null, "notes": "dense 31B, licence: Gemma"},
    {"name": "qwen3-coder-30b-a3b", "alias": "qwen3-coder-30b-a3b", "role": "candidate",
     "gguf": "Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf", "mmproj": null,
     "ctx": 16384, "extra_args": "",
     "hf": {"repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "files": ["Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"]},
     "notes": "MoE 3B active, no thinking"},
    {"name": "glm-4.7-flash", "alias": "glm-4.7-flash", "role": "candidate",
     "gguf": "GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf", "mmproj": null,
     "ctx": 16384, "extra_args": "",
     "hf": {"repo": "unsloth/GLM-4.7-Flash-GGUF", "files": ["GLM-4.7-Flash-UD-Q4_K_XL.gguf"]},
     "notes": "MoE 3B active, thinking on by default (kept)"},
    {"name": "devstral-small-2", "alias": "devstral-small-2", "role": "candidate",
     "gguf": "Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf",
     "mmproj": "Devstral-Small-2-24B-Instruct-2512-GGUF/mmproj-F16.gguf",
     "ctx": 16384, "extra_args": "",
     "hf": {"repo": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF", "files": ["Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf", "mmproj-F16.gguf"]},
     "notes": "dense 24B agentic coder"},
    {"name": "gpt-oss-20b", "alias": "gpt-oss-20b", "role": "candidate",
     "gguf": "gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf", "mmproj": null,
     "ctx": 16384, "extra_args": "--chat-template-kwargs {\"reasoning_effort\":\"low\"}",
     "hf": {"repo": "ggml-org/gpt-oss-20b-GGUF", "files": ["gpt-oss-20b-MXFP4.gguf"]},
     "notes": "native MXFP4, reasoning low"},
    {"name": "qwen2.5-coder-7b", "alias": "qwen2.5-coder-7b", "role": "floor",
     "gguf": "Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf", "mmproj": null,
     "ctx": 16384, "extra_args": "",
     "hf": {"repo": "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF", "files": ["Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"]},
     "notes": "current fast rung as a llama.cpp arm"}
  ]
}
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_arms.py
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import arms


def test_arms_file_is_well_formed():
    a = arms.load_arms()
    assert {"qwen3.8-27b-nothink", "gemma-4-31b", "qwen3-coder-30b-a3b", "glm-4.7-flash",
            "devstral-small-2", "gpt-oss-20b", "qwen2.5-coder-7b"} <= set(a)
    for arm in a.values():
        assert arm["gguf"].endswith(".gguf") and arm["ctx"] >= 8192 and arm["alias"]


def test_render_env_has_every_key():
    arm = arms.load_arms()["gpt-oss-20b"]
    env = arms.render_env(arm)
    for key in ("MODEL=", "MMPROJ=", "CTX=16384", "PORT=8088", "ALIAS=gpt-oss-20b", "EXTRA_ARGS="):
        assert key in env
    assert "/mnt/nvme-apps/LinuxModels/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf" in env


def test_apply_arm_writes_env_and_cad_json(tmp_path):
    arm = arms.load_arms()["gemma-4-31b"]
    cad_json = tmp_path / "cad.json"; cad_json.write_text(json.dumps({"code_model": None}))
    env_path = tmp_path / "maker.env"
    arms.apply_arm(arm, cad_json, env_path)
    cfg = json.loads(cad_json.read_text())
    assert cfg["maker"] == {"enabled": True, "port": 8088, "alias": "gemma-4-31b", "arm": "gemma-4-31b"}
    assert cfg["code_model"] is None          # untouched keys survive
    assert "ALIAS=gemma-4-31b" in env_path.read_text()


def test_restore_disables_maker(tmp_path):
    cad_json = tmp_path / "cad.json"
    cad_json.write_text(json.dumps({"maker": {"enabled": True, "port": 8088, "alias": "x", "arm": "x"}}))
    arms.disable_maker(cad_json)
    assert json.loads(cad_json.read_text())["maker"]["enabled"] is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_arms.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'arms'`.

- [ ] **Step 4: Implement scripts/arms.py**

```python
#!/usr/bin/env python3
"""Candidate model arms for the Maker Agent card (docs/MAKER-1.0-CAMPAIGN.md 4.1a).

  arms.py list
  arms.py download <name>      # hf download into the NVMe store
  arms.py use <name>           # write ~/.openclaw/maker.env + cad.json maker block, start maker-server
  arms.py restore              # maker.enabled=false, stop maker-server, start the resident
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ARMS_FILE = HERE / "benchmarks" / "arms.json"
CAD_JSON = Path(os.environ.get("CAD_CONFIG_FILE", Path.home() / ".openclaw" / "cad.json"))
ENV_PATH = Path(os.environ.get("MAKER_ENV", Path.home() / ".openclaw" / "maker.env"))
PORT = 8088


def load_arms(path: Path = ARMS_FILE) -> dict[str, dict]:
    data = json.loads(path.read_text())
    store = Path(data["store"])
    out = {}
    for arm in data["arms"]:
        arm = dict(arm)
        arm["model_path"] = str(store / arm["gguf"])
        arm["mmproj_path"] = str(store / arm["mmproj"]) if arm.get("mmproj") else ""
        arm["store"] = str(store)
        out[arm["name"]] = arm
    return out


def render_env(arm: dict) -> str:
    return "\n".join([
        f"MODEL={arm['model_path']}",
        f"MMPROJ={arm['mmproj_path']}",
        f"CTX={arm['ctx']}",
        f"PORT={PORT}",
        f"ALIAS={arm['alias']}",
        f"EXTRA_ARGS={arm.get('extra_args', '')}",
        "",
    ])


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return {}


def _write_json(p: Path, data: dict) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(p)


def apply_arm(arm: dict, cad_json: Path = CAD_JSON, env_path: Path = ENV_PATH) -> None:
    env_path.write_text(render_env(arm))
    cfg = _read_json(cad_json)
    cfg["maker"] = {"enabled": True, "port": PORT, "alias": arm["alias"], "arm": arm["name"]}
    _write_json(cad_json, cfg)


def disable_maker(cad_json: Path = CAD_JSON) -> None:
    cfg = _read_json(cad_json)
    m = cfg.get("maker") or {}
    m["enabled"] = False
    cfg["maker"] = m
    _write_json(cad_json, cfg)


def _wait(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                json.loads(r.read()); return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"{url} not healthy after {timeout}s")


def cmd_list(a: dict) -> None:
    for name, arm in a.items():
        have = Path(arm["model_path"]).exists()
        print(f"{'ok ' if have else '-- '}{name:24s} {arm['role']:9s} {arm['alias']:22s} {arm['gguf']}")


def cmd_download(arm: dict) -> None:
    if not arm.get("hf"):
        print(f"{arm['name']}: no hf entry (expected on disk at {arm['model_path']})"); return
    dest = Path(arm["store"]) / Path(arm["gguf"]).parent
    dest.mkdir(parents=True, exist_ok=True)
    for f in arm["hf"]["files"]:
        if (dest / f).exists():
            print(f"have {dest / f}"); continue
        subprocess.run(["hf", "download", arm["hf"]["repo"], f, "--local-dir", str(dest)], check=True)
    for f in arm["hf"]["files"]:
        print(f"{dest / f}: {(dest / f).stat().st_size} bytes")


def cmd_use(arm: dict, start: bool = True) -> None:
    if not Path(arm["model_path"]).exists():
        raise SystemExit(f"{arm['model_path']} missing; run arms.py download {arm['name']}")
    apply_arm(arm)
    if start:
        subprocess.run(["systemctl", "--user", "stop", "qwen38-server"], check=False)
        subprocess.run(["systemctl", "--user", "restart", "maker-server"], check=True)
        _wait(f"http://127.0.0.1:{PORT}/health", 900)
    print(f"maker-server -> {arm['name']} ({arm['alias']}) on :{PORT}")


def cmd_restore() -> None:
    disable_maker()
    subprocess.run(["systemctl", "--user", "stop", "maker-server"], check=False)
    subprocess.run(["systemctl", "--user", "start", "qwen38-server"], check=False)
    _wait("http://127.0.0.1:8086/health", 300)
    print("resident restored on :8086; maker disabled")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("download").add_argument("name")
    u = sub.add_parser("use"); u.add_argument("name"); u.add_argument("--no-start", action="store_true")
    sub.add_parser("restore")
    ns = ap.parse_args()
    a = load_arms()
    if ns.cmd == "list": cmd_list(a)
    elif ns.cmd == "download": cmd_download(a[ns.name])
    elif ns.cmd == "use": cmd_use(a[ns.name], start=not ns.no_start)
    elif ns.cmd == "restore": cmd_restore()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_arms.py -q`
Expected: 4 PASS.

- [ ] **Step 6: Live smoke on the control arm (no download needed)**

```bash
cd ~/.openclaw/skills/cad-builder
python3 scripts/arms.py list
python3 scripts/arms.py use qwen3.8-27b-nothink
python3 -c "import cad_engine; print(cad_engine.maker_config())"
python3 scripts/fluid_gen.py build "a solid cube 20mm on a side" --coder strong --json | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['ok'], r['code_model'], r['gate_hard'], r.get('usage'))"
python3 scripts/arms.py restore
systemctl --user is-active qwen38-server maker-server
```
Expected: `{'enabled': True, 'port': 8088, ...}`, then `True local:qwen3.8-27b [] {...usage...}`, then `active` / `inactive`.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/arms.json scripts/arms.py tests/test_arms.py
git commit -m "arms: candidate model declarations + download/use/restore for the maker server"
```

---

### Task 5: external suites (CADPrompt, CAD Arena, optional Text-to-CadQuery)

**Files:**
- Create: `scripts/fetch_external.py`, `benchmarks/cad-arena/specs.json` (checked in, 12 public prompts), `benchmarks/cadprompt/` (generated, git-ignored except `README.md`), `benchmarks/text2cadquery/` (generated, optional)
- Modify: `scripts/harvest_census.py:158-168` (`suite_slugs()` list), `.gitignore`
- Test: `tests/test_external_suites.py`

**Interfaces:**
- Produces: suite dirs in the existing shape: `specs.json` = `[{"id", "name", "tier", "spec", "source"}]`, `acceptance.json` = `{id: {"reference_stl": "refs/<id>.stl", "solids": 1}}` (reference optional for cad-arena). Library: `cadprompt_to_suite(src_dir: Path, out_dir: Path, limit: int, seed: int) -> int` (returns count), `arena_specs() -> list[dict]`.
- Extends `harvest_census.suite_slugs()` to scan `["text-to-cad", "organic", "heldout-cqe", "hard-eval", "cadprompt", "cad-arena", "text2cadquery"]`.

- [ ] **Step 1: Write the failing test with a fixture**

```python
# tests/test_external_suites.py
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import fetch_external as fx


def _fake_cadprompt(root: Path, n: int = 3):
    for i in range(n):
        d = root / f"{i:08d}"; d.mkdir(parents=True)
        (d / "Natural_Language_Descriptions_Prompt_with_specific_measurements.txt").write_text(
            f"Create a cylinder of radius {0.5+i} units and height 1 unit.")
        (d / "Natural_Language_Descriptions_Prompt.txt").write_text("Create a cylinder.")
        (d / "Ground_Truth.stl").write_bytes(b"solid x\nendsolid x\n")
        (d / "Ground_Truth.json").write_text(json.dumps({"Number_of_Faces": 3, "Is_Solid": True}))
    return root


def test_cadprompt_to_suite_builds_specs_and_refs(tmp_path):
    src = _fake_cadprompt(tmp_path / "CADPrompt")
    out = tmp_path / "suite"
    n = fx.cadprompt_to_suite(src, out, limit=2, seed=7)
    specs = json.loads((out / "specs.json").read_text())
    acc = json.loads((out / "acceptance.json").read_text())
    assert n == 2 and len(specs) == 2
    s = specs[0]
    assert set(s) >= {"id", "name", "tier", "spec", "source"} and "units" in s["spec"]
    assert acc[s["id"]]["reference_stl"] == f"refs/{s['id']}.stl"
    assert (out / "refs" / f"{s['id']}.stl").exists()


def test_cadprompt_sampling_is_deterministic(tmp_path):
    src = _fake_cadprompt(tmp_path / "CADPrompt", n=6)
    a = fx.cadprompt_to_suite(src, tmp_path / "a", limit=3, seed=1)
    b = fx.cadprompt_to_suite(src, tmp_path / "b", limit=3, seed=1)
    ids = lambda p: [s["id"] for s in json.loads((p / "specs.json").read_text())]
    assert a == b == 3 and ids(tmp_path / "a") == ids(tmp_path / "b")


def test_arena_has_twelve_public_prompts_with_tiers():
    specs = fx.arena_specs()
    assert len(specs) == 12 and sorted({s["tier"] for s in specs}) == [1, 2, 3, 4]
    assert any("spur gear" in s["spec"] for s in specs)


def test_suite_slugs_cover_external_dirs():
    import harvest_census as hc
    assert {"cadprompt", "cad-arena", "text2cadquery"} <= set(hc.CARD_SUITES)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_external_suites.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fetch_external'`.

- [ ] **Step 3: Implement scripts/fetch_external.py**

```python
#!/usr/bin/env python3
"""Fetch and convert public text-to-CAD suites into benchmarks/<suite>/ (card inputs).

  fetch_external.py cadprompt [--limit 100] [--seed 20260915]
      git sparse-clone github.com/Kamel773/CAD_Code_Generation (folder CADPrompt/, 200 items,
      each with Ground_Truth.stl and a dimensioned prompt) -> benchmarks/cadprompt/
  fetch_external.py arena
      the 12 CAD Arena prompts published on cadarena.dev (tiers 1-4) -> benchmarks/cad-arena/
  fetch_external.py text2cadquery [--limit 100] [--seed 20260915]
      hf download ricemonster/IEEE-T-ASE data/data_test.jsonl; reference STLs need cadquery in
      benchmarks/external/.venv-cq (created on demand); skipped with a note if that install fails.

Evaluation-only data: nothing here is redistributed (CADPrompt has no licence file; Text-to-CadQuery's
licence is unconfirmed). Generated dirs are git-ignored except README.md.
"""
from __future__ import annotations
import argparse, json, random, re, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BENCH = HERE / "benchmarks"
CACHE = HERE / "benchmarks" / "external" / "cache"
CADPROMPT_REPO = "https://github.com/Kamel773/CAD_Code_Generation"
T2CQ_REPO = "ricemonster/IEEE-T-ASE"

_ARENA = [
    (1, "A cube 20 x 20 x 20 mm"),
    (1, "A cylinder 10 mm diameter, 30 mm tall"),
    (1, "A hollow sphere, outer radius 20 mm, wall 2 mm"),
    (2, "A rectangular plate 50 x 30 x 5 mm with a centered hole 8 mm diameter"),
    (2, "An L-shaped bracket, 40 mm arms, 5 mm thick, 30 mm tall"),
    (2, "A hex bolt head 10 mm across flats, M6 thread, 20 mm shaft"),
    (3, "A flanged shaft with 3 equally-spaced M4 bolt holes on the flange"),
    (3, "A box with a snap-fit lid, 50 x 40 x 30 mm"),
    (3, "A spur gear: 20 teeth, module 2, 10 mm thick, 8 mm center bore"),
    (4, "A parametric living hinge, 100 mm span, 0.3 mm flex zone"),
    (4, "An S-curve pipe fitting, 15 mm inner diameter, 45 degree bend"),
    (4, "A 3-part snap-fit assembly: housing, PCB carrier, and lid"),
]


def arena_specs() -> list[dict]:
    return [{"id": f"arena-{i+1:02d}", "name": text[:48], "tier": tier, "spec": text,
             "source": "cadarena.dev (public examples, 12 of 20)"} for i, (tier, text) in enumerate(_ARENA)]


def _write_suite(out: Path, specs: list[dict], acceptance: dict, readme: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "specs.json").write_text(json.dumps(specs, indent=2) + "\n")
    (out / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    (out / "README.md").write_text(readme)


def cadprompt_to_suite(src_dir: Path, out_dir: Path, limit: int, seed: int) -> int:
    items = sorted(p for p in src_dir.iterdir() if p.is_dir() and (p / "Ground_Truth.stl").exists())
    rng = random.Random(seed)
    rng.shuffle(items)
    items = sorted(items[:limit], key=lambda p: p.name)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "refs").mkdir(parents=True)
    specs, acc = [], {}
    for p in items:
        prompt = (p / "Natural_Language_Descriptions_Prompt_with_specific_measurements.txt").read_text().strip()
        sid = f"cp-{p.name}"
        shutil.copy(p / "Ground_Truth.stl", out_dir / "refs" / f"{sid}.stl")
        specs.append({"id": sid, "name": prompt[:48], "tier": 0, "spec": prompt,
                      "source": f"CADPrompt/{p.name}"})
        acc[sid] = {"reference_stl": f"refs/{sid}.stl", "solids": 1}
    _write_suite(out_dir, specs, acc,
                 f"# CADPrompt slice\n\n{len(specs)} of 200 items, seed {seed}, dimensioned prompts, "
                 f"DeepCAD units (not mm: card scores are unit-normalised). Source {CADPROMPT_REPO} "
                 "(no licence file; evaluation only, not redistributed).\n")
    return len(specs)


def cmd_cadprompt(limit: int, seed: int) -> None:
    repo = CACHE / "CAD_Code_Generation"
    if not repo.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        CADPROMPT_REPO, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "CADPrompt"], check=True)
    n = cadprompt_to_suite(repo / "CADPrompt", BENCH / "cadprompt", limit, seed)
    print(f"benchmarks/cadprompt: {n} specs")


def cmd_arena() -> None:
    specs = arena_specs()
    _write_suite(BENCH / "cad-arena", specs, {s["id"]: {"solids": 1} for s in specs if s["tier"] < 4},
                 "# CAD Arena public prompts\n\n12 of the 20 prompts published on https://cadarena.dev "
                 "(3 per tier). Validity smoke set, no reference geometry.\n")
    print("benchmarks/cad-arena: 12 specs")


def cmd_text2cadquery(limit: int, seed: int) -> None:
    dest = CACHE / "text2cadquery"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["hf", "download", T2CQ_REPO, "data/data_test.jsonl", "--local-dir", str(dest)], check=True)
    venv = BENCH / "external" / ".venv-cq"
    if not (venv / "bin" / "python").exists():
        subprocess.run(["uv", "venv", str(venv), "--python", "3.12"], check=True)
        r = subprocess.run(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "cadquery"])
        if r.returncode != 0:
            print("text2cadquery: cadquery install failed; suite skipped (card notes it)"); return
    rows = [json.loads(l) for l in (dest / "data" / "data_test.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random(seed); rng.shuffle(rows); rows = rows[:limit]
    out = BENCH / "text2cadquery"
    if out.exists(): shutil.rmtree(out)
    (out / "refs").mkdir(parents=True)
    specs, acc = [], {}
    for i, row in enumerate(rows):
        sid = f"t2cq-{i:04d}"
        code = re.sub(r"exporters\.export\([^\n]*\n", "", row["output"])
        code += f"\nfrom cadquery import exporters\nexporters.export(part_1 if 'part_1' in dir() else result, r'{out / 'refs' / (sid + '.stl')}')\n"
        r = subprocess.run([str(venv / "bin" / "python"), "-c", code], capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not (out / "refs" / f"{sid}.stl").exists():
            continue
        specs.append({"id": sid, "name": row["input"][:48], "tier": 0, "spec": row["input"], "source": T2CQ_REPO})
        acc[sid] = {"reference_stl": f"refs/{sid}.stl", "solids": 1}
    _write_suite(out, specs, acc, f"# Text-to-CadQuery test slice\n\n{len(specs)} items with references rebuilt "
                 "by executing the reference CadQuery code. Licence unconfirmed; evaluation only.\n")
    print(f"benchmarks/text2cadquery: {len(specs)} specs with references")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cadprompt", "text2cadquery"):
        p = sub.add_parser(name); p.add_argument("--limit", type=int, default=100); p.add_argument("--seed", type=int, default=20260915)
    sub.add_parser("arena")
    ns = ap.parse_args()
    if ns.cmd == "cadprompt": cmd_cadprompt(ns.limit, ns.seed)
    elif ns.cmd == "arena": cmd_arena()
    else: cmd_text2cadquery(ns.limit, ns.seed)


if __name__ == "__main__":
    main()
```

Note for the Text-to-CadQuery export line: the reference programs name their final object inconsistently; the loader tries `part_1` then `result`, and any item whose code fails to export is skipped (count reported). That is acceptable for an evaluation slice.

- [ ] **Step 4: Extend the contamination guard**

In `scripts/harvest_census.py:158-168`, replace the hard-coded list with a module constant and use it:

```python
CARD_SUITES = ["text-to-cad", "organic", "heldout-cqe", "hard-eval",
               "cadprompt", "cad-arena", "text2cadquery"]   # every suite the card reads; never train on these


def suite_slugs() -> set[str]:
    out: set[str] = set()
    for name in CARD_SUITES:
        p = BENCH / name / "specs.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        items = data["benchmarks"] if isinstance(data, dict) and "benchmarks" in data else data
        out.update(_slug(it["spec"], 40) for it in items)
    return out
```

(Keep whatever `BENCH` path variable the file already uses; if it uses a different name, use that.)

Add to `.gitignore`:

```
benchmarks/external/cache/
benchmarks/external/.venv-cq/
benchmarks/cadprompt/
benchmarks/text2cadquery/
!benchmarks/cadprompt/README.md
!benchmarks/text2cadquery/README.md
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_external_suites.py -q`
Expected: 4 PASS.

- [ ] **Step 6: Fetch the real suites**

```bash
cd ~/.openclaw/skills/cad-builder
python3 scripts/fetch_external.py arena
python3 scripts/fetch_external.py cadprompt --limit 100
python3 scripts/fetch_external.py text2cadquery --limit 100 || true
ls benchmarks/cadprompt/refs | wc -l; head -c 600 benchmarks/cadprompt/specs.json
python3 -c "import sys; sys.path.insert(0,'scripts'); import harvest_census as hc; print(len(hc.suite_slugs()))"
```
Expected: 12 arena specs, 100 CADPrompt refs, a text2cadquery count or the skip note, and the slug count grew by the new specs.

- [ ] **Step 7: Commit**

```bash
git add scripts/fetch_external.py scripts/harvest_census.py .gitignore benchmarks/cad-arena/ tests/test_external_suites.py
git commit -m "benchmarks: public suites (CADPrompt slice, CAD Arena prompts, optional Text-to-CadQuery) + guard coverage"
```

---

### Task 5b: BenchCAD official harness against each arm

**Files:**
- Create: `scripts/run_benchcad.py`, `benchmarks/external/benchcad/local_adapter.py` (copied into the harness clone), `benchmarks/external/benchcad/README.md`
- Modify: `scripts/card_report.py` (add `render_benchcad_md`), `.gitignore` (`benchmarks/external/benchcad/BenchCAD-main/`)
- Test: `tests/test_card_report.py` (extend)

**Interfaces:**
- Consumes: `scripts/arms.py` (`load_arms`, `cmd_use`, `cmd_restore`), the maker server on `http://127.0.0.1:8088/v1`.
- Produces: `run_benchcad.py --arms a,b --tasks codeedit,codeqa,vision2code --num 150 --seed 42 --out <card dir>` writing `<card dir>/benchcad.json` = `{arm: {task: {"score": float, "n": int, "exec_rate": float|None, "raw": <harness summary>}}}`; `card_report.render_benchcad_md(results: dict) -> str` (one row per arm, one column per task, plus the published leaderboard reference rows for Gemma-4-31B-it and gpt-oss-120b copied from `LEADERBOARD.md` at run time).
- Vision2Code runs only for arms whose `mmproj` is set (the harness sends images); text-only arms get `null` for it.

Facts from the 2026-09-15 investigation (verify against the clone, do not trust blindly): harness `https://github.com/BenchCAD/BenchCAD-main` (MIT), data `BenchCAD/BenchCAD` on HF (CC-BY-4.0, configs `code_gen` 17,900 / `edit-bench` 748 / `QA` 2,400), `uv sync` with pinned `cadquery==2.3.0`, `cadquery-ocp==7.9.3.0`, `numpy==1.26.4` (own venv, isolated from ours). Run form: `uv run python benchcad.py --task <t> --num N --seed S --model <name>`. Model adapters in `benchcad_core/models/`; `openrouter_adapter.py` uses `openai.OpenAI(base_url=...)` + `chat.completions.create`, which llama-server speaks; `openai_adapter.py` uses the Responses API and will not work. Scores: Vision2Code = voxel IoU x exec rate (64^3, bbox-normalised STEP); CodeEdit = headroom-normalised IoU improvement; Code-QA = ratio accuracy. Leaderboard rows to quote: Code-QA Gemma-4-31B-it 0.664, gpt-oss-120b 0.689, Nemotron-3 120B 0.671, GPT-4o 0.726, Gemini 3.1 Pro 0.838; CodeEdit gpt-oss-120b 0.561, Nemotron-3 120B 0.608, GPT-5.3 (thinking) 0.865; Vision2Code Qwen3-VL-2B 0.0005, GPT-4o 0.1823, Gemini 3.1 Pro 0.2890.

- [ ] **Step 1: Clone and sync the harness in its own venv**

```bash
cd ~/.openclaw/skills/cad-builder/benchmarks/external && mkdir -p benchcad && cd benchcad
git clone --depth 1 https://github.com/BenchCAD/BenchCAD-main
cd BenchCAD-main && uv sync && cp .env.example .env
ls benchcad_core/models/ && sed -n '1,80p' benchcad_core/models/openrouter_adapter.py
grep -rn "openrouter" benchcad_core/models/__init__.py benchcad_core/models/*.py | head   # find the registry
```

- [ ] **Step 2: Write the local adapter**

Clone `openrouter_adapter.py` to `benchcad_core/models/local_adapter.py` (keep a copy at `benchmarks/external/benchcad/local_adapter.py` in our repo so the clone can be recreated): `base_url = os.environ.get("BENCHCAD_BASE_URL", "http://127.0.0.1:8088/v1")`, `api_key = "local"`, model name = `os.environ.get("BENCHCAD_MODEL", "maker")`, drop the OpenRouter reasoning-suffix parsing, keep image handling (base64 `image_url` parts) so Vision2Code works on vision arms, pass `extra_body={"chat_template_kwargs": json.loads(os.environ.get("BENCHCAD_TEMPLATE_KWARGS", "{}"))}` so the 27B thinking arms and gpt-oss reasoning level are honoured, set `timeout=900`. Register it in the model dispatch under the name `local` (follow how `openrouter` is registered). Smoke: `uv run python benchcad.py --task codeqa --num 3 --model local` with the control arm up (`python3 ../../../scripts/arms.py use qwen3.8-27b-nothink`).

- [ ] **Step 3: Write scripts/run_benchcad.py**

```python
#!/usr/bin/env python3
"""Official BenchCAD harness (benchcad.com, MIT) against each arm, through the maker server.

  run_benchcad.py --arms all --tasks codeedit,codeqa,vision2code --num 150 --seed 42 --out benchmarks/results/card/phase0

Writes <out>/benchcad.json. Vision2Code runs only for arms with an mmproj. The resident is restored in a finally.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
HARNESS = HERE / "benchmarks" / "external" / "benchcad" / "BenchCAD-main"
sys.path.insert(0, str(HERE / "scripts"))
import arms as arms_mod  # noqa: E402

TASKS = {"codeedit": "codeedit", "codeqa": "codeqa", "vision2code": "vision2code"}


def run_task(arm: dict, task: str, num: int, seed: int) -> dict:
    env = {**os.environ, "BENCHCAD_BASE_URL": "http://127.0.0.1:8088/v1", "BENCHCAD_MODEL": arm["alias"],
           "BENCHCAD_TEMPLATE_KWARGS": template_kwargs(arm)}
    cmd = ["uv", "run", "python", "benchcad.py", "--task", TASKS[task], "--num", str(num), "--seed", str(seed), "--model", "local"]
    p = subprocess.run(cmd, cwd=HARNESS, env=env, capture_output=True, text=True, timeout=6 * 3600)
    return parse_summary(p.stdout + "\n" + p.stderr, task)


def template_kwargs(arm: dict) -> str:
    m = re.search(r"--chat-template-kwargs\s+(\S+)", arm.get("extra_args", "") or "")
    return m.group(1) if m else "{}"


def parse_summary(text: str, task: str) -> dict:
    """The harness prints a final summary line per task; adapt these regexes to the real output after Step 2."""
    score = re.findall(r"(?:score|iou|accuracy)[^0-9]*([01]\.\d+)", text, flags=re.I)
    execr = re.findall(r"exec[^0-9]*(\d+(?:\.\d+)?)%", text, flags=re.I)
    return {"score": float(score[-1]) if score else None, "exec_rate": float(execr[-1]) / 100 if execr else None,
            "raw": text[-1500:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="all"); ap.add_argument("--tasks", default="codeedit,codeqa,vision2code")
    ap.add_argument("--num", type=int, default=150); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()
    all_arms = arms_mod.load_arms()
    names = list(all_arms) if ns.arms == "all" else ns.arms.split(",")
    out = Path(ns.out); out.mkdir(parents=True, exist_ok=True)
    res_path = out / "benchcad.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    try:
        for name in names:
            arm = all_arms[name]; results.setdefault(name, {})
            todo = [t for t in ns.tasks.split(",") if t not in results[name] and (t != "vision2code" or arm.get("mmproj"))]
            if not todo:
                continue
            arms_mod.cmd_use(arm)
            for t in todo:
                print(f"== {name} / {t}")
                results[name][t] = {**run_task(arm, t, ns.num, ns.seed), "n": ns.num}
                res_path.write_text(json.dumps(results, indent=2) + "\n")
            if not arm.get("mmproj"):
                results[name]["vision2code"] = None
    finally:
        arms_mod.cmd_restore()
        res_path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

The implementer must replace `parse_summary` with parsing of the harness's actual result artefact (it writes per-run JSON under its `results/` or `runs/` dir; read that file rather than scraping stdout if it exists, and say which in the README).

- [ ] **Step 4: Renderer + test**

Add to `tests/test_card_report.py`:

```python
def test_render_benchcad_md_rows_and_reference():
    res = {"a": {"codeedit": {"score": 0.5, "n": 10, "exec_rate": None}, "codeqa": {"score": 0.6, "n": 10, "exec_rate": None}, "vision2code": None}}
    md = cr.render_benchcad_md(res)
    assert "| a | 0.500 | 0.600 | - |" in md and "Gemma-4-31B-it" in md and "0.664" in md
```

and in `scripts/card_report.py`:

```python
BENCHCAD_REFERENCE = [  # published leaderboard rows (benchcad.com LEADERBOARD.md, read 2026-09-15)
    ("Gemma-4-31B-it (published)", None, 0.664, None),
    ("gpt-oss-120b (published)", 0.561, 0.689, None),
    ("GPT-4o (published)", None, 0.726, 0.1823),
    ("Gemini 3.1 Pro (published)", 0.837, 0.838, 0.2890),
]


def _f(x):
    return "-" if x is None else f"{x:.3f}"


def render_benchcad_md(results: dict) -> str:
    lines = ["## Official BenchCAD (benchcad.com harness, local adapter)", "",
             "| arm | CodeEdit | Code-QA | Vision2Code IoU |", "|---|---|---|---|"]
    for arm in sorted(results):
        r = results[arm] or {}
        g = lambda t: (r.get(t) or {}).get("score") if isinstance(r.get(t), dict) else None
        lines.append(f"| {arm} | {_f(g('codeedit'))} | {_f(g('codeqa'))} | {_f(g('vision2code'))} |")
    for name, ce, qa, v2c in BENCHCAD_REFERENCE:
        lines.append(f"| {name} | {_f(ce)} | {_f(qa)} | {_f(v2c)} |")
    return "\n".join(lines) + "\n"
```

`run_card.py` (Task 8) appends `render_benchcad_md` output to `card.md` when `<out>/benchcad.json` exists; the Lab view (Task 9) shows the same table under the card table.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_benchcad.py scripts/card_report.py tests/test_card_report.py benchmarks/external/benchcad/local_adapter.py benchmarks/external/benchcad/README.md .gitignore
git commit -m "benchcad: official harness runner against each arm via a local llama.cpp adapter"
```

---

### Task 6: unit-normalised Chamfer bands

**Files:**
- Modify: `scripts/geom_bands.py:86-117` (`score_against_reference`)
- Test: `tests/test_geom_normalize.py`

**Interfaces:**
- Produces: `normalize_stl(src: Path, dst: Path, target_diag: float = 100.0) -> float` (returns the scale factor applied); `score_against_reference(candidate, reference_stl, expected_components=1, normalize=False) -> dict` where `normalize=True` scales BOTH meshes to bbox diagonal 100 before the existing checks and adds `"normalized": True, "scale_candidate": f, "scale_reference": f` to the result.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geom_normalize.py
import sys
from pathlib import Path
import trimesh
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import geom_bands as gb


def _box(tmp_path, name, extents):
    p = tmp_path / name
    trimesh.creation.box(extents=extents).export(p)
    return p


def test_normalize_scales_to_target_diagonal(tmp_path):
    src = _box(tmp_path, "small.stl", (0.3, 0.4, 0.5))
    dst = tmp_path / "norm.stl"
    f = gb.normalize_stl(src, dst, target_diag=100.0)
    m = trimesh.load(dst)
    assert abs(float(m.bounding_box.primitive.extents @ m.bounding_box.primitive.extents) ** 0.5 - 100.0) < 1e-3
    assert abs(f - 100.0 / (0.3**2 + 0.4**2 + 0.5**2) ** 0.5) < 1e-6


def test_same_shape_different_units_matches_when_normalized(tmp_path):
    ref = _box(tmp_path, "ref.stl", (1.0, 2.0, 3.0))       # DeepCAD-style units
    cand = _box(tmp_path, "cand.stl", (10.0, 20.0, 30.0))  # the agent built it in mm
    raw = gb.score_against_reference(cand, ref)
    norm = gb.score_against_reference(cand, ref, normalize=True)
    assert raw["band"] == "fail"
    assert norm["band"] == "match" and norm["normalized"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_geom_normalize.py -q`
Expected: FAIL with `AttributeError: module 'geom_bands' has no attribute 'normalize_stl'`.

- [ ] **Step 3: Implement**

In `scripts/geom_bands.py` add above `score_against_reference`:

```python
def normalize_stl(src: Path, dst: Path, target_diag: float = 100.0) -> float:
    """Scale a mesh so its bounding-box diagonal is target_diag (about the origin). Returns the factor."""
    import trimesh
    m = trimesh.load(str(src), force="mesh")
    ext = m.bounding_box.primitive.extents
    diag = float((ext @ ext) ** 0.5)
    f = target_diag / diag if diag > 0 else 1.0
    m.apply_scale(f)
    m.export(str(dst))
    return f
```

and change the signature and body start of `score_against_reference` to:

```python
def score_against_reference(candidate: Path, reference_stl: Path, expected_components: int = 1,
                            normalize: bool = False) -> dict:
    extra: dict = {}
    if normalize:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="geomnorm-"))
        fc = normalize_stl(candidate, tmp / "cand.stl")
        fr = normalize_stl(reference_stl, tmp / "ref.stl")
        candidate, reference_stl = tmp / "cand.stl", tmp / "ref.stl"
        extra = {"normalized": True, "scale_candidate": fc, "scale_reference": fr}
```

and merge `extra` into every returned dict (`{**result, **extra}` on both the success and the exception return). If `candidate` is a STEP (existing code calls `step_to_stl` first), normalise AFTER that conversion.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_geom_normalize.py -q`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/geom_bands.py tests/test_geom_normalize.py
git commit -m "geom_bands: unit-normalised scoring (both meshes to diagonal 100) for DeepCAD-unit suites"
```

---

### Task 7: card report functions (pure)

**Files:**
- Create: `scripts/card_report.py`
- Test: `tests/test_card_report.py`

**Interfaces:**
- Produces: `summarise(rows: list[dict]) -> dict` keyed `f"{arm}|{suite}"` with `{"arm", "suite", "n", "valid", "invalid_ratio", "gate_clean", "acceptance", "bands": {"match","valid","near_miss","fail"}, "median_wall_s", "tokens_out"}`; `render_md(summary: dict, meta: dict) -> str`. Row schema (written by Task 8): `{"arm", "suite", "id", "tier", "ok": bool, "gate_hard": int, "gate_spec": int, "acc_passed": int, "acc_total": int, "band": str|None, "wall_s": float, "tokens_out": int|None, "build_dir": str, "error": str|None}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_card_report.py
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import card_report as cr

ROWS = [
    {"arm": "a", "suite": "s", "id": "1", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0, "acc_passed": 2, "acc_total": 2, "band": "match", "wall_s": 10.0, "tokens_out": 100, "build_dir": "", "error": None},
    {"arm": "a", "suite": "s", "id": "2", "tier": 1, "ok": True, "gate_hard": 1, "gate_spec": 0, "acc_passed": 1, "acc_total": 2, "band": "fail", "wall_s": 30.0, "tokens_out": 300, "build_dir": "", "error": None},
    {"arm": "a", "suite": "s", "id": "3", "tier": 2, "ok": False, "gate_hard": 0, "gate_spec": 0, "acc_passed": 0, "acc_total": 2, "band": None, "wall_s": 20.0, "tokens_out": None, "build_dir": "", "error": "no STEP"},
]


def test_summarise_counts():
    s = cr.summarise(ROWS)["a|s"]
    assert s["n"] == 3 and s["valid"] == 2
    assert abs(s["invalid_ratio"] - 1/3) < 1e-9
    assert s["gate_clean"] == 1                       # ok and gate_hard==0 and gate_spec==0
    assert abs(s["acceptance"] - 3/6) < 1e-9          # pooled checks, like run_benchmarks
    assert s["bands"] == {"match": 1, "valid": 0, "near_miss": 0, "fail": 1}
    assert s["median_wall_s"] == 20.0 and s["tokens_out"] == 400


def test_render_md_has_one_line_per_arm_suite():
    md = cr.render_md(cr.summarise(ROWS), {"stamp": "t", "mode": "oneshot"})
    assert "| a | s | 3 |" in md and "invalid" in md.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_card_report.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python3
"""Pure summarising and rendering for the Maker Agent card (no I/O)."""
from __future__ import annotations
from statistics import median

BANDS = ("match", "valid", "near_miss", "fail")


def summarise(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(f"{r['arm']}|{r['suite']}", []).append(r)
    out = {}
    for key, rs in groups.items():
        n = len(rs)
        valid = sum(1 for r in rs if r["ok"])
        acc_p = sum(r["acc_passed"] for r in rs); acc_t = sum(r["acc_total"] for r in rs)
        out[key] = {
            "arm": rs[0]["arm"], "suite": rs[0]["suite"], "n": n, "valid": valid,
            "invalid_ratio": (n - valid) / n if n else None,
            "gate_clean": sum(1 for r in rs if r["ok"] and r["gate_hard"] == 0 and r["gate_spec"] == 0),
            "acceptance": acc_p / acc_t if acc_t else None,
            "bands": {b: sum(1 for r in rs if r.get("band") == b) for b in BANDS},
            "median_wall_s": median(r["wall_s"] for r in rs) if rs else None,
            "tokens_out": sum(r["tokens_out"] or 0 for r in rs),
        }
    return out


def _pct(x):
    return "-" if x is None else f"{100*x:.0f}%"


def render_md(summary: dict, meta: dict) -> str:
    lines = [f"# Maker Agent card {meta.get('stamp','')}", "",
             f"Mode: {meta.get('mode','')}. One row per arm and suite. invalid = no solid produced; "
             "gate clean = solid with zero hard and zero [spec] findings; acceptance = pooled checks; "
             "bands = unit-normalised Chamfer vs reference where one exists.", "",
             "| arm | suite | n | valid | invalid | gate clean | acceptance | match | valid band | near miss | fail | median s | tokens out |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(summary):
        s = summary[key]; b = s["bands"]
        lines.append(f"| {s['arm']} | {s['suite']} | {s['n']} | {s['valid']} | {_pct(s['invalid_ratio'])} | "
                     f"{s['gate_clean']} | {_pct(s['acceptance'])} | {b['match']} | {b['valid']} | {b['near_miss']} | "
                     f"{b['fail']} | {s['median_wall_s']:.0f} | {s['tokens_out']} |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests, commit**

Run: `python3 -m pytest tests/test_card_report.py -q` (2 PASS), then:

```bash
git add scripts/card_report.py tests/test_card_report.py
git commit -m "card_report: pure summarise + markdown renderer for the benchmark card"
```

---

### Task 8: scripts/run_card.py (the runner)

**Files:**
- Create: `scripts/run_card.py`
- Uses: `scripts/arms.py` (`load_arms`, `cmd_use`, `cmd_restore`), `scripts/run_benchmarks.py:score_acceptance` (import by path), `scripts/geom_bands.py:score_against_reference`, `scripts/card_report.py`, `scripts/harvest_census.py:_slug`, `scripts/fluid_gen.py` (subprocess), `cad_v5` (subprocess for `--mode agent`)

**Interfaces:**
- CLI: `run_card.py --arms a,b --suites text-to-cad,organic,hard-eval,heldout-cqe,cadprompt,cad-arena,text2cadquery --mode oneshot|agent --limit N --out benchmarks/results/card/<stamp> --resume --timeout 900`. Writes `rows.jsonl` (append, resumable by `(arm,suite,id)`), `card.json` (`{"meta", "summary"}`), `card.md`, and `latest` symlink `benchmarks/results/card/latest -> <stamp>`.
- Refuses to start if any suite spec slug appears in `~/.openclaw/cad-sftpairs.jsonl`, `~/.openclaw/cad-examples.jsonl` or `~/.openclaw/cad-sft-train.jsonl` (prints the clashes, exit 2), unless `--allow-contaminated` is given for a smoke run.

- [ ] **Step 1: Implement**

```python
#!/usr/bin/env python3
"""The Maker Agent card: every arm on every suite, one JSON + one markdown table.

  run_card.py --arms qwen3.8-27b-nothink --limit 2                # smoke
  run_card.py --arms all --mode oneshot                            # the Phase 0 shootout
  run_card.py --arms qwen3.8-27b-nothink --mode agent --suites text-to-cad,organic,hard-eval,heldout-cqe

oneshot = scripts/fluid_gen.py build (one codegen, one salvage, gate, no critic): base-model capability.
agent   = python3 -m cad_v5 --once --json (full observe-edit loop): the agent baseline.
The resident is restored in a finally: whatever happens. Rows append to rows.jsonl so --resume continues.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
BENCH = HERE / "benchmarks"
CARD_DIR = BENCH / "results" / "card"
sys.path.insert(0, str(SCRIPTS)); sys.path.insert(0, str(HERE))
import arms as arms_mod            # noqa: E402
import card_report                 # noqa: E402
import geom_bands                  # noqa: E402
import harvest_census as hc        # noqa: E402

_rb_spec = importlib.util.spec_from_file_location("run_benchmarks", SCRIPTS / "run_benchmarks.py")
_rb = importlib.util.module_from_spec(_rb_spec); _rb_spec.loader.exec_module(_rb)
score_acceptance = _rb.score_acceptance

INTERNAL = ["text-to-cad", "organic", "hard-eval", "heldout-cqe"]
EXTERNAL = ["cadprompt", "cad-arena", "text2cadquery"]
TRAIN_FILES = [Path.home() / ".openclaw" / n for n in ("cad-sftpairs.jsonl", "cad-examples.jsonl", "cad-sft-train.jsonl")]


def load_suite(name: str) -> tuple[list[dict], dict]:
    d = BENCH / name
    if not (d / "specs.json").exists():
        return [], {}
    data = json.loads((d / "specs.json").read_text())
    specs = data["benchmarks"] if isinstance(data, dict) and "benchmarks" in data else data
    acc = json.loads((d / "acceptance.json").read_text()) if (d / "acceptance.json").exists() else {}
    return specs, acc


def contamination(specs_by_suite: dict[str, list[dict]]) -> list[str]:
    train = set()
    for f in TRAIN_FILES:
        if f.exists():
            for line in f.read_text().splitlines():
                try:
                    train.add(hc._slug(json.loads(line).get("spec", ""), 40))
                except Exception:
                    pass
    clashes = []
    for suite, specs in specs_by_suite.items():
        for s in specs:
            if hc._slug(s["spec"], 40) in train:
                clashes.append(f"{suite}/{s['id']}: {s['spec'][:70]}")
    return clashes


def build_once(spec: str, mode: str, timeout: int) -> tuple[dict, float, str]:
    env = {**os.environ, "CAD_BENCH": "1"}
    if mode == "oneshot":
        cmd = [sys.executable, str(SCRIPTS / "fluid_gen.py"), "build", spec, "--coder", "strong", "--json"]
    else:
        cmd = [sys.executable, "-m", "cad_v5", spec, "--once", "--json", "--coder", "strong", "--target", "file"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=HERE, env=env)
        wall = time.time() - t0
        last = [l for l in p.stdout.splitlines() if l.startswith("{")]
        return (json.loads(last[-1]) if last else {"ok": False, "error": f"rc={p.returncode} no json"}), wall, p.stderr[-800:]
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout {timeout}s"}, time.time() - t0, ""


def geometry_of(res: dict, mode: str) -> dict:
    if mode == "oneshot":
        f = res.get("facts") or {}
        return {"solids": f.get("solids"), "bbox_mm": f.get("bbox_mm"), "faces": f.get("faces"),
                "cyl_faces": f.get("cyl_faces"), "through_holes": f.get("through_holes")}
    return {k: res.get(k) for k in ("solids", "bbox_mm", "faces", "cyl_faces", "through_holes")}


def step_of(res: dict, mode: str) -> Path | None:
    if mode == "agent":
        return Path(res["step_local"]) if res.get("step_local") else None
    bd = res.get("build_dir")
    if not bd:
        return None
    steps = sorted(Path(bd).glob("*.step"))
    return steps[-1] if steps else None


def run_row(arm: str, suite: str, spec: dict, crit: dict | None, mode: str, timeout: int) -> dict:
    res, wall, stderr = build_once(spec["spec"], mode, timeout)
    ok = bool(res.get("ok")) and (res.get("facts", {}).get("solids", 1) if mode == "oneshot" else res.get("has_bodies", True))
    if mode == "oneshot":
        gate_hard = len(res.get("gate_hard") or []); gate_spec = len(res.get("gate_spec") or [])
    else:
        gate_hard = 0 if res.get("converged") else 1; gate_spec = 0
    acc = score_acceptance(geometry_of(res, mode), {k: v for k, v in (crit or {}).items() if k != "reference_stl"}) if ok else {"passed": 0, "total": len([k for k in (crit or {}) if k != "reference_stl"])}
    band = None
    ref = (crit or {}).get("reference_stl")
    step = step_of(res, mode)
    if ok and ref and step:
        try:
            band = geom_bands.score_against_reference(step, BENCH / suite / ref, normalize=True)["band"]
        except Exception as e:
            band = "fail"; stderr += f"\nband error: {e}"
    usage = res.get("usage") or {}
    return {"arm": arm, "suite": suite, "id": spec["id"], "tier": spec.get("tier", 0), "ok": ok,
            "gate_hard": gate_hard, "gate_spec": gate_spec,
            "acc_passed": acc.get("passed", 0), "acc_total": acc.get("total", 0), "band": band,
            "wall_s": round(wall, 1), "tokens_out": usage.get("completion_tokens"),
            "build_dir": res.get("build_dir") or res.get("build_dir", ""), "error": res.get("error"),
            "stderr_tail": stderr[-300:] if not ok else ""}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--suites", default=",".join(INTERNAL + EXTERNAL))
    ap.add_argument("--mode", choices=["oneshot", "agent"], default="oneshot")
    ap.add_argument("--limit", type=int, default=0, help="specs per suite (0 = all)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-contaminated", action="store_true")
    ns = ap.parse_args()

    all_arms = arms_mod.load_arms()
    names = list(all_arms) if ns.arms == "all" else ns.arms.split(",")
    suites = {s: load_suite(s) for s in ns.suites.split(",")}
    suites = {k: v for k, v in suites.items() if v[0]}
    if ns.limit:
        suites = {k: (v[0][: ns.limit], v[1]) for k, v in suites.items()}
    clashes = contamination({k: v[0] for k, v in suites.items()})
    if clashes and not ns.allow_contaminated:
        print("REFUSING: card specs present in training data:\n  " + "\n  ".join(clashes)); sys.exit(2)

    stamp = ns.out or str(CARD_DIR / datetime.now().strftime("%Y%m%d-%H%M"))
    out = Path(stamp); out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    done = set()
    rows: list[dict] = []
    if ns.resume and rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line); rows.append(r); done.add((r["arm"], r["suite"], r["id"]))
    meta = {"stamp": out.name, "mode": ns.mode, "arms": names, "suites": list(suites), "limit": ns.limit,
            "started": datetime.now().isoformat(timespec="seconds")}
    try:
        for name in names:
            todo = [(s, sp, acc.get(sp["id"])) for s, (specs, acc) in suites.items() for sp in specs if (name, s, sp["id"]) not in done]
            if not todo:
                continue
            print(f"== arm {name}: {len(todo)} builds"); arms_mod.cmd_use(all_arms[name])
            for suite, spec, crit in todo:
                row = run_row(name, suite, spec, crit, ns.mode, ns.timeout)
                rows.append(row)
                with rows_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {suite}/{spec['id']}: ok={row['ok']} hard={row['gate_hard']} spec={row['gate_spec']} band={row['band']} {row['wall_s']}s")
    finally:
        arms_mod.cmd_restore()
        summary = card_report.summarise(rows)
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
        (out / "card.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2) + "\n")
        (out / "card.md").write_text(card_report.render_md(summary, meta))
        latest = CARD_DIR / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out.name)
        print((out / "card.md").read_text())


if __name__ == "__main__":
    main()
```

Implementation notes for the worker: check `fluid_gen.py`'s `facts` dict keys (the engine's `parse_facts`) and map them to what `score_acceptance` expects (`solids`, `bbox_mm`, `cyl_faces`, `through_holes`, `faces`); adjust `geometry_of` to the real key names. Check that `fluid_gen.py build` prints exactly one JSON line to stdout when `--json` is passed (the runner takes the last line starting with `{`).

- [ ] **Step 2: Smoke run on the control arm (live, 2 specs per suite)**

```bash
cd ~/.openclaw/skills/cad-builder
python3 scripts/run_card.py --arms qwen3.8-27b-nothink --limit 2 --out benchmarks/results/card/smoke --allow-contaminated
cat benchmarks/results/card/smoke/card.md
systemctl --user is-active qwen38-server maker-server
```
Expected: one row per suite with n=2, bands present for heldout-cqe and cadprompt, resident `active`, maker `inactive`. If contamination was reported for the internal suites, list the clashing ids in the task report (they are real findings for Phase 3).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_card.py
git commit -m "run_card: arms x suites card runner (oneshot/agent), resumable, contamination refusal, resident restored in finally"
```

---

### Task 9: Lab view in the web UI (read-only card)

**Files:**
- Modify: `webui/app.py` (add route near `api_job` at :684), `webui/static/index.html` (Lab view + stale label at :290)

**Interfaces:**
- Produces: `GET /api/lab/card` returning `{"meta", "summary"}` from `benchmarks/results/card/latest/card.json` (404 if none); a "Lab" button in the rail header that shows `labEl` (a third top-level view beside `homeEl`/`threadEl`) rendering the summary as a table with the same columns as `card.md`.

- [ ] **Step 1: Route**

```python
CARD_LATEST = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "card" / "latest" / "card.json"


@app.get("/api/lab/card")
def api_lab_card():
    if not CARD_LATEST.exists():
        raise HTTPException(404, "no card yet: run scripts/run_card.py")
    return json.loads(CARD_LATEST.read_text())
```

- [ ] **Step 2: View**

In `static/index.html`: next to the existing "New creation" control in the rail header add `<button id="labBtn" class="ghost" type="button">Lab</button>`. Add a hidden view after `threadEl`:

```html
<section id="lab" class="view hidden">
  <header><h2>Lab</h2><p class="muted">Latest benchmark card. Rows are arm and suite; invalid means no solid was produced.</p></header>
  <div id="labCard"><p class="muted">Loading the card...</p></div>
</section>
```

and JS, following the `newCreation()` pattern (hide `homeEl` and `threadEl`, show `#lab`):

```js
const labEl = document.getElementById('lab');
document.getElementById('labBtn').addEventListener('click', async () => {
  homeEl.classList.add('hidden'); threadEl.classList.add('hidden'); labEl.classList.remove('hidden');
  const box = document.getElementById('labCard');
  try {
    const r = await fetch('/api/lab/card'); if (!r.ok) throw new Error(await r.text());
    const {meta, summary} = await r.json();
    const cols = ['arm','suite','n','valid','invalid_ratio','gate_clean','acceptance','median_wall_s'];
    const pct = v => v == null ? '-' : Math.round(v*100) + '%';
    const rows = Object.values(summary).sort((a,b)=> (a.arm+a.suite).localeCompare(b.arm+b.suite));
    box.innerHTML = `<p class="muted">${meta.stamp} · mode ${meta.mode}</p><table class="card"><thead><tr>${cols.map(c=>`<th>${c.replace('_',' ')}</th>`).join('')}<th>match</th><th>fail</th></tr></thead><tbody>` +
      rows.map(s => `<tr><td>${s.arm}</td><td>${s.suite}</td><td>${s.n}</td><td>${s.valid}</td><td>${pct(s.invalid_ratio)}</td><td>${s.gate_clean}</td><td>${pct(s.acceptance)}</td><td>${Math.round(s.median_wall_s)}</td><td>${s.bands.match}</td><td>${s.bands.fail}</td></tr>`).join('') + '</tbody></table>';
  } catch (e) { box.innerHTML = `<p class="muted">${e.message}</p>`; }
});
```

Make `newCreation()` and `openJob()` also add `hidden` to `labEl` so the views stay exclusive. Add minimal CSS: `table.card{border-collapse:collapse;font-size:13px} table.card th,table.card td{padding:4px 8px;border-bottom:1px solid var(--line,#333);text-align:left}`.

Change line 290 to `<option value="strong">strong · local 27B (maker)</option>`.

- [ ] **Step 3: Verify live**

```bash
systemctl --user restart cad-web
curl -s http://127.0.0.1:8090/api/lab/card | head -c 300
curl -s http://127.0.0.1:8090/ | grep -c 'id="labBtn"'
```
Expected: the smoke card JSON, and `1`. Open https://hp-z2-tower-g4-workstation.taila0e0e0.ts.net:8443 and click Lab (owner does this; report that the route works).

- [ ] **Step 4: Commit**

```bash
git add webui/app.py webui/static/index.html
git commit -m "webui: Lab view (latest benchmark card) + strong-rung label"
```

---

### Task 10: README truth lines

**Files:**
- Modify: `README.md:3`, `:7`, `:59`

- [ ] **Step 1: Edit**

Line 3 becomes:
`**A fully-local text-to-CAD agent: plain-English spec, verified, manufacturable part. Developed on one 24 GB consumer GPU; a 16 GB ship target is the next campaign.**`

In line 7 replace the sentence starting "Everything runs on a single AMD RX 6600" with:
`Everything runs on a single RTX 3090 (24 GB, CUDA): a resident 27B via llama.cpp for chat and a swappable "maker" coder server for builds. No cloud calls at build time. (The 8 GB RX 6600 era, with a resident 35B MoE plus Ollama guests, ended 2026-09-04.)`

Line 59 becomes:
`Requires Python 3.10+, a llama.cpp build with `llama-server`, and the build123d stack (`pip install build123d`). Models: a small coder (e.g. Qwen2.5-Coder-7B Q4), a multimodal critic (Gemma 4 class), nomic-embed-text-v1.5 for retrieval via a local embed server, and a strong rung served by llama.cpp (Qwen3.8-27B today; the Maker Agent 1.0 campaign in `docs/MAKER-1.0-CAMPAIGN.md` is choosing and training it).`

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "README: hardware truth (RTX 3090, 24GB dev / 16GB target, llama.cpp not Ollama)"
```

---

### Task 11: downloads

- [ ] **Step 1: Download the four candidates and the floor**

```bash
cd ~/.openclaw/skills/cad-builder
df -h /mnt/nvme-apps | tail -1
for a in qwen3-coder-30b-a3b glm-4.7-flash devstral-small-2 gpt-oss-20b qwen2.5-coder-7b; do python3 scripts/arms.py download $a; done
python3 scripts/arms.py list
df -h /mnt/nvme-apps | tail -1
```
Expected sizes (bytes): Qwen3-Coder 17,665,334,432; GLM 17,520,169,312; Devstral 14,506,151,072 + mmproj 878,054,048; gpt-oss 12,109,566,624; 7B 4,683,073,504. Every arm shows `ok`.

- [ ] **Step 2: Health-check each new arm for 60 seconds**

```bash
for a in qwen3-coder-30b-a3b glm-4.7-flash devstral-small-2 gpt-oss-20b qwen2.5-coder-7b gemma-4-31b; do
  python3 scripts/arms.py use $a && curl -s http://127.0.0.1:8088/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"Reply with the single word OK"}],"max_tokens":16}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"][:40])'
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
python3 scripts/arms.py restore
```
Expected: every arm answers; record VRAM per arm in the task report. If an arm fails to load, record the error and mark it `"skip": true` in arms.json with the reason.

---

### Task 12: the Phase 0 shootout and the decision

- [ ] **Step 1: Agent baseline on the control (full loop, internal suites)**

```bash
cd ~/.openclaw/skills/cad-builder
nohup python3 scripts/run_card.py --arms qwen3.8-27b-nothink --mode agent --suites text-to-cad,organic,hard-eval,heldout-cqe --out benchmarks/results/card/agent-baseline > benchmarks/results/card/agent-baseline.log 2>&1 &
```

- [ ] **Step 2: The shootout (one-shot, all arms, all suites), resumable**

```bash
nohup python3 scripts/run_card.py --arms all --mode oneshot --resume --out benchmarks/results/card/phase0 > benchmarks/results/card/phase0.log 2>&1 &
```
Run only when the previous job finished (one GPU). The resident is down while an arm runs; chat requests queue in the gpu-proxy for up to 15 minutes. Check progress with `tail -3 benchmarks/results/card/phase0.log`; on interruption rerun the same command (rows resume).

- [ ] **Step 3: Write the decision**

Create `benchmarks/results/card/phase0/DECISION.md` applying spec section 6 in writing: the ranked table, the 3-point band, the tie-breaks used, the chosen base, and the arms deleted from disk. Delete losers' GGUFs only for arms with `hf` entries (never the resident's file), and only after the owner has seen the card.

- [ ] **Step 4: Commit results**

```bash
git add benchmarks/results/card/phase0/card.json benchmarks/results/card/phase0/card.md benchmarks/results/card/phase0/DECISION.md benchmarks/results/card/agent-baseline/card.md
git commit -m "card: Phase 0 shootout results + base model decision"
```

---

## Self-review

- Spec 4.1 (loaders, metrics, arms file, output, Lab render): Tasks 4, 5, 7, 8, 9.
- Spec 4.1a (arm list, deletions): Tasks 4, 11, 12.
- Spec 4.2 (maker server, cad.json maker block, hooks, rollback): Tasks 1, 2, 3. Rollback = `arms.py restore` or editing `maker.enabled`.
- Spec 5 Phase 0 exit ("one command", 3 local models, 2+ public suites, stale labels, README, base chosen): Tasks 8 (`run_card.py --arms all`), 5 (cadprompt + cad-arena, text2cadquery optional), 9, 10, 12.
- Spec 6 (rule applied in writing): Task 12 step 3.
- Types: `load_arms` returns dict keyed by name (Tasks 4, 8); `score_against_reference(..., normalize=True)` (Tasks 6, 8); row schema identical in Tasks 7 and 8; `maker_config()` keys identical in Tasks 1, 3, 4.
- Not in this plan (later phases): thinking A/B beyond the two control arms, critic swap, data engine, training, Bridge tile.
