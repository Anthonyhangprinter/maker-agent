"""One-off recovery script: rebuild ~/.openclaw/cad-web/sessions.json (the web UI's
creation rail) after it was overwritten with `[]` at 2026-09-26 23:58 with no backup.

The job store itself is gone, but every build's artefacts still live on disk under
~/.openclaw/cad-builds/fluid_*/ (build.png/build.step/build.stl/build_source.py/fluid.json).
This script re-derives job rows from those artefacts, in EXACTLY the shape `webui/app.py`
persists (`_persist`) and rehydrates (`_load`) — see that file's job dict literal in
`api_build` and `_result_public`.

Telling a real web UI creation apart from a benchmark/harvest run:
fluid_gen.py's `cmd_build` (scripts/fluid_gen.py) is the SAME code path for every caller —
the web UI, the CLI, and every benchmark/harvest script all create a `fluid_<UTC ts>` dir
the same way, so the directory name alone proves nothing. What does distinguish them,
checked empirically against all 455 fluid_* dirs before writing this script:

  - Benchmark/harvest runs replay a FIXED set of prompts, sometimes verbatim across
    several different days (regression-testing the engine after a change). Across the
    corpus, 453 of 455 dirs share a `user_spec` with at least one other dir (127 distinct
    specs, each appearing 2 to 18+ times) — and the repeats land in tight ~1-3 minute
    clusters, consistent with a script iterating a prompt list, not someone typing at a
    keyboard.
  - Exactly 2 dirs have a `user_spec` that appears NOWHERE else in the corpus:
    fluid_20260922_110239 ("a 120x80x40 enclosure with 4x m3 mounting hole and 2mm thick
    walls") and fluid_20260923_103658 ("make this v-bracket", image-conditioned — matches
    the owner's own memory of a V-bracket build). Both also sit outside every dense
    same-spec cluster in time.
  - `fluid_gatetest/` (Aug 11) has no fluid.json at all — a manual gate smoke-test
    artefact, not a build — and is excluded by the same "must have fluid.json with a
    user_spec" gate the task asked for as a fallback rule.

So the rule this script applies: a fluid_* dir counts as a web UI creation iff it has a
fluid.json with a non-empty user_spec AND that user_spec is unique across the whole
fluid_* corpus. On this data that yields exactly the 2 dirs above. If the corpus changes
(more real creations happen before this script is ever re-run), the rule still holds:
re-run it and it re-derives from whatever is unique at that point. Everything excluded is
printed with its reason so the owner can sanity-check the call.

No chat/revise history is reconstructed: every fluid.json's "history" list was empty
across the whole corpus (checked before writing this), and no fluid_* dir contains a
turn<N>.* snapshot file, so there is nothing to replay into job["turns"]/job["chat"].

Usage:
    python3 -X utf8 webui/restore_history.py [--dry-run] [--sessions-file PATH]

Procedure (see the task): stop cad-web.service, run this script, start cad-web.service,
then verify with curl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent   # …/skills/cad-builder
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

# Pure `ast` over build_source.py, no build123d/OCP needed — safe to import from any
# interpreter (see cad_v5/design_assistant.py's own module docstring).
from cad_v5.design_assistant import extract_build123d_params  # noqa: E402

# Must match webui/app.py exactly — this script rebuilds rows for THAT loader.
BUILDS_DIR = (Path.home() / ".openclaw" / "cad-builds").resolve()
SESSIONS_FILE_DEFAULT = Path.home() / ".openclaw" / "cad-web" / "sessions.json"
MAX_SESSIONS = 200

# app.py's _result_public artifact key -> filename map.
ARTIFACT_MAP = (("render", "build.png"), ("step", "build.step"), ("stl", "build.stl"),
                ("dxf", "build.dxf"), ("scad", "build.scad"),
                ("reference", "reference.jpg"), ("source", "build_source.py"))


def _fallback_title(spec: str) -> str:
    spec = (spec or "").strip()
    if not spec:
        return "Image-only build"
    return spec[:45].rstrip() + ("…" if len(spec) > 45 else "")


def _epoch_from_dirname(name: str, fallback_mtime: float) -> float:
    """fluid_gen.py names build dirs `fluid_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}` —
    parse that back into an epoch timestamp. Falls back to the directory's own mtime if
    the name doesn't parse (should not happen for a real fluid_* dir, but never crash a
    restore over one oddly-named directory)."""
    try:
        ts = name[len("fluid_"):]
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return fallback_mtime


def _scan() -> tuple[list[dict], list[tuple[str, str]]]:
    """Returns (candidate rows' raw ingredients, [(dir_name, exclusion_reason), ...])."""
    dirs = sorted(p for p in BUILDS_DIR.glob("fluid_*") if p.is_dir())
    parsed = {}
    excluded: list[tuple[str, str]] = []
    for d in dirs:
        meta_f = d / "fluid.json"
        if not meta_f.is_file():
            excluded.append((d.name, "no fluid.json (not a fluid_gen build — e.g. a "
                                      "manual gate-test artefact)"))
            continue
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
        except Exception as e:
            excluded.append((d.name, f"fluid.json unreadable ({e})"))
            continue
        user_spec = (meta.get("user_spec") or "").strip()
        if not user_spec:
            excluded.append((d.name, "empty user_spec (image-only build with no fewshot "
                                      "text — can't disambiguate batch vs. interactive; "
                                      "none found in this corpus)"))
            continue
        parsed[d.name] = {"dir": d, "meta": meta, "user_spec": user_spec}

    counts = Counter(v["user_spec"] for v in parsed.values())
    kept = []
    for name, v in parsed.items():
        n = counts[v["user_spec"]]
        if n > 1:
            excluded.append((name, f"user_spec repeated in {n} fluid_* dirs — a batch/"
                                    f"benchmark prompt list, not a one-off web UI creation"))
            continue
        kept.append(v)
    return kept, sorted(excluded)


def _build_row(entry: dict) -> dict:
    d, meta = entry["dir"], entry["meta"]
    spec = meta.get("spec") or entry["user_spec"]
    coder = meta.get("coder") or "strong"
    created_at = _epoch_from_dirname(d.name, d.stat().st_mtime)

    artifacts = {}
    for key, fname in ARTIFACT_MAP:
        if (d / fname).is_file():
            artifacts[key] = f"/artifacts/{d.name}/{fname}"

    result = {"ok": True, "mode": "fluid", "build_dir_fs": str(d), "build_id": d.name,
              "artifacts": artifacts}
    src = d / "build_source.py"
    if src.is_file():
        try:
            params = extract_build123d_params(src.read_text(encoding="utf-8"))
            if params:
                result["params"] = params
        except Exception as e:
            print(f"[restore] {d.name}: param extraction failed ({e}) — skipping params",
                  flush=True)

    # reference.jpg in the build dir is direct proof an image conditioned this build (the
    # engine only writes it when `--image` was passed) — use it as job["image"] rather
    # than guessing which ~/.openclaw/cad-web/uploads/*.png file it came from.
    ref = d / "reference.jpg"
    image = str(ref) if ref.is_file() else None

    job = {
        "id": uuid.uuid4().hex[:12], "spec": spec, "coder": coder, "image": image,
        "candidates": "", "fewshots": True, "lang": "build123d", "engine_mode": "fluid",
        "composed_spec": None, "parameters": None, "user": "local",
        "status": "done", "result": result, "error": None,
        "created_at": created_at, "updated_at": created_at,
        "title": _fallback_title(spec), "titled": False,
        "turns": [], "chat": [],
    }
    return job


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                     help="scan and print what would be restored, write nothing")
    ap.add_argument("--sessions-file", default=str(SESSIONS_FILE_DEFAULT),
                     help="override the sessions.json path (matches "
                          "CAD_WEB_SESSIONS_FILE if app.py has been updated to honour it)")
    args = ap.parse_args()
    sessions_file = Path(args.sessions_file)

    kept, excluded = _scan()
    print(f"[restore] {len(kept)} web UI creation(s) identified out of "
          f"{len(kept) + len(excluded)} fluid_* dirs scanned.")
    for name, reason in excluded:
        print(f"  excluded {name}: {reason}")

    rows = [_build_row(e) for e in kept]
    rows.sort(key=lambda j: j["created_at"])
    rows = rows[-MAX_SESSIONS:]

    print(f"[restore] restoring {len(rows)} row(s):")
    for j in rows:
        print(f"  {j['id']}  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(j['created_at']))}  "
              f"{j['title']!r}  build_id={j['result']['build_id']}")

    if args.dry_run:
        print("[restore] --dry-run: nothing written")
        return 0

    if sessions_file.is_file():
        bak = sessions_file.with_name(
            f"{sessions_file.name}.bak-{time.strftime('%Y%m%d%H%M%S')}")
        bak.write_bytes(sessions_file.read_bytes())
        print(f"[restore] backed up existing {sessions_file} -> {bak}")

    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = sessions_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows), encoding="utf-8")
    os.replace(tmp, sessions_file)
    print(f"[restore] wrote {len(rows)} row(s) to {sessions_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
