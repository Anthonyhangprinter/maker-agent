#!/usr/bin/env python3
"""
Judge-eval Task 3: ask the resident (qwen3.8-27b) to judge every labelled, rebuilt program in
two modes (T = text only: spec+code+facts; V = vision: same plus the two-panel render), one
request at a time, and save every request/response to judgements.jsonl (image bytes NOT saved,
per the task). Resumable by (program_id, mode) — safe to re-run after an interruption.

Preconditions checked before any call: qwen38-server must be active AND the resident's own
health endpoint must answer, or this script refuses to run (that is the shared safety gate: if
another job is about to take the GPU, we must not start a long judging run).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import judge_client  # noqa: E402
from judge_prompt import JUDGE_SYSTEM_PROMPT, build_user_content  # noqa: E402

RUN_ID = "2026-09-19"
RESULTS_DIR = HERE / "results" / RUN_ID
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
JUDGEMENTS_PATH = RESULTS_DIR / "judgements.jsonl"


def check_resident():
    r = subprocess.run(["systemctl", "--user", "is-active", "qwen38-server"],
                        capture_output=True, encoding="utf-8")
    active = r.stdout.strip() == "active"
    healthy = False
    try:
        r2 = subprocess.run(["curl", "-s", "-m", "5", "localhost:8086/health"],
                             capture_output=True, encoding="utf-8")
        healthy = '"status"' in r2.stdout and "ok" in r2.stdout
    except Exception:
        pass
    return active, healthy


def load_manifest():
    p = RESULTS_DIR / "build_manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_done():
    done = set()
    if JUDGEMENTS_PATH.exists():
        with open(JUDGEMENTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add((rec["program_id"], rec["mode"]))
                except Exception:
                    pass
    return done


def append_result(rec):
    with open(JUDGEMENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    active, healthy = check_resident()
    print(f"qwen38-server active={active} health={healthy}")
    if not active or not healthy:
        print("REFUSING TO RUN: resident is not active/healthy. Stopping without calling the "
              "judge endpoint.")
        sys.exit(1)

    labelled = {r["program_id"]: r for r in json.loads((HERE / "labelled.json").read_text(encoding="utf-8"))}
    manifest = load_manifest()
    done = load_done()

    todo = []
    for pid, rec in labelled.items():
        m = manifest.get(pid)
        if not m or not m.get("ok"):
            continue  # not rebuilt (excluded / failed) — no judge needed
        for mode in ("text", "vision"):
            if (pid, mode) not in done:
                todo.append((pid, mode))

    print(f"{len(labelled)} labelled, {sum(1 for m in manifest.values() if m.get('ok'))} rebuilt, "
          f"{len(todo)} judge calls to make (of {2 * sum(1 for m in manifest.values() if m.get('ok'))} total)")

    for idx, (pid, mode) in enumerate(todo, 1):
        # Re-check health before every call — a family chat or another job could take the GPU
        # mid-run; better to stop cleanly than to error into a shared server.
        active, healthy = check_resident()
        if not active or not healthy:
            print(f"STOPPING at {idx}/{len(todo)}: resident no longer active/healthy "
                  f"(active={active} healthy={healthy}). Progress so far is saved.")
            break

        rec = labelled[pid]
        m = manifest[pid]
        artifact_dir = Path(m["step"]).parent if m.get("step") else None
        facts = m.get("facts", {})
        facts_json = json.dumps(facts, indent=2)
        code = Path(rec["code_path"]).read_text(encoding="utf-8")
        spec = rec["spec"]

        image_url = None
        if mode == "vision":
            png_path = m.get("png")
            if not png_path or not Path(png_path).exists():
                print(f"SKIP {pid} vision: no render found")
                continue
            image_url = judge_client.image_data_url(png_path)

        user_content = build_user_content(spec, code, facts_json, mode, image_url)

        t0 = time.time()
        result = judge_client.call_judge(JUDGE_SYSTEM_PROMPT, user_content)
        wall_s = time.time() - t0

        verdict_obj = result.get("verdict_obj")
        verdict = None
        confidence = None
        problems = None
        if verdict_obj:
            v = str(verdict_obj.get("verdict", "")).strip().lower()
            if v in ("correct", "wrong"):
                verdict = v
            try:
                confidence = float(verdict_obj.get("confidence"))
            except (TypeError, ValueError):
                confidence = None
            problems = verdict_obj.get("problems")

        out_rec = {
            "program_id": pid,
            "mode": mode,
            "writer": rec["writer"],
            "writer_family": rec["writer_family"],
            "suite": rec["suite"],
            "spec_id": rec["spec_id"],
            "truth": rec["truth"],
            "ok": result["ok"],
            "error": result.get("error"),
            "verdict": verdict if verdict else "abstain",
            "confidence": confidence,
            "problems": problems,
            "content": result.get("content"),
            "reasoning_content": result.get("reasoning_content"),
            "usage": result.get("usage"),
            "wall_s": wall_s,
            "finish_reason": result.get("finish_reason"),
        }
        append_result(out_rec)
        print(f"[{idx}/{len(todo)}] {pid} [{mode}] -> verdict={out_rec['verdict']} "
              f"(truth={rec['truth']}) conf={confidence} {wall_s:.1f}s")

    print("Done (or stopped early — see message above).")


if __name__ == "__main__":
    main()
