#!/usr/bin/env python3
"""The Maker Agent card: every arm on every suite, one JSON + one markdown table.

  run_card.py --arms qwen3.8-27b-nothink --limit 2                # smoke
  run_card.py --arms all --mode oneshot                            # the Phase 0 shootout
  run_card.py --arms qwen3.8-27b-nothink --mode agent --suites text-to-cad,organic,hard-eval,heldout-cqe

oneshot = scripts/fluid_gen.py build (one codegen, one salvage, gate, no critic): base-model capability.
agent   = python3 -m cad_v5 --once --json (full observe-edit loop): the agent baseline.
The resident is restored in a finally: whatever happens. Rows append to rows.jsonl so --resume continues.

Every build subprocess (either mode) gets CAD_KEEP_MAKER=1 alongside CAD_BENCH=1: cad_engine's
_ensure_default_server/_resume_default_server treat that as "a card runner owns the maker-server
lifecycle" and skip their own per-build stop/start, so the arm arms_mod.cmd_use started at the top
of the arm loop stays warm across every build in that arm instead of cold-loading per build. This
runner's own finally is the ONLY place that calls arms_mod.cmd_restore(): an interrupted card
(Ctrl-C, a crash mid-arm) leaves the maker up and the resident down until `python3 scripts/arms.py
restore` is run by hand.
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
import cad_engine as engine        # noqa: E402  (parse_facts + run_inspect: cheap, stdlib-only imports)

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
    # CAD_KEEP_MAKER=1: see the module docstring, the arm stays warm for the whole arm
    # loop instead of cad_engine cold-loading/evicting it around every single build.
    env = {**os.environ, "CAD_BENCH": "1", "CAD_KEEP_MAKER": "1"}
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


def step_of(res: dict, mode: str) -> Path | None:
    if mode == "agent":
        return Path(res["step_local"]) if res.get("step_local") else None
    bd = res.get("build_dir")
    if not bd:
        return None
    steps = sorted(Path(bd).glob("*.step"))
    return steps[-1] if steps else None


def geometry_of(res: dict, mode: str) -> dict:
    """Map a build result's own key names onto what score_acceptance expects: solids,
    bbox_mm, faces, cyl_faces, through_holes.

    oneshot: fluid_gen's `facts` dict IS cad_engine.parse_facts's output (see
    scripts/inspect's FACTS_JSON block), its size key is `bbox`, not `bbox_mm`.

    agent: cad_v5's `--once --json` result carries NO geometry facts at all. loop.run
    computes them locally (cad_v5/loop.py `_facts_for`) only to print the console summary
    and delta, and never puts them in the returned result dict. So for agent mode the
    runner re-derives facts the same way fluid_gen does: parse_facts(run_inspect(step)),
    off the STEP the build actually produced (`step_local`)."""
    if mode == "oneshot":
        f = res.get("facts") or {}
    else:
        f = {}
        step = step_of(res, mode)
        if step and step.exists():
            try:
                f = engine.parse_facts(engine.run_inspect(step)["output"])
            except Exception:
                f = {}
    return {"solids": f.get("solids"), "bbox_mm": f.get("bbox"), "faces": f.get("faces"),
            "cyl_faces": f.get("cyl_faces"), "through_holes": f.get("through_holes")}


def run_row(arm: str, suite: str, spec: dict, crit: dict | None, mode: str, timeout: int) -> dict:
    res, wall, stderr = build_once(spec["spec"], mode, timeout)
    # bool(x) and y returns y verbatim when x is truthy (Python's `and` short-circuits to
    # the operand, not to a bool). solids is an int, so `ok` came out as e.g. 1 instead
    # of True for oneshot rows. Wrap the whole thing so ok is always a real bool.
    ok = bool(bool(res.get("ok")) and (res.get("facts", {}).get("solids", 1) if mode == "oneshot" else res.get("has_bodies", True)))
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
            "build_dir": res.get("build_dir") or res.get("step_local") or "", "error": res.get("error"),
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
        card_md = card_report.render_md(summary, meta)
        benchcad_path = out / "benchcad.json"
        if benchcad_path.exists():
            # Optional companion run: scripts/run_benchcad.py (the official BenchCAD harness,
            # Task 5b) writes benchcad.json into the same --out dir when driven alongside this
            # runner. If it's there, fold its table into the same card.md instead of leaving a
            # second file the reader has to know to look for.
            try:
                card_md += "\n" + card_report.render_benchcad_md(json.loads(benchcad_path.read_text()))
            except Exception as e:
                card_md += f"\n(benchcad.json present but could not be rendered: {e})\n"
        (out / "card.md").write_text(card_md)
        latest = CARD_DIR / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out.name)
        print((out / "card.md").read_text())


if __name__ == "__main__":
    main()
