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
import argparse, dataclasses, importlib.util, json, os, subprocess, sys, time
from dataclasses import dataclass
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

# Derived from harvest_census.CARD_SUITES (the contamination guard's own list) so the two
# cannot drift: a suite added there is automatically carded and automatically never trained on.
# The first four entries are the internal suites, the rest the public ones.
INTERNAL = hc.CARD_SUITES[:4]
EXTERNAL = hc.CARD_SUITES[4:]
# Keys in an acceptance entry that are NOT criteria for score_acceptance. reference_stl names
# the band reference; normalized is the band's unit policy (see geom_bands.score_against_reference);
# the rest are provenance/annotation that older suites carry. Counting them in a denominator
# (as the failed-row branch used to) silently inflates every failure's total.
NON_CRITERIA = {"reference_stl", "normalized", "checks", "scoring", "source", "bbox_notes",
                "min_holes_nulls", "heldout", "notes", "_meta"}
TRAIN_FILES = [Path.home() / ".openclaw" / n for n in ("cad-sftpairs.jsonl", "cad-examples.jsonl", "cad-sft-train.jsonl")]

# Phase 1 stratified subset: the first N specs of each suite, in file order, so the same
# subset is reproduced every run without a random seed. "full" (the default) runs everything.
SUBSETS = {
    "full": None,
    "phase1": {"cadprompt": 30, "text2cadquery": 30, "heldout-cqe": 25, "text-to-cad": 10,
               "organic": 5, "hard-eval": 15, "cad-arena": 12},
    "phase1think": {"cadprompt": 15, "text2cadquery": 15, "heldout-cqe": 10, "text-to-cad": 5,
                    "organic": 3, "hard-eval": 5, "cad-arena": 6},
}


@dataclass(frozen=True)
class Knobs:
    """The per-run configuration knobs a card variant sweeps, threaded from argparse down to
    the child build subprocess. `variant` labels the arm (e.g. "bo3") so several configurations
    of the same arm can coexist in one rows.jsonl; the rest become env vars / argv the child
    build reads, none of them mandatory (empty/zero = engine default)."""
    variant: str = ""
    candidates: int = 0
    no_fewshots: bool = False
    critic: str = ""
    critic_url: str = ""

    def env(self) -> dict:
        e = {}
        if self.candidates > 0:
            e["CAD_CANDIDATES"] = str(self.candidates)
        if self.critic:
            e["CAD_CRITIC_MODEL"] = self.critic
        if self.critic_url:
            e["CAD_CRITIC_URL"] = self.critic_url
        return e

    def argv(self) -> list[str]:
        return ["--no-fewshots"] if self.no_fewshots else []


def apply_subset(suites: dict, name: str) -> dict:
    """Cap each suite to the first N specs (file order, deterministic). `name` "full" is a
    no-op; a suite not named in the subset's per-suite caps is left uncapped."""
    caps = SUBSETS[name]
    if caps is None:
        return suites
    return {s: (specs[: caps.get(s, len(specs))], acc) for s, (specs, acc) in suites.items()}


def labelled(name: str, knobs: Knobs) -> str:
    """The row/resume-key arm label: the arm name, plus "+variant" when a variant is given."""
    return f"{name}+{knobs.variant}" if knobs.variant else name


def load_suite(name: str) -> tuple[list[dict], dict]:
    d = BENCH / name
    if not (d / "specs.json").exists():
        return [], {}
    data = json.loads((d / "specs.json").read_text())
    specs = data["benchmarks"] if isinstance(data, dict) and "benchmarks" in data else data
    acc = json.loads((d / "acceptance.json").read_text()) if (d / "acceptance.json").exists() else {}
    return specs, acc


def contamination(specs_by_suite: dict[str, list[dict]]) -> list[str]:
    """Card specs that also appear in the training data.

    Identity is hc._key (sha1 of the whitespace-collapsed full text), never a bare hc._slug:
    the 40-char slug is degenerate on the public suites (text2cadquery still has one slug
    covering 28 different specs), so slug-keyed, one training row sharing an opening would
    condemn 28 unrelated specs while a genuine collision past character 40 went unseen.

    But exact matching alone is too weak for the case this guard exists for: the real known
    clash is text-to-cad/05, whose training copy is the SAME part reworded, not the same
    string. So a slug match is also reported, as a near duplicate, and only when that slug
    identifies exactly one spec in its own suite. That condition is what keeps the degenerate
    public-suite buckets out: a slug shared by 28 specs is not evidence about any of them."""
    train = set()
    for f in TRAIN_FILES:
        if f.exists():
            for line in f.read_text().splitlines():
                try:
                    train.add(json.loads(line).get("spec", ""))
                except Exception:
                    pass
    train_keys = {hc._key(t) for t in train}
    train_slugs = {hc._slug(t, 40) for t in train}
    clashes = []
    for suite, specs in specs_by_suite.items():
        slug_counts: dict[str, int] = {}
        for s in specs:
            slug_counts[hc._slug(s["spec"], 40)] = slug_counts.get(hc._slug(s["spec"], 40), 0) + 1
        for s in specs:
            slug = hc._slug(s["spec"], 40)
            if hc._key(s["spec"]) in train_keys:
                clashes.append(f"{suite}/{s['id']}: {s['spec'][:70]}")
            elif slug in train_slugs and slug_counts[slug] == 1:
                clashes.append(f"{suite}/{s['id']} (near duplicate, same opening): {s['spec'][:70]}")
    return clashes


def build_once(spec: str, mode: str, timeout: int, knobs: Knobs | None = None) -> tuple[dict, float, str]:
    if knobs is None:
        knobs = Knobs()
    # CAD_KEEP_MAKER=1: see the module docstring, the arm stays warm for the whole arm
    # loop instead of cad_engine cold-loading/evicting it around every single build.
    env = {**os.environ, "CAD_BENCH": "1", "CAD_KEEP_MAKER": "1", **knobs.env()}
    if mode == "oneshot":
        cmd = [sys.executable, str(SCRIPTS / "fluid_gen.py"), "build", spec, "--coder", "strong", "--json"]
    else:
        cmd = [sys.executable, "-m", "cad_v5", spec, "--once", "--json", "--coder", "strong", "--target", "file"]
    cmd += knobs.argv()
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
    if not f:
        # {} not a dict of Nones. score_acceptance's contract is "no geometry => 0 of N", which
        # it reads off `bool(geom)`; a dict of Nones looks like geometry and then raises
        # TypeError on `cyl >= 0` when a min_holes criterion is present.
        return {}
    return {"solids": f.get("solids"), "bbox_mm": f.get("bbox"), "faces": f.get("faces"),
            "cyl_faces": f.get("cyl_faces"), "through_holes": f.get("through_holes")}


def criteria_of(crit: dict | None) -> dict:
    """The acceptance entry stripped to the keys score_acceptance actually scores."""
    return {k: v for k, v in (crit or {}).items() if k not in NON_CRITERIA}


def step_in(path: str) -> Path | None:
    """The STEP named by a row's build_dir: the newest *.step inside it when it is a
    directory (oneshot), the file itself when it is the STEP (agent), else None."""
    p = Path(path) if path else None
    if p and p.is_dir():
        steps = sorted(p.glob("*.step"))
        return steps[-1] if steps else None
    return p if p and p.is_file() else None


def band_of(step: Path, reference: Path, crit: dict | None) -> str:
    """Chamfer band, normalised only where the suite says so.

    normalize=True rescales both meshes to a common diagonal, which is right for the public
    suites (DeepCAD-style units, not mm) and WRONG for the mm-specified internal suites: it
    would forgive a part built at half the specified size. The acceptance entry carries the
    policy (`normalized`), written by scripts/fetch_external.py."""
    return geom_bands.score_against_reference(
        step, reference, normalize=bool((crit or {}).get("normalized")))["band"]


def run_row(arm: str, suite: str, spec: dict, crit: dict | None, mode: str, timeout: int,
            knobs: Knobs | None = None) -> dict:
    if knobs is None:
        knobs = Knobs()
    res, wall, stderr = build_once(spec["spec"], mode, timeout, knobs)
    # bool(x) and y returns y verbatim when x is truthy (Python's `and` short-circuits to
    # the operand, not to a bool). solids is an int, so `ok` came out as e.g. 1 instead
    # of True for oneshot rows. Wrap the whole thing so ok is always a real bool.
    ok = bool(bool(res.get("ok")) and (res.get("facts", {}).get("solids", 1) if mode == "oneshot" else res.get("has_bodies", True)))
    if mode == "oneshot":
        gate_hard = len(res.get("gate_hard") or []); gate_spec = len(res.get("gate_spec") or [])
    else:
        gate_hard = 0 if res.get("converged") else 1; gate_spec = 0
    # ONE denominator for ok and failed rows alike: score_acceptance itself, whose docstring
    # guarantees 0/N for a build with no geometry. The old failed-row branch counted raw
    # criteria keys instead, which both included non-criterion keys and included criteria
    # score_acceptance would have skipped, so a failure's total did not match a success's.
    acc = score_acceptance(geometry_of(res, mode) if ok else {}, criteria_of(crit))
    band = None
    ref = (crit or {}).get("reference_stl")
    step = step_of(res, mode)
    if ok and ref and step:
        try:
            band = band_of(step, BENCH / suite / ref, crit)
        except Exception as e:
            band = "fail"; stderr += f"\nband error: {e}"
    usage = res.get("usage") or {}
    return {"arm": labelled(arm, knobs), "suite": suite, "id": spec["id"], "tier": spec.get("tier", 0), "ok": ok,
            "gate_hard": gate_hard, "gate_spec": gate_spec,
            "acc_passed": acc.get("passed", 0), "acc_total": acc.get("total", 0), "band": band,
            "helper": bool(res.get("helper")),
            "wall_s": round(wall, 1), "tokens_out": usage.get("completion_tokens"),
            "build_dir": res.get("build_dir") or res.get("step_local") or "", "error": res.get("error"),
            "stderr_tail": stderr[-300:] if not ok else ""}


def write_card(out: Path, rows: list[dict], meta: dict) -> str:
    """Write card.json + card.md into `out` and return the markdown. Pure file work: no
    systemctl, no model, so it is safe to call before the resident restore in the finally."""
    summary = card_report.summarise(rows)
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
    if out.resolve().parent == CARD_DIR.resolve():
        # `latest` lives in CARD_DIR and points at a sibling by name, so it is only meaningful
        # for runs written there. An --out somewhere else used to retarget it at a name that
        # does not exist inside CARD_DIR, leaving a dangling symlink.
        latest = CARD_DIR / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out.name)
    return card_md


def rescore(out: Path) -> str:
    """Recompute the derived columns of an existing run from rows.jsonl, no rebuilding.

    Fixes rows written by an older runner: acceptance totals for failed rows (which used to
    count non-criterion keys) and bands (which used to normalise every suite). A band is only
    recomputed when the row's STEP is still on disk; otherwise the stored value is kept.
    rows.jsonl is backed up to rows.jsonl.bak before being rewritten."""
    rows_path = out / "rows.jsonl"
    if not rows_path.exists():
        raise SystemExit(f"{rows_path} not found")
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    crits = {s: load_suite(s)[1] for s in {r["suite"] for r in rows}}
    changed = 0
    for r in rows:
        crit = crits.get(r["suite"], {}).get(r["id"])
        before = (r.get("acc_passed"), r.get("acc_total"), r.get("band"))
        if not r.get("ok"):
            acc = score_acceptance({}, criteria_of(crit))
            r["acc_passed"], r["acc_total"] = acc["passed"], acc["total"]
        ref = (crit or {}).get("reference_stl")
        if r.get("ok") and ref:
            # A row's build_dir is a directory (oneshot) or the STEP itself (agent): the
            # writer stores `build_dir or step_local`, so accept either shape here.
            step = step_in(r.get("build_dir") or "")
            if step:
                try:
                    r["band"] = band_of(step, BENCH / r["suite"] / ref, crit)
                except Exception:
                    r["band"] = "fail"
        r.setdefault("helper", False)
        if (r.get("acc_passed"), r.get("acc_total"), r.get("band")) != before:
            changed += 1
    rows_path.replace(rows_path.with_suffix(".jsonl.bak"))
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    card = out / "card.json"
    meta = json.loads(card.read_text())["meta"] if card.exists() else {"stamp": out.name}
    meta["rescored"] = datetime.now().isoformat(timespec="seconds")
    md = write_card(out, rows, meta)
    print(f"rescored {len(rows)} row(s), {changed} changed; backup {rows_path.with_suffix('.jsonl.bak').name}")
    return md


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
    ap.add_argument("--variant", default="", help="label appended to the arm as +LABEL, so several "
                     "configurations of the same arm can share one rows.jsonl (default: none)")
    ap.add_argument("--candidates", type=int, default=0,
                     help="CAD_CANDIDATES for the child build, best-of-N first turn (0 = engine default)")
    ap.add_argument("--no-fewshots", action="store_true", help="pass --no-fewshots to the child build")
    ap.add_argument("--critic", default="", help="CAD_CRITIC_MODEL for the child build (default: engine default)")
    ap.add_argument("--critic-url", default="", help="CAD_CRITIC_URL for the child build, e.g. "
                     "http://127.0.0.1:8092/v1/chat/completions to run the critic on deploy/critic-server "
                     "instead of riding the coder server (default: engine default)")
    ap.add_argument("--subset", choices=sorted(SUBSETS), default="full",
                     help="cap each suite to a stratified subset, first N specs in file order "
                          "(default: full, no cap)")
    ap.add_argument("--rescore", default="", metavar="DIR",
                    help="recompute acceptance/bands for an existing run dir from rows.jsonl and "
                         "rewrite its card; builds nothing, touches no GPU")
    ns = ap.parse_args()

    if ns.rescore:
        print(rescore(Path(ns.rescore)))
        return

    knobs = Knobs(variant=ns.variant, candidates=ns.candidates, no_fewshots=ns.no_fewshots,
                  critic=ns.critic, critic_url=ns.critic_url)

    all_arms = arms_mod.load_arms()
    names = list(all_arms) if ns.arms == "all" else ns.arms.split(",")
    suites = {s: load_suite(s) for s in ns.suites.split(",")}
    suites = {k: v for k, v in suites.items() if v[0]}
    if ns.limit:
        suites = {k: (v[0][: ns.limit], v[1]) for k, v in suites.items()}
    suites = apply_subset(suites, ns.subset)
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
    meta.update(variant=knobs.variant, knobs=dataclasses.asdict(knobs), subset=ns.subset)
    try:
        for name in names:
            if all_arms[name].get("skip"):
                print(f"== arm {name}: skipped (\"skip\": true in benchmarks/arms.json)")
                meta.setdefault("skipped_arms", {})[name] = "skip: true in arms.json"
                continue
            label = labelled(name, knobs)
            todo = [(s, sp, acc.get(sp["id"])) for s, (specs, acc) in suites.items() for sp in specs if (label, s, sp["id"]) not in done]
            if not todo:
                continue
            print(f"== arm {label}: {len(todo)} builds")
            try:
                arms_mod.cmd_use(all_arms[name])
            except BaseException as exc:
                # One arm that will not load (missing GGUF, a llama.cpp flag it rejects, an
                # OOM on load) used to abort the whole --arms all card, losing every arm
                # after it. Record why and move to the next one; cmd_use has already
                # restored the resident on its way out.
                print(f"   arm {name} unavailable: {exc}")
                meta.setdefault("skipped_arms", {})[name] = str(exc)[:200]
                continue
            for suite, spec, crit in todo:
                row = run_row(name, suite, spec, crit, ns.mode, ns.timeout, knobs=knobs)
                rows.append(row)
                with rows_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {suite}/{spec['id']}: ok={row['ok']} hard={row['gate_hard']} spec={row['gate_spec']} band={row['band']} {row['wall_s']}s")
    finally:
        # Write the card BEFORE restoring: cmd_restore shells out to systemctl and waits up to
        # 5 minutes on a health endpoint, so a failure or a Ctrl-C there used to take the whole
        # card down with it after every build had already been paid for.
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
        card_md = write_card(out, rows, meta)
        try:
            arms_mod.cmd_restore()
        except BaseException as exc:
            print(f"RESIDENT NOT RESTORED: {exc}\n  run: python3 scripts/arms.py restore")
        print(card_md)


if __name__ == "__main__":
    main()
