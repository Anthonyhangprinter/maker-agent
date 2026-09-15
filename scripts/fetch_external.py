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

EXECUTES DATASET CODE. Text-to-CadQuery ships its references as CadQuery programs, and the only
way to get reference geometry out of them is to run them. This script runs each one with the
operator's own privileges, in the private venv at benchmarks/external/.venv-cq, with no sandbox
and no network isolation. That is acceptable only because this is an evaluation-only, operator-run
fetch of a dataset the operator chose: nothing here runs unattended, from a service, or on
untrusted input. The HF download is pinned to a revision (--revision) so the code executed is the
code that was reviewed.

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
# No tag is published on this repo, so `main` is the only available pin (a branch, not an
# immutable commit): recorded in the suite README so a moved main is detectable.
T2CQ_REVISION = "main"

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


# Upstream CADPrompt prompts are written for a code-generation task, not a CAD spec: every one
# opens "Write Python code using CADQuery to ...". Fed verbatim to the maker arms that instruction
# asks for a CadQuery program while the harness measures build123d geometry, and it makes the
# first 40 characters of all 100 specs identical (which is what made the old slug contamination
# key degenerate). Strip it and restore a normal sentence opening.
_CP_PREAMBLE = re.compile(r"^\s*write (?:a )?python code using cadquery to\s*", re.IGNORECASE)


def strip_cadprompt_preamble(text: str) -> str:
    out = _CP_PREAMBLE.sub("", text.strip(), count=1).lstrip()
    return out[:1].upper() + out[1:] if out else text.strip()


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
        prompt = strip_cadprompt_preamble(
            (p / "Natural_Language_Descriptions_Prompt_with_specific_measurements.txt").read_text())
        sid = f"cp-{p.name}"
        shutil.copy(p / "Ground_Truth.stl", out_dir / "refs" / f"{sid}.stl")
        specs.append({"id": sid, "name": prompt[:48], "tier": 0, "spec": prompt,
                      "source": f"CADPrompt/{p.name}"})
        # normalized: this suite is in DeepCAD units, not mm, so its Chamfer band must be
        # scored shape-only (both meshes scaled to a common diagonal). The mm-specified
        # internal suites carry no such flag and are scored at true scale.
        acc[sid] = {"reference_stl": f"refs/{sid}.stl", "solids": 1, "normalized": True}
    _write_suite(out_dir, specs, acc,
                 f"# CADPrompt slice\n\n{len(specs)} of 200 items, seed {seed}, dimensioned prompts, "
                 f"DeepCAD units (not mm: card scores are unit-normalised, and every acceptance "
                 f"entry carries `normalized: true`). Source {CADPROMPT_REPO} "
                 "(no licence file; evaluation only, not redistributed).\n\n"
                 "Prompts are the upstream `..._with_specific_measurements.txt` text with the "
                 "leading code-generation instruction (\"Write Python code using CADQuery to \") "
                 "stripped and the next letter capitalised. The rest of the sentence is verbatim: "
                 "the suite asks for a part, not for a CadQuery program.\n")
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
    # --revision pins what gets executed below. The repo publishes no tag, so `main` is the
    # only available pin: it names a branch, not an immutable commit, so a later fetch can pull
    # different code. Re-review the dataset if this is ever re-run against a moved main.
    subprocess.run(["hf", "download", T2CQ_REPO, "data/data_test.jsonl",
                    "--revision", T2CQ_REVISION, "--local-dir", str(dest)], check=True)
    venv = BENCH / "external" / ".venv-cq"
    py = venv / "bin" / "python"
    if not py.exists():
        subprocess.run(["uv", "venv", str(venv), "--python", "3.12"], check=True)
    # An existing venv is not proof of a usable one: a half-finished or hand-cleaned .venv-cq
    # used to skip the install and then fail EVERY reference build, producing an empty suite
    # with no error. Probe the import and install when it is missing.
    if subprocess.run([str(py), "-c", "import cadquery"], capture_output=True).returncode != 0:
        r = subprocess.run(["uv", "pip", "install", "--python", str(py), "cadquery"])
        if r.returncode != 0 or subprocess.run(
                [str(py), "-c", "import cadquery"], capture_output=True).returncode != 0:
            print("text2cadquery: cadquery not importable in .venv-cq; suite skipped (card notes it)")
            return
    rows = [json.loads(l) for l in (dest / "data" / "data_test.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random(seed); rng.shuffle(rows); rows = rows[:limit]
    out = BENCH / "text2cadquery"
    if out.exists(): shutil.rmtree(out)
    (out / "refs").mkdir(parents=True)
    specs, acc = [], {}
    for i, row in enumerate(rows):
        sid = f"t2cq-{i:04d}"
        code = re.sub(r"exporters\.export\([^\n]*\n", "", row["output"])
        stl_path = out / "refs" / f"{sid}.stl"
        export_line = (
            "\nfrom cadquery import exporters\n"
            "for __name in ('part_1', 'result', 'solid', 'shape', 'assembly'):\n"
            "    if __name in dir() and __name not in ('exporters',):\n"
            f"        exporters.export(eval(__name), r'{stl_path}')\n"
            "        break\n"
        )
        code += export_line
        # Executes reference dataset code with the operator's privileges: see the module docstring.
        r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not stl_path.exists():
            continue
        specs.append({"id": sid, "name": row["input"][:48], "tier": 0, "spec": row["input"], "source": T2CQ_REPO})
        # normalized: the dataset's units are not mm, so bands are shape-only (see cadprompt).
        acc[sid] = {"reference_stl": f"refs/{sid}.stl", "solids": 1, "normalized": True}
    _write_suite(out, specs, acc, f"# Text-to-CadQuery test slice\n\n{len(specs)} items with references rebuilt "
                 f"by EXECUTING the reference CadQuery code in the private venv at "
                 f"benchmarks/external/.venv-cq, with the operator's own privileges (evaluation-only, "
                 f"operator-run; HF revision {T2CQ_REVISION}). Every acceptance entry carries "
                 "`normalized: true`: units are not mm, so Chamfer bands are shape-only. "
                 "Licence unconfirmed; evaluation only.\n")
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
