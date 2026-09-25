#!/usr/bin/env python3
"""lab/eval_round.py -- one-command round evaluation: an arm vs stock, on round 1's exact
85-spec public harness plus the 18 owner references, writing a DECISION.md.

PREP-ONLY as of 2026-09-25 (owner instruction): round 2a training was cancelled before it
started, so there is no new arm to evaluate today. This script exists so that whenever a
real round-2 arm IS trained and registered (benchmarks/arms.json), grading it against
stock is one command, using the SAME harness and scorer round 1 used (round1/DECISION.md):

  python3 lab/eval_round.py --arm <NEW_ARM> --baseline gemma-4-31b \\
      --out-dir benchmarks/results/card/<round>

That single call:
  1. Materialises the owner-refs suite (benchmarks/owner-refs/{specs.json,acceptance.json,
     reference/*.stl}) from ~/CAD/references/<name>/{spec.txt,model.step|model.stl} --
     CPU-only, idempotent (a name whose reference STL already exists is left alone), never
     touches ~/CAD/references or the shared lab/state/ bank. Safe to run standalone:
       python3 lab/eval_round.py --materialize-only
  2. Runs scripts/run_card.py, UNCHANGED, for --arm then --baseline, on the public suites
     (cadprompt, text2cadquery, heldout-cqe) capped to round 1's "phase1" subset (n=85
     after lift_report's public-suite filter) plus the owner-refs suite (n=18, uncapped:
     "owner-refs" is not a key in run_card.SUBSETS, so apply_subset leaves it whole).
     Public suites are read via --suite-root pointed at the MAIN checkout (this worktree
     has no cadprompt/text2cadquery specs.json of its own -- see MAIN_CHECKOUT below);
     owner-refs is read from THIS worktree's own benchmarks/owner-refs/ in a second,
     un-suite-rooted call into the same --out rows.jsonl (--resume).
  3. Computes two lift tables via scripts/lift_report.py's own build_report(): the public
     85-spec one exactly as round 1's own `lift_report.py <out> --baseline <arm>` would
     (written to the same LIFT.json/LIFT.md names, byte-for-byte the documented command),
     and a second one scoped to public_suites=("owner-refs",) for the 18 references
     (written to OWNER-REFS-LIFT.json/.md) -- same match/invalid/flip formulas, not a
     hand-rolled second implementation (lift_report.lift_table takes public_suites as a
     parameter for exactly this).
  4. Writes DECISION.md in round 1's format (benchmarks/results/card/round1/DECISION.md):
     match %, invalid %, paired flips for both suites, and a SHIP/NO-SHIP verdict by round
     1's own stated rule (docs/plans and round1/DECISION.md section 4 restated in
     _verdict()): beat stock on invalid ratio first, then match share, with the paired
     flips column in favour, no validity regression -- on BOTH suites, not just the public
     one, since the owner references are the harder, hand-verified half of this check.

GPU: steps 2-3 build real parts through the maker-server (arms.py switches it, like every
other run_card.py card) -- this is the part of the plan the owner said NOT to run today.
--materialize-only and --score-existing (below) are the two ways to exercise everything
else with no GPU at all; see the module's own "CPU smoke test" section for what was
actually run on 2026-09-25.

  --score-existing STEP REF   score one candidate STEP against one reference STL/STEP
                               with geom_bands.score_against_reference and print the band
                               -- proves the scoring half of step 3 without any build.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
BENCH = HERE / "benchmarks"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))

import geom_bands  # noqa: E402
import lift_report  # noqa: E402

DEFAULT_REFS_DIR = Path.home() / "CAD" / "references"
OWNER_REFS_SUITE_DIR = BENCH / "owner-refs"
OWNER_REFS_SUITE_NAME = "owner-refs"
PUBLIC_SUITES = ("cadprompt", "text2cadquery", "heldout-cqe")
# This worktree carries no specs.json for the two big public suites (they are gitignored,
# regenerated once by scripts/fetch_external.py and living only where that was last run) --
# see run_card.py's own --suite-root docstring for why. The main checkout has them.
MAIN_CHECKOUT = Path.home() / ".openclaw" / "skills" / "cad-builder"


# ---------------------------------------------------------------------------
# Step 1: materialise the owner-refs suite (CPU only)
# ---------------------------------------------------------------------------

def materialize_owner_refs_suite(refs_dir: Path = DEFAULT_REFS_DIR,
                                 suite_dir: Path = OWNER_REFS_SUITE_DIR) -> dict:
    """Reads ~/CAD/references/<name>/{spec.txt, model.step|model.stl}, writes
    benchmarks/owner-refs/{specs.json, acceptance.json, reference/<name>.stl} in the same
    shape run_card.load_suite() already reads for every other suite (see e.g.
    benchmarks/heldout-cqe/{specs.json,acceptance.json}). Read-only against refs_dir and
    the shared lab/state/ bank -- this never calls lab/specbank.py's import_references,
    which mutates lab/state/specs.jsonl; this suite is a private, gitignored copy scoped
    to this worktree's own benchmarks/, not a bank row.

    Idempotent: a name whose reference/<name>.stl already exists is left untouched (same
    "materialise once" contract as lab/specbank.py's _materialize_reference_stl)."""
    suite_dir.mkdir(parents=True, exist_ok=True)
    ref_out_dir = suite_dir / "reference"
    ref_out_dir.mkdir(parents=True, exist_ok=True)

    specs: list[dict] = []
    acceptance: dict = {"_meta": {
        "source": f"owner references, {refs_dir}",
        "checks": "band (geom_bands.score_against_reference against the owner's own "
                  "model.step/model.stl, mm-scale, not normalised) via lab/eval_round.py.",
    }}
    materialized, skipped, reused = [], [], []
    for folder in sorted(p for p in refs_dir.iterdir() if p.is_dir()):
        spec_file = folder / "spec.txt"
        step_file = folder / "model.step"
        stl_file = folder / "model.stl"
        if not spec_file.exists() or not (step_file.exists() or stl_file.exists()):
            skipped.append(folder.name)
            continue
        lines = spec_file.read_text(encoding="utf-8").splitlines()
        tier = 3
        spec_lines = lines
        if lines and lines[0].strip().lower().startswith("tier:"):
            try:
                tier = int(lines[0].split(":", 1)[1].strip())
            except Exception:
                pass
            spec_lines = lines[1:]
        spec = "\n".join(spec_lines).strip()
        if not spec:
            skipped.append(folder.name)
            continue

        dest_stl = ref_out_dir / f"{folder.name}.stl"
        if dest_stl.exists():
            reused.append(folder.name)
        else:
            if stl_file.exists():
                shutil.copyfile(stl_file, dest_stl)
            else:
                geom_bands.step_to_stl(step_file, dest_stl)
            materialized.append(folder.name)

        specs.append({"id": folder.name, "name": folder.name, "tier": tier, "spec": spec})
        acceptance[folder.name] = {"solids": 1, "reference_stl": f"reference/{folder.name}.stl",
                                   "normalized": False}

    (suite_dir / "specs.json").write_text(json.dumps({"benchmarks": specs}, indent=2), encoding="utf-8")
    (suite_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    return {"suite_dir": str(suite_dir), "n_specs": len(specs), "materialized": materialized,
            "reused": reused, "skipped": skipped}


# ---------------------------------------------------------------------------
# Step 2: build (GPU) -- thin subprocess wrappers around the UNCHANGED run_card.py, the
# same harness round 1 used.
# ---------------------------------------------------------------------------

def _run_card(arm: str, suites: str, out_dir: Path, *, suite_root: Path | None,
             subset: str, resume: bool, mode: str = "oneshot", timeout: int = 900) -> None:
    cmd = [sys.executable, "-X", "utf8", str(SCRIPTS / "run_card.py"),
          "--arms", arm, "--suites", suites, "--mode", mode, "--subset", subset,
          "--timeout", str(timeout), "--out", str(out_dir)]
    if suite_root:
        cmd += ["--suite-root", str(suite_root)]
    if resume:
        cmd.append("--resume")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=HERE)


def build_arm(arm: str, out_dir: Path, *, resume_public: bool, resume_owner: bool,
              subset: str = "phase1", timeout: int = 900,
              suite_root: Path = MAIN_CHECKOUT) -> None:
    """Two run_card.py calls into the SAME --out dir's rows.jsonl: public suites (from
    suite_root, the main checkout) then owner-refs (from this worktree's own
    benchmarks/owner-refs/, materialize_owner_refs_suite() must have run first)."""
    _run_card(arm, ",".join(PUBLIC_SUITES), out_dir, suite_root=suite_root, subset=subset,
             resume=resume_public, timeout=timeout)
    _run_card(arm, OWNER_REFS_SUITE_NAME, out_dir, suite_root=None, subset=subset,
             resume=True, timeout=timeout)


# ---------------------------------------------------------------------------
# Step 3: score (lift tables) + step 4: DECISION.md
# ---------------------------------------------------------------------------

def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _flip(f) -> str:
    return "-" if not f else f"+{f['improved']}/-{f['worsened']}"


def _verdict(public_row: dict, owner_row: dict) -> tuple[str, list[str]]:
    """Round 1's own ship rule (round1/DECISION.md section 4): promote only when the new
    arm beats stock on invalid ratio first, then match share, with the paired flips column
    in its favour -- applied to BOTH the public 85-spec table and the owner 18-reference
    table, since the references are the harder, hand-verified half of this check and a
    lift on the easy public suites alone proved nothing in round 1."""
    reasons = []
    ok = True
    for name, row in (("public-85", public_row), ("owner-refs-18", owner_row)):
        inv_flips = row["flips"]["invalid_ratio"]
        match_flips = row["flips"]["match_rate"]
        if inv_flips["worsened"] > inv_flips["improved"]:
            ok = False
            reasons.append(f"{name}: invalid ratio regressed (flips {_flip(inv_flips)})")
        if match_flips["improved"] <= match_flips["worsened"]:
            ok = False
            reasons.append(f"{name}: match share did not improve net of flips "
                           f"({_flip(match_flips)})")
    return ("SHIP" if ok else "NO-SHIP"), reasons


def write_decision_md(out_dir: Path, arm: str, baseline: str, public_table: dict,
                      owner_table: dict) -> str:
    pub = public_table[arm]
    own = owner_table[arm]
    verdict, reasons = _verdict(pub, own)
    lines = [
        f"# {arm} vs {baseline} -- round evaluation",
        "",
        f"Arm: `{arm}`. Baseline: `{baseline}`. Harness: `scripts/run_card.py` "
        f"`--subset phase1 --mode oneshot` (round 1's own harness, unchanged); scorer: "
        f"`scripts/geom_bands.py score_against_reference` via `scripts/lift_report.py`'s "
        f"`lift_table` (unchanged). One sample per arm per spec at the engine's fixed "
        f"serving temperature -- the same 'one sample each' round 1 used.",
        "",
        "## Public suites (cadprompt, text2cadquery, heldout-cqe; round 1's own 85 specs)",
        "",
        "| arm | n | invalid | gate clean | acceptance | match (ref n) | flips (invalid) | "
        "flips (match) | median s |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {baseline} (baseline) | {public_table[baseline]['n']} | "
        f"{_pct(public_table[baseline]['invalid_ratio'])} | "
        f"{_pct(public_table[baseline]['gate_clean_rate'])} | "
        f"{_pct(public_table[baseline]['acceptance'])} | "
        f"{_pct(public_table[baseline]['match_rate'])} ({public_table[baseline]['ref_n']}) | - | - | "
        f"{public_table[baseline]['median_wall_s']} |",
        f"| {arm} | {pub['n']} | {_pct(pub['invalid_ratio'])} | {_pct(pub['gate_clean_rate'])} | "
        f"{_pct(pub['acceptance'])} | {_pct(pub['match_rate'])} ({pub['ref_n']}) | "
        f"{_flip(pub['flips']['invalid_ratio'])} | {_flip(pub['flips']['match_rate'])} | "
        f"{pub['median_wall_s']} |",
        "",
        "## Owner references (18 parts, ~/CAD/references, hand-verified, never trained on)",
        "",
        "| arm | n | invalid | gate clean | acceptance | match (ref n) | flips (invalid) | "
        "flips (match) | median s |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {baseline} (baseline) | {owner_table[baseline]['n']} | "
        f"{_pct(owner_table[baseline]['invalid_ratio'])} | "
        f"{_pct(owner_table[baseline]['gate_clean_rate'])} | "
        f"{_pct(owner_table[baseline]['acceptance'])} | "
        f"{_pct(owner_table[baseline]['match_rate'])} ({owner_table[baseline]['ref_n']}) | - | - | "
        f"{owner_table[baseline]['median_wall_s']} |",
        f"| {arm} | {own['n']} | {_pct(own['invalid_ratio'])} | {_pct(own['gate_clean_rate'])} | "
        f"{_pct(own['acceptance'])} | {_pct(own['match_rate'])} ({own['ref_n']}) | "
        f"{_flip(own['flips']['invalid_ratio'])} | {_flip(own['flips']['match_rate'])} | "
        f"{own['median_wall_s']} |",
        "",
        f"## Verdict: {verdict}",
        "",
        ("Round 1's rule: promote only when the new arm beats stock on invalid ratio "
         "first, then match share, with the paired flips column in its favour, on BOTH "
         "tables above -- a lift on the public suites alone is not enough (round 1's own "
         "confound: the public suites are the easy half)." if not reasons else
         "Round 1's rule (invalid ratio first, then match share, paired flips in favour, "
         "on BOTH tables) was not met:"),
    ] + [f"- {r}" for r in reasons]
    text = "\n".join(lines) + "\n"
    (out_dir / "DECISION.md").write_text(text, encoding="utf-8")
    return text


def score_and_decide(out_dir: Path, arm: str, baseline: str) -> None:
    rows_path = out_dir / "rows.jsonl"
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]

    public_report = lift_report.build_report(rows, baseline, public_suites=PUBLIC_SUITES)
    (out_dir / "LIFT.json").write_text(json.dumps(public_report, indent=2) + "\n")
    (out_dir / "LIFT.md").write_text(lift_report.render_lift_md(public_report["table"], baseline))

    owner_report = lift_report.build_report(rows, baseline, public_suites=(OWNER_REFS_SUITE_NAME,))
    (out_dir / "OWNER-REFS-LIFT.json").write_text(json.dumps(owner_report, indent=2) + "\n")
    (out_dir / "OWNER-REFS-LIFT.md").write_text(
        lift_report.render_lift_md(owner_report["table"], baseline))

    md = write_decision_md(out_dir, arm, baseline, public_report["table"], owner_report["table"])
    print(md)


# ---------------------------------------------------------------------------
# CPU-only smoke test path: score one existing candidate against one reference, no build.
# ---------------------------------------------------------------------------

def score_existing(step_path: Path, reference_stl: Path) -> dict:
    result = geom_bands.score_against_reference(step_path, reference_stl)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="")
    ap.add_argument("--baseline", default="gemma-4-31b")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--refs-dir", type=Path, default=DEFAULT_REFS_DIR)
    ap.add_argument("--suite-root", type=Path, default=MAIN_CHECKOUT)
    ap.add_argument("--subset", default="phase1")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--materialize-only", action="store_true",
                    help="CPU only: (re)build benchmarks/owner-refs/ and stop")
    ap.add_argument("--score-only", action="store_true",
                    help="skip the builds, only (re)compute LIFT tables + DECISION.md from "
                         "an existing rows.jsonl in --out-dir")
    ap.add_argument("--score-existing", nargs=2, metavar=("STEP", "REFERENCE_STL"),
                    help="CPU only: score one existing candidate STEP/STL against one "
                         "reference STL/STEP and print the band -- no build, no --out-dir")
    a = ap.parse_args()

    if a.score_existing:
        score_existing(Path(a.score_existing[0]), Path(a.score_existing[1]))
        return

    materialize_report = materialize_owner_refs_suite(a.refs_dir)
    print(json.dumps(materialize_report, indent=2))
    if a.materialize_only:
        return

    if not a.arm or not a.out_dir:
        raise SystemExit("--arm and --out-dir are required unless --materialize-only or "
                         "--score-existing")

    if not a.score_only:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        build_arm(a.arm, a.out_dir, resume_public=False, resume_owner=True,
                 subset=a.subset, timeout=a.timeout, suite_root=a.suite_root)
        build_arm(a.baseline, a.out_dir, resume_public=True, resume_owner=True,
                 subset=a.subset, timeout=a.timeout, suite_root=a.suite_root)

    score_and_decide(a.out_dir, a.arm, a.baseline)


if __name__ == "__main__":
    main()
