#!/usr/bin/env python3
"""harvest_census.py — retroactive M6' fine-tune data miner + census.

Mines the existing ~/.openclaw/cad-builds/ artifact dirs into cad-sftpairs.jsonl "good"
pairs (spec -> verified code, + render where present) and reports a census of every data
source feeding the budget-gated M6' QLoRA. The census NUMBER is the point: it decides
whether the local harvest is thin (lean on adapted public data like GenCAD-Code) or
plentiful (the agent's own builds carry the mix).

Spec recovery: pre-2026-07-19 build dirs never stored the full spec — the dir name holds a
40-char slug. But cad-agent.log logs every "[vX] build: <spec>" with a timestamp, and dirs
are named YYYYMMDD-HHMMSS-<slug>, so specs are matched back by timestamp (±120s) + slug
prefix. Unmatched dirs keep the slug and are flagged truncated (excluded from mining).

Eval protection: any build whose spec/slug matches a benchmark suite spec (text-to-cad,
organic, heldout-cqe) is EXCLUDED — benchmark builds land in the same dir as organic ones
with no persisted marker, and suite specs must never leak into training data (the same
contamination rule as Stage C's CAD_BENCH guard; heldout-cqe/acceptance.json _meta says
NEVER promote these).

Usage:
  python3 scripts/harvest_census.py            # census only (read-only, default)
  python3 scripts/harvest_census.py --mine     # also append new pairs to cad-sftpairs.jsonl
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import sys
from datetime import datetime

HERE      = Path(__file__).resolve().parent.parent
_OPENCLAW = Path.home() / ".openclaw"
BUILDS    = _OPENCLAW / "cad-builds"
LOG       = _OPENCLAW / "cad-agent.log"
PAIRS     = _OPENCLAW / "cad-sftpairs.jsonl"
PAIRS_DIR = _OPENCLAW / "cad-sftpairs"
CORPUS    = _OPENCLAW / "cad-examples.jsonl"
LESSONS   = _OPENCLAW / "cad-lessons.jsonl"

_BUILD_LINE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[INFO\] \[v[\d.]+\] build: (.+?)(?:  \[reference: .*)?$")


def _slug(text: str, n: int = 40) -> str:
    """Mirror of the engine's dir-name slugging (lowercase, non-alnum -> '-')."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:n]


def load_log_specs() -> list[tuple[datetime, str, bool]]:
    """(start_ts, spec, converged) per logged build. Convergence is read from the log lines
    between one 'build:' line and the next: 'Not converged'/'Build failed' mark a best-effort
    or failed build — meta-less dirs mined as 'good' pairs must be log-confirmed converged."""
    out = []
    if LOG.exists():
        cur = None
        for line in LOG.read_text(errors="replace").splitlines():
            m = _BUILD_LINE.match(line)
            if m:
                if cur:
                    out.append(tuple(cur))
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                cur = [ts, m.group(2), True] if m.group(2) != "<image-only>" else None
            elif cur and ("Not converged" in line or "Build failed" in line
                          or "stopping NOT converged" in line):
                cur[2] = False
        if cur:
            out.append(tuple(cur))
    return out


def recover_spec(dirname: str, log_specs: list) -> tuple[str, bool, bool]:
    """(spec, full?, converged) for a build dir named YYYYMMDD-HHMMSS-<slug>."""
    m = re.match(r"^(\d{8})-(\d{6})-(.*)$", dirname)
    if not m:
        return dirname, False, False
    ts = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    slug = m.group(3)
    best = None
    for lts, spec, conv in log_specs:
        dt = abs((ts - lts).total_seconds())
        # The log line is written at build START, the dir is created at persist time —
        # a build can run many minutes, so accept the nearest earlier-or-near start whose
        # slug agrees.
        if lts <= ts and dt < 3600 and _slug(spec).startswith(slug[:30]):
            if best is None or dt < best[0]:
                best = (dt, spec, conv)
    if best:
        return best[1], True, best[2]
    return slug.replace("-", " ").strip(), False, False


_DIMS3_RE = re.compile(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*mm")
_LONG_RE  = re.compile(r"(\d+(?:\.\d+)?)\s*mm\s+(?:long|wide|tall|high)")
_HOLES_RE = re.compile(r"(an?|one|two|three|four|five|six|\d+)\s+(\d+(?:\.\d+)?)\s*mm\s+"
                       r"(?:\w+\s+){0,2}?(?:through[- ]?)?(?:holes?|bores?)", re.I)
_HOLLOW_RE = re.compile(r"open-?top|hollow|(?:\d+\s*mm\s+walls?)|enclosure|\bbox\b", re.I)
_COUNTS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def verify_pair(code: str, spec: str) -> tuple[bool, str]:
    """Deterministic sanity of a mined pair: execute the code and check the measured solid
    (inspect's FACTS_JSON) against numbers written IN the spec. Needed because 'converged'
    in old logs is not 'correct' — a 2026-07-19 audit of 7 log-converged organic builds
    found 4 measurably wrong (solid brick as an open-top box; '80mm long' brackets at
    100/168mm; five Ø10 holes where one was asked). Same spirit as the engine's [spec]
    advisories, applied offline."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "s.py", Path(td) / "o.step"
        src.write_text(code)
        p = subprocess.run([sys.executable, str(HERE / "scripts" / "step"), str(src), str(out)],
                           capture_output=True, text=True, timeout=120)
        if p.returncode or not out.exists():
            return False, "code failed to run"
        q = subprocess.run([sys.executable, str(HERE / "scripts" / "inspect"), str(out)],
                           capture_output=True, text=True, timeout=60)
    m = re.search(r"^FACTS_JSON:\s*(\{.*\})", q.stdout, re.M)
    if not m:
        return False, "inspect produced no FACTS_JSON"
    facts = json.loads(m.group(1))
    bbox = sorted(facts.get("bbox") or [])
    vol = facts.get("volume")
    groups = facts.get("hole_groups") or []

    # Only enforce NxNxN when the spec states exactly one — sub-part dims of a compound part
    # ("base flange ... and ... upright") are not overall extents (2026-07-19 audit lesson).
    d3 = _DIMS3_RE.search(spec) if len(_DIMS3_RE.findall(spec)) == 1 else None
    if d3 and bbox:
        want = sorted(float(v) for v in d3.groups())
        if any(abs(w - g) > max(1.0, 0.02 * w) for w, g in zip(want, bbox)):
            return False, f"bbox {bbox} != spec {want}"
    for m in _LONG_RE.finditer(spec):
        want = float(m.group(1))
        if bbox and all(abs(want - g) > max(1.0, 0.05 * want) for g in bbox):
            return False, f"no bbox axis matches '{m.group(0)}' (bbox {bbox})"
    cyl = facts.get("cyl_faces") or 0
    for m in _HOLES_RE.finditer(spec):
        n_want = _COUNTS.get(m.group(1).lower()) or int(m.group(1))
        d_want = float(m.group(2))
        n_got = sum(g["n"] for g in groups if abs(g["d"] - d_want) <= max(0.5, 0.05 * d_want))
        if n_got > n_want:
            return False, f"spec wants {n_want}×Ø{d_want} holes, measured {n_got}"
        # Undercount: hole_groups misses non-axial bores (radial flange holes measure as 0,
        # 2026-07-19), so trust raw cylindrical faces as the fallback before failing.
        if n_got < n_want and cyl < n_want:
            return False, f"spec wants {n_want}×Ø{d_want} holes, measured {n_got} ({cyl} cyl faces)"
    if d3 and vol is not None and _HOLLOW_RE.search(spec):
        full = 1.0
        for v in d3.groups():
            full *= float(v)
        if vol > 0.9 * full:
            return False, f"spec implies hollow but volume {vol:.0f} ≈ solid {full:.0f}"
    return True, "ok"


def suite_slugs() -> set[str]:
    slugs = set()
    for suite in ("text-to-cad", "organic", "heldout-cqe", "hard-eval"):
        f = HERE / "benchmarks" / suite / "specs.json"
        if f.exists():
            data = json.loads(f.read_text())
            if isinstance(data, dict):   # text-to-cad wraps the list in {"benchmarks": [...]}
                data = data.get("benchmarks", [])
            for b in data:
                slugs.add(_slug(b["spec"], 40))
    return slugs


def existing_build_ids() -> set[str]:
    ids = set()
    if PAIRS.exists():
        for line in PAIRS.read_text().splitlines():
            try:
                ids.add(json.loads(line).get("build_id", ""))
            except Exception:
                pass
    return ids


def _count_jsonl(path: Path) -> int:
    return sum(1 for ln in path.read_text().splitlines() if ln.strip()) if path.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", action="store_true",
                    help="append newly mined good pairs to cad-sftpairs.jsonl (default: census only)")
    a = ap.parse_args()

    log_specs = load_log_specs()
    bench = suite_slugs()
    already = existing_build_ids()

    dirs = sorted(p for p in BUILDS.iterdir() if p.is_dir()) if BUILDS.exists() else []
    census = {"dirs": len(dirs), "minable": 0, "mined_new": 0, "bench_excluded": 0,
              "truncated_spec": 0, "not_converged": 0, "verify_failed": 0, "no_source": 0,
              "with_render": 0, "with_image_ref": 0, "already_mined": 0}
    mined_rows = []
    for d in dirs:
        src = d / "build_source.py"
        if not (src.exists() and (d / "build.step").exists()):
            census["no_source"] += 1
            continue
        meta = {}
        if (d / "build_meta.json").exists():
            try:
                meta = json.loads((d / "build_meta.json").read_text())
            except Exception:
                pass
        if meta.get("spec"):
            spec, full, conv = meta["spec"], True, bool(meta.get("converged"))
        else:
            spec, full, conv = recover_spec(d.name, log_specs)
        if _slug(spec, 40) in bench or _slug(d.name.split("-", 2)[-1], 40) in {s[:40] for s in bench}:
            census["bench_excluded"] += 1
            continue
        if not full:
            census["truncated_spec"] += 1
            continue
        if not conv:
            census["not_converged"] += 1
            continue   # a best-effort/failed build's final code is not a "good" pair
        ok, why = verify_pair(src.read_text(), spec)
        if not ok:
            census["verify_failed"] += 1
            print(f"  ✗ {d.name[:50]}: {why}", file=sys.stderr)
            continue
        census["minable"] += 1
        if (d / "build.png").exists():
            census["with_render"] += 1
        if (d / "reference.jpg").exists():
            census["with_image_ref"] += 1
        if d.name in already:
            census["already_mined"] += 1
            continue
        image = ""
        if a.mine and (d / "build.png").exists():
            PAIRS_DIR.mkdir(parents=True, exist_ok=True)
            dst = PAIRS_DIR / f"{d.name}-final.png"
            shutil.copy(d / "build.png", dst)
            image = str(dst)
        mined_rows.append({
            "kind": "good", "source": "retro", "spec": spec,
            "code": src.read_text(), "image": image, "build_id": d.name,
            "code_model": meta.get("code_model", ""),
            "accepted_via": meta.get("accepted_via", ""),
            "timestamp": meta.get("built_at", "")})

    if a.mine and mined_rows:
        with PAIRS.open("a") as f:
            for r in mined_rows:
                f.write(json.dumps(r) + "\n")
        census["mined_new"] = len(mined_rows)

    pair_rows = _count_jsonl(PAIRS)
    print(json.dumps(census, indent=1))
    print(f"\n─ M6' data census {datetime.now():%Y-%m-%d} "
          f"({'mined' if a.mine else 'census only — rerun with --mine to write'})")
    print(f"  cad-sftpairs.jsonl rows:   {pair_rows}"
          + (f"  (+{census['mined_new']} new)" if census["mined_new"] else ""))
    print(f"  corpus (cad-examples):     {_count_jsonl(CORPUS)}")
    print(f"  lessons (cad-lessons):     {_count_jsonl(LESSONS)}")
    print(f"  build dirs minable:        {census['minable']}/{census['dirs']} "
          f"(bench-excluded {census['bench_excluded']}, truncated-spec {census['truncated_spec']}, "
          f"no-source {census['no_source']})")
    if not a.mine and mined_rows:
        print(f"  would mine now:            {len(mined_rows)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
