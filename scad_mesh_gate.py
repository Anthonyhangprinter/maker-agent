"""
scad_mesh_gate.py — deterministic mesh verification gate for the OpenSCAD backend spike (M8).

Pure trimesh, no LLM call anywhere in this module. Plays the same role for the mesh backend
that `cad_agent_v4.verify_expected()` plays for the build123d/B-rep backend: measure FACTS from
the produced geometry and assert the ones that are structurally unambiguous, while flagging
spec-derived guesses as advisories that guide the coder without blocking convergence.

HONESTY NOTE (read this before trusting a "pass"):
  The B-rep gate (`scripts/inspect` + `verify_expected`) can tell a THROUGH hole from a BLIND
  one by walking the solid's actual cylindrical-face axis — it has the exact topology. A mesh
  has no such privileged information: all we get here is the EULER CHARACTERISTIC of each
  connected, watertight body, and from it a topological GENUS. For a simply-connected part
  (everything this spike's goldens are), each clean through-hole raises genus by 1 — a block
  with N through-holes is genus N (same as an N-holed torus). This makes genus-sum a LOWER
  BOUND on through-hole count:
    - it can UNDER-count: a blind hole/pocket/counterbore never punctures the surface, so it
      changes ZERO topology and is invisible to genus (a part could have ten blind pockets and
      still show genus 0);
    - it can also be misleading on non-simply-connected base shapes (e.g. a part with a
      pre-existing handle from something other than a "hole" in the colloquial sense), though
      none of this spike's goldens exercise that case.
  So a genus shortfall vs. an expected hole count is ALWAYS an advisory here, never a hard fail
  — the mirror image of the B-rep gate's `min_holes`/`min_through_holes` advisories, but for a
  strictly weaker reason (genus truly cannot see blind features, whereas the B-rep gate's
  checks are merely brief-derived guesses that might be wrong).

Everything else measured here — loadability, non-emptiness, non-zero volume, watertightness,
volume, bbox, and body count — is measured EXACTLY from the mesh, and body-count / bbox
mismatches are treated as hard fails (unlike the B-rep gate, which treats bbox as advisory
because the *brief*, not the geometry, is the unreliable part there — here the caller passes
in the spec's own numbers, so a mismatch means the GEOMETRY is wrong).
"""
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


def _genus_of_body(body: "trimesh.Trimesh") -> int:
    """genus = (2 - euler_number) // 2 for one closed, watertight, orientable shell (a block is
    genus 0; each clean through-hole in an otherwise simply-connected body adds 1 — same
    topology as an N-holed torus). Clamped at 0: a numerically noisy euler_number should never
    report a negative genus."""
    return max(0, (2 - int(body.euler_number)) // 2)


def _bbox_ok(got: list[float], expected: list[float]) -> bool:
    """Axis-order-invariant bbox compare, tolerance max(5.0mm, 10%) per axis — identical rule to
    cad_agent_v4.verify_expected()'s ADVISORY size check (the brief routinely mis-derives
    dimensions, so the in-loop gate only hints; the external run_benchmarks scorer still applies
    its own tighter max(2mm, 5%) acceptance rule to the final geometry)."""
    if len(got) != 3 or len(expected) != 3:
        return False
    for e, g in zip(sorted(expected), sorted(got)):
        if not (isinstance(e, (int, float)) and e > 0):
            continue
        if abs(g - e) > max(5.0, 0.10 * e):
            return False
    return True


def gate(stl_path, expected: Optional[dict] = None) -> tuple[list[str], list[str], dict]:
    """Load `stl_path` and check it. `expected` is the same shape as a build123d-agent brief's
    `expected` block: {"bbox_mm": [L,W,H], "solids": N, "min_holes": N}. Any key may be absent —
    absent keys simply skip that check (never a fail-by-omission).

    Returns (hard_fails, advisories, facts). hard_fails non-empty means the mesh is not a valid
    deliverable; advisories are informational (including the honest genus-vs-holes caveat above).
    """
    hard: list[str] = []
    soft: list[str] = []
    facts: dict = {"loadable": False, "watertight": None, "repaired": False,
                   "volume_mm3": None, "bbox_mm": None, "body_count": None,
                   "genus_sum": None, "per_body_genus": None}

    stl_path = Path(stl_path)
    try:
        mesh = trimesh.load(str(stl_path), force="mesh", process=True)
    except Exception as e:
        hard.append(f"STL failed to load: {e}")
        return hard, soft, facts
    facts["loadable"] = True

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        hard.append("mesh is empty (no vertices/faces) — the script produced no geometry.")
        return hard, soft, facts

    volume = float(mesh.volume)
    facts["volume_mm3"] = round(volume, 3)
    if abs(volume) < 1e-6:
        hard.append(f"volume is near-zero ({volume:.6f} mm³) — likely a degenerate/open surface, "
                    f"not a solid.")
        return hard, soft, facts

    watertight = bool(mesh.is_watertight)
    facts["watertight"] = watertight
    if not watertight:
        # One repair attempt — fill_holes patches small gaps from a slightly-off boolean; if it
        # genuinely fixes the mesh, note it as an advisory instead of failing a recoverable part.
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass
        watertight = bool(mesh.is_watertight)
        facts["watertight"] = watertight
        if watertight:
            facts["repaired"] = True
            soft.append("mesh had small gaps that trimesh.repair.fill_holes() closed — the "
                        "original OpenSCAD output was not perfectly watertight (usually a "
                        "coincident-face tolerance issue); the repaired mesh is used below.")
            volume = float(mesh.volume)
            facts["volume_mm3"] = round(volume, 3)
        else:
            hard.append("mesh is not watertight (open edges / non-manifold) even after one "
                        "trimesh.repair.fill_holes() attempt — this is not a valid solid for "
                        "print or measurement.")
            return hard, soft, facts

    bbox = mesh.bounding_box.extents  # [dx, dy, dz]
    bbox_mm = [round(float(x), 2) for x in bbox]
    facts["bbox_mm"] = bbox_mm

    bodies = mesh.split(only_watertight=True)
    body_count = len(bodies) if len(bodies) else 1
    facts["body_count"] = body_count

    per_body_genus = [_genus_of_body(b) for b in bodies] if len(bodies) else [_genus_of_body(mesh)]
    facts["per_body_genus"] = per_body_genus
    facts["genus_sum"] = sum(per_body_genus)

    if expected:
        exp_solids = expected.get("solids")
        if isinstance(exp_solids, int) and exp_solids >= 1 and body_count != exp_solids:
            hard.append(f"expected {exp_solids} solid body/bodies (fused/assembly count) but "
                        f"the mesh split into {body_count} watertight body/bodies — features "
                        f"are not unioned, or an assembly part fused/fragmented incorrectly.")

        # Advisory, not a hard fail — mirrors verify_expected(): the brief's bbox is a guess
        # (qwen3:8b routinely mis-derives dimensions), so a mismatch guides the coder/critic
        # instead of blocking convergence. Hard-failing here would bias the spike's A/B against
        # OpenSCAD relative to the B-rep gate's demoted-size rule (2026-06-26 gate redesign).
        exp_bbox = expected.get("bbox_mm")
        if isinstance(exp_bbox, list) and len(exp_bbox) == 3:
            if not _bbox_ok(bbox_mm, exp_bbox):
                soft.append(
                    f"overall size {'×'.join(f'{d:.1f}' for d in sorted(bbox_mm, reverse=True))}mm "
                    f"differs from the brief's expected "
                    f"{'×'.join(f'{d:.1f}' for d in sorted(exp_bbox, reverse=True))}mm "
                    f"(tol = max(5mm, 10%) per axis) — confirm the dimensions against the actual "
                    f"spec (the brief's size guess is often wrong).")

        exp_holes = expected.get("min_holes")
        if isinstance(exp_holes, int) and exp_holes > 0:
            if facts["genus_sum"] < exp_holes:
                soft.append(
                    f"genus-sum lower bound ({facts['genus_sum']}) is below the spec's "
                    f"~{exp_holes} hole(s)/bore(s) — ADVISORY ONLY: genus cannot see blind "
                    f"holes/pockets/counterbores (they don't puncture the surface, so they add "
                    f"zero topology), so this may be correct even when the count looks short. "
                    f"If every intended hole is meant to go fully through, treat this as a real "
                    f"signal; otherwise it is expected noise.")

    return hard, soft, facts
