#!/usr/bin/env python3
"""scripts/measure_part.py -- deterministic geometry measurement for the code-first teacher
pipeline (lab/teacher_codefirst.py).

Adapted from the exploration script that first proved this out
(/tmp/.../scratchpad/measure2.py, 2026-09-23 session): pure build123d/OCP, no subprocess.
Given a single-solid STEP file, reports the bounding box, precise (non-tessellated) volume,
and every cylindrical feature (hole/shaft) with its diameter, axis, centre point, depth, and
through/blind classification, plus TORUS faces (fillets) and CONE faces (chamfers/tapers) --
the "drawing callout" facts a human would read off the part to write a spec from measurements
alone, the way lab/teacher_codefirst.py's SPEC call is meant to.

Library:
    from measure_part import measure, write_reference_volume_sidecar
CLI (for manual inspection):
    python3 scripts/measure_part.py <part.step>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def measure(step_path: Path) -> dict:
    """Measure a single-solid STEP file. Never assumes the caller already checked
    n_solids == 1 -- that is reported (`n_solids`) so a caller can reject a multi-body
    result itself; this function still measures whatever is there."""
    from build123d import import_step
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    step_path = Path(step_path)
    shape = import_step(str(step_path))
    solids = shape.solids()
    faces = shape.faces()
    bbox = shape.bounding_box()
    size = bbox.size
    bmin = np.array([bbox.min.X, bbox.min.Y, bbox.min.Z])
    bmax = np.array([bbox.max.X, bbox.max.Y, bbox.max.Z])
    volume_mm3 = float(sum(float(s.volume) for s in solids))

    all_verts = shape.vertices()
    vpts = np.array([[v.X, v.Y, v.Z] for v in all_verts], dtype=float) if all_verts \
        else np.zeros((0, 3))

    dims = np.array([size.X, size.Y, size.Z])
    thin_axis = int(np.argmin(dims))
    in_plane_axes = [i for i in range(3) if i != thin_axis]

    # Cylindrical faces are classified by TWO independent, physically obvious signals:
    #   arc span   -- a FULL circle (~360 deg) is a hole/bore (concave) or shaft/boss (convex);
    #                 a PARTIAL arc (a straight prismatic edge rounded off, e.g.
    #                 fillet(edges().filter_by(Axis.Z))) is an edge fillet, not a hole at all.
    #   convexity  -- already computed below (face normal vs the outward radial direction).
    # This replaces an earlier "own_coverage" heuristic that tried to infer the same thing
    # from face AREA vs a hypothetical full-circle area and, in doing so, silently DROPPED
    # every partial-arc fillet face outright (found live, 2026-09-24: cf13/cf14/cf44 lost
    # their outer-corner fillets -- design_code had fillet() calls, measurements.json showed
    # zero fillets, so the spec never mentioned them and a geometrically-faithful blind
    # rebuild correctly left them out; the tiny volume delta let it through as band "match").
    # A fillet on a CURVED edge (not a straight prismatic one) instead leaves a TORUS face --
    # handled separately below, unchanged.
    FULL_ARC_TOL_DEG = 3.0     # within this of 360 deg counts as a full circle
    MIN_FACE_AREA_MM2 = 0.02  # drop near-zero-area tessellation/seam slivers, either kind

    raw = []          # full-arc cylindrical faces -> holes/shafts (grouped by axis+position)
    fillet_raw = []   # partial-arc cylindrical faces + TORUS faces -> fillets (grouped by radius)
    chamfers = []
    for f in faces:
        if f.area < MIN_FACE_AREA_MM2:
            continue
        gt = str(f.geom_type)
        if gt == "GeomType.TORUS":
            surf = BRepAdaptor_Surface(f.wrapped)
            try:
                torus = surf.Torus()
                fillet_raw.append({"radius": float(torus.MinorRadius()), "convex": True,
                                   "area": float(f.area), "kind": "corner blend (torus)"})
            except Exception:
                pass
            continue
        if gt == "GeomType.CONE":
            # A conical face on a mechanical part is almost always a chamfer (edge bevel) or
            # a countersink -- both read the same way in a drawing callout, so both are
            # reported together as "chamfers", the caller/spec-writer can name it either way.
            surf = BRepAdaptor_Surface(f.wrapped)
            try:
                cone = surf.Cone()
                semi_deg = round(float(np.degrees(cone.SemiAngle())), 2)
                chamfers.append({"semi_angle_deg": abs(semi_deg),
                                  "ref_radius_mm": round(float(cone.RefRadius()), 3),
                                  "area_mm2": round(float(f.area), 3)})
            except Exception:
                pass
            continue
        if gt != "GeomType.CYLINDER":
            continue
        surf = BRepAdaptor_Surface(f.wrapped)
        cyl = surf.Cylinder()
        ax = cyl.Axis()
        loc = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
        d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        d = d / np.linalg.norm(d)
        r = float(cyl.Radius())
        c = f.center()
        cpt = np.array([c.X, c.Y, c.Z])
        t_c = float(np.dot(cpt - loc, d))
        axis_pt_at_c = loc + t_c * d
        radial_outward = cpt - axis_pt_at_c
        rn = np.linalg.norm(radial_outward)
        if rn > 1e-9:
            radial_outward = radial_outward / rn
        n = f.normal_at(c)
        nvec = np.array([n.X, n.Y, n.Z])
        convex = bool(np.dot(nvec, radial_outward) > 0)

        u0, u1 = surf.FirstUParameter(), surf.LastUParameter()
        span_deg = float(np.degrees(abs(u1 - u0)))
        is_full_circle = span_deg >= (360.0 - FULL_ARC_TOL_DEG)

        if not is_full_circle:
            fillet_raw.append({
                "radius": r, "convex": convex, "area": float(f.area),
                "kind": "edge fillet" if convex else "internal fillet/round",
            })
            continue

        fv = f.vertices()
        fp = np.array([[v.X, v.Y, v.Z] for v in fv], dtype=float) if fv else cpt.reshape(1, 3)

        d_key = d if d[np.argmax(np.abs(d))] >= 0 else -d
        perp = loc - np.dot(loc, d_key) * d_key

        raw.append({"radius": r, "diameter": round(r * 2, 3), "convex": convex,
                    "d": d_key, "perp": perp, "fp": fp, "area": float(f.area)})

    # ── fillets: group by (kind bucket, radius) -- position doesn't matter, count does ────
    fillets = []
    fillet_groups: list[dict] = []
    for rf in fillet_raw:
        placed = False
        for g in fillet_groups:
            if g["kind"] == rf["kind"] and abs(g["radius"] - rf["radius"]) < 0.1:
                g["radius"] = (g["radius"] * g["count"] + rf["radius"]) / (g["count"] + 1)
                g["count"] += 1
                g["area"] += rf["area"]
                placed = True
                break
        if not placed:
            fillet_groups.append({"radius": rf["radius"], "count": 1, "area": rf["area"],
                                  "kind": rf["kind"]})
    for g in fillet_groups:
        fillets.append({"radius_mm": round(g["radius"], 3), "count": g["count"],
                        "kind": g["kind"], "area_mm2": round(g["area"], 3)})
    fillets.sort(key=lambda g: -g["radius_mm"])

    # ── holes/shafts: only full-arc faces reach here now, grouped by axis + position ───────
    groups = []
    for rf in raw:
        placed = False
        for g in groups:
            same_d = np.linalg.norm(g["d"] - rf["d"]) < 0.02
            same_perp = np.linalg.norm(g["perp"] - rf["perp"]) < 0.05
            same_dia = abs(g["diameter"] - rf["diameter"]) < 0.05
            if same_d and same_perp and same_dia:
                g["fp"] = np.vstack([g["fp"], rf["fp"]])
                g["area"] += rf["area"]
                g["members"] += 1
                placed = True
                break
        if not placed:
            groups.append({**rf, "members": 1})

    holes, shafts = [], []
    for g in groups:
        t_face = np.dot(g["fp"] - g["perp"], g["d"])
        t_lo, t_hi = float(t_face.min()), float(t_face.max())
        if vpts.shape[0]:
            t_solid = np.dot(vpts - g["perp"], g["d"])
            t_solid_lo, t_solid_hi = float(t_solid.min()), float(t_solid.max())
        else:
            t_solid_lo, t_solid_hi = t_lo, t_hi
        margin_lo = t_lo - t_solid_lo
        margin_hi = t_solid_hi - t_hi
        span = t_hi - t_lo

        entry_pt = g["perp"] + t_lo * g["d"]
        exit_pt = g["perp"] + t_hi * g["d"]
        axis_line_pt = g["perp"]

        axis_is_thin = abs(g["d"][thin_axis]) > 0.9
        pos_in_plane = None
        if axis_is_thin:
            pos_in_plane = {
                f"axis{in_plane_axes[0]}_from_min_mm": round(
                    float(axis_line_pt[in_plane_axes[0]] - bmin[in_plane_axes[0]]), 3),
                f"axis{in_plane_axes[1]}_from_min_mm": round(
                    float(axis_line_pt[in_plane_axes[1]] - bmin[in_plane_axes[1]]), 3),
            }

        entry = {
            "diameter_mm": round(g["diameter"], 3),
            "depth_mm": round(span, 3),
            "margin_lo_mm": round(margin_lo, 3),
            "margin_hi_mm": round(margin_hi, 3),
            "axis_dir": [round(float(x), 4) for x in g["d"]],
            "axis_point_mm": [round(float(x), 3) for x in axis_line_pt],
            "entry_point_mm": [round(float(x), 3) for x in entry_pt],
            "exit_point_mm": [round(float(x), 3) for x in exit_pt],
            "pos_in_plane_from_bbox_min_mm": pos_in_plane,
        }
        if g["convex"]:
            entry["kind"] = "shaft/boss (OD)"
            shafts.append(entry)
        else:
            through = margin_lo < 0.5 and margin_hi < 0.5
            entry["kind"] = "through hole" if through else "blind hole"
            entry["through"] = through
            holes.append(entry)

    holes.sort(key=lambda h: -h["diameter_mm"])
    shafts.sort(key=lambda h: -h["diameter_mm"])

    return {
        "step_file": str(step_path),
        "bbox_min_mm": [round(float(x), 3) for x in bmin],
        "bbox_max_mm": [round(float(x), 3) for x in bmax],
        "size_mm": [round(float(x), 3) for x in dims],
        "thin_axis": "XYZ"[thin_axis],
        "in_plane_axes": ["XYZ"[i] for i in in_plane_axes],
        "volume_mm3": round(volume_mm3, 3),
        "n_solids": len(solids),
        "n_faces": len(faces),
        "holes": holes,
        "shafts": shafts,
        "fillets": fillets,
        "chamfers": chamfers,
    }


def write_reference_volume_sidecar(step_path: Path, ref_stl: Path, sidecar_json: Path,
                                    measurements: dict | None = None) -> dict:
    """Export `step_path` to `ref_stl` (the mesh geom_bands.score_against_reference compares
    a blind rebuild against) and write a small JSON sidecar next to it recording the
    STEP-derived volume/bbox -- the EXACT build123d number, not trimesh's tessellation
    approximation of the mesh it sits beside. Purely an audit record; score_against_reference
    itself recomputes its own reference_volume from `ref_stl` directly and never reads this
    file, so nothing downstream breaks if it goes missing."""
    from geom_bands import step_to_stl

    step_path, ref_stl, sidecar_json = Path(step_path), Path(ref_stl), Path(sidecar_json)
    step_to_stl(step_path, ref_stl)
    m = measurements if measurements is not None else measure(step_path)
    sidecar = {
        "reference_step": str(step_path),
        "reference_stl": str(ref_stl),
        "volume_mm3": m["volume_mm3"],
        "bbox_mm": m["size_mm"],
        "n_solids": m["n_solids"],
        "source": "step-measured (build123d Solid.volume, not the STL tessellation)",
    }
    sidecar_json.parent.mkdir(parents=True, exist_ok=True)
    sidecar_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def _flat_radii(measurements: dict, list_key: str, radius_key: str) -> list[float]:
    """Expand a feature list into one radius/diameter value per physical instance -- a
    grouped entry (fillets carry `count`) becomes `count` copies of its radius, so two
    measurements that group the SAME features slightly differently (e.g. one run's 4
    identical fillets in one group vs another's 2+2 split by floating-point radius drift)
    still compare as equal flat multisets rather than by how many groups each has."""
    out: list[float] = []
    for e in (measurements.get(list_key) or []):
        v = e.get(radius_key)
        if v is None:
            continue
        n = int(e.get("count", 1) or 1)
        out.extend([float(v)] * n)
    return sorted(out)


def compare_features(design: dict, rebuild: dict, radius_tol_mm: float = 0.2) -> tuple[bool, list[str]]:
    """True (+ []) only when `rebuild`'s measured feature counts match `design`'s exactly --
    same number of holes, shafts, fillets and chamfers, each within `radius_tol_mm` (holes/
    shafts compared as diameters, so 2x that tolerance). Otherwise False + one message per
    mismatched feature type.

    Purely a function of two measure()-shaped dicts; never builds or gates anything itself.
    Exists because band=="match" (a volume/chamfer-distance score) can pass a rebuild that
    silently dropped a small feature -- a handful of corner fillets barely move total volume.
    Real incident, 2026-09-24: cf13/cf14/cf44 were kept as "match" with their outer-corner
    fillets missing entirely, because the SPEC never mentioned them (see the fillet-detection
    fix above and the SPEC-prompt fix in lab/teacher_codefirst.py) -- this is the second,
    independent check that would have caught it even if a spec omission slipped through."""
    problems: list[str] = []

    def _check(list_key: str, radius_key: str, tol: float, label: str) -> None:
        d = _flat_radii(design, list_key, radius_key)
        r = _flat_radii(rebuild, list_key, radius_key)
        if len(d) != len(r):
            problems.append(f"{label} count differs: design={len(d)} rebuild={len(r)}")
            return
        for dv, rv in zip(d, r):
            if abs(dv - rv) > tol:
                problems.append(f"{label} size mismatch: design={dv:g} rebuild={rv:g} "
                               f"(tol {tol:g}mm)")

    _check("holes", "diameter_mm", 2 * radius_tol_mm, "holes")
    _check("shafts", "diameter_mm", 2 * radius_tol_mm, "shafts")
    _check("fillets", "radius_mm", radius_tol_mm, "fillets")
    _check("chamfers", "ref_radius_mm", radius_tol_mm, "chamfers")
    return (not problems), problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step_file")
    a = ap.parse_args()
    m = measure(Path(a.step_file))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
