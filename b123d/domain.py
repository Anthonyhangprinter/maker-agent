"""
Domain library — structural sections, spur gears, hex bolts as build123d functions.
All dimensions in mm.  All functions return a build123d Part object.

Usage in model-generated code:
    from b123d.domain import structural_section, spur_gear, hex_bolt
    result = structural_section(200, 100, 10, 6, 1500)
"""

from __future__ import annotations
import math
from typing import Optional


def structural_section(
    d_mm: float,
    bf_mm: float,
    tf_mm: float,
    tw_mm: float,
    length_mm: float = 1000.0,
    mitre: bool = False,
    name: str = "Section",
):
    """
    I/H structural section.
    d_mm    — total depth (flange to flange)
    bf_mm   — flange width
    tf_mm   — flange thickness
    tw_mm   — web thickness
    length_mm — member length (extruded in +Z)
    mitre   — cut 45° isosceles triangles at each end for L-joints
    """
    # Input guards — fail with a clear message the agent loop can act on
    for label, val in (("d_mm", d_mm), ("bf_mm", bf_mm), ("tf_mm", tf_mm),
                       ("tw_mm", tw_mm), ("length_mm", length_mm)):
        if val is None or val <= 0:
            raise ValueError(f"structural_section: {label} must be > 0 (got {val}).")
    if 2 * tf_mm >= d_mm:
        raise ValueError(
            f"structural_section: flanges overlap — 2*tf_mm ({2*tf_mm}) must be < d_mm ({d_mm})."
        )
    if tw_mm >= bf_mm:
        raise ValueError(
            f"structural_section: web thicker than flange — tw_mm ({tw_mm}) must be < bf_mm ({bf_mm})."
        )

    from build123d import (
        BuildPart, BuildSketch, BuildLine, Polyline,
        extrude, make_face, Plane, Mode,
        Box, add,
    )

    h  = d_mm / 2
    fw = bf_mm / 2
    ft = tf_mm
    wt = tw_mm / 2

    # 12-vertex I/H profile — CCW in XY plane
    pts = [
        (-fw, -h),      # 0 bottom-left
        ( fw, -h),      # 1 bottom-right
        ( fw, -h + ft), # 2
        ( wt, -h + ft), # 3
        ( wt,  h - ft), # 4
        ( fw,  h - ft), # 5
        ( fw,  h),      # 6 top-right
        (-fw,  h),      # 7 top-left
        (-fw,  h - ft), # 8
        (-wt,  h - ft), # 9
        (-wt, -h + ft), # 10
        (-fw, -h + ft), # 11
    ]

    with BuildPart() as bp:
        with BuildSketch(Plane.XY):
            with BuildLine():
                Polyline(pts, close=True)
            make_face()
        extrude(amount=length_mm)

        if mitre:
            _apply_mitre_b123d(bp, h, fw, d_mm, bf_mm, length_mm)

    return bp.part


def _apply_mitre_b123d(bp, h, fw, d_mm, bf_mm, length_mm):
    """Cut 45° triangular wedges at each end of a beam already in BuildPart context."""
    from build123d import (
        BuildSketch, BuildLine, Polyline,
        extrude, make_face, Plane, Mode,
    )
    margin = d_mm * 0.02 + 2.0  # safety margin so cut fully clears flanges

    # Near end (at Z=0): triangle in YZ plane — removes wedge with hypotenuse at 45°
    # In YZ plane (Y=depth, Z=length): points at (Y=+h+m, Z=0), (Y=-h-m, Z=0), (Y=-h-m, Z=d_mm)
    with BuildSketch(Plane.YZ):
        with BuildLine():
            Polyline(
                [(h + margin, 0), (-h - margin, 0), (-h - margin, d_mm), (h + margin, 0)]
            )
        make_face()
    extrude(amount=fw + margin, both=True, mode=Mode.SUBTRACT)

    # Far end (at Z=length_mm): mirror cut
    with BuildSketch(Plane.YZ.offset(length_mm)):
        with BuildLine():
            Polyline(
                [(h + margin, 0), (-h - margin, 0), (-h - margin, -d_mm), (h + margin, 0)]
            )
        make_face()
    extrude(amount=fw + margin, both=True, mode=Mode.SUBTRACT)


def spur_gear(
    teeth: int,
    module_mm: float,
    width_mm: float,
    bore_mm: float = 0.0,
    pressure_angle: float = 20.0,
    name: str = "Gear",
):
    """
    Involute spur gear.
    teeth         — number of teeth (min 4)
    module_mm     — tooth size in mm (same as meshing partner)
    width_mm      — face width (extrusion depth)
    bore_mm       — shaft bore diameter (0 = solid)
    pressure_angle — degrees, default 20
    """
    # Input guards
    if int(teeth) < 4:
        raise ValueError(f"spur_gear: teeth must be >= 4 (got {teeth}).")
    if module_mm <= 0 or width_mm <= 0:
        raise ValueError(
            f"spur_gear: module_mm ({module_mm}) and width_mm ({width_mm}) must be > 0."
        )
    if bore_mm < 0:
        raise ValueError(f"spur_gear: bore_mm must be >= 0 (got {bore_mm}).")
    if not (0 < pressure_angle < 45):
        raise ValueError(
            f"spur_gear: pressure_angle must be between 0 and 45 deg (got {pressure_angle})."
        )

    from build123d import (
        BuildPart, BuildSketch, BuildLine, BuildLine as _BL,
        Polyline, Circle, extrude, make_face, Plane, Mode,
    )

    N = max(4, int(teeth))
    alpha = math.radians(pressure_angle)
    m = module_mm  # mm

    r_p = m * N / 2
    r_a = r_p + m
    r_f = max(r_p - 1.25 * m, 0.3 * m)
    r_b = r_p * math.cos(alpha)

    if bore_mm > 0 and bore_mm / 2 >= r_f:
        raise ValueError(
            f"spur_gear: bore ({bore_mm} mm) too large — must be < root diameter "
            f"({2*r_f:.1f} mm). Reduce bore or increase teeth/module."
        )

    if r_b >= r_p:
        raise ValueError(
            f"Base circle ({r_b:.2f} mm) >= pitch circle ({r_p:.2f} mm). "
            "Reduce pressure_angle or increase teeth."
        )

    r_inv_start = max(r_f, r_b)
    tooth_angle = 2 * math.pi / N

    def inv(t):
        return r_b * math.cos(t) + t * r_b * math.sin(t), \
               r_b * math.sin(t) - t * r_b * math.cos(t)

    def t_at(r):
        return math.sqrt(max(0.0, (r / r_b) ** 2 - 1.0))

    t_root = t_at(r_inv_start)
    t_tip  = t_at(r_a)
    t_p    = t_at(r_p)
    x_p, y_p = inv(t_p)
    angle_at_pitch = math.atan2(y_p, x_p)
    half_tooth = math.pi / (2 * N)

    rot_right = -(half_tooth + angle_at_pitch)
    rot_left  =   half_tooth + angle_at_pitch

    def rot2d(x, y, theta):
        c, s = math.cos(theta), math.sin(theta)
        return x * c - y * s, x * s + y * c

    N_INV = 8
    N_ARC = 5

    def involute_strip(t0, t1, rotation, flip_y=False):
        pts = []
        for i in range(N_INV):
            t = t0 + (t1 - t0) * i / (N_INV - 1)
            x, y = inv(t)
            if flip_y:
                y = -y
            pts.append(rot2d(x, y, rotation))
        return pts

    def arc_strip(r, a0, a1):
        pts = []
        for i in range(N_ARC):
            a = a0 + (a1 - a0) * i / (N_ARC - 1)
            pts.append((r * math.cos(a), r * math.sin(a)))
        return pts

    right_local = involute_strip(t_root, t_tip, rot_right)
    left_local  = involute_strip(t_tip, t_root, rot_left, flip_y=True)

    rr_local_angle = math.atan2(right_local[0][1], right_local[0][0])
    lr_local_angle = math.atan2(left_local[-1][1], left_local[-1][0])
    rt_local_angle = math.atan2(right_local[-1][1], right_local[-1][0])
    lt_local_angle = math.atan2(left_local[0][1],   left_local[0][0])

    all_pts = []
    for i in range(N):
        ca = i * tooth_angle
        for x, y in right_local:
            all_pts.append(rot2d(x, y, ca))

        a0_tip = rt_local_angle + ca
        a1_tip = lt_local_angle + ca
        while a1_tip < a0_tip:
            a1_tip += 2 * math.pi
        for pt in arc_strip(r_a, a0_tip, a1_tip)[1:]:
            all_pts.append(pt)

        for x, y in left_local[1:]:
            all_pts.append(rot2d(x, y, ca))

        a0_root = lr_local_angle + ca
        a1_root = rr_local_angle + (i + 1) * tooth_angle
        while a1_root < a0_root:
            a1_root += 2 * math.pi
        for pt in arc_strip(r_f, a0_root, a1_root)[1:-1]:
            all_pts.append(pt)

    # Deduplicate near-consecutive points
    MIN_SEG = 1e-6  # mm
    clean = []
    for pt in all_pts:
        if not clean or math.hypot(pt[0]-clean[-1][0], pt[1]-clean[-1][1]) > MIN_SEG:
            clean.append(pt)
    while clean and math.hypot(clean[-1][0]-clean[0][0], clean[-1][1]-clean[0][1]) < MIN_SEG:
        clean.pop()

    with BuildPart() as bp:
        with BuildSketch(Plane.XY):
            with BuildLine():
                Polyline(clean, close=True)
            make_face()
            if bore_mm > 0:
                Circle(bore_mm / 2, mode=Mode.SUBTRACT)
        extrude(amount=width_mm)

    return bp.part


def hex_bolt(
    size_mm: float,
    length_mm: float = 80.0,
    pitch_mm: Optional[float] = None,
    name: Optional[str] = None,
):
    """
    M-series hex head bolt.
    size_mm   — nominal diameter in mm (e.g. 8 for M8)
    length_mm — shank length in mm
    pitch_mm  — thread pitch (defaults to ISO coarse)
    """
    # Input guards
    if size_mm is None or size_mm <= 0:
        raise ValueError(f"hex_bolt: size_mm must be > 0 (got {size_mm}).")
    if length_mm is None or length_mm <= 0:
        raise ValueError(f"hex_bolt: length_mm must be > 0 (got {length_mm}).")

    from build123d import (
        BuildPart, BuildSketch, BuildLine,
        RegularPolygon, Circle, extrude, make_face, Plane, Mode,
        add,
    )
    from .conventions import coarse_pitch

    if pitch_mm is None:
        pitch_mm = coarse_pitch(size_mm)

    # ISO standard head dimensions
    af_mm   = round(size_mm * 1.5, 1)    # across-flats
    head_h  = round(size_mm * 0.64, 1)   # head height
    r_vtx   = (af_mm / 2) / math.cos(math.radians(30))  # circumscribed radius (vertex to centre)

    with BuildPart() as bp:
        # Hex head — extrudes in +Z
        with BuildSketch(Plane.XY):
            RegularPolygon(r_vtx, 6)
        extrude(amount=head_h)

        # Shank — extrudes in -Z (away from head)
        with BuildSketch(Plane.XY):
            Circle(size_mm / 2)
        extrude(amount=-length_mm, mode=Mode.ADD)

    return bp.part


# ── Feature cutters & braces (compose with algebra mode: result -= Pos(...)*cutter) ──────
# These return a Part. Cutters are oriented so the hole OPENING sits at local z=0 and the
# feature extends DOWN (−Z); place with Pos(x, y, top_z) and subtract. Make `depth` ≥ the
# part thickness so the through-portion fully pierces it.

def countersink_cutter(through_d: float, head_d: float, depth: float,
                       head_angle: float = 90.0):
    """Cutter for a COUNTERSUNK hole (flat-head screw): a through-shaft of `through_d`
    plus a conical recess widening to `head_d` at the top face (local z=0).
    Use: `result -= Pos(x, y, top_z) * countersink_cutter(through_d, head_d, depth)`."""
    for label, val in (("through_d", through_d), ("head_d", head_d), ("depth", depth)):
        if val is None or val <= 0:
            raise ValueError(f"countersink_cutter: {label} must be > 0 (got {val}).")
    if head_d <= through_d:
        raise ValueError(f"countersink_cutter: head_d ({head_d}) must be > through_d ({through_d}).")
    if not (0 < head_angle < 180):
        raise ValueError(f"countersink_cutter: head_angle must be 0–180 deg (got {head_angle}).")
    from build123d import Cylinder, Cone, Pos
    r, R = through_d / 2, head_d / 2
    csk = (R - r) / math.tan(math.radians(head_angle / 2))   # cone height for the included angle
    shaft = Pos(0, 0, -depth / 2) * Cylinder(r, depth)        # spans −depth..0
    cone  = Pos(0, 0, -csk / 2) * Cone(r, R, csk)             # r at bottom (−csk) → R at top (0)
    return shaft + cone


def counterbore_cutter(through_d: float, bore_d: float, bore_depth: float, depth: float):
    """Cutter for a COUNTERBORED hole (socket-head screw): a through-shaft of `through_d`
    plus a flat cylindrical recess of `bore_d` × `bore_depth` at the top face (local z=0).
    Use: `result -= Pos(x, y, top_z) * counterbore_cutter(through_d, bore_d, bore_depth, depth)`."""
    for label, val in (("through_d", through_d), ("bore_d", bore_d),
                       ("bore_depth", bore_depth), ("depth", depth)):
        if val is None or val <= 0:
            raise ValueError(f"counterbore_cutter: {label} must be > 0 (got {val}).")
    if bore_d <= through_d:
        raise ValueError(f"counterbore_cutter: bore_d ({bore_d}) must be > through_d ({through_d}).")
    from build123d import Cylinder, Pos
    shaft = Pos(0, 0, -depth / 2) * Cylinder(through_d / 2, depth)
    bore  = Pos(0, 0, -bore_depth / 2) * Cylinder(bore_d / 2, bore_depth)
    return shaft + bore


def bolt_circle(count: int, bolt_circle_d: float, hole_d: float, depth: float,
                start_angle: float = 0.0):
    """Union of `count` cylindrical hole-cutters equally spaced on a bolt circle of
    diameter `bolt_circle_d`, axis +Z, centred on the origin. Subtract from a part to make a
    bolt-hole pattern (flanges, engine cylinders, gear hubs).
    Use: `result -= bolt_circle(6, 60, 6, 20)`  (depth ≥ part thickness; centred on z=0)."""
    if int(count) < 1:
        raise ValueError(f"bolt_circle: count must be >= 1 (got {count}).")
    for label, val in (("bolt_circle_d", bolt_circle_d), ("hole_d", hole_d), ("depth", depth)):
        if val is None or val <= 0:
            raise ValueError(f"bolt_circle: {label} must be > 0 (got {val}).")
    if hole_d >= bolt_circle_d:
        raise ValueError(f"bolt_circle: hole_d ({hole_d}) must be < bolt_circle_d ({bolt_circle_d}).")
    from build123d import Cylinder, Pos
    holes = None
    bcr = bolt_circle_d / 2
    for i in range(int(count)):
        a = math.radians(start_angle + 360.0 * i / int(count))
        c = Pos(bcr * math.cos(a), bcr * math.sin(a), 0) * Cylinder(hole_d / 2, depth)
        holes = c if holes is None else holes + c
    return holes


def gusset(leg_h: float, leg_v: float, thickness: float):
    """Right-triangle reinforcing brace (gusset). Triangle legs run +X (`leg_h`) and +Z
    (`leg_v`) from the inner corner at the local origin; extruded ±Y by `thickness/2`.
    Add it to brace an L-corner: `result += Pos(corner_x, y, corner_z) * gusset(leg_h, leg_v, t)`
    (rotate with Rotation(...) for other corners)."""
    for label, val in (("leg_h", leg_h), ("leg_v", leg_v), ("thickness", thickness)):
        if val is None or val <= 0:
            raise ValueError(f"gusset: {label} must be > 0 (got {val}).")
    from build123d import (BuildPart, BuildSketch, BuildLine, Polyline,
                           make_face, extrude, Plane)
    with BuildPart() as bp:
        with BuildSketch(Plane.XZ):
            with BuildLine():
                Polyline([(0, 0), (leg_h, 0), (0, leg_v), (0, 0)])
            make_face()
        extrude(amount=thickness / 2, both=True)
    return bp.part


def cross_bore(diameter: float, length: float, axis: str = "x"):
    """Cutting cylinder for a bore through the SIDE of a part — a radial / cross / wrist-pin hole.
    A plain Cylinder() points up +Z and would bore the TOP/BOTTOM; this lays the cutter on its side
    so its axis is HORIZONTAL. SUBTRACT it, positioned at the bore's height:
        result -= Pos(0, 0, 22) * cross_bore(20, 200)        # Ø20 radial bore at z=22, axis +X
        result -= Pos(0, 0, 22) * cross_bore(20, 200, 'y')   # ... axis +Y
    Make `length` >= the part's full width so it pierces both walls. General primitive — used by
    shafts, pulleys, axles, hinges, manifolds, pistons, any hole that crosses a wall."""
    if diameter is None or diameter <= 0:
        raise ValueError(f"cross_bore: diameter must be > 0 (got {diameter}).")
    if length is None or length <= 0:
        raise ValueError(f"cross_bore: length must be > 0 (got {length}).")
    from build123d import Cylinder, Rotation
    cyl = Cylinder(diameter / 2, length)
    ax = (axis or "x").lower()
    if ax == "x":
        return Rotation(0, 90, 0) * cyl     # axis +Z -> +X
    if ax == "y":
        return Rotation(90, 0, 0) * cyl     # axis +Z -> +Y
    raise ValueError(f"cross_bore: axis must be 'x' or 'y' (got {axis!r}).")


def ring_groove(part_radius: float, depth: float, width: float):
    """Cutter for a shallow circumferential ring groove on the OUTER surface of a cylinder — piston
    ring grooves, snap-ring / circlip grooves, O-ring glands. It is a thin annular ring (a tube)
    that, when subtracted, carves a groove of radial `depth` and axial `width` into the surface — it
    does NOT bore out the core (the #1 mistake: subtracting a full Cylinder hollows the part). Axis
    +Z, concentric with the body. SUBTRACT one at each groove height:
        for z in (40, 47, 54):
            result -= Pos(0, 0, z) * ring_groove(40, 3, 3)   # r=40 body: 3mm-deep x 3mm-wide grooves
    General primitive — any circumferential groove on a round part."""
    if part_radius is None or part_radius <= 0:
        raise ValueError(f"ring_groove: part_radius must be > 0 (got {part_radius}).")
    if depth is None or not (0 < depth < part_radius):
        raise ValueError(f"ring_groove: depth must be in (0, part_radius) (got {depth}).")
    if width is None or width <= 0:
        raise ValueError(f"ring_groove: width must be > 0 (got {width}).")
    from build123d import Cylinder
    return Cylinder(part_radius + 1.0, width) - Cylinder(part_radius - depth, width)
