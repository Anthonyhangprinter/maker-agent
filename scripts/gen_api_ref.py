#!/usr/bin/env python3
"""scripts/gen_api_ref.py -- regenerate b123d/api_ref.json, the compact API reference the
coder prompt carries (api_ref.py selects the entries relevant to a spec).

Every signature and one-line doc is read by INTROSPECTION from the installed build123d and
the project helper modules (b123d/domain.py, b123d/warehouse.py) -- nothing is typed from
memory. The entry list itself is the set of names the model actually uses: the gold
few-shots in ~/.openclaw/cad-examples.jsonl plus every API named in the stock Gemma
baseline's 112 crash messages (codefirst-scale-2026-09-25/gemma_baseline.jsonl).

The few `facts` lines are also derived here, each asserted against the installed package
at generation time, so a build123d upgrade that changes one fails this script loudly
instead of shipping a stale hint.

tests/test_api_ref.py regenerates in memory and compares with the checked-in JSON, so the
file cannot silently drift from the installed package.

Usage: python3 -X utf8 scripts/gen_api_ref.py [--check]
"""
from __future__ import annotations

import inspect
import json
import re
import sys
import typing
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
OUT = HERE / "b123d" / "api_ref.json"

# (key, dotted target) -- target resolved against build123d unless it starts with b123d.
ENTRIES = [
    # edge selection + edge features (the dominant crash family on fillet/chamfer specs)
    ("fillet", "fillet"), ("chamfer", "chamfer"),
    ("ShapeList.filter_by", "ShapeList.filter_by"), ("ShapeList.group_by", "ShapeList.group_by"),
    ("ShapeList.sort_by", "ShapeList.sort_by"), ("Edge.center", "Edge.center"),
    ("Edge.start_point", "Edge.start_point"), ("Vector", "Vector"),
    ("Shape.bounding_box", "Shape.bounding_box"),
    # placement
    ("Pos", "Pos"), ("Rotation", "Rotation"),
    # primitives the spec may call for beyond Box/Cylinder (already in the system prompt)
    ("Cone", "Cone"), ("Torus", "Torus"),
    # profile-based solids
    ("extrude", "extrude"), ("revolve", "revolve"), ("loft", "loft"),
    ("make_face", "make_face"), ("Polyline", "Polyline"), ("Plane", "Plane"),
    ("mirror", "mirror"), ("Compound", "Compound"),
    # project helpers
    ("bolt_circle", "b123d.domain.bolt_circle"), ("cross_bore", "b123d.domain.cross_bore"),
    ("counterbore_cutter", "b123d.domain.counterbore_cutter"),
    ("countersink_cutter", "b123d.domain.countersink_cutter"),
    ("gusset", "b123d.domain.gusset"), ("ring_groove", "b123d.domain.ring_groove"),
    ("spur_gear", "b123d.domain.spur_gear"), ("hex_bolt", "b123d.domain.hex_bolt"),
    ("structural_section", "b123d.domain.structural_section"),
    ("iso_thread", "b123d.warehouse.iso_thread"), ("screw", "b123d.warehouse.screw"),
    ("nut", "b123d.warehouse.nut"), ("ball_bearing", "b123d.warehouse.ball_bearing"),
]

# Parameters that are generic noise on nearly every build123d object; dropped to save tokens.
_NOISE_PARAMS = {"self", "rotation", "align", "mode", "clean", "name", "label", "color",
                 "material", "joints", "parent", "obj", "tolerance"}


def _resolve(target: str):
    if target.startswith("b123d."):
        mod, _, attr = target.rpartition(".")
        import importlib
        return importlib.import_module(mod), attr, getattr(importlib.import_module(mod), attr)
    import build123d
    obj = build123d
    for part in target.split("."):
        obj = getattr(obj, part)
    return build123d, target, obj


def _short_default(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    r = repr(v)
    m = re.match(r"<(\w+\.\w+)(: .*)?>", r)   # <Mode.ADD: 1> / <CenterOf.GEOMETRY> -> Mode.ADD
    if m:
        return m.group(1)
    if len(r) > 24:
        return "..."
    return r


def _render_sig(sig: inspect.Signature, drop_noise: bool = True) -> str:
    parts = []
    for p in sig.parameters.values():
        if p.name == "self" or (drop_noise and p.name in _NOISE_PARAMS):
            continue
        if p.kind is p.VAR_POSITIONAL:
            parts.append("*" + p.name)
        elif p.kind is p.VAR_KEYWORD:
            parts.append("**" + p.name)
        elif p.default is p.empty:
            parts.append(p.name)
        else:
            parts.append(f"{p.name}={_short_default(p.default)}")
    out = "(" + ", ".join(parts) + ")"
    ra = sig.return_annotation
    if ra is not sig.empty:
        rs = ra if isinstance(ra, str) else getattr(ra, "__name__", str(ra))
        rs = str(rs).split("[")[0].split(".")[-1].strip("'\"")
        if rs in ("Vector", "ShapeList", "GroupBy", "BoundBox", "Part", "Sketch", "Face"):
            out += f" -> {rs}"
    return out


def _signature(key: str, obj) -> str:
    name = key.split(".")[-1]
    prefix = key if "." in key else name
    if isinstance(obj, property):
        return f"{prefix}  (property)"
    if inspect.isclass(obj):
        overloads = typing.get_overloads(obj.__init__)
        if overloads:
            # Constructor overloads ARE the API (Rotation(rotation) vs Rotation(X, Y, Z)):
            # keep every parameter, drop exact duplicates and the empty form.
            sigs = []
            for o in overloads:
                osig = inspect.signature(o)
                if any(n.startswith("gp_") for n in osig.parameters):
                    continue           # raw OCP-handle constructors: never model-facing
                s = _render_sig(osig, drop_noise=False)
                if s != "()" and s not in sigs:
                    sigs.append(s)
            return " | ".join(f"{name}{s}" for s in sigs)
    try:
        return prefix + _render_sig(inspect.signature(obj))
    except (TypeError, ValueError):
        return prefix + "(...)"


def _first_sentence(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    paras = [p for p in doc.strip().split("\n\n") if p.strip()]
    # build123d leads with a category banner ("Part Operation: loft", "Applies to 2 and 3
    # dimensional objects.") -- skip those to reach the sentence that says what it does.
    while len(paras) > 1 and re.match(r"^(\w+ (Operation|Object): |Applies to )", paras[0]):
        paras = paras[1:]
    para = paras[0] if paras else ""
    para = " ".join(ln.strip() for ln in para.splitlines())
    m = re.match(r"(.+?[.:;])(\s|$)", para)
    s = m.group(1) if m else para
    s = s.rstrip(":;")
    return s if len(s) <= 110 else s[:107].rstrip() + "..."


def _facts() -> dict:
    """Short, introspection-VERIFIED facts for entries whose signature alone does not show
    the trap the model fell into. Each assert is the evidence for its sentence."""
    import build123d as b
    from build123d.topology.shape_core import GroupBy
    v = b.Vector(1, 2, 3)
    assert hasattr(v, "X") and not hasattr(v, "x")
    assert inspect.isfunction(b.Edge.center) and inspect.isfunction(b.Edge.start_point)
    assert not hasattr(GroupBy, "sort_by") and not hasattr(GroupBy, "filter_by")
    assert hasattr(GroupBy, "__getitem__")
    sig_f = inspect.signature(b.fillet)
    assert list(sig_f.parameters)[:2] == ["objects", "radius"]
    return {
        "Vector": "components are .X .Y .Z (uppercase; .x/.y/.z do not exist)",
        "Edge.center": "a METHOD: e.center().Z, not e.center.z",
        "Edge.start_point": "a METHOD (also end_point()): e.start_point().X",
        "ShapeList.group_by": "returns GroupBy: index it ([0] / [-1]) to get a ShapeList "
                              "before .sort_by/.filter_by or fillet/chamfer",
        "fillet": "takes the EDGES and radius=; the part is implied by the edges: "
                  "fillet(part.edges().filter_by(...), radius=r)",
    }


def build() -> dict:
    facts = _facts()
    entries = {}
    for key, target in ENTRIES:
        mod, attr, obj = _resolve(target)
        is_helper = target.startswith("b123d.")
        # properties must be fetched statically so we see the descriptor, not a value
        if "." in key and not is_helper:
            cls_name, meth = key.split(".")
            import build123d
            obj = inspect.getattr_static(getattr(build123d, cls_name), meth)
            if isinstance(obj, (staticmethod, classmethod)):
                obj = obj.__func__
        e = {"sig": _signature(key, obj), "doc": _first_sentence(obj)}
        if is_helper:
            e["import"] = f"from {mod.__name__} import {attr}"
        if key in facts:
            e["fact"] = facts[key]
        entries[key] = e
    import build123d
    return {"build123d_version": getattr(build123d, "__version__", "?"), "entries": entries}


def main() -> int:
    data = build()
    text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != text:
            print("api_ref.json is STALE: run scripts/gen_api_ref.py", file=sys.stderr)
            return 1
        print("api_ref.json up to date")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(data['entries'])} entries, build123d "
          f"{data['build123d_version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
