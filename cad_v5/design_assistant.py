"""Design assistant — CADAM/Adam-CAD style "Parameters" panel (2026-09-21).

Two independent capabilities live here:

1. PRE-BUILD proposal (`propose_parameters` / `compose_spec`): turns a non-expert's vague
   request into named, editable parameters (label, value, unit, min/max/step) BEFORE any
   code is generated, the way CADAM's Customizer panel lets a user see and correct the
   machine's assumptions before committing GPU time to them. One LLM call, through the
   engine's existing strong-rung seam (`cad_engine._ollama`) — same call shape as
   `expand_spec`/`triage_ambiguity`, same graceful-degrade-to-nothing contract.

2. POST-BUILD live sliders for build123d (`extract_build123d_params` /
   `substitute_build123d_params`): the build123d equivalent of the OpenSCAD Customizer
   sliders in `scripts/openscad_gen.py`. `_CODE_SYSTEM` in cad_engine.py already tells the
   coder to write every key dimension as a named constant at the top of the script
   ("wall = 2.0") specifically so it is editable without an AI — this is that promise kept:
   pure `ast` parsing, zero LLM, substitution by exact source position (never a regex over
   the whole file, which could also rewrite the same number inside a comment or a
   different constant).

Both halves are deliberately dependency-light (stdlib `ast`/`re`/`json` + `cad_engine`
only) so this module can be imported from a small script under the system interpreter
without pulling in build123d/OCP — see scripts/assist_gen.py and scripts/fluid_gen.py.
"""
from __future__ import annotations

import ast
import logging
import os
import re
from typing import Optional

import cad_engine as engine

log = logging.getLogger("cad_v5.design_assistant")

MAX_PARAMETERS = 14

# ── Mode selection ──────────────────────────────────────────────────────────────

_MODES = ("off", "auto", "always")
_DEFAULT_MODE = "auto"


def design_assistant_mode(requested: Optional[str] = None) -> str:
    """Resolve the effective mode: "off" | "auto" | "always".

    CAD_BENCH=1 (benchmarks and the Phase 3 harvest) ALWAYS forces "off" — the assistant
    must never touch a scored or harvested run, no exceptions, checked before anything
    else. Otherwise an explicit `requested` value wins (the web UI's own select), then
    cad.json's `design_assistant` key, then the "auto" default.
    """
    if os.environ.get("CAD_BENCH"):
        return "off"
    if requested in _MODES:
        return requested
    try:
        cfg = engine._load_config().get("cad", {})
        val = cfg.get("design_assistant", _DEFAULT_MODE)
        return val if val in _MODES else _DEFAULT_MODE
    except Exception:
        return _DEFAULT_MODE


# ── Pre-build proposal ──────────────────────────────────────────────────────────

_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "title":   {"type": "string"},
        "summary": {"type": "string"},
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key":    {"type": "string"},
                    "label":  {"type": "string"},
                    "value":  {"type": "number"},
                    "unit":   {"type": "string", "enum": ["mm", "deg", "", "count"]},
                    "min":    {"type": "number"},
                    "max":    {"type": "number"},
                    "step":   {"type": "number"},
                    "source": {"type": "string", "enum": ["user", "assumed"]},
                    "why":    {"type": "string"},
                },
                "required": ["key", "label", "value", "unit", "min", "max", "step",
                             "source", "why"],
            },
        },
        "features": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "parameters", "features"],
}

_PROPOSAL_SYSTEM = """\
You turn a non-expert's CAD request into a short list of NAMED, EDITABLE parameters — a
"Parameters" panel a beginner can read and correct before anything is built, like a
spreadsheet of sliders: "Cylinder Count 9", "Wall Thickness 2 mm".

RULES (violating any of these makes the panel useless or misleading):
- Every number the user actually TYPED must appear as its own parameter, with the EXACT
  value they gave and source="user". Never round it, never "improve" it.
- Every other parameter you add to make the part buildable (a count, a proportion, a
  clearance) is source="assumed", with a one-clause "why" a beginner can read
  ("assumed — keeps the wall thicker than a 3D-printer nozzle").
- Assumed values must be PROPORTION-CONSISTENT with the user's own numbers and with each
  other (a wall thickness assumed for an 8mm-wide part must not be 5mm).
- Counts are integers, min is at least 1.
- min < value < max for every parameter, with headroom to actually explore (roughly a
  quarter to triple the value, tighter for something like a bolt count).
- unit is exactly one of "mm", "deg", "" (dimensionless ratios/flags), or "count"
  (integer quantities).
- At most 14 parameters. Group the essentials only — do not enumerate every possible
  feature of the part.
- NEVER add a feature, part, material, tolerance or manufacturing note the user did not
  ask for and the part does not need to function as asked. Naming an assumed feature the
  object doesn't need (e.g. deciding a plain bracket should be hollow) is the exact
  failure this panel exists to prevent — state only what the geometry needs.
- "features" lists the short, functional feature phrases the build will need (e.g. "4
  corner mounting holes"), not marketing language.

Reply with ONLY JSON:
{"title": "...", "summary": "one plain sentence describing the part",
 "parameters": [{"key": "snake_case_name", "label": "Human Label", "value": 0,
                 "unit": "mm"|"deg"|""|"count", "min": 0, "max": 0, "step": 0,
                 "source": "user"|"assumed", "why": "short reason"}],
 "features": ["short feature phrase", ...]}"""

_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(mm|millimeters?|millimetres?|deg|degrees?|°)?",
                     re.IGNORECASE)


def _user_numbers(spec: str) -> list[float]:
    """Every literal number the user typed, in the order they appear. Used to verify the
    model's proposal kept each one verbatim (see RULES above) — the known failure mode
    (an old v1 "brief" silently asserted a plate should be hollow) was a model quietly
    editing or dropping what the user actually said, so this is checked in code, not
    trusted from the prompt alone."""
    out = []
    for m in _NUM_RE.finditer(spec or ""):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            continue
    return out


def _user_unit_hint(spec: str, index: int) -> Optional[str]:
    """The unit written right after the `index`-th user number (mm/deg), or None when the
    text gives no hint — used only when a number must be inserted as a brand-new parameter
    (the model dropped it entirely), so the inserted slider gets a sensible unit instead of
    always guessing "count" for a whole number."""
    matches = list(_NUM_RE.finditer(spec or ""))
    if index >= len(matches):
        return None
    suffix = (matches[index].group(2) or "").lower()
    if suffix.startswith("mm") or suffix.startswith("millimet"):
        return "mm"
    if suffix.startswith("deg") or suffix == "°":
        return "deg"
    return None


def _label_from_key(key: str) -> str:
    return " ".join(w.capitalize() for w in re.split(r"[_\s]+", key.strip()) if w) or "Value"


def _repair_bounds(p: dict) -> dict:
    """min < value < max, always. Widens whichever side is broken rather than discarding
    the parameter — a slightly-wrong range is still useful, a missing slider is not."""
    v = p["value"]
    lo, hi = p.get("min"), p.get("max")
    if not isinstance(lo, (int, float)) or lo >= v:
        lo = v - abs(v) * 0.75 if v else -1.0
    if not isinstance(hi, (int, float)) or hi <= v:
        hi = v + abs(v) * 2.0 if v else 1.0
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    p["min"], p["max"] = (lo, hi) if lo < hi else (hi, lo)
    step = p.get("step")
    if not isinstance(step, (int, float)) or step <= 0:
        step = 1 if p.get("unit") == "count" else round(max((p["max"] - p["min"]) / 100, 0.01), 4)
    p["step"] = step
    return p


def _sanitize_proposal(spec: str, data: dict) -> dict:
    """Enforce every RULES clause from `_PROPOSAL_SYSTEM` in code — a prompt is a request,
    not a guarantee. Degrades to the empty "no proposal" shape on anything unrecoverable;
    never raises (mirrors expand_spec's/triage_ambiguity's contract)."""
    empty = {"title": "", "summary": "", "parameters": [], "features": []}
    if not isinstance(data, dict):
        return empty
    raw_params = data.get("parameters")
    if not isinstance(raw_params, list):
        return empty

    cleaned: list[dict] = []
    seen_keys: set[str] = set()
    for item in raw_params:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        key = re.sub(r"[^a-z0-9_]", "_", key.lower()).strip("_")
        if not key or key in seen_keys:
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        unit = item.get("unit")
        if unit not in ("mm", "deg", "", "count"):
            unit = "count" if float(value).is_integer() and unit not in ("mm", "deg") else "mm"
        source = item.get("source") if item.get("source") in ("user", "assumed") else "assumed"
        is_count = unit == "count"
        p = {
            "key": key,
            "label": str(item.get("label") or _label_from_key(key)).strip() or _label_from_key(key),
            "value": int(round(value)) if is_count else value,
            "unit": unit,
            "min": item.get("min"), "max": item.get("max"), "step": item.get("step"),
            "source": source,
            "why": str(item.get("why") or "").strip()[:160],
        }
        if is_count:
            p["value"] = max(1, int(round(value)))
            p["min"] = max(1, p["min"]) if isinstance(p.get("min"), (int, float)) else 1
        _repair_bounds(p)
        if is_count:
            p["min"] = max(1, int(round(p["min"])))
            p["max"] = max(p["min"] + 1, int(round(p["max"])))
            p["step"] = max(1, int(round(p["step"])))
        cleaned.append(p)
        seen_keys.add(key)

    # Every number the user actually typed must survive, verbatim, marked source=user.
    # Pass 1: an exact value match is the correct, common case — just flip its source.
    user_nums = _user_numbers(spec)
    matched = set()             # indices into user_nums already accounted for
    matched_param_ids = set()   # id() of params that matched a real user number
    for p in cleaned:
        target = round(p["value"], 6)
        for i, num in enumerate(user_nums):
            if i not in matched and round(num, 6) == target:
                p["source"] = "user"
                matched.add(i)
                matched_param_ids.add(id(p))
                break
    # Pass 2: a leftover user number means the model dropped it OR silently changed a
    # value it had itself flagged source=user (the failure this rule exists to catch —
    # "if the model changed or dropped one, overwrite/insert it and mark it user").
    # Repair a mismatched user-flagged parameter IN PLACE first, so a single altered
    # number is corrected rather than shipping alongside a duplicate "Value N" entry;
    # only fall back to inserting a brand-new parameter once no such candidate is left.
    # "Suspicious" = claims source=user but did NOT match any real user number above.
    suspicious = [p for p in cleaned if p["source"] == "user" and id(p) not in matched_param_ids]
    for i, num in enumerate(user_nums):
        if i in matched:
            continue
        if suspicious:
            p = suspicious.pop(0)
            is_count = p.get("unit") == "count"
            p["value"] = int(round(num)) if is_count else num
            p["source"] = "user"
            _repair_bounds(p)
            if is_count:
                p["min"] = max(1, int(round(p["min"])))
                p["max"] = max(p["min"] + 1, int(round(p["max"])))
                p["step"] = max(1, int(round(p["step"])))
            continue
        hint = _user_unit_hint(spec, i)
        unit = hint or ("count" if float(num).is_integer() and abs(num) < 1000 else "mm")
        is_count = unit == "count"
        p = {
            "key": f"user_value_{len(cleaned) + 1}",
            "label": f"Value {len(cleaned) + 1}",
            "value": int(num) if is_count else num,
            "unit": unit,
            "min": None, "max": None, "step": None,
            "source": "user", "why": "the number you typed",
        }
        _repair_bounds(p)
        cleaned.append(p)

    if len(cleaned) > MAX_PARAMETERS:
        # Keep every user-stated value (non-negotiable) and trim assumed ones first.
        user_ps = [p for p in cleaned if p["source"] == "user"]
        assumed_ps = [p for p in cleaned if p["source"] != "user"]
        cleaned = (user_ps + assumed_ps)[:MAX_PARAMETERS]

    features = [str(f).strip() for f in (data.get("features") or [])
                if isinstance(f, (str, int, float)) and str(f).strip()][:12]

    return {
        "title": str(data.get("title") or "").strip()[:80],
        "summary": str(data.get("summary") or "").strip()[:240],
        "parameters": cleaned,
        "features": features,
    }


def propose_parameters(spec: str) -> dict:
    """One LLM call, strong rung, schema-constrained, thinking off — same shape as
    `expand_spec`. Returns {"title", "summary", "parameters": [...], "features": [...]};
    on any failure (timeout, malformed JSON, empty spec) degrades to the "no proposal"
    shape ({"title": "", "summary": "", "parameters": [], "features": []}), never raises —
    a broken proposal must fall back to building the spec as typed, not block the build."""
    empty = {"title": "", "summary": "", "parameters": [], "features": []}
    spec = (spec or "").strip()
    if not spec:
        return empty
    try:
        raw = engine._ollama(engine.CODE_MODEL_STRONG, _PROPOSAL_SYSTEM, f"Request: {spec}",
                              timeout=engine.LLM_TIMEOUT, temperature=0.2,
                              fmt=_PROPOSAL_SCHEMA, no_think=True)
        data = engine._extract_json(raw) or {}
    except Exception as e:
        log.warning("[design-assistant] proposal call failed (%s) — no proposal.", e)
        return empty
    try:
        return _sanitize_proposal(spec, data)
    except Exception as e:
        log.warning("[design-assistant] proposal sanitize failed (%s) — no proposal.", e)
        return empty


def compose_spec(original_spec: str, parameters: list[dict]) -> str:
    """The user's own words, verbatim and FIRST, then a "Parameters:" block so the gate's
    [spec] checks can enforce the accepted numbers exactly the way expand_spec's assumptions
    become enforceable. Assumed-but-unedited values are marked "(assumed)" so a human
    re-reading the composed spec later can tell what the user actually asked for."""
    original = (original_spec or "").strip()
    lines = []
    for p in (parameters or []):
        if not isinstance(p, dict):
            continue
        label = str(p.get("label") or p.get("key") or "").strip()
        if not label:
            continue
        value = p.get("value")
        unit = str(p.get("unit") or "").strip()
        val_text = str(value)
        if isinstance(value, float) and value.is_integer():
            val_text = str(int(value))
        tag = "" if p.get("source") == "user" else " (assumed)"
        lines.append(f"{label}: {val_text}{(' ' + unit) if unit else ''}{tag}")
    if not lines:
        return original
    return (original + "\n\nParameters:\n" if original else "Parameters:\n") + "\n".join(lines)


# ── Post-build live sliders for build123d ────────────────────────────────────────
# The build123d counterpart of scripts/openscad_gen.py's Customizer-parameter regex —
# _CODE_SYSTEM in cad_engine.py tells the coder to define every key dimension as a plain
# numeric assignment at the top of the script for exactly this reason.

_NUMERIC_ARITH_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def _num_const(node: ast.AST):
    """The literal number `node` holds if it is a plain (optionally +/- signed) numeric
    constant, else None. Distinguishes int from float via the literal's own Python type,
    so `rib_count = 6` stays an integer slider and `wall = 2.0` stays a float one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _num_const(node.operand)
        if inner is not None:
            return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _is_pure_arith(node: ast.AST, known: set) -> bool:
    """True for expressions built only from numeric constants, +/-/*// and names already
    collected as earlier parameters in this same block — the shape a coordinate-calculation
    line takes in the generated scripts (`flange_z = -60.5 + (flange_t / 2)`). A Call, a
    comparison, or a name that isn't a known parameter fails this — that boundary IS where
    the parameter block ends and the geometry statements begin."""
    if _num_const(node) is not None:
        return True
    if isinstance(node, ast.Name):
        return node.id in known
    if isinstance(node, ast.BinOp) and isinstance(node.op, _NUMERIC_ARITH_OPS):
        return _is_pure_arith(node.left, known) and _is_pure_arith(node.right, known)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_pure_arith(node.operand, known)
    return False


def _eval_arith(node: ast.AST, env: dict):
    v = _num_const(node)
    if v is not None:
        return v
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.UnaryOp):
        inner = _eval_arith(node.operand, env)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if isinstance(node, ast.BinOp):
        left, right = _eval_arith(node.left, env), _eval_arith(node.right, env)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right if right else None
    return None


def _label(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split("_") if w) or name


def _unit_for(name: str, is_int: bool) -> str:
    if name.endswith("_deg"):
        return "deg"
    if name.endswith("_mm"):
        return "mm"
    if name.endswith("_count") or is_int:
        return "count"
    return "mm"    # _CODE_SYSTEM's contract: every dimension here is millimetres by default


MAX_BUILD123D_PARAMS = 40


def extract_build123d_params(source: str) -> list[dict]:
    """Top-of-script named constants, per _CODE_SYSTEM's "every key dimension is a named
    constant at the top" contract. Zero LLM — pure `ast`.

    Scans top-level statements in order, tolerating the header noise real generated files
    carry (the `sys.path.insert` shim, imports, a docstring) before the first constant.
    Once inside the parameter block: a plain numeric assignment is a substitutable slider;
    an assignment that is pure arithmetic over numbers and already-known parameter names
    (a coordinate calculation) is kept read-only with its source expression for display;
    anything else — a Call, an unknown name, a non-Assign statement — ends the block, since
    that is where geometry construction starts.
    """
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return []

    params: list[dict] = []
    known: dict = {}
    started = False

    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            if started:
                break
            continue
        if isinstance(stmt, ast.Expr):
            # A bare call (sys.path.insert(...)) or a string/docstring line — only
            # tolerated as header noise before the parameter block has started.
            if started:
                break
            continue
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            break
        name = stmt.targets[0].id
        value = stmt.value
        lit = _num_const(value)
        if lit is not None:
            is_int = isinstance(lit, int)
            known[name] = lit
            lo, hi = lit * 0.25, lit * 3.0
            lo, hi = (lo, hi) if lo <= hi else (hi, lo)
            if lo == hi:
                lo, hi = lo - 1, hi + 1
            params.append({
                "name": name, "label": _label(name), "value": lit,
                "unit": _unit_for(name, is_int), "is_int": is_int,
                "substitutable": True, "expr": None,
                "min": int(lo) if is_int else round(lo, 4),
                "max": int(hi) if is_int else round(hi, 4),
                "step": 1 if is_int else round(max((hi - lo) / 100, 0.01), 4),
                "lineno": value.lineno, "col_offset": value.col_offset,
                "end_lineno": getattr(value, "end_lineno", value.lineno),
                "end_col_offset": getattr(value, "end_col_offset", None),
            })
            started = True
            continue
        if _is_pure_arith(value, set(known)):
            val = _eval_arith(value, known)
            if val is not None:
                known[name] = val
            params.append({
                "name": name, "label": _label(name), "value": val,
                "unit": _unit_for(name, False), "is_int": False,
                "substitutable": False, "expr": ast.unparse(value),
                "min": None, "max": None, "step": None,
                "lineno": None, "col_offset": None,
                "end_lineno": None, "end_col_offset": None,
            })
            started = True
            continue
        break
    return params[:MAX_BUILD123D_PARAMS]


def _format_literal(value: float, is_int: bool) -> str:
    if is_int:
        return str(int(round(value)))
    text = f"{value:g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def substitute_build123d_params(source: str, values: dict) -> str:
    """Replace exactly the value node's source span for each substitutable parameter named
    in `values` — AST position, never a regex over the whole file (a value that happens to
    also appear in a comment, or as a different constant, must not be touched). Unknown
    names, non-substitutable ("expression") parameters, and unparsable values are silently
    skipped rather than raising — a partial rescale is still useful; a raised exception is
    not, since the caller's job is to build whatever comes back."""
    params = {p["name"]: p for p in extract_build123d_params(source)}
    edits = []
    for name, raw in (values or {}).items():
        p = params.get(str(name))
        if not p or not p.get("substitutable") or p.get("lineno") is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        text = _format_literal(v, p["is_int"])
        edits.append((p["lineno"], p["col_offset"], p["end_lineno"], p["end_col_offset"], text))
    if not edits:
        return source
    lines = source.splitlines(keepends=True)
    by_line: dict = {}
    for e in edits:
        if e[0] != e[2]:
            continue   # a value node spanning multiple lines is not a shape we generate —
                        # skip rather than risk corrupting the file with a partial edit
        by_line.setdefault(e[0], []).append(e)
    for lineno, edits_on_line in by_line.items():
        idx = lineno - 1
        if idx < 0 or idx >= len(lines):
            continue
        line = lines[idx]
        for (_, col, _end_lineno, end_col, text) in sorted(edits_on_line, key=lambda e: e[1],
                                                            reverse=True):
            line = line[:col] + text + line[end_col:]
        lines[idx] = line
    return "".join(lines)
