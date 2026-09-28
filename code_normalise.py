"""code_normalise.py -- deterministic SPELLING and IMPORT normalisation of model-written
build123d code, called from cad_engine._patch_code on every codegen / revise.

WHY (2026-09-26): stock Gemma-4-31B's baseline over 673 verified teacher pairs
(benchmarks/results/card/codefirst-scale-2026-09-25/gemma_baseline.jsonl) crashed 112
times, and roughly half of those crashes were not design mistakes at all but spelling:
  * `e.center().z` -- build123d Vectors spell their components X/Y/Z (uppercase);
  * `e.center.z` / `e.start_point.x` -- a zero-arg METHOD read as an attribute;
  * `bolt_circle(...)` / `cross_bore(...)` / `cos(...)` -- a project helper or math function
    used without its import line;
  * `spur_gear(module=..., width=...)` -- the helper's keyword is `module_mm` / `width_mm`.

Owner principle (memory feedback_cad_agent_design): the LLM makes every design decision;
nothing here may change WHAT is built. Every rewrite below is one the Python interpreter
would otherwise reject with an AttributeError / NameError / TypeError, and the rewrite is
the only reading the model could have meant. Anything ambiguous is left alone, so the
normal crash -> repair path still sees it.

All edits are surgical text edits at AST node positions (never ast.unparse), so comments,
formatting and line numbers in tracebacks survive -- the patched code is also what a
training pair records, and it should read like the model wrote it.
"""
from __future__ import annotations

import ast
import builtins
import functools
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ── Vector-typed expressions (build123d 0.10, verified by introspection 2026-09-26) ──────
# Zero-arg METHODS that return a Vector. Read without "()" they are bound methods, so
# `.center.z` raises "'function' object has no attribute 'z'".
METHODS_AS_ATTR = frozenset({"center", "start_point", "end_point"})
# Methods (any args) returning a Vector.
VEC_METHODS = METHODS_AS_ATTR | {"position_at", "tangent_at", "normal_at"}
# Properties returning a Vector (Location.position/.orientation, Shape.position,
# Edge.arc_center).
VEC_PROPS = frozenset({"position", "orientation", "arc_center"})
# BoundBox properties returning a Vector; only trusted on a `.bounding_box()` result.
BBOX_PROPS = frozenset({"min", "max", "size"})
LOWER_COMPONENTS = {"x": "X", "y": "Y", "z": "Z"}
COMPONENTS = frozenset(LOWER_COMPONENTS) | frozenset(LOWER_COMPONENTS.values())

# Bare math names that are safe to import when referenced but never bound. Deliberately
# excludes one-letter / ambiguous constants (e, inf, nan, tau): importing math.e for an
# unbound `e` would turn a loud NameError into silently wrong geometry.
MATH_NAMES = frozenset({
    "cos", "sin", "tan", "acos", "asin", "atan", "atan2", "radians", "degrees", "sqrt",
    "pi", "hypot", "floor", "ceil", "exp", "log", "log10", "fabs",
})

_HELPER_MODULES = {
    "b123d.domain": _HERE / "b123d" / "domain.py",
    "b123d.warehouse": _HERE / "b123d" / "warehouse.py",
}


@functools.lru_cache(maxsize=1)
def helper_exports() -> dict:
    """{name: (module, [param names])} for every public top-level function in the project
    helper modules. Read with ast (no import) so the engine process never pays for, or gets
    its locale reset by, a build123d import."""
    out: dict = {}
    for mod, path in _HELPER_MODULES.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                a = node.args
                params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
                out.setdefault(node.name, (mod, params))
    return out


@functools.lru_cache(maxsize=1)
def build123d_star_names() -> frozenset:
    """build123d's __all__ (what `from build123d import *` binds), read without importing
    build123d. Empty set if it cannot be found -- callers only use it to AVOID rewrites."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("build123d")
        init = Path(spec.origin)
        tree = ast.parse(init.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                return frozenset(ast.literal_eval(node.value))
    except Exception:
        pass
    return frozenset()


# ── binding analysis (scope-insensitive on purpose: conservative) ─────────────────────────

def _target_names(t, out: set) -> None:
    if isinstance(t, ast.Name):
        out.add(t.id)
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            _target_names(e, out)
    elif isinstance(t, ast.Starred):
        _target_names(t.value, out)


def _bindings(tree) -> tuple[dict, set, list]:
    """(name -> list of bound-value nodes or None for a non-Assign binding,
        imported names, star-import module names)"""
    binds: dict = {}
    imported: set = set()
    stars: list = []

    def bind(name, value=None):
        binds.setdefault(name, []).append(value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    bind(t.id, node.value)
                else:
                    s: set = set()
                    _target_names(t, s)
                    for n in s:
                        bind(n)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            s = set()
            _target_names(node.target, s)
            for n in s:
                bind(n)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            s = set()
            _target_names(node.target, s)
            for n in s:
                bind(n)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(node.name)
        elif isinstance(node, ast.arguments):
            for a in node.posonlyargs + node.args + node.kwonlyargs:
                bind(a.arg)
            if node.vararg:
                bind(node.vararg.arg)
            if node.kwarg:
                bind(node.kwarg.arg)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    s = set()
                    _target_names(item.optional_vars, s)
                    for n in s:
                        bind(n)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bind(node.name)
        elif isinstance(node, ast.NamedExpr):
            bind(node.target.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                bind(n)
        elif isinstance(node, ast.Import):
            for a in node.names:
                nm = (a.asname or a.name).split(".")[0]
                imported.add(nm)
                bind(nm)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    stars.append(node.module or "")
                else:
                    imported.add(a.asname or a.name)
                    bind(a.asname or a.name)
    return binds, imported, stars


# ── text edits at AST positions (col offsets are UTF-8 byte offsets) ─────────────────────

def _apply_edits(code: str, edits: list) -> str:
    """edits: (lineno, byte_col_start, byte_col_end, replacement). Applied right-to-left."""
    if not edits:
        return code
    lines = code.splitlines(keepends=True)
    blines = [ln.encode("utf-8") for ln in lines]
    for lineno, c0, c1, rep in sorted(set(edits), key=lambda e: (e[0], e[1]), reverse=True):
        b = blines[lineno - 1]
        blines[lineno - 1] = b[:c0] + rep.encode("utf-8") + b[c1:]
    return b"".join(blines).decode("utf-8")


class _VecTyper:
    def __init__(self, binds: dict, imported: set):
        self.binds = binds
        # `Vector(...)` is only trusted when the name is build123d's (star/explicit import),
        # never a user-defined Vector class/variable.
        self.vector_is_b3d = "Vector" not in binds or "Vector" in imported
        self._memo: dict = {}

    def is_vec(self, node, _depth: int = 0) -> bool:
        if _depth > 8:
            return False
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in VEC_METHODS:
                return True
            if isinstance(f, ast.Name) and f.id == "Vector" and self.vector_is_b3d:
                return True
            return False
        if isinstance(node, ast.Attribute):
            if node.attr in METHODS_AS_ATTR or node.attr in VEC_PROPS:
                return True
            if node.attr in BBOX_PROPS and self.is_bbox(node.value):
                return True
            return False
        if isinstance(node, ast.Name):
            return self._name_all(node.id, self.is_vec, _depth)
        return False

    def is_bbox(self, node, _depth: int = 0) -> bool:
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Attribute) and node.func.attr == "bounding_box"
        if isinstance(node, ast.Name):
            return self._name_all(node.id, self.is_bbox, _depth)
        return False

    def _name_all(self, name: str, pred, depth: int) -> bool:
        key = (name, pred.__name__)
        if key in self._memo:
            return self._memo[key]
        self._memo[key] = False            # cycle guard
        vals = self.binds.get(name)
        ok = bool(vals) and all(v is not None and pred(v, depth + 1) for v in vals)
        self._memo[key] = ok
        return ok


def _fix_vectors(tree, binds: dict, imported: set, fixes: list) -> list:
    """(a) lowercase .x/.y/.z on a Vector-typed expression -> .X/.Y/.Z;
       (b) `.center.<comp>` (zero-arg Vector method read as an attribute) -> `.center().<comp>`."""
    typer = _VecTyper(binds, imported)
    edits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in COMPONENTS):
            continue
        inner = node.value
        if (isinstance(inner, ast.Attribute) and inner.attr in METHODS_AS_ATTR
                and inner.end_lineno is not None):
            edits.append((inner.end_lineno, inner.end_col_offset, inner.end_col_offset, "()"))
            fixes.append(f"method-as-attribute .{inner.attr} -> .{inner.attr}()")
        if node.attr in LOWER_COMPONENTS and typer.is_vec(inner):
            if node.end_lineno is not None:
                c1 = node.end_col_offset
                edits.append((node.end_lineno, c1 - 1, c1, LOWER_COMPONENTS[node.attr]))
                fixes.append(f"Vector .{node.attr} -> .{LOWER_COMPONENTS[node.attr]}")
    return edits


def _fix_helper_kwargs(tree, binds: dict, imported: set, fixes: list,
                       auto_imported: set) -> list:
    """(d) a project-helper keyword missing only its `_mm` unit suffix (spur_gear(module=..)
    -> module_mm=..). Only on calls to a name that IS the helper: imported from b123d or
    about to be auto-imported, never a same-named user function."""
    helpers = helper_exports()
    edits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        name = node.func.id
        if name not in helpers:
            continue
        user_defined = any(v is None for v in binds.get(name, [])) and name not in imported
        if user_defined or not (name in imported or name in auto_imported):
            continue
        params = helpers[name][1]
        passed = {k.arg for k in node.keywords if k.arg}
        for kw in node.keywords:
            if not kw.arg or kw.arg in params:
                continue
            target = kw.arg + "_mm"
            if target in params and target not in passed and kw.lineno is not None:
                edits.append((kw.lineno, kw.col_offset, kw.col_offset + len(kw.arg.encode()),
                              target))
                fixes.append(f"{name}({kw.arg}=) -> {name}({target}=)")
    return edits


def _missing_imports(tree, binds: dict, stars: list) -> tuple[dict, list]:
    """(c) names referenced but never bound that a project helper module or math exports.
    Returns ({module: [names]}, notes)."""
    known_star = {"build123d", "math", "b123d.domain", "b123d.warehouse"}
    if any(s not in known_star for s in stars):
        return {}, []          # an unknown star import could define anything: hands off
    star_bound = set()
    if "build123d" in stars:
        star_bound |= build123d_star_names()
    helpers = helper_exports()
    for s in stars:
        if s == "math":
            star_bound |= MATH_NAMES
        elif s.startswith("b123d."):
            star_bound |= {n for n, (m, _) in helpers.items() if m == s}
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    need: dict = {}
    for name in sorted(used):
        if name in binds or name in star_bound or hasattr(builtins, name):
            continue
        if name in helpers:
            need.setdefault(helpers[name][0], []).append(name)
        elif name in MATH_NAMES:
            need.setdefault("math", []).append(name)
    return need, []


def _import_insert_line(tree) -> int:
    """0-based line index to insert import lines at: right after the first top-level
    `from build123d import ...` / `import build123d`, else after a module docstring and any
    `from __future__` imports, else line 0."""
    after = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module == "build123d":
            return node.end_lineno
        if isinstance(node, ast.Import) and any(a.name == "build123d" for a in node.names):
            return node.end_lineno
    for i, node in enumerate(tree.body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(
                getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            after = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            after = node.end_lineno
        else:
            break
    return after


def normalise_api_spelling(code: str) -> tuple[str, list]:
    """Return (code, fixes). Never raises; unparseable code comes back unchanged."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code, []
    fixes: list = []
    binds, imported, stars = _bindings(tree)
    need, _ = _missing_imports(tree, binds, stars)
    auto = {n for names in need.values() for n in names}

    edits = _fix_vectors(tree, binds, imported, fixes)
    edits += _fix_helper_kwargs(tree, binds, imported, fixes, auto)
    out = _apply_edits(code, edits)

    if need:
        lines = out.splitlines(keepends=True)
        at = _import_insert_line(tree)
        if at > 0 and at <= len(lines) and not lines[at - 1].endswith("\n"):
            lines[at - 1] += "\n"
        new = [f"from {mod} import {', '.join(names)}\n" for mod, names in sorted(need.items())]
        lines[at:at] = new
        out = "".join(lines)
        for mod, names in sorted(need.items()):
            fixes.append(f"auto-import from {mod}: {', '.join(names)}")
    return out, fixes
