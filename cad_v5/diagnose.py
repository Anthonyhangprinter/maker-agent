"""B3 — table-driven failure taxonomy (CADSmith-style structured diagnostics).

Every failed turn gets classified into a category with a repair hint tuned for a small coder.
Two consumers:
  1. The revise prompt — a targeted hint beats a raw traceback for a 7B.
  2. The per-build category counts (`failure_categories` in the result) — the histogram over
     benchmark runs tells the fine-tune track (D) exactly which failure modes to train on.

Categories are drawn from failures actually observed in this project's logs/lessons, not
invented: keep them few, mutually distinguishable, and each with ONE actionable hint.
"""
import re

# (category, compiled pattern, repair hint for the coder — "" means the original message
#  already carries the repair instruction, e.g. the gate's own messages)
_TAXONOMY: list[tuple[str, re.Pattern, str]] = [
    ("timeout", re.compile(r"\btime[d]? ?out\b|\btimeout\b", re.I),
     "The build exceeded its time budget. Simplify: fewer features per attempt, no loops over "
     "many faces, prefer plain primitives and a single boolean chain."),

    ("no_result", re.compile(r"never assigns it to `?result`?|No build123d Part/Compound found", re.I),
     ""),  # the step script's teaching error already names the fix and the candidate variable

    ("fillet_chamfer", re.compile(r"fillet|chamfer|no suitable edges", re.I),
     "This is a FILLET/CHAMFER failure — it is cosmetic and must not fail the part. Wrap EVERY "
     "fillet/chamfer in try/except and keep the unfilleted solid on failure, OR remove the fillet "
     "entirely, OR use a much smaller radius on a specific edge set (filter_by/group_by). Do NOT "
     "repeat the same fillet call that just failed."),

    ("syntax", re.compile(r"SyntaxError|invalid syntax|unexpected indent|unmatched", re.I),
     "Python syntax error — rewrite the whole script cleanly; do not patch around the broken line."),

    ("api_misuse", re.compile(
        r"has no attribute|unexpected keyword|not defined|cannot import|No module named|"
        r"missing \d+ required|takes \d+ positional|NameError|AttributeError|TypeError", re.I),
     "You used a build123d API that does not exist or with wrong arguments. Stick to the "
     "documented algebra-mode basics: Box/Cylinder/Cone/Sphere, Pos()/Rot() placement, + - & "
     "booleans, fillet/chamfer on edge lists. Do not invent methods or keyword arguments."),

    ("degenerate_input", re.compile(
        r"must be (?:greater|positive)|zero|negative (?:radius|height|thickness)|"
        r"BRep_API.*command not done|null shape|"
        # real OCP kernel errors for degenerate inputs, captured live: Box(0,0,0) raises a bare
        # Standard_DomainError; other zero/negative dims raise ConstructionError/Failure
        r"Standard_(?:DomainError|Failure|ConstructionError|NullObject|RangeError)", re.I),
     "A primitive or boolean got a degenerate dimension (zero/negative size, or a cut that "
     "misses the solid entirely). Re-check every dimension is positive and every subtracted "
     "feature actually overlaps the material — primitives are CENTRED at the origin."),

    ("invalid_geometry", re.compile(
        r"No solid bodies|near-zero|likely a surface|geometry is invalid|open or degenerate", re.I),
     "The script ran but produced no valid solid. Usually a boolean chain that subtracted "
     "everything, or an unclosed sketch. Build the base solid first, verify each subtraction "
     "removes only what it should, and keep `result` a single fused solid."),

    ("gate_structural", re.compile(
        r"built nothing|fragmented|disconnected|unfused|separate (?:bodies|pieces)|"
        r"no solid at all", re.I),
     "The part is in pieces (or empty) when ONE fused solid was expected. Join touching solids "
     "with `+` so everything fuses; make sure joined parts actually overlap or share a face."),

    ("gate_feature", re.compile(
        r"no bore of that size|must be RADIAL|must be AXIAL|ring groove|bolt circle|"
        r"came out blind|blind hole", re.I),
     ""),  # gate feature messages already carry the specific repair (cross_bore/ring_groove/etc.)

    ("gate_dimension", re.compile(
        r"bbox|bounding box|overall size|differs from|wall(?:s| thickness)|too (?:thick|thin)", re.I),
     "A measured dimension is off the spec target. Set the named parameter constants at the top "
     "of the script to EXACTLY the requested numbers and derive everything else from them."),
]

CATEGORIES = [name for name, _, _ in _TAXONOMY] + ["unknown"]


def diagnose(error_text: str) -> tuple[str, str]:
    """Classify a failure message → (category, repair_hint). hint == "" means the original
    message is already the best instruction — pass it through unchanged."""
    text = error_text or ""
    for name, pat, hint in _TAXONOMY:
        if pat.search(text):
            return name, hint
    return "unknown", ""
