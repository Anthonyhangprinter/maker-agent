"""Single seam to the validated build123d pipeline engine.

The v4.3 engine (`cad_agent_v4.py`) holds the validated brief / codegen / gate / critic / loop /
learning logic — the part with hard-won decision-rules (escalation guards, advisory-vs-hard gate,
the measured learning loop). v5 re-architects the OUTPUT layer and the USER FLOW around that engine
rather than re-deriving it. This module is the ONLY place v5 imports the engine, so the rest of v5
depends on stable names here; the engine internals can later be physically split into
`geometry.py / gate.py / brief.py / …` behind these same names with no churn upstream.
"""
import importlib
from . import config  # noqa: F401  (ensures sys.path + logging are set up before importing engine)

_engine = importlib.import_module("cad_agent_v4")

# Core loop + refine
build          = _engine.build            # (spec, coder, use_fewshots, do_upload, final_render,
                                          #  brief_override) -> result
merge_spec     = _engine.merge_spec       # (original, feedback, history) -> new spec
store_feedback = _engine.store_feedback   # (result, rating, comment)

# N2 (ambiguity gate) / N3 (brief-as-contract) — see cad_agent_v4 for the full contracts.
triage_ambiguity  = _engine.triage_ambiguity     # (spec) -> [questions] ([] if buildable/failed)
patch_brief       = _engine.patch_brief          # (brief, feedback) -> (patched_brief|None, delta)
apply_brief_patch = _engine.apply_brief_patch    # (brief, changes, +features, -features) -> (brief, delta)

# Geometry/inspection (used by the v5 loop for summaries + diffs + undo)
run_inspect    = _engine.run_inspect
run_diff       = _engine.run_diff
run_stl        = _engine.run_stl
parse_facts    = _engine.parse_facts

# Brief / codegen / critic (exposed for future v5 callers / tests)
build_brief      = _engine.build_brief
generate_code    = _engine.generate_code
verify_expected  = _engine.verify_expected
visual_critique  = _engine.visual_critique
wants_section    = _engine.wants_section

ENGINE_VERSION = getattr(_engine, "VERSION", "?")
