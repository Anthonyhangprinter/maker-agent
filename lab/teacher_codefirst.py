#!/usr/bin/env python3
"""lab/teacher_codefirst.py -- the "code-first" teacher-data runner.

WHY (owner finding, 2026-09-24): verified-good CAD data is the bottleneck, and a spec that
does not fully define its part produces an unverifiable guess -- a human or a second model
grading "does this match the spec" is grading against ambiguity, not ground truth. This
script reverses the direction:

  1. DESIGN  -- the teacher gets a short seed idea (part type + tier) and writes build123d
               code, using the SAME production code-generation prompt the engine uses.
  2. BUILD + GATE -- CPU-only, via fluid_gen._materialize + lab/harvest.py's _regate.
               Reject on crash, gate_hard, or anything other than exactly one solid.
  3. MEASURE -- scripts/measure_part.py reads the built STEP directly (bbox, precise
               volume, every cylindrical hole/shaft with diameter/axis/centre/depth/
               through-or-blind, fillets, chamfers). Render multi-view PNGs via scripts/render.
  4. SPEC    -- a FRESH call gets the measurement JSON + the render and writes ONE spec in
               plain engineer prose (mm, named datum, every hole/pattern/depth, chamfers,
               fillets), phrased as a user request. No commentary.
  5. BLIND REBUILD -- a FRESH call (no design context) gets only that spec plus the SAME
               production code prompt and writes code. Built, gated, and scored against the
               design's own STEP-derived reference geometry (geom_bands.score_against_reference).
               The pair (spec, blind-rebuild code) is kept ONLY if band == "match" and the
               rebuild's gate is clean (no gate_hard, no [spec] findings). This makes every
               kept pair verified by construction, against a reference the teacher itself
               produced -- no human and no second model in the loop.

Everything CPU-only: never touches ~/.openclaw/cad-build.lock, the GPU, systemd units, or the
main checkout's lab/state/ (this script writes only under benchmarks/results/card/). The OCCT
locale trap (see ~/CLAUDE.md's "process trap" note) is worked around exactly as
scripts/pilot_opus55.py did it: force PYTHONUTF8 and default every Path.write_text/read_text
call to utf-8, process-wide, before any build123d import.

Usage:
    python3 -X utf8 lab/teacher_codefirst.py --model claude-sonnet-5 --budget 0.60 --limit 2
    python3 -X utf8 lab/teacher_codefirst.py --model claude-opus-5-5 --budget 6.00
    python3 -X utf8 lab/teacher_codefirst.py --model claude-sonnet-5 --budget 3.00 --batch
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTHONUTF8", "1")

# ── OCCT/OCP locale trap workaround (see cad-builder/scripts/pilot_opus55.py) ─────────────
# Once build123d's native code runs once, the process locale can get reset to C, and any
# Path.write_text/read_text call made WITHOUT an explicit encoding then silently degrades to
# ASCII. Force utf-8 as the default for EVERY call in this process, before any build123d
# import happens anywhere below.
import pathlib as _pathlib  # noqa: E402
_orig_write_text = _pathlib.Path.write_text
_orig_read_text = _pathlib.Path.read_text


def _write_text_utf8(self, data, encoding=None, errors=None, newline=None):
    return _orig_write_text(self, data, encoding=encoding or "utf-8", errors=errors, newline=newline)


def _read_text_utf8(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors or "replace")


_pathlib.Path.write_text = _write_text_utf8
_pathlib.Path.read_text = _read_text_utf8

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "lab"))

import cad_engine as engine          # noqa: E402
import fluid_gen                     # noqa: E402
import harvest as lab_harvest        # noqa: E402  (_regate -- "the way harvest does")
import geom_bands                    # noqa: E402
import measure_part                  # noqa: E402
import harvest_census as hc          # noqa: E402
from lab import specbank             # noqa: E402

import anthropic                     # noqa: E402

# ── model config ─────────────────────────────────────────────────────────────────────────
# List prices, USD / MTok. Deliberately list price, not any promotional rate -- a budget cap
# must never under-count.
MODEL_PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5": (2.0, 10.0),
}
MAX_TOKENS = 32000

HERE_LAB = Path(__file__).resolve().parent
SPEND_LEDGER = Path.home() / ".openclaw" / "cad-cloud-spend.jsonl"
OUT_ROOT = HERE / "benchmarks" / "results" / "card" / "codefirst-pilot-2026-09-24"

BATCH_POLL_S = 20
BATCH_MAX_WAIT_S = 3600


class BudgetStop(Exception):
    pass


# ── seed bank: 50 diverse, engineering-plausible mechanical parts, tiers 1-4 ──────────────
# Paraphrased inspiration from lab/state/specs.jsonl's own tier examples (brackets, flanges,
# shafts, plates with hole patterns, ...), never copied verbatim -- these are SEEDS (a part
# idea the teacher designs FROM), not finished specs, so verbatim collision is unlikely, but
# every seed and every generated spec is still run through the contamination guard below
# before any code is kept.
SEEDS = [
    # tier 1 -- one or two features, simple envelope
    {"id": "cf01", "tier": 1, "idea": "a rectangular mounting plate with one central through hole"},
    {"id": "cf02", "tier": 1, "idea": "a flat rectangular spacer shim with no holes"},
    {"id": "cf03", "tier": 1, "idea": "a short cylindrical standoff post with a through bore"},
    {"id": "cf04", "tier": 1, "idea": "a plain flat washer-like disc with a centre hole"},
    {"id": "cf05", "tier": 1, "idea": "a straight metal strap with two through holes near each end"},
    {"id": "cf06", "tier": 1, "idea": "a square panel with a single square cutout in the middle"},
    {"id": "cf07", "tier": 1, "idea": "a plain solid cylindrical shaft with no features"},
    {"id": "cf08", "tier": 1, "idea": "a rectangular block with one counterbored hole"},
    {"id": "cf09", "tier": 1, "idea": "a small square tab with one mounting hole and rounded corners"},
    {"id": "cf10", "tier": 1, "idea": "a solid rectangular spacer block with four corner holes"},
    {"id": "cf11", "tier": 1, "idea": "a short tube segment, open at both ends, constant wall thickness"},
    {"id": "cf12", "tier": 1, "idea": "a flat disc plate with three equally spaced mounting holes"},

    # tier 2 -- moderate feature count, still one clear function
    {"id": "cf13", "tier": 2, "idea": "a slotted bar with one elongated slot along its length"},
    {"id": "cf14", "tier": 2, "idea": "a bored plate with a central bore and a bolt circle of holes around it"},
    {"id": "cf15", "tier": 2, "idea": "an L-shaped angle bracket with holes in both legs"},
    {"id": "cf16", "tier": 2, "idea": "a gusseted corner bracket with a triangular stiffening rib"},
    {"id": "cf17", "tier": 2, "idea": "a tie plate with a hole at each end and rounded outside corners"},
    {"id": "cf18", "tier": 2, "idea": "an open-top rectangular tray with thin walls and a flat floor"},
    {"id": "cf19", "tier": 2, "idea": "a hollow electronics enclosure case, open on top, with a mounting lip"},
    {"id": "cf20", "tier": 2, "idea": "an open-top box with four cylindrical standoff bosses inside"},
    {"id": "cf21", "tier": 2, "idea": "a straight rectangular duct segment with constant wall thickness"},
    {"id": "cf22", "tier": 2, "idea": "a stepped shaft with two different diameters along its length"},
    {"id": "cf23", "tier": 2, "idea": "a shaft with a single keyway slot cut along part of its length"},
    {"id": "cf24", "tier": 2, "idea": "a shaft with a narrow retaining-ring groove near one end"},
    {"id": "cf25", "tier": 2, "idea": "a spindle with a chamfer on both ends"},
    {"id": "cf26", "tier": 2, "idea": "a round flange with a central bore and a bolt circle of holes"},
    {"id": "cf27", "tier": 2, "idea": "a square flange plate with a central bore and four corner bolt holes"},
    {"id": "cf28", "tier": 2, "idea": "a blind flange with a bolt circle of holes and no centre bore"},
    {"id": "cf29", "tier": 2, "idea": "a pillow block style bearing mount with a raised boss and base holes"},
    {"id": "cf30", "tier": 2, "idea": "a clevis fork with two parallel ears and a pin hole through both"},
    {"id": "cf31", "tier": 2, "idea": "a U-shaped mounting bracket with holes in both side walls"},
    {"id": "cf32", "tier": 2, "idea": "a stepped spacer sleeve with a shoulder and a through bore"},
    {"id": "cf33", "tier": 2, "idea": "a cylindrical bushing with a flanged head and a through bore"},
]

# extend to a round 50 -- tier 2 continued, then tier 3 and tier 4
SEEDS += [
    {"id": "cf34", "tier": 2, "idea": "a cable clamp saddle bracket with a curved seat and two mounting feet"},
    {"id": "cf35", "tier": 2, "idea": "a single-groove pulley wheel with a central bore and a set-screw hole"},
    {"id": "cf36", "tier": 2, "idea": "a pipe coupling collar with a bore through its length and a split slot"},
    {"id": "cf37", "tier": 2, "idea": "a bearing end cap with a central bore and a bolt circle of small holes"},

    # tier 3 -- harder features (loft/revolve/sweep, gear teeth, raised faces)
    {"id": "cf38", "tier": 3, "idea": "a pipe flange with a raised sealing face and a bolt circle of holes"},
    {"id": "cf39", "tier": 3, "idea": "a two-part assembly of a base plate with a separate cylindrical post that plugs into it"},
    {"id": "cf40", "tier": 3, "idea": "a square-to-round transition duct that lofts from a square base to a round top"},
    {"id": "cf41", "tier": 3, "idea": "a shallow revolved bowl shape with a rounded rim"},
    {"id": "cf42", "tier": 3, "idea": "a short rod swept along a smooth S-shaped curved path"},
    {"id": "cf43", "tier": 3, "idea": "a small spur gear with straight teeth around its rim and a centre bore"},
    {"id": "cf44", "tier": 3, "idea": "a hinge knuckle segment for a folding bracket, with a pin bore through its barrel"},
    {"id": "cf45", "tier": 3, "idea": "a perforated disc with a hexagonal grid pattern of small holes"},
    {"id": "cf46", "tier": 3, "idea": "a rectangular ventilation panel with rows of rounded slots"},
    {"id": "cf47", "tier": 3, "idea": "a cam plate with an off-centre eccentric bore for a follower"},
    {"id": "cf48", "tier": 3, "idea": "a ratchet pawl plate with a pivot bore and a hooked profile"},
    {"id": "cf49", "tier": 3, "idea": "a T-slot rail segment with a captured channel along its length"},

    # tier 4 -- hardest: multi-feature mechanisms
    {"id": "cf50", "tier": 4, "idea": "a three-knuckle piano hinge segment with a pin bore running through all three knuckles"},
]

assert len(SEEDS) == 50, f"expected 50 seeds, got {len(SEEDS)}"
assert len({s['id'] for s in SEEDS}) == 50, "duplicate seed id"


# ── contamination guard (lab/specbank.py's own rule, never a second copy of it) ───────────
def check_contamination(text: str) -> Optional[str]:
    """None when `text` may be used; otherwise the reason it must be refused. Mirrors
    scripts/teacher_gen.py's guard_contamination / lab/specbank.py's refusal_reason, using
    the SAME contamination_sets() primitive so a collision rule change is felt here too."""
    suite_keys, suite_slugs = specbank.contamination_sets()
    if hc._key(text) in suite_keys:
        return "suite-exact-match"
    if hc._slug(text, 40) in suite_slugs:
        return "suite-unique-slug-match"
    return None


def guard_seeds() -> None:
    clashes = [s for s in SEEDS if check_contamination(s["idea"])]
    if clashes:
        print("CONTAMINATION GUARD TRIPPED -- these seeds collide with an eval suite:",
              file=sys.stderr)
        for s in clashes:
            print(f"  {s['id']}: {s['idea']}", file=sys.stderr)
        sys.exit(2)


# ── API key: read into process env only, never print/write ────────────────────────────────
def _load_api_key() -> str:
    cfg = json.loads((Path.home() / ".openclaw" / "openclaw.json").read_text(encoding="utf-8"))
    key = cfg.get("env", {}).get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in openclaw.json env block")
    return key


_client: Optional["anthropic.Anthropic"] = None


def client() -> "anthropic.Anthropic":
    global _client
    if _client is None:
        os.environ["ANTHROPIC_API_KEY"] = _load_api_key()
        _client = anthropic.Anthropic()
    return _client


# ── single-instance lock on the output dir ─────────────────────────────────────────────────
_lock_fh = None


def acquire_single_instance_lock(out_dir: Path) -> None:
    global _lock_fh
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / "_run.lock"
    _lock_fh = open(lock_path, "w")
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise RuntimeError(
            f"another teacher_codefirst.py is already running against {out_dir} "
            f"(lock held on {lock_path}) -- refusing to start a second writer")
    _lock_fh.write(str(os.getpid()))
    _lock_fh.flush()


# ── spend tracking (shared ledger, filtered to this model + this run's start time) ────────
def ledger_spend_so_far(model: str, since_ts: datetime) -> tuple[float, int]:
    if not SPEND_LEDGER.exists():
        return 0.0, 0
    total, n = 0.0, 0
    for line in SPEND_LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("model") != model:
            continue
        try:
            ts = datetime.fromisoformat(r["ts"])
        except Exception:
            continue
        if ts < since_ts:
            continue
        total += float(r.get("usd", 0) or 0)
        n += 1
    return total, n


def budget_check(model: str, since_ts: datetime, budget: float, state: dict,
                  reserve_calls: int = 1) -> None:
    """Re-syncs against the shared ledger every time (protects against a second writer
    that slipped past the lock, or a concurrent run of the same model) and refuses the
    NEXT call before it is sent if it would breach `budget`. `reserve_calls` lets a batch
    submission reserve headroom for every request in the batch at once, since a batch
    cannot be stopped mid-way once created."""
    if not budget:
        return
    ledger_total, ledger_n = ledger_spend_so_far(model, since_ts)
    if ledger_total > state["total_usd"]:
        state["total_usd"] = ledger_total
        state["calls"] = max(state["calls"], ledger_n)
    if state["calls"] == 0:
        pin, pout = MODEL_PRICES[model]
        per_call = MAX_TOKENS * pout / 1_000_000 + 15000 * pin / 1_000_000
    else:
        per_call = state["max_call_usd"] * 1.2
    estimate = per_call * reserve_calls
    if state["total_usd"] + estimate > budget:
        raise BudgetStop(
            f"stopping before next call: total so far ${state['total_usd']:.4f} "
            f"(ledger-verified), next-call estimate ${estimate:.4f} "
            f"({reserve_calls} call(s)), budget ${budget:.2f}")


def record_spend(model: str, usage, wall_s: float, state: dict) -> dict:
    pin, pout = MODEL_PRICES[model]
    tin = int(getattr(usage, "input_tokens", 0) or 0)
    tout = int(getattr(usage, "output_tokens", 0) or 0)
    cread = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cwrite = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cost = ((tin + 0.1 * cread + 1.25 * cwrite) * pin + tout * pout) / 1_000_000
    state["total_usd"] += cost
    state["calls"] += 1
    state["max_call_usd"] = max(state["max_call_usd"], cost)
    row = {"ts": datetime.now(timezone.utc).isoformat(), "model": model,
           "in": tin, "out": tout, "usd": round(cost, 6)}
    try:
        SPEND_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with SPEND_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:
        print(f"WARNING: could not write shared spend ledger: {e}", file=sys.stderr)
    return {"in": tin, "out": tout, "cost": cost, "wall_s": wall_s}


# ── one non-batch call ──────────────────────────────────────────────────────────────────
def _call_kwargs(model: str) -> dict:
    """claude-opus-5-5: adaptive thinking, cannot be disabled, so the `thinking` param is
    OMITTED entirely (passing it would try to override what the API refuses to change).
    claude-sonnet-5: same effort/shape but thinking is opt-in, so it is passed explicitly.
    Neither takes temperature/top_p/top_k or an assistant prefill."""
    if model == "claude-opus-5-5":
        return {"output_config": {"effort": "high"}}
    if model == "claude-sonnet-5":
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    raise ValueError(f"unsupported model {model!r} (only claude-opus-5-5 / claude-sonnet-5)")


def call_model(model: str, system: str, content, since_ts: datetime, budget: float,
                state: dict) -> dict:
    """One streamed call, no thinking override beyond what the model requires, no sampling
    params, no prefill. `content` is either a prompt string or a list of content blocks
    (text + image) for the multimodal SPEC call."""
    budget_check(model, since_ts, budget, state)
    t0 = time.monotonic()
    with client().messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": content}],
        **_call_kwargs(model),
    ) as stream:
        final = stream.get_final_message()
    wall = time.monotonic() - t0
    meter = record_spend(model, final.usage, wall, state)
    text = "".join(b.text for b in final.content if getattr(b, "type", "") == "text")
    return {"text": text, "stop_reason": final.stop_reason, **meter}


# ── batch calls (Message Batches API) -- implemented for all 3 stages, not exercised by
# the smoke pilot (the task runs non-batch); see run_batch_stage(). ─────────────────────
def _batch_request(custom_id: str, model: str, system: str, content) -> dict:
    return {"custom_id": custom_id,
            "params": {"model": model, "max_tokens": MAX_TOKENS, "system": system,
                       "messages": [{"role": "user", "content": content}],
                       **_call_kwargs(model)}}


def run_batch_stage(model: str, requests: list[tuple[str, str, object]], since_ts: datetime,
                     budget: float, state: dict) -> dict[str, dict]:
    """requests: list of (custom_id, system, content). Submits ALL of them as one Batch,
    polls processing_status until "ended", then reads results keyed by custom_id (never by
    position -- the Batches API makes no ordering guarantee on `results`). A batch cannot be
    partially cancelled once cost is committed, so the WHOLE batch's worst-case cost is
    checked against the budget BEFORE it is created (reserve_calls=len(requests)). Batch
    pricing is roughly half of synchronous pricing in practice; this deliberately still
    meters at the synchronous MODEL_PRICES rate so the budget cap stays conservative rather
    than under-counting a real bill."""
    budget_check(model, since_ts, budget, state, reserve_calls=len(requests))
    batch = client().messages.batches.create(
        requests=[_batch_request(cid, model, sysprompt, content)
                  for cid, sysprompt, content in requests])
    deadline = time.monotonic() + BATCH_MAX_WAIT_S
    while True:
        b = client().messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f"batch {batch.id} did not end within {BATCH_MAX_WAIT_S}s "
                                f"(status={b.processing_status})")
        time.sleep(BATCH_POLL_S)
    out: dict[str, dict] = {}
    for entry in client().messages.batches.results(batch.id):
        cid = entry.custom_id
        if entry.result.type != "succeeded":
            out[cid] = {"text": "", "stop_reason": f"batch-{entry.result.type}",
                        "in": 0, "out": 0, "cost": 0.0, "wall_s": 0.0}
            continue
        msg = entry.result.message
        meter = record_spend(model, msg.usage, 0.0, state)
        text = "".join(c.text for c in msg.content if getattr(c, "type", "") == "text")
        out[cid] = {"text": text, "stop_reason": msg.stop_reason, **meter}
    return out


# ── production prompt capture (byte-identical to the live engine) ─────────────────────────
def capture_codegen_prompt(spec_text: str) -> dict:
    """The exact (system, prompt) the live loop would send for this spec text -- captured by
    stubbing engine._ollama (never actually called) and reading engine._LAST_PROMPT
    afterwards, the same trick scripts/pilot_opus55.py and scripts/teacher_gen.py use."""
    orig = engine._ollama
    engine._ollama = lambda *a, **k: ""
    try:
        notes = engine.retrieval_notes_for(spec_text, use_fewshots=True)
        engine.generate_code_raw(spec_text, notes)
    finally:
        engine._ollama = orig
    return dict(engine._LAST_PROMPT)


def to_code(raw_text: str, spec_text: str) -> str:
    return engine._patch_code(engine._strip_fences(raw_text),
                              wants=engine._wanted_edge_features(spec_text))


# ── build + gate (CPU only, no build lock) ─────────────────────────────────────────────────
def build_and_gate(code: str, build_dir: Path, spec_text: str) -> dict:
    build_dir.mkdir(parents=True, exist_ok=True)
    m = fluid_gen._materialize(code, build_dir, spec_text)
    return lab_harvest._regate(m, build_dir, spec_text)


def design_accept(gate: dict) -> tuple[bool, str]:
    """Reject on crash, gate_hard, or anything other than exactly one solid."""
    if gate.get("error"):
        return False, f"crash: {gate['error'][:200]}"
    if gate.get("unscored_reason"):
        return False, f"unscored: {gate['unscored_reason']}"
    if gate.get("gate_hard"):
        return False, "gate_hard: " + "; ".join(gate["gate_hard"])[:300]
    n_solids = gate.get("facts", {}).get("solids")
    if n_solids != 1:
        return False, f"n_solids={n_solids} (need exactly 1)"
    return True, ""


def keep_pair(band: Optional[str], gate: dict) -> bool:
    """The blind-rebuild pair is kept ONLY if it matches the reference geometry AND its own
    gate is clean (no gate_hard, no [spec] contradiction). A near-miss render, or a build
    that happens to match geometry but trips a [spec] check, is not verified-good data."""
    if band != "match":
        return False
    if gate.get("error") or gate.get("unscored_reason"):
        return False
    if gate.get("gate_hard") or gate.get("gate_spec"):
        return False
    return True


# ── multi-view render ───────────────────────────────────────────────────────────────────
def render_views(step_path: Path, out_png: Path) -> bool:
    try:
        subprocess.run(["/usr/bin/python3", str(HERE / "scripts" / "render"),
                        str(step_path), str(out_png), "--section"],
                       timeout=120, capture_output=True, encoding="utf-8", errors="replace")
    except Exception:
        return False
    return out_png.exists()


def _image_block(png_path: Path) -> dict:
    import base64
    data = base64.b64encode(png_path.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


# ── the SPEC call ───────────────────────────────────────────────────────────────────────
_SPEC_SYSTEM = """\
You are a mechanical engineer describing a manufactured part from its inspection measurements,
the way you would phrase a request to a machine shop or a CAD drafter to have this exact part
built again from scratch.

You are given: (1) a JSON block of deterministic measurements taken directly off the solid --
bounding box, precise volume, every cylindrical hole/shaft (diameter, axis, depth, position,
through-or-blind), every fillet, every chamfer; (2) a multi-view render (isometric, top-down,
and a mid-plane section showing internal features) of the same part.

Write ONE spec, in plain engineering prose, in millimetres, as if you are the person ASKING for
this part to be built -- phrase it as a request ("A flanged bushing with a 40mm OD..."), never
as a description of an existing object ("the part has..." or "this part contains...").

Rules:
- State a named datum or overall envelope first (e.g. "a 90x60x8mm plate", "a 40mm diameter
  shaft 120mm long"), then every other feature relative to it.
- State EVERY hole: diameter, whether it is through or blind (and its depth if blind), and its
  position (from a named edge/corner/centre, or as a bolt circle with a hole count and
  diameter, whichever the measurements show).
- State every chamfer and fillet with its size.
- Every number must come from the measurement JSON -- do not invent or round away a stated
  dimension, but you may use ordinary shop rounding (e.g. 39.98mm to "40mm") when the
  measurement is clearly a rounding artefact of the build, not a real dimension.
- No commentary, no headings, no bullet points, no restating that this is a "spec" -- just the
  request text itself, 2-6 sentences.
"""


def build_spec_prompt(measurements: dict) -> str:
    return ("Measurements (JSON, all lengths in mm):\n"
            + json.dumps(measurements, indent=2)
            + "\n\nWrite the part request now.")


# ── per-seed pipeline ───────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_row(resp: dict) -> dict:
    return {"stop_reason": resp["stop_reason"], "tokens_in": resp["in"], "tokens_out": resp["out"],
            "cost_usd": round(resp["cost"], 6), "wall_s": round(resp["wall_s"], 1)}


def run_seed(seed: dict, model: str, out_dir: Path, since_ts: datetime, budget: float,
             state: dict) -> dict:
    sid, tier, idea = seed["id"], seed["tier"], seed["idea"]
    part_dir = out_dir / "builds" / sid
    part_dir.mkdir(parents=True, exist_ok=True)
    row: dict = {"id": sid, "tier": tier, "idea": idea, "model": model, "ts": _now()}

    coll = check_contamination(idea)
    if coll:
        row.update(outcome="refused-contamination-seed", reason=coll)
        return row

    # 1. DESIGN call -- production codegen prompt, seed idea as the "spec"
    p_design = capture_codegen_prompt(idea)
    (part_dir / "design_prompt.json").write_text(
        json.dumps({"seed": idea, **p_design}, indent=2), encoding="utf-8")
    resp = call_model(model, p_design["system"], p_design["prompt"], since_ts, budget, state)
    row["design_call"] = _usage_row(resp)
    if resp["stop_reason"] == "refusal" or not resp["text"].strip():
        row.update(outcome="design-refused", reason=f"stop_reason={resp['stop_reason']}")
        return row
    design_code = to_code(resp["text"], idea)
    (part_dir / "design_code.py").write_text(design_code, encoding="utf-8")

    # 2. BUILD + GATE (design)
    design_build_dir = part_dir / "design_build"
    gate = build_and_gate(design_code, design_build_dir, idea)
    ok, reason = design_accept(gate)
    row["design_gate"] = {"gate_hard": gate.get("gate_hard"), "gate_spec": gate.get("gate_spec"),
                          "error": gate.get("error"), "unscored_reason": gate.get("unscored_reason"),
                          "solids": gate.get("facts", {}).get("solids")}
    if not ok:
        row.update(outcome="design-rejected", reason=reason)
        return row

    # 3. MEASURE + render
    step_path = design_build_dir / "build.step"
    try:
        measurements = measure_part.measure(step_path)
    except Exception as e:
        row.update(outcome="measure-failed", reason=str(e)[:300])
        return row
    (part_dir / "measurements.json").write_text(json.dumps(measurements, indent=2),
                                                 encoding="utf-8")
    views_png = design_build_dir / "views.png"
    have_views = render_views(step_path, views_png)
    if not have_views:
        fallback = design_build_dir / "build.png"
        views_png = fallback if fallback.exists() else None

    ref_stl = part_dir / "reference.stl"
    ref_sidecar = part_dir / "reference_volume.json"
    try:
        measure_part.write_reference_volume_sidecar(step_path, ref_stl, ref_sidecar, measurements)
    except Exception as e:
        row.update(outcome="reference-export-failed", reason=str(e)[:300])
        return row

    # 4. SPEC call -- measurements + render, no design context
    spec_prompt_text = build_spec_prompt(measurements)
    content: list = [{"type": "text", "text": spec_prompt_text}]
    if views_png is not None:
        content.append(_image_block(views_png))
    resp2 = call_model(model, _SPEC_SYSTEM, content, since_ts, budget, state)
    row["spec_call"] = _usage_row(resp2)
    if resp2["stop_reason"] == "refusal" or not resp2["text"].strip():
        row.update(outcome="spec-refused", reason=f"stop_reason={resp2['stop_reason']}")
        return row
    spec_text = resp2["text"].strip()
    (part_dir / "spec.txt").write_text(spec_text, encoding="utf-8")

    coll2 = check_contamination(spec_text)
    if coll2:
        row.update(outcome="refused-contamination-spec", reason=coll2)
        return row

    # 5. BLIND REBUILD -- fresh call, spec only, same production code prompt
    p_rebuild = capture_codegen_prompt(spec_text)
    (part_dir / "rebuild_prompt.json").write_text(
        json.dumps({"spec": spec_text, **p_rebuild}, indent=2), encoding="utf-8")
    resp3 = call_model(model, p_rebuild["system"], p_rebuild["prompt"], since_ts, budget, state)
    row["rebuild_call"] = _usage_row(resp3)
    if resp3["stop_reason"] == "refusal" or not resp3["text"].strip():
        row.update(outcome="rebuild-refused", reason=f"stop_reason={resp3['stop_reason']}")
        return row
    rebuild_code = to_code(resp3["text"], spec_text)
    (part_dir / "rebuild_code.py").write_text(rebuild_code, encoding="utf-8")

    rebuild_build_dir = part_dir / "rebuild_build"
    gate2 = build_and_gate(rebuild_code, rebuild_build_dir, spec_text)
    row["rebuild_gate"] = {"gate_hard": gate2.get("gate_hard"), "gate_spec": gate2.get("gate_spec"),
                           "error": gate2.get("error"), "unscored_reason": gate2.get("unscored_reason"),
                           "solids": gate2.get("facts", {}).get("solids")}
    rebuild_step = rebuild_build_dir / "build.step"
    if gate2.get("error") or not rebuild_step.exists():
        row.update(outcome="rebuild-crashed", reason=gate2.get("error") or "no build.step",
                  band="fail")
        return row

    score = geom_bands.score_against_reference(rebuild_step, ref_stl)
    row["score"] = score
    row["band"] = score.get("band")

    kept = keep_pair(score.get("band"), gate2)
    row["kept"] = kept
    if not kept:
        row["outcome"] = "rebuild-mismatch" if score.get("band") != "match" else "rebuild-gate-dirty"
        return row

    row["outcome"] = "kept"
    pair = {"id": sid, "tier": tier, "model": model, "idea": idea, "spec": spec_text,
           "code": rebuild_code, "design_code": design_code, "band": score.get("band"),
           "chamfer_mm": score.get("chamfer_mm"), "volume_diff_pct": score.get("volume_diff_pct"),
           "measurements": measurements, "timestamp": _now()}
    with (out_dir / "pairs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(pair, default=str) + "\n")
    return row


# ── results.jsonl I/O (resumable: skip ids already attempted by this model) ────────────────
def append_result(out_dir: Path, row: dict) -> None:
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def done_ids(out_dir: Path, model: str) -> set:
    results = out_dir / "results.jsonl"
    if not results.exists():
        return set()
    done = set()
    for line in results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("model") == model:
            done.add(r.get("id"))
    return done


# ── batch pipeline (--batch): the SAME 5 steps, but each call stage is submitted as one
# Batch across every surviving seed instead of one synchronous call per seed. The CPU-only
# work in between (build/gate/measure/render/score) stays per-item and sequential -- there is
# nothing to batch there, it never touches the API. ─────────────────────────────────────────
def run_batch_pipeline(seeds: list[dict], model: str, out_dir: Path, since_ts: datetime,
                       budget: float, state: dict) -> dict[str, dict]:
    rows: dict[str, dict] = {}

    def _reject(sid: str, seed: dict, outcome: str, reason: str = "", **extra) -> None:
        # merge onto whatever call-usage fields (design_call/spec_call) this sid's row
        # already has -- never discard spend accounting for a call that already happened.
        row = rows.setdefault(sid, {})
        row.update(id=sid, tier=seed["tier"], idea=seed["idea"], model=model, ts=_now(),
                  outcome=outcome, reason=reason, **extra)

    live = {}
    for seed in seeds:
        sid = seed["id"]
        coll = check_contamination(seed["idea"])
        if coll:
            _reject(sid, seed, "refused-contamination-seed", coll)
            continue
        live[sid] = seed

    # ── Stage A: DESIGN batch ───────────────────────────────────────────────────────────
    design_prompts = {sid: capture_codegen_prompt(seed["idea"]) for sid, seed in live.items()}
    for sid, p in design_prompts.items():
        part_dir = out_dir / "builds" / sid
        part_dir.mkdir(parents=True, exist_ok=True)
        (part_dir / "design_prompt.json").write_text(
            json.dumps({"seed": live[sid]["idea"], **p}, indent=2), encoding="utf-8")
    design_results = run_batch_stage(
        model, [(sid, p["system"], p["prompt"]) for sid, p in design_prompts.items()],
        since_ts, budget, state)

    measured = {}   # sid -> {"measurements":..., "ref_stl":..., "views_png":..., "part_dir":...}
    for sid, seed in list(live.items()):
        part_dir = out_dir / "builds" / sid
        resp = design_results.get(sid) or {"stop_reason": "batch-missing", "text": "",
                                           "in": 0, "out": 0, "cost": 0.0, "wall_s": 0.0}
        rows.setdefault(sid, {})["design_call"] = _usage_row(resp)
        if resp["stop_reason"] == "refusal" or not resp["text"].strip():
            _reject(sid, seed, "design-refused", f"stop_reason={resp['stop_reason']}")
            del live[sid]
            continue
        design_code = to_code(resp["text"], seed["idea"])
        (part_dir / "design_code.py").write_text(design_code, encoding="utf-8")
        design_build_dir = part_dir / "design_build"
        gate = build_and_gate(design_code, design_build_dir, seed["idea"])
        ok, reason = design_accept(gate)
        rows[sid]["design_gate"] = {
            "gate_hard": gate.get("gate_hard"), "gate_spec": gate.get("gate_spec"),
            "error": gate.get("error"), "unscored_reason": gate.get("unscored_reason"),
            "solids": gate.get("facts", {}).get("solids")}
        if not ok:
            _reject(sid, seed, "design-rejected", reason)
            del live[sid]
            continue
        step_path = design_build_dir / "build.step"
        try:
            measurements = measure_part.measure(step_path)
        except Exception as e:
            _reject(sid, seed, "measure-failed", str(e)[:300])
            del live[sid]
            continue
        (part_dir / "measurements.json").write_text(json.dumps(measurements, indent=2),
                                                     encoding="utf-8")
        views_png = design_build_dir / "views.png"
        if not render_views(step_path, views_png):
            fallback = design_build_dir / "build.png"
            views_png = fallback if fallback.exists() else None
        ref_stl = part_dir / "reference.stl"
        ref_sidecar = part_dir / "reference_volume.json"
        try:
            measure_part.write_reference_volume_sidecar(step_path, ref_stl, ref_sidecar,
                                                         measurements)
        except Exception as e:
            _reject(sid, seed, "reference-export-failed", str(e)[:300])
            del live[sid]
            continue
        measured[sid] = {"measurements": measurements, "views_png": views_png,
                         "ref_stl": ref_stl, "part_dir": part_dir, "design_code": design_code}

    # ── Stage B: SPEC batch ─────────────────────────────────────────────────────────────
    spec_reqs = []
    for sid in live:
        m = measured[sid]
        content: list = [{"type": "text", "text": build_spec_prompt(m["measurements"])}]
        if m["views_png"] is not None:
            content.append(_image_block(m["views_png"]))
        spec_reqs.append((sid, _SPEC_SYSTEM, content))
    spec_results = run_batch_stage(model, spec_reqs, since_ts, budget, state)

    specced = {}
    for sid in list(live):
        seed = live[sid]
        resp = spec_results.get(sid) or {"stop_reason": "batch-missing", "text": "",
                                         "in": 0, "out": 0, "cost": 0.0, "wall_s": 0.0}
        rows[sid]["spec_call"] = _usage_row(resp)
        if resp["stop_reason"] == "refusal" or not resp["text"].strip():
            _reject(sid, seed, "spec-refused", f"stop_reason={resp['stop_reason']}")
            del live[sid]
            continue
        spec_text = resp["text"].strip()
        (measured[sid]["part_dir"] / "spec.txt").write_text(spec_text, encoding="utf-8")
        coll2 = check_contamination(spec_text)
        if coll2:
            _reject(sid, seed, "refused-contamination-spec", coll2)
            del live[sid]
            continue
        specced[sid] = spec_text

    # ── Stage C: BLIND REBUILD batch ────────────────────────────────────────────────────
    rebuild_prompts = {sid: capture_codegen_prompt(spec_text) for sid, spec_text in specced.items()}
    for sid, p in rebuild_prompts.items():
        (measured[sid]["part_dir"] / "rebuild_prompt.json").write_text(
            json.dumps({"spec": specced[sid], **p}, indent=2), encoding="utf-8")
    rebuild_results = run_batch_stage(
        model, [(sid, p["system"], p["prompt"]) for sid, p in rebuild_prompts.items()],
        since_ts, budget, state)

    for sid, spec_text in specced.items():
        seed = live[sid]
        part_dir = measured[sid]["part_dir"]
        resp = rebuild_results.get(sid) or {"stop_reason": "batch-missing", "text": "",
                                            "in": 0, "out": 0, "cost": 0.0, "wall_s": 0.0}
        rows[sid]["rebuild_call"] = _usage_row(resp)
        if resp["stop_reason"] == "refusal" or not resp["text"].strip():
            _reject(sid, seed, "rebuild-refused", f"stop_reason={resp['stop_reason']}")
            continue
        rebuild_code = to_code(resp["text"], spec_text)
        (part_dir / "rebuild_code.py").write_text(rebuild_code, encoding="utf-8")
        rebuild_build_dir = part_dir / "rebuild_build"
        gate2 = build_and_gate(rebuild_code, rebuild_build_dir, spec_text)
        rows[sid]["rebuild_gate"] = {
            "gate_hard": gate2.get("gate_hard"), "gate_spec": gate2.get("gate_spec"),
            "error": gate2.get("error"), "unscored_reason": gate2.get("unscored_reason"),
            "solids": gate2.get("facts", {}).get("solids")}
        rebuild_step = rebuild_build_dir / "build.step"
        if gate2.get("error") or not rebuild_step.exists():
            rows[sid].update(id=sid, tier=seed["tier"], idea=seed["idea"], model=model, ts=_now(),
                            outcome="rebuild-crashed", reason=gate2.get("error") or "no build.step",
                            band="fail")
            continue
        score = geom_bands.score_against_reference(rebuild_step, measured[sid]["ref_stl"])
        kept = keep_pair(score.get("band"), gate2)
        rows[sid].update(id=sid, tier=seed["tier"], idea=seed["idea"], model=model, ts=_now(),
                        score=score, band=score.get("band"), kept=kept,
                        outcome="kept" if kept else (
                            "rebuild-mismatch" if score.get("band") != "match"
                            else "rebuild-gate-dirty"))
        if kept:
            pair = {"id": sid, "tier": seed["tier"], "model": model, "idea": seed["idea"],
                   "spec": spec_text, "code": rebuild_code,
                   "design_code": measured[sid]["design_code"], "band": score.get("band"),
                   "chamfer_mm": score.get("chamfer_mm"), "volume_diff_pct": score.get("volume_diff_pct"),
                   "measurements": measured[sid]["measurements"], "timestamp": _now()}
            with (out_dir / "pairs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(pair, default=str) + "\n")

    return rows


# ── main ─────────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_PRICES))
    ap.add_argument("--budget", type=float, required=True,
                    help="hard USD cap for this run, checked against the shared ledger "
                         "before every call")
    ap.add_argument("--limit", type=int, default=0, help="max seeds to attempt (0 = all 50)")
    ap.add_argument("--only", default="", help="comma-separated seed ids")
    ap.add_argument("--batch", action="store_true",
                    help="use the Message Batches API for each of the 3 call stages "
                         "instead of one synchronous call per seed")
    a = ap.parse_args()

    guard_seeds()

    out_dir = OUT_ROOT / a.model
    acquire_single_instance_lock(out_dir)
    for sub in ("builds",):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    seeds = SEEDS
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        seeds = [s for s in seeds if s["id"] in want]
    if a.limit:
        seeds = seeds[:a.limit]

    since_ts = datetime.now(timezone.utc)
    state = {"total_usd": 0.0, "calls": 0, "max_call_usd": 0.0}
    already = done_ids(out_dir, a.model)
    seeds = [s for s in seeds if s["id"] not in already]

    print(f"[codefirst] model={a.model} seeds={len(seeds)} (skipping {len(already)} already "
          f"done) budget=${a.budget:.2f} batch={a.batch}")

    kept, stopped = 0, None

    if a.batch:
        # The whole batch's worst-case cost is checked ONCE up front per stage (see
        # run_batch_stage's own budget_check with reserve_calls=len(requests)) -- a batch
        # cannot be stopped mid-way once created, so there is no per-seed BudgetStop here.
        try:
            rows = run_batch_pipeline(seeds, a.model, out_dir, since_ts, a.budget, state)
        except BudgetStop as e:
            print(f"\n[codefirst] BUDGET STOP before batch submission: {e}")
            return 0
        for i, seed in enumerate(seeds, 1):
            row = rows.get(seed["id"], {"id": seed["id"], "outcome": "missing"})
            append_result(out_dir, row)
            if row.get("kept"):
                kept += 1
            print(f"  [{i}/{len(seeds)}] {row['id']:>5}  {row.get('outcome'):<26} "
                  f"band={row.get('band')}  kept={row.get('kept', False)}  "
                  f"spend=${state['total_usd']:.4f}", flush=True)
    else:
        for i, seed in enumerate(seeds, 1):
            try:
                row = run_seed(seed, a.model, out_dir, since_ts, a.budget, state)
            except BudgetStop as e:
                print(f"\n[codefirst] BUDGET STOP during {seed['id']}: {e}")
                stopped = str(e)
                break
            except Exception as e:
                row = {"id": seed["id"], "tier": seed["tier"], "idea": seed["idea"],
                      "model": a.model, "ts": _now(), "outcome": "error", "reason": str(e)[:300]}
            append_result(out_dir, row)
            if row.get("kept"):
                kept += 1
            print(f"  [{i}/{len(seeds)}] {row['id']:>5}  {row.get('outcome'):<26} "
                  f"band={row.get('band')}  kept={row.get('kept', False)}  "
                  f"spend=${state['total_usd']:.4f}", flush=True)

    print(f"\nTOTAL SPEND: ${state['total_usd']:.4f} over {state['calls']} call(s) "
          f"(max single call ${state['max_call_usd']:.4f})")
    print(f"kept pairs: {kept} / {len(seeds)} attempted")
    if stopped:
        print(f"stopped early: {stopped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
