#!/usr/bin/env python3
"""
generate_teacher_data.py  —  Claude teacher data generator for CAD agent RL training.

Generates ~50 high-quality (spec -> plan JSON) training pairs using Claude as teacher.
Output: ~/.openclaw/cad-teacher-data.jsonl
        ~/.openclaw/cad-teacher-failures.jsonl  (parse errors / invalid plans, never trained)

Usage:
    python3 generate_teacher_data.py                  # generate all 50 (uses claude CLI)
    python3 generate_teacher_data.py --start 20       # resume from spec #20
    python3 generate_teacher_data.py --dry-run        # print prompts only, no API calls
    python3 generate_teacher_data.py --spec "hollow tube 50mm OD 3mm wall 200mm long"
    python3 generate_teacher_data.py --api-key sk-...  # use direct API instead of CLI
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
# anthropic is imported lazily in run() only when --cli is not used

_OPENCLAW      = Path.home() / ".openclaw"
TEACHER_FILE   = _OPENCLAW / "cad-teacher-data.jsonl"
TEACH_FAIL_FILE = _OPENCLAW / "cad-teacher-failures.jsonl"

MODEL = "claude-sonnet-4-6"

# ── 50-spec curriculum (simple → complex) ───────────────────────────────────────

SPECS = [
    # Tier 1 — single operation (1-3 tools)
    "solid cylinder 60mm diameter 80mm tall",
    "steel round bar 25mm diameter 500mm long",
    "flat rectangular plate 150x100mm 10mm thick",
    "spur gear 24 teeth module 2 12mm thick",
    "M12 hex bolt 80mm shank",
    "250UC72.9 I-beam 1000mm",
    "thin disc 120mm diameter 5mm thick",
    "square bar 30x30mm cross-section 400mm long",
    "M8 hex bolt 50mm shank",
    "spur gear 48 teeth module 2 20mm thick",

    # Tier 2 — two operations
    "hollow tube 50mm OD 3mm wall 200mm long",
    "round disc 100mm diameter 15mm thick with central 20mm bore",
    "rectangular box 120x80mm 60mm tall wall 4mm open top",
    "steel plate 200x150mm 8mm thick with 10mm hole at centre",
    "flat plate 100x100mm 5mm thick all edges filleted 3mm",
    "cylindrical boss 30mm diameter 25mm tall on 80x60mm plate 8mm thick",
    "round rod 20mm diameter 300mm long with 6mm centre hole through length",
    "rectangular plate 180x120mm 6mm thick with 12mm slot 40mm long at centre",
    "hollow square tube 40x40mm OD 3mm wall 300mm long",
    "L-bracket 80x60mm legs 5mm thick 150mm long",

    # Tier 3 — three operations
    "cylindrical boss 40mm diameter 30mm tall centred on 120x80mm plate 10mm thick",
    "steel plate 160x120mm 10mm thick with 4 corner holes 8mm diameter",
    "symmetric bracket 100x40mm 5mm thick 2 bolt holes 8mm filleted corners 2mm",
    "flanged cylinder 80mm OD 50mm bore 40mm tall with 10mm flange ring at base",
    "gear shaft 20mm diameter 150mm long with 8mm bore 20mm deep at one end",
    "stepped shaft 30mm then 20mm sections 60mm each",
    "rectangular housing 100x80mm 60mm tall 4mm wall with 20mm circular boss on top",
    "round plate 150mm diameter 12mm thick with 6 holes 8mm diameter on 120mm bolt circle",
    "square plate 100x100mm 8mm thick with central 30mm boss 20mm tall",
    "tube with flange 60mm OD 4mm wall 150mm long with 90mm flange disc 8mm thick",

    # Tier 4 — four operations / compound shapes
    "hollow cylindrical container 80mm OD 3mm wall 100mm tall open top with 5mm base flange",
    "rectangular box 100x60x40mm 3mm wall with 4 counterbored M5 lid mounting holes",
    "stepped shaft 30mm then 20mm then 15mm sections 60mm each",
    "plate 200x150mm 8mm thick with 6-hole 60mm bolt circle 8mm holes",
    "symmetric H-frame two 40x40mm posts 200mm tall 80mm apart joined by 20mm crossbar",
    "flanged hollow cylinder 100mm OD 70mm bore 50mm tall 130mm flange OD 10mm thick with 4 bolt holes 10mm",
    "bracket 120x80mm 6mm thick with central rectangular pocket 60x40mm 3mm deep",
    "cylindrical cap 50mm OD 3mm wall 30mm tall with 5mm rim flange and filleted outer edges",
    "shaft coupling two 25mm diameter shafts 40mm long joined by 50mm diameter hub 60mm long",
    "rectangular plate 200x150mm 10mm thick with 3 rows of 4 holes 8mm diameter on 40mm grid",

    # Tier 5 — complex / stress test
    "spur gear pair 3:1 ratio module 2 20mm face width on common shaft",
    "tube manifold block 80x50mm 40mm tall with 3 circular ports 20mm diameter on each face",
    "motor mount bracket 100x80mm base 5mm thick 60mm tall rear wall 5mm thick 4 M6 mounting holes",
    "weld-prep I-beam 200UB25.4 500mm with 45 degree flange chamfers both ends",
    "stepped reducer bushing 40mm OD to 25mm OD 30mm long total with 10mm bore through",
    "square frame 200x200mm outer 180x180mm inner 10mm thick 300mm tall four corner fillets 5mm",
    "heat sink base 80x60mm 5mm thick with 8 rectangular fins 2mm thick 15mm tall 8mm pitch",
    "flanged coupling two bolt circles 80mm PCD and 60mm PCD 6 holes each 8mm 15mm thick",
    "pump impeller disc 100mm diameter 15mm thick with 6 radial vane profiles 3mm thick",
    "M16 hex bolt 100mm shank with M16 hex nut for assembly",
]

assert len(SPECS) == 50, f"Expected 50 specs, got {len(SPECS)}"

# ── System prompt (identical to cad_agent_v2 planner, minus the RAG block) ──────

TOOL_SCHEMA_TEXT = """
Available tools and when to use them:

create_document: Always first. Creates Onshape document. params: {name: string}
get_part_studio: Call immediately after create_document. params: {documentId, workspaceId}
create_sketch_circle: Circular profile for cylinders/bosses. params: {center:[x,y] inches, radius: inches, plane: Top|Front|Right, name}
create_sketch_rectangle: Rectangular profile. params: {corner1:[x,y] inches, corner2:[x,y] inches, plane, name}
create_sketch: Multi-entity sketch (circles + lines mixed). params: {entities:[{type,center/corner1/corner2/radius}], plane, name}
create_extrude: Extrude sketch to solid. params: {sketchFeatureId, depth: inches, operationType: NEW|ADD|REMOVE, name}
create_hole: Circular through-hole/bore. params: {sketchFeatureId, depth: inches, name}
create_stepped_extrude: Counterbore — stepped hole. params: {center:[x,y], radii:[r1,r2], depths:[d1,d2], plane, namePrefix}
create_fillet: Round edges. params: {edgeIds:[], radius: inches, name}
create_gear: Spur gear. params: {numTeeth: int, module: mm, pressureAngle, thickness: inches, boreDiameter: inches, centerX: inches, centerY: inches, plane, name}
create_bolt: Hex bolt + threads. params: {size: mm, pitch: mm, shankLength: mm, plane, name}
create_section: Structural steel section by dimensions. params: {d_mm, bf_mm, tf_mm, tw_mm, length_mm, plane, name}
mirror_part: Mirror bodies about a plane. params: {plane: Front|Right|Top, operationType: NEW|ADD, name}
get_mass_properties: Always include as final step. params: {documentId, workspaceId, elementId}

UNIT RULES — CRITICAL:
- All params to create_sketch_circle, create_sketch_rectangle, create_sketch, create_extrude: INCHES
- mm → inches: divide by 25.4. E.g. 80mm = 3.150 inches, 10mm = 0.394 inches
- create_gear: module in mm, all other spatial params in INCHES
- create_bolt, create_section: ALL dimensions in mm

SHAPE TYPES (choose one for shape_type field):
cylinder, round_bar, plate, gear, bolt, i_beam, hollow_rect, container, stepped_shaft
"""

SYSTEM_PROMPT = f"""You are an expert CAD planner generating JSON build plans for an Onshape CAD agent.

{TOOL_SCHEMA_TEXT}

CONSTRAINTS:
1. Steps MUST start with create_document → get_part_studio.
2. Every sketch step must be immediately followed by create_extrude (or create_hole).
3. For a hollow cylinder/container: sketch outer circle → extrude(NEW) → sketch inner circle → extrude(REMOVE).
4. get_mass_properties MUST be the last step.
5. All positions and depths in INCHES (divide mm by 25.4).
6. operationType: NEW for first body, ADD to join, REMOVE to cut.
7. sketchFeatureId in extrude/hole steps: write "AUTO_FROM_CTX" — the executor injects it at runtime.

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences, no explanation:
{{"steps":[{{"tool":"<name>","params":{{...}},"validates_after":true|false}}],"max_calls":20,"timeout_s":300,"shape_type":"<str>","expected_mass_g":<number|null>}}"""


def _load_existing() -> set[str]:
    """Return set of specs already in teacher file."""
    done = set()
    if TEACHER_FILE.exists():
        for line in TEACHER_FILE.read_text().splitlines():
            try:
                row = json.loads(line)
                done.add(row["spec"])
            except Exception:
                pass
    return done


def _validate_plan(plan: dict) -> tuple[bool, str]:
    """Basic structural validation."""
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return False, "steps missing or too short"
    tools = [s.get("tool") for s in steps]
    if tools[0] != "create_document":
        return False, "first step must be create_document"
    if tools[1] != "get_part_studio":
        return False, "second step must be get_part_studio"
    if tools[-1] != "get_mass_properties":
        return False, "last step must be get_mass_properties"
    if not plan.get("shape_type"):
        return False, "missing shape_type"
    return True, ""


def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def generate_plan_api(client, spec: str) -> dict:
    """Use anthropic SDK (direct API key)."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Build plan for: {spec}"}],
    )
    raw = _strip_fences(msg.content[0].text.strip())
    return json.loads(raw)


def generate_plan_cli(spec: str) -> dict:
    """Use the claude CLI subprocess (uses your Claude Code subscription)."""
    prompt = f"{SYSTEM_PROMPT}\n\nBuild plan for: {spec}"
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "text"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI error: {result.stderr.strip()[:200]}")
    raw = _strip_fences(result.stdout.strip())
    return json.loads(raw)


def generate_plan(client, spec: str, dry_run: bool = False, use_cli: bool = False) -> dict | None:
    """Generate a plan for the given spec. Returns parsed plan dict or error dict."""
    if dry_run:
        print(f"  [dry-run] Would call Claude for: {spec}")
        return None
    try:
        if use_cli:
            return generate_plan_cli(spec)
        return generate_plan_api(client, spec)
    except json.JSONDecodeError as e:
        return {"_error": f"JSON parse failed: {e}"}
    except Exception as e:
        return {"_error": str(e)}


def run(args):
    use_cli = args.use_cli
    client = None
    if not use_cli:
        import anthropic
        client = anthropic.Anthropic()
    done = _load_existing()
    specs = args.specs if args.specs else SPECS[args.start - 1:]

    total = 0
    success = 0
    for i, spec in enumerate(specs, start=args.start):
        if spec in done:
            print(f"  [{i:02d}] SKIP (already done): {spec[:60]}")
            continue

        print(f"  [{i:02d}] Generating: {spec[:70]}")
        t0 = time.monotonic()
        plan = generate_plan(client, spec, dry_run=args.dry_run, use_cli=use_cli)
        elapsed = round(time.monotonic() - t0, 1)

        if args.dry_run:
            continue

        total += 1
        if plan is None:
            continue

        if "_error" in plan:
            row = {
                "spec": spec, "error": plan["_error"],
                "raw": plan.get("_raw", ""), "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(TEACH_FAIL_FILE, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"       ERROR: {plan['_error'][:80]}")
            continue

        valid, reason = _validate_plan(plan)
        if not valid:
            row = {
                "spec": spec, "error": f"invalid plan: {reason}",
                "plan": plan, "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(TEACH_FAIL_FILE, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"       INVALID: {reason}")
            continue

        row = {
            "spec":       spec,
            "plan_full":  plan,
            "plan_steps": [s["tool"] for s in plan.get("steps", [])],
            "shape_type": plan.get("shape_type", "unknown"),
            "source":     "claude_teacher",
            "model":      MODEL,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "elapsed_s":  elapsed,
        }
        with open(TEACHER_FILE, "a") as f:
            f.write(json.dumps(row) + "\n")

        steps_preview = " → ".join(s["tool"] for s in plan["steps"])
        print(f"       OK  ({elapsed}s): {steps_preview[:80]}")
        success += 1

        # Rate-limit: 3 req/s is well within Claude free tier
        time.sleep(0.4)

    if not args.dry_run:
        print(f"\nDone: {success}/{total} valid plans written to {TEACHER_FILE}")
        if TEACH_FAIL_FILE.exists():
            fail_lines = len(TEACH_FAIL_FILE.read_text().splitlines())
            if fail_lines:
                print(f"Failures logged: {fail_lines} → {TEACH_FAIL_FILE}")


def _resolve_api_key() -> str | None:
    """Resolve API key from env → openclaw.json → None."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    cfg_path = _OPENCLAW / "openclaw.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            return cfg.get("anthropic_api_key") or cfg.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate CAD teacher data via Claude API")
    parser.add_argument("--start", type=int, default=1,
                        help="Start from spec number N (1-indexed, default=1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts only, no API calls")
    parser.add_argument("--spec", dest="specs", action="append",
                        help="Generate plan for a single spec (can repeat)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--cli", dest="use_cli", action="store_true",
                        help="Use 'claude -p' CLI instead of API key (uses your Claude Code subscription)")
    args = parser.parse_args()

    if not args.dry_run and not args.use_cli:
        key = args.api_key or _resolve_api_key()
        if not key:
            print("No API key found. Use one of:", file=sys.stderr)
            print("  --cli                       (use your Claude Code subscription)", file=sys.stderr)
            print("  --api-key sk-ant-...        (direct API key)", file=sys.stderr)
            print("  ANTHROPIC_API_KEY=sk-ant-.. (env var)", file=sys.stderr)
            sys.exit(1)
        os.environ["ANTHROPIC_API_KEY"] = key

    _OPENCLAW.mkdir(exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
