---
name: cad-builder
description: "Build Onshape CAD models from natural language specs. Two-stage pipeline: qwen2.5:14b planner → qwen2.5:14b executor with per-step validation, repair pass, and Telegram feedback. Handles: gears/assemblies, structural sections, airfoils, bolts, freeform FeatureScript shapes. Use when: user sends /cad <spec>, asks to build a part, modify a previous build, or check a CAD model."
metadata:
  {
    "openclaw":
      {
        "emoji": "🔩",
        "requires": { "bins": ["python3"] },
      },
  }
---

# CAD Builder Skill

Build and iterate on Onshape CAD models from text specs. All inference runs locally via Ollama (qwen2.5-coder:7b). No Claude API calls.

## When to Use

✅ **USE this skill when:**
- `/cad <spec>` — build a new model
- "make it longer/wider/heavier", "add mitre cuts", "resize to 250UC72.9" — modify the last build
- "is the CAD model correct?", "verify the last build" — check for errors
- "what did we just build?", "show session" — recall last build state

❌ **DON'T use this skill when:**
- User wants to open/view Onshape in browser (just give them the URL)
- User wants to delete a document (Onshape UI only)

## Scripts

```
V2=~/.openclaw/skills/cad-builder/cad_agent_v2.py      # two-stage pipeline (default)
V1=~/.openclaw/skills/cad-builder/onshape_cad_agent.py  # legacy fallback
```

Credentials loaded from `~/.openclaw/openclaw.json` env section.
Log: `~/.openclaw/cad-agent.log`
Feedback: `~/.openclaw/cad-feedback.jsonl`

## Workflow: New Build (v2)

```bash
# Dry-run planner only (no Onshape API calls)
python3 $V2 plan "spur gear 20 teeth module 2 meshing pinion 3:1 ratio"

# Full build — planner → executor → repair pass → Telegram feedback
python3 $V2 build "W200x100 I-beam 2000mm long"
python3 $V2 build "spur gear 20 teeth module 2" --chat-id 7788781234

# Stress test (gear assembly)
python3 $V2 test
```

### Pipeline stages
1. **Planner** (qwen2.5:14b): NL → `{steps:[{tool,params,validates_after}], max_calls, timeout_s}`
2. **Executor** (qwen2.5:14b): runs each step, validates after geometry steps, retries once with LLM-adjusted params on failure
3. **Repair pass**: reads back mass + bounding box, rebuilds if >50% deviation from expected
4. **Telegram feedback**: inline 1-5★ keyboard after every build; 4-5★ saved to feedback.jsonl for future few-shot context; 1-2★ prompts text description

### Fallback triggers (→ v1 agent)
- Planner outputs invalid JSON
- >2 consecutive tool failures
- Any tool call exceeds 10s wall-clock budget

### v1 build workflow (legacy)
```bash
python3 $V1 build "W200x100 I-beam 2000mm long"
python3 $V1 verify <DID> <WID> <EID>
```

## Workflow: Follow-up / Modification

```bash
# Check last session (was anything built recently?)
python3 $SCRIPT session
# Output: JSON with spec, parsed params, did, wid, eid, url, built_at

# Rebuild the same document with modified spec
python3 $SCRIPT rebuild <DID> <WID> <EID> "W200x100 I-beam 2500mm long"
# Clears all features and rebuilds in-place — same URL
```

## Workflow: Self-Correction

When `verify` returns `feature_errors` or `ok=false`:
1. Read `features` to see the full feature list and which ones errored
2. Determine which parameter caused the error (e.g. wrong sketch plane, bad dimensions)
3. `rebuild` with corrected parameters
4. `verify` again to confirm fix

```bash
# Detailed feature list
python3 $SCRIPT features <DID> <WID> <EID>
# Output: JSON list of {featureId, name, type, suppressed, notices[]}
```

## Supported Shapes — BTM Path (fast, deterministic)

| Shape | Key | Required params |
|---|---|---|
| Universal/wide-flange beam | `i_beam` | height_mm, flange_width_mm, flange_thickness_mm, web_thickness_mm, length_mm |
| C/U channel | `c_channel` | same as i_beam |
| Hollow rectangular section | `hollow_rect` | width_mm, height_mm, wall_thickness_mm, length_mm |
| Flat plate | `plate` | width_mm, height_mm, length_mm |
| Round bar | `round_bar` | diameter_mm, length_mm |
| NACA airfoil | `naca_airfoil` | naca_digits (e.g. "2412"), chord_mm, span_mm |
| ISO metric hex bolt | `bolt` | designation (e.g. "M18"), length_mm |

Optional: `root_radius_mm` (i_beam fillet), `mitre_cuts: true` (45° mitre both ends).
Bolt designations M3–M48: head dimensions auto-looked up from ISO 4014/4017 table.

## Supported Shapes — FeatureScript Path (arbitrary geometry)

Any shape not in the BTM list routes to the FeatureScript pipeline:
- **Revolved parts**: knobs, cups, discs, stepped shafts (`revolve.fs` example)
- **Lofted shapes**: spoons, forks, tapered handles, organic forms (`loft.fs` example)
- **Helix/threads**: threaded fasteners, springs, worm gears (`helix-thread.fs` example)
- **Swept profiles**: pipes, bent rods, handles, hooks (`sweep.fs` example)

FeatureScript workflow:
```bash
# Force FeatureScript path (bypasses shape detection)
python3 $SCRIPT fs "a wooden spoon with 80mm oval bowl and 180mm handle"

# Preview generated code without uploading
python3 $SCRIPT fs-preview "M18 bolt with threaded shank"
```

## Standard Section Designations

The LLM knows these — just pass the designation string:
- `W200x100` → h=210, bf=206, tf=14.5, tw=9.0
- `250UC72.9` → h=250, bf=254, tf=14.2, tw=8.6
- `UB203x102x23` → h=203, bf=102, tf=9.3, tw=5.4

## Replying to the User

After a successful build, reply with:
1. The Onshape URL
2. Key dimensions (from parsed params)
3. Mass and volume (from verify output)
4. Any warnings (from feature_errors)

If errors occurred and self-correction failed, explain what went wrong and ask for clarification.

## Session Continuity

The session file at `~/.openclaw/cad-session.json` persists the last build. Check it at the start of every `/cad` interaction to detect follow-ups vs new builds.

A message is a **follow-up** if:
- It doesn't describe a brand new part
- It references a previous build ("make it", "change the", "add to it")
- A session file exists and was written recently

A message is a **new build** if:
- It contains a full part specification
- User says "new" or "start fresh"
- No session file exists
