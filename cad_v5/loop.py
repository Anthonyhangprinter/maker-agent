"""The v5 unified flow: describe → build → ALWAYS refine.

One loop, no build/chat split. Every build completes, opens in the output target (local CAD Viewer
by default), prints a one-line geometry summary + the Δ from the previous build, then waits for
plain-English feedback. Feedback merges into the spec and rebuilds. Typing `done` after the first
build is the "one-shot" case. Refinement re-enters the same validated build loop — not a fragile
patch — so every round is gated and critiqued exactly like the first.
"""
import shutil
from pathlib import Path

from . import config, targets, engine
from .config import log, STEP_OUT, STL_OUT, DXF_OUT, _OPENCLAW

_PREV_STEP = _OPENCLAW / "cad-prev-build.step"   # kept so each refine can be diffed against the last

_PROMPT = "\nFeedback?  (describe a change · undo · done · rate N · onshape · show) > "


def _summary(facts: dict) -> str:
    bits = []
    if facts.get("bbox"):
        bits.append("×".join(f"{d:.0f}" for d in facts["bbox"]) + "mm")
    s = facts.get("solids")
    if isinstance(s, int):
        bits.append(f"{s} solid" + ("" if s == 1 else "s"))
    if facts.get("bores"):
        uniq = sorted({round(d, 1) for d in facts["bores"]})[:4]
        bits.append("Ø" + "/".join(f"{d:g}" for d in uniq))
    if facts.get("walls"):
        bits.append(f"walls {min(facts['walls']):.1f}mm")
    return " · ".join(bits) or "built"


def _facts_for(step: Path) -> dict:
    try:
        return engine.parse_facts(engine.run_inspect(step)["output"]) if step.exists() else {}
    except Exception as e:
        log.warning("[v5] inspect for Δ summary failed on %s: %s", step, e)
        return {}


def _name_of(result: dict, spec: str) -> str:
    return (result.get("brief", {}) or {}).get("name") or spec[:40]


def run(spec: str | None = None, coder: str = "auto", target_name: str | None = None,
        use_fewshots: bool = True, interactive: bool = True) -> dict | None:
    """Build `spec`, show it in the target, then refine interactively until the user is done."""
    target_name = target_name or targets.default_target_name()
    if not spec:
        try:
            spec = input("Describe the part to build > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not spec:
            print("Nothing to build.")
            return None

    history: list[dict] = []
    last: dict | None = None
    have_prev = False
    prev_spec: str | None = None   # spec that produced _PREV_STEP — restored by `undo`

    while True:
        print(f"\n… building ({coder} coder: brief → code → check → view) …")
        try:
            result = engine.build(spec, coder=coder, use_fewshots=use_fewshots,
                                  do_upload=False, final_render=False)
        except Exception as e:
            print(f"\n✗ Build failed: {e}")
            if not interactive:
                return last
            try:
                fb = input("\nRetry with a tweak? (describe a change · done) > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if fb.lower() in ("", "done", "q", "quit", "exit"):
                break
            spec = engine.merge_spec(spec, fb, history)
            history.append({"role": "user", "content": fb})
            continue

        last = result
        step = Path(result.get("step_local") or STEP_OUT)
        facts = _facts_for(step)

        # Δ from the previous build (deterministic confirmation of what the refine changed)
        delta = ""
        if have_prev and _PREV_STEP.exists():
            d = engine.run_diff(_PREV_STEP, step)
            if d:
                delta = "  Δ vs previous:\n    " + d.replace("\n", "\n    ")

        # Route to the output target (default: local CAD Viewer, auto-opened). The target's
        # outcome belongs IN the result — machine consumers (--json) need the URL or the error.
        try:
            view = targets.resolve_target(target_name)(step, _name_of(result, spec))
        except Exception as e:
            log.warning("[v5] output target %s failed: %s", target_name, e)
            view = {"url": "", "target": target_name, "target_error": str(e)[:300]}
        result["target"] = target_name
        if view.get("url"):
            result["url"] = view["url"]
        if view.get("target_error"):
            result["target_error"] = view["target_error"]

        print(f"\n✓ Built.  {_summary(facts)}")
        if view.get("url"):
            print(f"  Viewer: {view['url']}")
        else:
            print(f"  Saved:  {step}")
        outs = [p.name for p in (STEP_OUT, STL_OUT, DXF_OUT) if p.exists()]
        if outs:
            print(f"  Files:  {', '.join(outs)}  (in {_OPENCLAW})")
        if not result.get("converged", True):
            print("  ⚠ not fully converged — last critique: "
                  f"{(result.get('last_critique') or '')[:160]}")
        if delta:
            print(delta)

        if not interactive:
            return last

        # Inner prompt loop: show / rate / onshape just re-prompt (no rebuild); only a real change
        # (or done) leaves it. This is what stops a non-edit command from triggering a fresh build.
        change: str | None = None
        while True:
            try:
                fb = input(_PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                fb = "done"
            low = fb.lower()

            if low in ("", "done", "q", "quit", "exit"):
                break
            if low.split()[0] == "rate":
                toks = fb.split()
                try:
                    engine.store_feedback(result, int(toks[1]), " ".join(toks[2:]))
                    print(f"  ✓ rated {toks[1]}★"
                          + (" (stored as an example)" if int(toks[1]) >= 4 else ""))
                except Exception as e:
                    print(f"  (rate failed: {e} — use 'rate 5 optional note')")
                continue
            if low in ("onshape", "upload"):
                try:
                    r = targets.onshape_target(step, _name_of(result, spec),
                                               public=config.public_uploads())
                    print(f"  ✓ Onshape: {r['url']}")
                except Exception as e:
                    print(f"  Onshape upload failed: {e}")
                continue
            if low in ("show", "view"):
                print(f"  {view.get('url') or step}")
                continue
            if low == "undo":
                # Instant revert to the previous build — no rebuild, no LLM.
                if not (have_prev and _PREV_STEP.exists() and prev_spec is not None):
                    print("  (nothing to undo — this is the first build)")
                    continue
                spec = prev_spec
                if history:
                    history.pop()
                step = _PREV_STEP
                facts = _facts_for(step)
                try:
                    shutil.copy(step, STEP_OUT)               # keep last-build convention honest
                    engine.run_stl(STEP_OUT, STL_OUT)         # refresh the sliceable mesh too
                except Exception as e:
                    log.warning("[v5] undo artifact refresh: %s", e)
                view = targets.resolve_target(target_name)(step, _name_of(result, spec))
                print(f"  ↩ reverted to previous build.  {_summary(facts)}")
                if view.get("url"):
                    print(f"  Viewer: {view['url']}")
                have_prev = False   # only one step of history is kept
                continue
            # Anything else is a description of a change → leave the prompt loop and rebuild.
            change = fb
            break

        if change is None:   # done / EOF
            break

        # Snapshot this build (geometry + spec) for the next diff and for `undo`, then
        # merge the feedback and rebuild.
        try:
            if step != _PREV_STEP:
                shutil.copy(step, _PREV_STEP)
            prev_spec = spec
            have_prev = True
        except Exception as e:
            log.warning("[v5] prev-build snapshot failed: %s", e)
            have_prev = False
        spec = engine.merge_spec(spec, change, history)
        history.append({"role": "user", "content": change})

    print(f"\nSaved. STEP: {STEP_OUT}  (STL/DXF alongside in {_OPENCLAW})")
    return last
