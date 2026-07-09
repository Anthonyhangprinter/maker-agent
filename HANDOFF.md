# HANDOFF — CAD Agent session continuation

**This is the entry point for a fresh session.** It holds current state, the operating protocol,
and the ordered next steps. For *why* the system is built this way read `PROJECT.md` (architecture +
history); for *how to drive it* read `SKILL.md` (operator commands); for *the plan* read `ROADMAP.md`.
Repo root: `/home/theultimatecunt/.openclaw/skills/cad-builder` (a git repo). Everything below is
committed unless flagged; `git log --oneline -25` tells the story.

---

## SESSION PROTOCOL — do these in order, every session

1. **Read this whole file first**, then skim `ROADMAP.md` and the memory note
   `project_cad_phase1_roadmap`. Do not start editing before you know current state.

2. **Run the health checks** and read the output before touching anything:
   ```bash
   systemctl --user status cad-telegram openclaw-gateway hermes-gateway --no-pager
   ollama list | grep -E 'qwen3:8b|qwen2.5-coder:7b|qwen3-coder:30b|gemma4:e4b|nomic-embed'
   cd /home/theultimatecunt/.openclaw/skills/cad-builder && git status --short
   ```
   Expected: `cad-telegram.service` active (Satine); the five models present; a clean or
   known-dirty tree. `openclaw-gateway`/`hermes-gateway` may be active — leave them alone.

3. **NEVER edit engine files while a benchmark/showcase unit is active.** The runner shells the
   agent per build; editing `cad_agent_v4.py` or `cad_v5/*` mid-run corrupts the measurement. Check
   before every engine edit:
   ```bash
   systemctl --user list-units --type=service --state=running | grep -E 'cad-.*(bench|showcase)'
   ```
   Empty output = safe to edit. Non-empty = wait, or develop in a worktree (step 5).

4. **Long jobs run ONLY as detached systemd units** with a log file and a sentinel — harness
   background tasks DIE with the session. Template:
   ```bash
   systemd-run --user --unit=cad-mytask \
     --working-directory=/home/theultimatecunt/.openclaw/skills/cad-builder \
     bash -c 'python3 scripts/run_benchmarks.py --tiers 1,2 --coder fast \
       > benchmarks/results/mytask.log 2>&1; echo "=== ALL DONE ===" >> benchmarks/results/mytask.log'
   # watch:   tail -f benchmarks/results/mytask.log
   # status:  systemctl --user status cad-mytask --no-pager
   ```
   A run is finished only when its log ends with `=== ALL DONE ===`. Add a detached
   post-processor if results need archiving after the run.

5. **Develop in repo-local worktrees, never `/tmp`.** The Jul-5 reboot wiped `/tmp` and nearly lost
   E2. Use `.worktrees/` (repo-local, gitignored):
   ```bash
   git worktree add .worktrees/<feature> -b <feature>
   # …work, benchmark, then merge back and: git worktree remove .worktrees/<feature>
   ```

6. **No claimed improvement without a measured before/after** on the honest scorer. Failed builds
   score 0/N; `converged` is the headline number. Benchmark against `benchmarks/results/baseline_*.json`
   and commit the new result json. If a feature shows no benchmark lift, remove it — don't leave it dormant.

7. **Never make cloud LLM calls.** The user has NO API credit. The cloud rung (M3, built) stays
   dormant: do not set a `cloud` block in `~/.openclaw/cad.json`, do not pass `--coder cloud`. All
   inference is local via Ollama.

8. **Commit each coherent change** with a trailing
   `Co-Authored-By: Claude Opus <noreply@anthropic.com>` line. Commit or push only what the task
   asks for.

Two more hard-won rules (violating them cost real time):
- **One model in VRAM** (`OLLAMA_MAX_LOADED_MODELS=1`): no LLM/embedding calls (nomic-embed
  included) while a benchmark runs — evictions wreck timings.
- **`cad.*` config lives in `~/.openclaw/cad.json`, NOT `openclaw.json`.** A top-level `cad` key in
  `openclaw.json` makes the OpenClaw gateway REFUSE TO START (strict schema).
- **FreeCAD is the AppImage** at `~/Applications/FreeCAD_1.1.1-Linux-x86_64-py311.AppImage` (no
  `freecadcmd` on PATH). Console mode: `AppImage -c script.py` with `stdin=DEVNULL`; scripts must end
  `sys.exit(0)` (console never exits on its own) and success prints need `flush=True` (hard exit
  swallows buffers). `cad_v5/freecad_export.py::_freecad_invoke()` finds the AppImage automatically.
- **Satine restart rule:** editing `~/.openclaw/cad-telegram.py` needs
  `systemctl --user restart cad-telegram`. Engine edits (`cad_agent_v4.py`, `cad_v5/*`) do NOT — Satine
  shells the agent per request.

---

## Where the project stands (measured)

| Milestone | State | Proof |
|---|---|---|
| M0 reliability pass (9 commits) | done | `benchmarks/results/baseline_*.json` |
| M1 schema outputs + contracts + E1 params/regen | done | `m1_7b_tiers12.json`; zero schema fallbacks |
| M2 failure taxonomy + retrieval hygiene | done | lift A/B: 5/6 with retrieval vs 3/6 without (`m2-lift-summary.txt`) |
| M3 cloud rung | **built, DORMANT** | offline-verified; NO live calls (no API credit); enable = `cloud` block in `~/.openclaw/cad.json` — do not |
| M4 question-critique + section-axis | done (as M4.1) | `m41_7b_tiers12.json` 4/6 conv, 13/22 acc, zero critic false-blocks |
| M5 B7 v5 migration | done | benchmark defaults to v5 `--json`; Satine parses one JSON line; upload single-sourced in `cad_v5/targets.py` |
| M5 E2 FreeCAD target | done v1, live-verified | box-grammar STEP → parametric `.FCStd` (Params spreadsheet + Cut tree), verified by STEP re-export |
| M6 fine-tune (QLoRA 7B) | next big item | needs user go-ahead (rented GPU cost) |

**Coder ladder: 7B → 30B (2 rungs).** The 14B was measured OUT twice (`m1_14b_tiers12.json`: 3/6
at 583–804s — slower than the 30B MoE, weaker than the 7B); `--coder mid` stays manual-only.
Honest 7B level on tiers 1–2: **4/6 converged, 13/22 acceptance** (`m41_7b_tiers12.json`).

Latest full 30B showcase run (`--coder strong`, current stricter engine): **6/10 converged, 9/10
geometry, 22/31 acceptance (71%)** — `benchmarks/results/showcase_30b_full.json` (log
`showcase-30b.log`, gitignored; artifacts `benchmarks/results/artifacts/20260706_085550_strong/`).
The 6/10 vs the older 7/10 baseline reflects a stricter honest gate, not a regression.

---

## Open work, in order

1. **Gallery rebuild after the showcase run** — the published model gallery (Artifact URL below) is
   stale. Regenerate it from repo truth per the "gallery generator" spec below: one section per
   archived run json in `benchmarks/results/` (now including `showcase_30b_full.json`), a card per
   benchmark row. Renders may not survive per run — cards without a surviving PNG say so honestly.
   User wants it EXTENSIVE (every model ever made, incl. 30B tier-3 parts).
2. **E2 v2** — extend `cad_v5/freecad_export.py` grammar beyond box parts: cylinder-based parts
   (flanges, shafts), blind holes/pockets, fillet/chamfer as real FreeCAD features. Keep the
   verify-by-STEP-reexport rule (`freecad_export.convert` diffs volume/bbox vs the agent's own STEP).
3. **M6 fine-tune** — recipe in `ROADMAP.md` Track D. Ask the user before spending on a rented GPU.
4. **Horizon** — assemblies end-to-end validation, image-conditioned builds (Satine already stashes
   reference photos in sessions), photo→slicer advisor. See `ROADMAP.md` §7.

---

## Quick reference

```bash
cd /home/theultimatecunt/.openclaw/skills/cad-builder
# build via v5 CLI (local CAD Viewer target default; editable params, no LLM for regen)
cad "a 100x60x30 enclosure with 2mm walls"    # then refine by chatting; type: done
cad params                                     # list editable dims of the last build
cad regen wall=3                               # rebuild with new value (seconds, no LLM)
cad --target freecad "<spec>"                  # parametric .FCStd (box-grammar parts)
# honest benchmark (v5 --json entry is the default agent)
python3 scripts/run_benchmarks.py --tiers 1,2 --coder fast
# viewer down after a reboot? any v5 build restarts it on port 4178
```

---

## Gallery generator (recreate at the scratchpad — the source was /tmp-wiped)

`make_gallery.py` pattern: for each archived run json in `benchmarks/results/`, one section —
stats row (converged / acceptance / time) + one card per benchmark row (render PNG base64-embedded
from `artifacts/<run_tag>/<id>.png` or the matching `~/.openclaw/cad-builds/<ts>-<slug>/build.png`,
plus status chip, acceptance, wall time, file paths). Cards without a surviving render say so
honestly. Publish via the Artifact tool to the EXISTING gallery URL (favicon 🖼️); bump the visible
"updated" label in the eyebrow — the user checks that to confirm freshness.

## The four shareable documents (Artifact URLs)

- Roadmap: https://claude.ai/code/artifact/de34202b-4073-41df-bb83-63de43668610 (favicon 🔩)
- History v1→now: https://claude.ai/code/artifact/f46f3f6f-5b1f-46e2-b2c9-7b6ec1c47b6e (favicon 📜)
- User guide: https://claude.ai/code/artifact/d7674e5c-bbf3-4fd0-8402-ae1d1a7fcfb6 (favicon 📖)
- Model gallery: https://claude.ai/code/artifact/f65369c2-4c79-49f5-9a5b-de2a97d37da1 (favicon 🖼️)

Artifacts are private by default — the user flips the share toggle per page before emailing links.
The HTML sources lived in the /tmp scratchpad and were wiped by the Jul-5 reboot; to update a page,
rebuild it from repo truth (`ROADMAP.md` for the roadmap; regenerate the gallery per the spec above),
then publish with the Artifact tool's `url` parameter pointing at the EXISTING URL so it redeploys in
place. Known staleness: the roadmap page (Rev G) shows M5 as half-done — E2 completed 2026-07-06, so
bump it to done on the next publish. Keep favicons stable across redeploys.
