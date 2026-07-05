# HANDOFF — CAD Agent session continuation (written 2026-07-06)

Read this + `ROADMAP.md` + memory (`project_cad_phase1_roadmap`) before touching anything.
Everything below is committed; `git log --oneline -25` in this repo tells the story.

## Where the project stands

| Milestone | State | Proof |
|---|---|---|
| M0 reliability pass (9 commits) | done | `benchmarks/results/baseline_*.json` |
| M1 schema outputs + contracts + E1 params/regen | done | `m1_7b_tiers12.json` 5/6, 16/22; zero schema fallbacks |
| M2 failure taxonomy + retrieval hygiene | done | lift A/B: 5/6 with retrieval vs 3/6 without (`m2-lift-summary.txt`) |
| M3 cloud rung (Anthropic/OpenRouter) | built, **dormant** | offline-verified; NO live calls — user has no API credit; enable = `cloud` block in `~/.openclaw/cad.json` |
| M4 question-critique + section-axis | done (as M4.1) | regression postmortem → 2 gate fixes → `m41_7b_tiers12.json` 4/6, zero critic false-blocks |
| M5 B7 v5 migration | done | benchmark defaults to v5 `--json`; Satine parses one JSON line; upload single-sourced |
| M5 E2 FreeCAD target | **done v1, live-verified** | calibration block → parametric `.FCStd`, 7 spreadsheet params, ΔV 0.71% |
| M6 fine-tune (QLoRA 7B) | next big item | needs user go-ahead (rented GPU cost); failure histograms accumulating since M2 |

Honest current 7B level on tiers 1–2: **~4/6** (M1's 5/6 included a lenient brief roll; the
gate is stricter now). Coder ladder: 7B → 30B (14B measured out twice — see PROJECT.md).

## In flight right now

- **`cad-30b-showcase.service`** — full 10-benchmark run, `--coder strong`, regenerating all
  30B models for the gallery (killed once by the Jul-5 reboot, relaunched Jul-6 ~09:00).
  Log: `benchmarks/results/showcase-30b.log`, ends with `=== ALL DONE ===` (~2h).
  **When done:** archive the run json (`showcase_30b_full.json`), add a gallery section, republish.

## Open work, in order

1. **Gallery rebuild after the showcase run** — generator: scratchpad is wiped (reboot!), the
   generator lives only in this handoff's history… REGENERATE it: see "gallery generator" note
   below. Sections = every archived run json; artifacts now persist under
   `benchmarks/results/artifacts/<run_tag>/` and `~/.openclaw/cad-builds/` (retention 200).
   User wants it EXTENSIVE — every model ever made, including 30B tier-3 parts.
2. **E2 v2** — extend `cad_v5/freecad_export.py` grammar: cylinder-based parts (flanges,
   shafts), blind holes/pockets, fillet/chamfer as real FreeCAD features. Same verify-by-reexport rule.
3. **M6 fine-tune** — recipe in ROADMAP Track D. Ask the user before spending.
4. Horizon: assemblies validation, image-conditioned builds (Satine already stashes photos in
   sessions), photo→slicer advisor.

## Hard-won operational rules (violating these cost us real time)

- **Long jobs = detached systemd units** (`systemd-run --user --unit=<name> …` + log file +
  `=== ALL DONE ===` sentinel). Harness background tasks DIE with the session. Add a detached
  post-processor when results need archiving.
- **Never edit engine files while a benchmark runs** — it shells the agent per build. Develop
  in a `git worktree` and merge after. **Worktrees go in `.worktrees/` (repo-local), NEVER
  /tmp** — the Jul-5 reboot wiped /tmp and nearly lost E2.
- **One model in VRAM** (`OLLAMA_MAX_LOADED_MODELS=1`): no LLM/embedding calls while a
  benchmark runs; nomic-embed evictions count too.
- **No claimed improvement without a measured before/after** on the honest scorer; failed
  builds score 0/N. `converged` is the headline number.
- **cad.* config lives in `~/.openclaw/cad.json`** — a top-level `cad` key in openclaw.json
  makes the OpenClaw gateway REFUSE TO START (strict schema).
- **FreeCAD AppImage** (`~/Applications/FreeCAD_1.1.1*.AppImage`): console mode is
  `AppImage -c script.py` + `stdin=DEVNULL`; scripts must end `sys.exit(0)` (console never
  exits) and success prints need `flush=True` (hard exit swallows buffers).
- Telegram: Satine owns the cad bot token; `accounts.cad.enabled=false` in openclaw.json keeps
  the node gateway off it. After editing `~/.openclaw/cad-telegram.py`: restart
  `cad-telegram.service`. Engine edits need no restart (shelled per request).

## Quick reference

```bash
# health
systemctl --user status cad-telegram openclaw-gateway hermes-gateway
# benchmark (honest scorer, v5 entry default)
python3 scripts/run_benchmarks.py --tiers 1,2 --coder fast
# build via CLI (viewer target default)          # editable dims, no LLM
cad "a 100x60x30 enclosure with 2mm walls"  ·  cad params  ·  cad regen wall=3
# FreeCAD parametric export of the last build
cad --target freecad "<spec>"    # or: python3 -c "from cad_v5.freecad_export import convert; ..."
# viewer (if down after reboot): any v5 build restarts it, port 4178
```

## Gallery generator (recreate at scratchpad, it was /tmp-wiped)

`make_gallery.py` pattern: for each archived run json in `benchmarks/results/`, one section —
stats row (converged/acceptance/time) + one card per benchmark row (render PNG base64-embedded
from `artifacts/<run_tag>/<id>.png` or the matching `~/.openclaw/cad-builds/<ts>-<slug>/build.png`,
status chip, acceptance, wall time, file paths). Cards without surviving renders say so honestly.
Publish via the Artifact tool to the EXISTING gallery URL (f65369c2-…, favicon 🖼️); bump the
visible "updated" label in the eyebrow — the user checks that to confirm freshness.

## The four shareable documents (Artifact URLs)

- Roadmap: https://claude.ai/code/artifact/de34202b-4073-41df-bb83-63de43668610
- History v1→now: https://claude.ai/code/artifact/f46f3f6f-5b1f-46e2-b2c9-7b6ec1c47b6e
- User guide: https://claude.ai/code/artifact/d7674e5c-bbf3-4fd0-8402-ae1d1a7fcfb6
- Model gallery: https://claude.ai/code/artifact/f65369c2-4c79-49f5-9a5b-de2a97d37da1
(Artifacts are private by default — the user must flip the share toggle on each page before
emailing the links.)

## Artifact republishing note (added post-reboot)

The artifact HTML sources lived in the /tmp scratchpad and were wiped by the Jul-5 reboot.
The published pages are fine, but to update them: rebuild the page from repo truth
(`ROADMAP.md` for the roadmap; regenerate the gallery per the spec above; history/user-guide
only if content changed), then publish with the Artifact tool's `url` parameter pointing at
the EXISTING URL so it redeploys in place. Known staleness right now: the roadmap page (Rev G)
shows M5 as "half done" — E2 completed 2026-07-06, so bump it to done + Rev H on next publish.
Keep favicons stable: 🔩 roadmap, 📜 history, 📖 guide, 🖼️ gallery.
