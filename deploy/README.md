# deploy/

`maker-server` is the launcher for the swappable Maker Agent CAD coder llama-server (reads `~/.openclaw/maker.env`); `maker-server.service` is its systemd user unit, `Conflicts=qwen38-server.service` so the resident and the maker arm never share the single RTX 3090 at once.

Install: `install -m 755 deploy/maker-server ~/.local/bin/maker-server && install -m 644 deploy/maker-server.service ~/.config/systemd/user/maker-server.service && systemctl --user daemon-reload`

`RuntimeMaxSec=12h` is the dead-man switch: an abandoned arm (a card run killed mid-loop, a hand swap nobody restored) would otherwise hold the whole GPU indefinitely and the resident would never come back. The unit self-terminates after 12 hours, which is well clear of any real run, and the health watcher below then restores the resident within 5 minutes because no maker-server is active any more. It is not a substitute for `python3 scripts/arms.py restore`, just the floor under forgetting it.

The family health watcher (`~/family-ai-server/bin/health-watch.sh`, timer every 5 min) skips its resident-restart remedy while `maker-server.service` is active, so a card run (or any deliberate arm swap) is not fought and restarted mid-run; see that script's `check_qwen38`.

## critic-server (optional, :8092)

`critic-server` is a second, independent llama-server for a vision critic (`~/.openclaw/critic.env`, keys `MODEL`/`MMPROJ`/`CTX`/`PORT`/`ALIAS`/`EXTRA_ARGS`, same single-quoted-`source`d shape as `maker.env`). Candidates live in `benchmarks/arms.json`'s `critics` list: `minicpm-v-4.6` (cheap, ~3.5GB conservative) and `gemma-4-12b` (non-Qwen judge candidate, ~9.5GB conservative). Unlike `maker-server`, this launcher never evicts an Ollama guest and its unit carries no `Conflicts=`: it is meant to run alongside the maker or the resident, not instead of them.

Whether it fits is a real question on one 24GB card: the maker running `gemma-4-31b` at ctx 16384 alone measures 21,435 MiB used of 24,576, leaving under 3.2GB free, so a critic cannot coexist with that arm at that context. `scripts/arms.py critic use <name>` is the gate for this: it reads each critic's `vram_gb` field, parses `nvidia-smi --query-gpu=memory.free` for the real free MiB, and refuses with exit code 2 and a message naming the free MiB whenever free VRAM is below `vram_gb + 0.5`. Only when it proceeds does it write `critic.env`, restart `critic-server`, wait for `/health`, and print `CAD_CRITIC_MODEL=local:<alias>` and `CAD_CRITIC_URL=http://127.0.0.1:8092/v1/chat/completions` for the card to consume (`run_card.py --critic-url ...` or the engine's own `CAD_CRITIC_URL`). `scripts/arms.py critic off` stops the unit; `scripts/arms.py critic list` shows what is on disk.

Install: `install -m 755 deploy/critic-server ~/.local/bin/critic-server && install -m 644 deploy/critic-server.service ~/.config/systemd/user/critic-server.service && systemctl --user daemon-reload`

## lab-harvest (Phase 3 data engine, NOT installed/enabled by this change)

`lab/harvest_unit.sh` is the timer's launcher: a non-blocking `flock` on
`lab/state/harvest.lock` (a still-running unit makes the next tick skip, not queue), then
(fix round 1, H4) `python3 lab/harvest.py --check-gate` -- the SAME paused/night-window/
budget/gpu-proxy/bank-exhausted checks `--unit` re-runs once inside the window, plus a
non-blocking probe of the CAD build lock, all WITHOUT starting a GPU window -- and only on
a `--check-gate` exit 0 does it `exec lab/gpu_window.sh python3 lab/harvest.py --unit`. It
calls `gpu_window.sh` unmodified, the one thing on this box that ever HOLDS
`~/.openclaw/cad-build.lock`. No second locking scheme: `--check-gate`'s own lock probe
only opens and immediately releases the lock file, it never holds it past the probe.

**M2, closed by Task 3a (2026-09-19):** `--check-gate`'s lock probe is instant and never
waits, and the real arbiter is still `lab/gpu_window.sh`'s own `flock`, but that wait is
now a knob: `harvest_unit.sh` sets `GPU_WINDOW_LOCK_WAIT_SEC=120` before the `exec`, so a
tick that would merely queue behind an interactive CAD build gives up after two minutes and
exits 75 (`EX_TEMPFAIL`, declared in `SuccessExitStatus=`), which systemd logs as a skip
rather than a failure. The next tick is 30 minutes away.

`lab-harvest.service` is `Type=oneshot` with `KillMode=mixed` + `TimeoutStopSec=400`, so a
`systemctl --user stop lab-harvest.service` sends ONE SIGTERM to the unit's MAIN process
(the shell script, which has exec'd into `gpu_window.sh`), with a cgroup-wide SIGKILL only
once `TimeoutStopSec` expires. `gpu_window.sh` then handles it the same way it handles a
manual `kill <pid>`: TERM the job's whole process group, KILL that group after
`GPU_WINDOW_GRACE_SEC` (180s) if it ignores that, then restore the resident, in that order.
`TimeoutStopSec=400` covers that path (180s grace + up to about 90s of restore) with margin.
The dead-man cap is the window's own `GPU_WINDOW_MAX_SEC=2400`, set by `harvest_unit.sh`:
systemd's `RuntimeMaxSec=` has no effect on `Type=oneshot` units. `TimeoutStartSec=2800` is
the systemd-side belt above it, and `ExecStopPost=` starts the resident even when the wrapper
itself was SIGKILLed and its restore never ran.

`lab-harvest.timer` fires every 30 minutes but ONLY inside 22:00-07:00 (fix round 1, H4:
two `OnCalendar=` lines, `22..23:00/30` and `00..06:00/30`, matching `cad.json`'s
`lab.harvest.night_start`/`night_end` defaults exactly). **The timer and
`lab.harvest.night_window` must be changed together** -- widening or narrowing the config
window without editing these two lines leaves the timer either firing (and immediately
no-oping via `--check-gate`) outside the configured hours, or missing hours inside it.
Even within the window, `--check-gate`/`--unit` still decide whether a given tick actually
does anything (paused file, today's GPU-hour budget, the gpu-proxy's queue, the bank being
exhausted -- see `lab/harvest.py`'s own module docstring), so a tick landing inside the
window is not a guarantee of real work, just of being worth checking.

Install (paths below assume `maker-1.0/phase3` has been merged into the main
`~/.openclaw/skills/cad-builder` checkout -- NOT the `cad-builder-phase3` worktree these
files were authored in):

```
install -m 755 lab/harvest_unit.sh ~/.openclaw/skills/cad-builder/lab/harvest_unit.sh
install -m 644 deploy/lab-harvest.service ~/.config/systemd/user/lab-harvest.service
install -m 644 deploy/lab-harvest.timer ~/.config/systemd/user/lab-harvest.timer
systemctl --user daemon-reload
systemctl --user enable --now lab-harvest.timer
```

Do not enable the timer before at least one manual `--once --spec-id` smoke and one real
`--unit` run have been reviewed -- see `docs/plans/2026-09-19-phase3-data-engine.md`
Task 3's smoke step. `lab/state/paused` (any content, even empty) pauses the timer without
disabling it: every tick still fires and exits 0 immediately once `_unit_gate()` sees the
file. `python3 lab/harvest.py --status` reports `timer_active` (paused file absent) plus
budget/window/bank state without touching the GPU at all -- safe to run any time.

## Known limitation: preflight still needs Ollama reachable

`cad_engine.preflight()` calls `_installed_ollama_models()` before anything else, and only then
does `_preflight_models()` decide which models it actually has to find. Models on the `local:`
or `cloud/` rungs are skipped by that check, so a fully llama.cpp arm needs nothing from Ollama
at build time. But the roster query has already happened, and the fast coder rung
(`qwen2.5-coder:7b-instruct-q4_K_M`) is still an Ollama model, so the order cannot simply be
inverted: preflight runs before the coder rung is chosen, which means an installation that has
the Ollama fast rung configured still needs Ollama reachable for preflight to pass, even for a
run that will only ever use the strong rung on `maker-server`.

Not fixed here on purpose. `cad_engine.py` is imported live by every in-flight build and by the
benchmark chain, so reordering its startup path is a Phase 2 change, taken together with cutting
the Ollama fast rung out of the ladder rather than before it.
