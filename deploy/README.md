# deploy/

`maker-server` is the launcher for the swappable Maker Agent CAD coder llama-server (reads `~/.openclaw/maker.env`); `maker-server.service` is its systemd user unit, `Conflicts=qwen38-server.service` so the resident and the maker arm never share the single RTX 3090 at once.

Install: `install -m 755 deploy/maker-server ~/.local/bin/maker-server && install -m 644 deploy/maker-server.service ~/.config/systemd/user/maker-server.service && systemctl --user daemon-reload`

`RuntimeMaxSec=12h` is the dead-man switch: an abandoned arm (a card run killed mid-loop, a hand swap nobody restored) would otherwise hold the whole GPU indefinitely and the resident would never come back. The unit self-terminates after 12 hours, which is well clear of any real run, and the health watcher below then restores the resident within 5 minutes because no maker-server is active any more. It is not a substitute for `python3 scripts/arms.py restore`, just the floor under forgetting it.

The family health watcher (`~/family-ai-server/bin/health-watch.sh`, timer every 5 min) skips its resident-restart remedy while `maker-server.service` is active, so a card run (or any deliberate arm swap) is not fought and restarted mid-run; see that script's `check_qwen38`.

## critic-server (optional, :8092)

`critic-server` is a second, independent llama-server for a vision critic (`~/.openclaw/critic.env`, keys `MODEL`/`MMPROJ`/`CTX`/`PORT`/`ALIAS`/`EXTRA_ARGS`, same single-quoted-`source`d shape as `maker.env`). Candidates live in `benchmarks/arms.json`'s `critics` list: `minicpm-v-4.6` (cheap, ~3.5GB conservative) and `gemma-4-12b` (non-Qwen judge candidate, ~9.5GB conservative). Unlike `maker-server`, this launcher never evicts an Ollama guest and its unit carries no `Conflicts=`: it is meant to run alongside the maker or the resident, not instead of them.

Whether it fits is a real question on one 24GB card: the maker running `gemma-4-31b` at ctx 16384 alone measures 21,435 MiB used of 24,576, leaving under 3.2GB free, so a critic cannot coexist with that arm at that context. `scripts/arms.py critic use <name>` is the gate for this: it reads each critic's `vram_gb` field, parses `nvidia-smi --query-gpu=memory.free` for the real free MiB, and refuses with exit code 2 and a message naming the free MiB whenever free VRAM is below `vram_gb + 0.5`. Only when it proceeds does it write `critic.env`, restart `critic-server`, wait for `/health`, and print `CAD_CRITIC_MODEL=local:<alias>` and `CAD_CRITIC_URL=http://127.0.0.1:8092/v1/chat/completions` for the card to consume (`run_card.py --critic-url ...` or the engine's own `CAD_CRITIC_URL`). `scripts/arms.py critic off` stops the unit; `scripts/arms.py critic list` shows what is on disk.

Install: `install -m 755 deploy/critic-server ~/.local/bin/critic-server && install -m 644 deploy/critic-server.service ~/.config/systemd/user/critic-server.service && systemctl --user daemon-reload`

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
