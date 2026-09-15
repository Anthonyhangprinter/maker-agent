# deploy/

`maker-server` is the launcher for the swappable Maker Agent CAD coder llama-server (reads `~/.openclaw/maker.env`); `maker-server.service` is its systemd user unit, `Conflicts=qwen38-server.service` so the resident and the maker arm never share the single RTX 3090 at once.

Install: `install -m 755 deploy/maker-server ~/.local/bin/maker-server && install -m 644 deploy/maker-server.service ~/.config/systemd/user/maker-server.service && systemctl --user daemon-reload`

`RuntimeMaxSec=12h` is the dead-man switch: an abandoned arm (a card run killed mid-loop, a hand swap nobody restored) would otherwise hold the whole GPU indefinitely and the resident would never come back. The unit self-terminates after 12 hours, which is well clear of any real run, and the health watcher below then restores the resident within 5 minutes because no maker-server is active any more. It is not a substitute for `python3 scripts/arms.py restore`, just the floor under forgetting it.

The family health watcher (`~/family-ai-server/bin/health-watch.sh`, timer every 5 min) skips its resident-restart remedy while `maker-server.service` is active, so a card run (or any deliberate arm swap) is not fought and restarted mid-run; see that script's `check_qwen38`.
