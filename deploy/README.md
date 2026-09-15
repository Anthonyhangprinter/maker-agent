# deploy/

`maker-server` is the launcher for the swappable Maker Agent CAD coder llama-server (reads `~/.openclaw/maker.env`); `maker-server.service` is its systemd user unit, `Conflicts=qwen38-server.service` so the resident and the maker arm never share the single RTX 3090 at once.

Install: `install -m 755 deploy/maker-server ~/.local/bin/maker-server && install -m 644 deploy/maker-server.service ~/.config/systemd/user/maker-server.service && systemctl --user daemon-reload`

The family health watcher (`~/family-ai-server/bin/health-watch.sh`, timer every 5 min) skips its resident-restart remedy while `maker-server.service` is active, so a card run (or any deliberate arm swap) is not fought and restarted mid-run; see that script's `check_qwen38`.
