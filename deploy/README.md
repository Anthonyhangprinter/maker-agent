# deploy/

`maker-server` is the launcher for the swappable Maker Agent CAD coder llama-server (reads `~/.openclaw/maker.env`); `maker-server.service` is its systemd user unit, `Conflicts=qwen38-server.service` so the resident and the maker arm never share the single RTX 3090 at once.

Install: `install -m 755 deploy/maker-server ~/.local/bin/maker-server && install -m 644 deploy/maker-server.service ~/.config/systemd/user/maker-server.service && systemctl --user daemon-reload`
