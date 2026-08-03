# CAD Agent Web UI — Runbook

CADAM-style web front door for the v5 CAD engine. FastAPI app (`app.py`), one page
(`static/index.html`), builds run as `python3 -m cad_v5 … --once --json --target file`
subprocesses. In-memory job store (v1): a restart loses in-flight jobs (artifacts survive
in `~/.openclaw/cad-builds/`).

## URLs

- On the box: http://127.0.0.1:8090
- Tailnet (any of your devices): **https://hp-z2-tower-g4-workstation.taila0e0e0.ts.net:8443**

## Service

```bash
systemctl --user status cad-web        # unit: ~/.config/systemd/user/cad-web.service
systemctl --user restart cad-web
journalctl --user -u cad-web -f        # logs
curl -s localhost:8090/healthz         # {"ok":true,"queue":0}
```

Deps live in `webui/.venv` (created with `uv venv` — system python3 lacks ensurepip):
`uv pip install --python .venv/bin/python fastapi uvicorn python-multipart`.

## Tailscale exposure (one-time — NOT stored in any script)

`tailscale serve` state lives only inside tailscaled, so this command is recorded here
(the Family AI server learned this the hard way):

```bash
sudo tailscale serve --bg --https=8443 http://127.0.0.1:8090
tailscale serve status     # verify: 8443 → proxy http://127.0.0.1:8090
```

- Port 443 is taken by LibreChat (Family AI server) — this app uses **8443**.
- `serve` = tailnet-only. NEVER use `tailscale funnel` (public internet) for this.
- App binds 127.0.0.1 only; ufw already restricts inbound to the tailscale0 interface.

## Behaviour notes

- One build at a time: the app has its own queue AND the engine holds a machine-wide flock
  (`~/.openclaw/cad-build.lock`), so web builds serialize with Satine/CLI/benchmarks. A build
  blocked behind another frontend shows "waiting for GPU".
- Image uploads: jpg/png/webp by magic bytes, ≤10 MB, stored in `~/.openclaw/cad-web/uploads/`.
  The engine copies the downscaled reference into the build dir as `reference.jpg`.
- Artifacts served from `~/.openclaw/cad-builds/<id>/` (path-resolved + extension whitelist);
  old builds are pruned by the engine (KEEP_BUILDS=200), so old links 404 gracefully.
- Beta testers (2026-08-03): `tailscale serve` stamps requests with `Tailscale-User-Login`/
  `Tailscale-User-Name`, so every job records its submitter (shown as a pill on the job card).
  Jobs from any login NOT in `OWNER_LOGINS` (app.py) ping the owner's Telegram via Satine's
  bot (token read from openclaw.json, chat `OWNER_CHAT_ID`) on queue, chat-revise, and
  completion (with render photo). Localhost + owner logins stay silent. Friends get access
  via Tailscale **node sharing** (admin console → machine → Share); optionally scope
  `autogroup:shared` to port 8443 in the tailnet ACL so shared users can't reach LibreChat.
