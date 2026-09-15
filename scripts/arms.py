#!/usr/bin/env python3
"""Candidate model arms for the Maker Agent card (docs/MAKER-1.0-CAMPAIGN.md 4.1a).

  arms.py list
  arms.py download <name>      # hf download into the NVMe store
  arms.py use <name>           # write ~/.openclaw/maker.env + cad.json maker block, start maker-server
  arms.py restore              # maker.enabled=false, stop maker-server, start the resident
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ARMS_FILE = HERE / "benchmarks" / "arms.json"
CAD_JSON = Path(os.environ.get("CAD_CONFIG_FILE", Path.home() / ".openclaw" / "cad.json"))
ENV_PATH = Path(os.environ.get("MAKER_ENV", Path.home() / ".openclaw" / "maker.env"))
PORT = 8088


def load_arms(path: Path = ARMS_FILE) -> dict[str, dict]:
    data = json.loads(path.read_text())
    store = Path(data["store"])
    out = {}
    for arm in data["arms"]:
        arm = dict(arm)
        arm["model_path"] = str(store / arm["gguf"])
        arm["mmproj_path"] = str(store / arm["mmproj"]) if arm.get("mmproj") else ""
        arm["store"] = str(store)
        out[arm["name"]] = arm
    return out


def _q(value) -> str:
    """Single-quote a maker.env value. The launcher `source`s this file, so an
    unquoted value containing spaces (EXTRA_ARGS, e.g. a --chat-template-kwargs
    JSON blob) would break bash word-splitting / cause a syntax error. Arm values
    never contain a single quote (arms.json uses double quotes inside its JSON
    strings), so no escaping is needed — just assert that invariant holds."""
    s = str(value)
    assert "'" not in s, f"maker.env value must not contain a single quote: {s!r}"
    return f"'{s}'"


def render_env(arm: dict) -> str:
    return "\n".join([
        f"MODEL={_q(arm['model_path'])}",
        f"MMPROJ={_q(arm['mmproj_path'])}",
        f"CTX={_q(arm['ctx'])}",
        f"PORT={_q(PORT)}",
        f"ALIAS={_q(arm['alias'])}",
        f"EXTRA_ARGS={_q(arm.get('extra_args', ''))}",
        "",
    ])


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return {}


def _write_json(p: Path, data: dict) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(p)


def apply_arm(arm: dict, cad_json: Path = CAD_JSON, env_path: Path = ENV_PATH) -> None:
    env_path.write_text(render_env(arm))
    cfg = _read_json(cad_json)
    cfg["maker"] = {"enabled": True, "port": PORT, "alias": arm["alias"], "arm": arm["name"]}
    _write_json(cad_json, cfg)


def disable_maker(cad_json: Path = CAD_JSON) -> None:
    cfg = _read_json(cad_json)
    m = cfg.get("maker") or {}
    m["enabled"] = False
    cfg["maker"] = m
    _write_json(cad_json, cfg)


def _wait(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                json.loads(r.read()); return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"{url} not healthy after {timeout}s")


def cmd_list(a: dict) -> None:
    for name, arm in a.items():
        have = Path(arm["model_path"]).exists()
        print(f"{'ok ' if have else '-- '}{name:24s} {arm['role']:9s} {arm['alias']:22s} {arm['gguf']}")


def cmd_download(arm: dict) -> None:
    if not arm.get("hf"):
        print(f"{arm['name']}: no hf entry (expected on disk at {arm['model_path']})"); return
    dest = Path(arm["store"]) / Path(arm["gguf"]).parent
    dest.mkdir(parents=True, exist_ok=True)
    for f in arm["hf"]["files"]:
        if (dest / f).exists():
            print(f"have {dest / f}"); continue
        subprocess.run(["hf", "download", arm["hf"]["repo"], f, "--local-dir", str(dest)], check=True)
    for f in arm["hf"]["files"]:
        print(f"{dest / f}: {(dest / f).stat().st_size} bytes")


def cmd_use(arm: dict, start: bool = True) -> None:
    if not Path(arm["model_path"]).exists():
        raise SystemExit(f"{arm['model_path']} missing; run arms.py download {arm['name']}")
    apply_arm(arm)
    if start:
        # A maker-server that never becomes healthy (review finding on Task 4) must not
        # leave the box with no server at all: restore the resident before re-raising,
        # whatever the failure was (CalledProcessError from the restart, SystemExit from
        # _wait timing out, ...). run_card.py's own finally calls cmd_restore() too, so
        # this makes the failure safe even when cmd_use is called directly (arms.py use).
        try:
            subprocess.run(["systemctl", "--user", "stop", "qwen38-server"], check=False)
            subprocess.run(["systemctl", "--user", "restart", "maker-server"], check=True)
            _wait(f"http://127.0.0.1:{PORT}/health", 900)
        except BaseException:
            cmd_restore()
            raise
    print(f"maker-server -> {arm['name']} ({arm['alias']}) on :{PORT}")


def cmd_restore() -> None:
    disable_maker()
    subprocess.run(["systemctl", "--user", "stop", "maker-server"], check=False)
    subprocess.run(["systemctl", "--user", "start", "qwen38-server"], check=False)
    _wait("http://127.0.0.1:8086/health", 300)
    print("resident restored on :8086; maker disabled")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("download").add_argument("name")
    u = sub.add_parser("use"); u.add_argument("name"); u.add_argument("--no-start", action="store_true")
    sub.add_parser("restore")
    ns = ap.parse_args()
    a = load_arms()
    if ns.cmd == "list": cmd_list(a)
    elif ns.cmd == "download": cmd_download(a[ns.name])
    elif ns.cmd == "use": cmd_use(a[ns.name], start=not ns.no_start)
    elif ns.cmd == "restore": cmd_restore()


if __name__ == "__main__":
    main()
