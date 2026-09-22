#!/usr/bin/env python3
"""Candidate model arms for the Maker Agent card (docs/MAKER-1.0-CAMPAIGN.md 4.1a).

  arms.py list
  arms.py download <name>      # hf download into the NVMe store
  arms.py use <name>           # write ~/.openclaw/maker.env + cad.json maker block, start maker-server
                                #   (saves the pre-use maker block + maker.env once per run)
  arms.py restore              # stop maker-server, start the resident; re-applies the
                                #   saved pre-use maker block + maker.env if `use` saved one,
                                #   else maker.enabled=false
  arms.py restore --disable    # always maker.enabled=false, drop any saved pre-use state
  arms.py critic list
  arms.py critic use <name>    # VRAM check, write ~/.openclaw/critic.env, start critic-server (:8092)
  arms.py critic off           # stop critic-server
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ARMS_FILE = HERE / "benchmarks" / "arms.json"
CAD_JSON = Path(os.environ.get("CAD_CONFIG_FILE", Path.home() / ".openclaw" / "cad.json"))
ENV_PATH = Path(os.environ.get("MAKER_ENV", Path.home() / ".openclaw" / "maker.env"))
PORT = 8088
# The optional critic server (Task 4): a second, VRAM-checked llama-server that never
# evicts anything and never touches cad.json. Separate env file and port from the maker
# arm above so the two can coexist when the free VRAM allows it.
CRITIC_ENV_PATH = Path(os.environ.get("CRITIC_ENV", Path.home() / ".openclaw" / "critic.env"))
CRITIC_PORT_DEFAULT = 8092


def load_arms(path: Path = ARMS_FILE) -> dict[str, dict]:
    """Every arm in arms.json, resolved against the model store.

    Entries with `"skip": true` are KEPT here (so `arms.py list` still shows them and the
    file stays the single roster) and skipped by the runners: run_card.py and run_benchcad.py
    each print a line and move on. That is how an arm is parked without deleting its row."""
    data = json.loads(path.read_text())
    store = Path(data["store"])
    out = {}
    for arm in data["arms"]:
        arm = dict(arm)
        arm["skip"] = bool(arm.get("skip", False))
        arm["model_path"] = str(store / arm["gguf"])
        arm["mmproj_path"] = str(store / arm["mmproj"]) if arm.get("mmproj") else ""
        arm["store"] = str(store)
        out[arm["name"]] = arm
    return out


def load_critics(path: Path = ARMS_FILE) -> dict[str, dict]:
    """Every critic in arms.json's `critics` list, resolved against the model store.

    Same shape as load_arms: adds model_path/mmproj_path/store so callers never
    reconstruct a path from `gguf`/`mmproj` by hand. Critics have no `skip` concept
    (there is no shootout ladder for them) and no `role`."""
    data = json.loads(path.read_text())
    store = Path(data["store"])
    out = {}
    for critic in data.get("critics", []):
        critic = dict(critic)
        critic["model_path"] = str(store / critic["gguf"])
        critic["mmproj_path"] = str(store / critic["mmproj"]) if critic.get("mmproj") else ""
        critic["store"] = str(store)
        out[critic["name"]] = critic
    return out


def _q(value) -> str:
    """Single-quote a maker.env value. The launcher `source`s this file, so an
    unquoted value containing spaces (EXTRA_ARGS, e.g. a --chat-template-kwargs
    JSON blob) would break bash word-splitting / cause a syntax error. Arm values
    never contain a single quote (arms.json uses double quotes inside its JSON
    strings), so no escaping is needed, but the invariant is checked rather than
    assumed. ValueError, not assert: `python -O` strips asserts, and this one is a
    real input check on a file bash will execute, not a debug aid."""
    s = str(value)
    if "'" in s:
        raise ValueError(f"maker.env value must not contain a single quote: {s!r}")
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


def render_critic_env(critic: dict) -> str:
    """Same single-quoting discipline as render_env: deploy/critic-server `source`s this
    file, so every value must be quoted (EXTRA_ARGS can carry a --chat-template-kwargs
    JSON blob with spaces)."""
    return "\n".join([
        f"MODEL={_q(critic['model_path'])}",
        f"MMPROJ={_q(critic['mmproj_path'])}",
        f"CTX={_q(critic.get('ctx', 8192))}",
        f"PORT={_q(critic.get('port', CRITIC_PORT_DEFAULT))}",
        f"ALIAS={_q(critic['alias'])}",
        f"EXTRA_ARGS={_q(critic.get('extra_args', ''))}",
        "",
    ])


def free_vram_gb() -> float:
    """Free VRAM in GiB, parsed from `nvidia-smi --query-gpu=memory.free`.

    Raises (CalledProcessError / ValueError) rather than guessing when nvidia-smi is
    missing or its output is unparseable: a critic VRAM check exists to prevent an OOM
    on the single 3090, so "can't tell" must never be silently treated as "plenty free"."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    mib = int(out.strip().splitlines()[0])
    return mib / 1024.0


def _read_json(p: Path) -> dict:
    """Absent file: start from {} (first arm swap on a box with no cad.json).

    Present but unparseable: RAISE. cad.json carries the operator's own settings
    (code_model pins, the cloud block, print targets); treating a transient read error or a
    half-written file as "empty" would have _write_json replace all of it with a bare maker
    block. Better to fail the swap and leave the file alone."""
    try:
        text = p.read_text()
    except FileNotFoundError:
        return {}
    try:
        return json.loads(text)
    except Exception as e:
        raise SystemExit(f"{p} exists but is not valid JSON ({e}); refusing to overwrite it")


def _write_json(p: Path, data: dict) -> None:
    """Atomic replace, with a one-time .bak of the original.

    The backup is written only on the FIRST modification of an existing file (no .bak yet),
    so the pre-Maker cad.json survives; later arm swaps must not overwrite that snapshot with
    another maker-block-bearing copy."""
    bak = p.with_suffix(p.suffix + ".bak")
    if p.exists() and not bak.exists():
        bak.write_bytes(p.read_bytes())
    tmp = p.with_suffix(p.suffix + ".tmp")
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


def _pre_arm_paths(cad_json: Path, env_path: Path) -> tuple[Path, Path]:
    return (cad_json.with_suffix(cad_json.suffix + ".pre-arm"),
            env_path.with_suffix(env_path.suffix + ".pre-arm"))


def _save_pre_arm(cad_json: Path, env_path: Path) -> None:
    """Snapshot the maker block and maker.env before a `use` switches them, but only when
    no snapshot already exists (2026-09-19, the Phase 2 trap): a card that walks several
    arms calls `cmd_use` once per arm without restoring in between, so the FIRST call in
    that sequence is the one whose snapshot matters -- it is the state from before the
    whole run, and every later call in the same run must leave it alone.

    The maker.env side records whether the file existed at all, not just its bytes: a
    box that has never run `arms.py use` has no maker.env yet, and restore must be able
    to tell "put nothing back" apart from "put back an empty file"."""
    pre_cad, pre_env = _pre_arm_paths(cad_json, env_path)
    if not pre_cad.exists():
        cfg = _read_json(cad_json)
        pre_cad.write_text(json.dumps({"maker": cfg.get("maker")}, indent=2) + "\n")
    if not pre_env.exists():
        envelope = {
            "existed": env_path.exists(),
            "content": env_path.read_text() if env_path.exists() else None,
        }
        pre_env.write_text(json.dumps(envelope, indent=2) + "\n")


def _pop_pre_arm(cad_json: Path, env_path: Path) -> bool:
    """Re-apply the maker block and maker.env saved by `_save_pre_arm` and delete both
    snapshots. Returns False (no-op) when there is nothing to restore, which is the
    ordinary case on a box that has never run `arms.py use`."""
    pre_cad, pre_env = _pre_arm_paths(cad_json, env_path)
    if not pre_cad.exists():
        return False
    saved = json.loads(pre_cad.read_text())
    cfg = _read_json(cad_json)
    if saved.get("maker") is None:
        cfg.pop("maker", None)
    else:
        cfg["maker"] = saved["maker"]
    _write_json(cad_json, cfg)
    if pre_env.exists():
        envelope = json.loads(pre_env.read_text())
        if envelope.get("existed"):
            env_path.write_text(envelope.get("content") or "")
        else:
            env_path.unlink(missing_ok=True)
        pre_env.unlink()
    pre_cad.unlink()
    return True


def _clear_pre_arm(cad_json: Path, env_path: Path) -> None:
    """Drop any pre-arm snapshot without re-applying it: the `restore --disable` escape
    hatch back to today's behaviour."""
    pre_cad, pre_env = _pre_arm_paths(cad_json, env_path)
    pre_cad.unlink(missing_ok=True)
    pre_env.unlink(missing_ok=True)


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
        if arm.get("retired"):
            mark = "retired"
        elif arm.get("skip"):
            mark = "skip"
        else:
            mark = "ok " if have else "-- "
        print(f"{mark:8s}{name:24s} {arm['role']:9s} {arm['alias']:22s} {arm['gguf']}")


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


def unit_active(unit: str) -> bool:
    """True when `systemctl --user is-active <unit>` says active.

    Never raises: systemctl missing, the unit not installed, a DBus hiccup, all read as
    "not active". Both callers use this to decide whether to say something extra or stop
    one more unit, and neither is worth failing an arm swap over."""
    try:
        out = subprocess.run(["systemctl", "--user", "is-active", unit],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        return False
    return out == "active"


def cmd_use(arm: dict, start: bool = True, cad_json: Path = CAD_JSON, env_path: Path = ENV_PATH) -> None:
    if arm.get("retired"):
        note = arm.get("retired_note", "GGUF no longer on disk")
        raise SystemExit(f"{arm['name']} is retired ({note}); refusing to use it")
    if not Path(arm["model_path"]).exists():
        raise SystemExit(f"{arm['model_path']} missing; run arms.py download {arm['name']}")
    # Snapshot BEFORE writing the new arm, and regardless of --no-start: this is what
    # `restore` re-applies later, and the marker rule (only write it once) is what lets a
    # card that walks several arms in a row keep the state from before the whole run.
    _save_pre_arm(cad_json, env_path)
    apply_arm(arm, cad_json, env_path)
    if start:
        # The critic holds its own VRAM on the one card and cmd_use's arm load does not
        # account for it: an arm that fits on an empty 3090 can OOM on load, or silently
        # spill to system RAM, with a critic already resident. Say so rather than leaving
        # the operator to read it off a mystery slowdown.
        if unit_active("critic-server"):
            print("WARNING: critic-server is active and holds VRAM; this arm's load does not "
                  "account for it (python3 scripts/arms.py critic off to free it)", file=sys.stderr)
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
            try:
                cmd_restore(cad_json=cad_json, env_path=env_path)
            except BaseException as restore_exc:
                # The recovery must never replace the real cause: a restore that itself fails
                # (resident unhealthy too) would otherwise be the only error anyone sees.
                print(f"restore during cmd_use recovery also failed: {restore_exc}", file=sys.stderr)
            raise
    print(f"maker-server -> {arm['name']} ({arm['alias']}) on :{PORT}")


def cmd_restore(cad_json: Path = CAD_JSON, env_path: Path = ENV_PATH, disable: bool = False) -> None:
    """Back to the resident: maker-server and critic-server stopped, qwen38-server up and
    healthy.

    When a pre-arm snapshot exists (written by `cmd_use` before the switch this call is
    undoing) and `disable` was not asked for, the saved maker block and maker.env are
    re-applied and the snapshot deleted -- this is the Phase 2 trap fix: cad.json and
    maker.env travel together back to exactly what they held before the run, instead of
    cad.json going to plain-disabled while maker.env is left pointing at whatever GGUF
    was loaded last. With no snapshot on disk, or with `disable=True`, this falls back to
    today's plain disable (`maker.enabled=false`, alias/arm left as-is).

    critic-server is stopped too, and BEFORE the resident is started. The resident wants
    around 23 GB of the 24 GB card, so a critic left running from a card run is the
    difference between the resident loading and the resident OOMing or spilling to system
    RAM. "Restore" has to mean the box is back to its normal state, not the maker half of
    it: the card runner's finally calls only this."""
    restored = False
    if disable:
        disable_maker(cad_json)
        _clear_pre_arm(cad_json, env_path)
    else:
        restored = _pop_pre_arm(cad_json, env_path)
        if not restored:
            disable_maker(cad_json)
    subprocess.run(["systemctl", "--user", "stop", "maker-server"], check=False)
    subprocess.run(["systemctl", "--user", "stop", "critic-server"], check=False)
    subprocess.run(["systemctl", "--user", "start", "qwen38-server"], check=False)
    _wait("http://127.0.0.1:8086/health", 300)
    if restored:
        print("resident restored on :8086; pre-arm maker block and maker.env re-applied, critic-server stopped")
    else:
        print("resident restored on :8086; maker disabled, critic-server stopped")


def cmd_critic_list(critics: dict) -> None:
    for name, critic in critics.items():
        have = "ok " if Path(critic["model_path"]).exists() else "-- "
        print(f"{have:5s}{name:20s} {critic['alias']:16s} vram={critic.get('vram_gb')}GB {critic['gguf']}")


def cmd_critic_use(critic: dict, env_path: Path = CRITIC_ENV_PATH) -> None:
    """VRAM check, write critic.env, (re)start critic-server, wait for /health, print the
    two exports a card/run_card.py wires to CAD_CRITIC_MODEL/CAD_CRITIC_URL.

    Refuses with exit code 2 (not a bare SystemExit(str), which exits 1) and a message
    naming the free MiB when there is not enough headroom -- the whole point of this
    command is that a critic must never be started into an OOM."""
    if not Path(critic["model_path"]).exists():
        raise SystemExit(f"{critic['model_path']} missing; the critic GGUF must already be on disk")
    need_gb = float(critic.get("vram_gb", 0)) + 0.5
    free_gb = free_vram_gb()
    if free_gb < need_gb:
        free_mib = round(free_gb * 1024)
        print(f"critic {critic['name']} needs {critic.get('vram_gb')} GB (+0.5 GB headroom = "
              f"{need_gb:.1f} GB) but only {free_mib} MiB free on the GPU; refusing to start "
              f"critic-server.", file=sys.stderr)
        sys.exit(2)
    env_path.write_text(render_critic_env(critic))
    subprocess.run(["systemctl", "--user", "restart", "critic-server"], check=True)
    port = critic.get("port", CRITIC_PORT_DEFAULT)
    _wait(f"http://127.0.0.1:{port}/health", 300)
    print(f"CAD_CRITIC_MODEL=local:{critic['alias']}")
    print(f"CAD_CRITIC_URL=http://127.0.0.1:{port}/v1/chat/completions")


def cmd_critic_off() -> None:
    subprocess.run(["systemctl", "--user", "stop", "critic-server"], check=False)
    print("critic-server stopped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("download").add_argument("name")
    u = sub.add_parser("use"); u.add_argument("name"); u.add_argument("--no-start", action="store_true")
    r = sub.add_parser("restore")
    r.add_argument("--disable", action="store_true",
                    help="force the plain disable (ignore/clear any pre-arm snapshot)")
    c = sub.add_parser("critic")
    csub = c.add_subparsers(dest="critic_cmd", required=True)
    csub.add_parser("list")
    cu = csub.add_parser("use"); cu.add_argument("name")
    csub.add_parser("off")
    ns = ap.parse_args()
    if ns.cmd == "critic":
        critics = load_critics()
        if ns.critic_cmd == "list": cmd_critic_list(critics)
        elif ns.critic_cmd == "use": cmd_critic_use(critics[ns.name])
        elif ns.critic_cmd == "off": cmd_critic_off()
        return
    a = load_arms()
    if ns.cmd == "list": cmd_list(a)
    elif ns.cmd == "download": cmd_download(a[ns.name])
    elif ns.cmd == "use": cmd_use(a[ns.name], start=not ns.no_start)
    elif ns.cmd == "restore": cmd_restore(disable=ns.disable)


if __name__ == "__main__":
    main()
