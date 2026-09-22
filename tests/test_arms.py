import json, sys
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import arms


def test_arms_file_is_well_formed():
    a = arms.load_arms()
    assert {"qwen3.8-27b-nothink", "gemma-4-31b", "qwen3-coder-30b-a3b", "glm-4.7-flash",
            "devstral-small-2", "gpt-oss-20b", "qwen2.5-coder-7b"} <= set(a)
    for arm in a.values():
        assert arm["gguf"].endswith(".gguf") and arm["ctx"] >= 8192 and arm["alias"]


def test_render_env_has_every_key():
    arm = arms.load_arms()["gpt-oss-20b"]
    env = arms.render_env(arm)
    for key in ("MODEL=", "MMPROJ=", "CTX='16384'", "PORT='8088'", "ALIAS='gpt-oss-20b'", "EXTRA_ARGS="):
        assert key in env
    assert "/mnt/nvme-apps/LinuxModels/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf" in env


def test_render_env_values_are_single_quoted():
    # correction (controller ruling): the launcher `source`s this file, so every value
    # must be single-quoted or unquoted values containing spaces break bash.
    arm = arms.load_arms()["qwen3.8-27b-nothink"]
    env = arms.render_env(arm)
    for line in env.splitlines():
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        assert value.startswith("'") and value.endswith("'"), f"{key} value is not single-quoted: {value!r}"


def test_apply_arm_writes_env_and_cad_json(tmp_path):
    arm = arms.load_arms()["gemma-4-31b"]
    cad_json = tmp_path / "cad.json"; cad_json.write_text(json.dumps({"code_model": None}))
    env_path = tmp_path / "maker.env"
    arms.apply_arm(arm, cad_json, env_path)
    cfg = json.loads(cad_json.read_text())
    assert cfg["maker"] == {"enabled": True, "port": 8088, "alias": "gemma-4-31b", "arm": "gemma-4-31b"}
    assert cfg["code_model"] is None          # untouched keys survive
    assert "ALIAS='gemma-4-31b'" in env_path.read_text()


def test_restore_disables_maker(tmp_path):
    cad_json = tmp_path / "cad.json"
    cad_json.write_text(json.dumps({"maker": {"enabled": True, "port": 8088, "alias": "x", "arm": "x"}}))
    arms.disable_maker(cad_json)
    assert json.loads(cad_json.read_text())["maker"]["enabled"] is False


def test_cmd_use_restores_resident_if_maker_never_healthy(tmp_path, monkeypatch):
    """Review finding on Task 4: a maker-server that never becomes healthy used to leave
    the box with the resident stopped AND the maker still down (SystemExit from _wait
    propagated straight out of cmd_use, no restore). cmd_use must now restore the resident
    before re-raising, same as run_card.py's own finally does when the whole run blows up."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(arms, "apply_arm", lambda *a, **kw: None)  # no real cad.json/maker.env writes
    monkeypatch.setattr(arms.subprocess, "run", fake_run)

    wait_calls = {"n": 0}

    def fake_wait(url, timeout):
        wait_calls["n"] += 1
        if wait_calls["n"] == 1:
            raise SystemExit(f"{url} not healthy after {timeout}s")
        # second call is cmd_restore's own _wait on the resident — let it "succeed" so the
        # test never touches the network.

    monkeypatch.setattr(arms, "_wait", fake_wait)

    model = tmp_path / "fake.gguf"
    model.write_text("x")
    arm = {"name": "fake-arm", "alias": "fake-alias", "model_path": str(model)}
    # Real tmp cad.json/maker.env: apply_arm is stubbed out above, but _save_pre_arm and
    # the cmd_restore recovery path below are NOT, so this must never fall back to the
    # module's real ~/.openclaw defaults.
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"

    with pytest.raises(SystemExit):
        arms.cmd_use(arm, cad_json=cad_json, env_path=env_path)

    assert wait_calls["n"] == 2   # cmd_use's own wait, then cmd_restore's wait
    cmds = [" ".join(c) for c in calls]
    assert cmds[-3] == "systemctl --user stop maker-server"
    assert cmds[-2] == "systemctl --user stop critic-server"
    assert cmds[-1] == "systemctl --user start qwen38-server"


def test_write_json_backs_up_the_original_once(tmp_path):
    """I9: the pre-Maker cad.json must survive the first arm swap, and later swaps must not
    overwrite that snapshot with another maker-block-bearing copy."""
    cad_json = tmp_path / "cad.json"
    cad_json.write_text(json.dumps({"code_model": "pinned", "cloud": {"provider": "anthropic"}}))
    arm = arms.load_arms()["gemma-4-31b"]
    arms.apply_arm(arm, cad_json, tmp_path / "maker.env")
    bak = tmp_path / "cad.json.bak"
    assert json.loads(bak.read_text()) == {"code_model": "pinned", "cloud": {"provider": "anthropic"}}
    arms.apply_arm(arms.load_arms()["gpt-oss-20b"], cad_json, tmp_path / "maker.env")
    assert "maker" not in json.loads(bak.read_text())   # still the original, not the first swap
    assert json.loads(cad_json.read_text())["maker"]["alias"] == "gpt-oss-20b"


def test_read_json_absent_file_is_empty_but_unparseable_raises(tmp_path):
    """Absent: a box with no cad.json yet, proceed from {}. Present but broken: refuse, or
    _write_json would replace the operator's real settings with a bare maker block."""
    assert arms._read_json(tmp_path / "nope.json") == {}
    broken = tmp_path / "cad.json"
    broken.write_text('{"cad": ')
    with pytest.raises(SystemExit):
        arms._read_json(broken)
    assert broken.read_text() == '{"cad": '       # untouched


def test_apply_arm_refuses_to_clobber_an_unparseable_cad_json(tmp_path):
    cad_json = tmp_path / "cad.json"
    cad_json.write_text("{ not json")
    with pytest.raises(SystemExit):
        arms.apply_arm(arms.load_arms()["gemma-4-31b"], cad_json, tmp_path / "maker.env")
    assert cad_json.read_text() == "{ not json"


def test_single_quote_invariant_raises_value_error_not_assert():
    """M5: `python -O` strips asserts, and this guards a file bash will source."""
    with pytest.raises(ValueError):
        arms._q("it's broken")


def test_load_arms_keeps_skipped_entries(tmp_path):
    src = json.loads((HERE / "benchmarks" / "arms.json").read_text())
    src["arms"] = [{**arm, "skip": False} for arm in src["arms"]]   # the shipped file may skip real arms
    src["arms"][0] = {**src["arms"][0], "skip": True}
    p = tmp_path / "arms.json"; p.write_text(json.dumps(src))
    a = arms.load_arms(p)
    first = src["arms"][0]["name"]
    assert a[first]["skip"] is True                   # kept in the roster
    assert all(a[n]["skip"] is False for n in a if n != first)


def test_load_critics_shape():
    c = arms.load_critics()
    assert set(c) == {"minicpm-v-4.6", "gemma-4-12b"}
    store = json.loads(arms.ARMS_FILE.read_text())["store"]
    for critic in c.values():
        assert critic["gguf"].endswith(".gguf")
        assert critic["model_path"] == str(Path(store) / critic["gguf"])
    minicpm = c["minicpm-v-4.6"]
    assert minicpm["model_path"].endswith("MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q8_0.gguf")
    assert minicpm["mmproj_path"].endswith("MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf")
    assert minicpm["vram_gb"] == 3.5
    assert minicpm["port"] == 8092
    gemma = c["gemma-4-12b"]
    assert gemma["model_path"].endswith("gemma-4-12B-it-GGUF/gemma-4-12b-it-UD-Q4_K_XL.gguf")
    assert gemma["mmproj_path"].endswith("gemma-4-12B-it-GGUF/mmproj-F16.gguf")
    assert gemma["vram_gb"] == 9.5
    assert gemma["port"] == 8092


def test_render_critic_env_keys_and_quoting():
    critic = arms.load_critics()["gemma-4-12b"]
    env = arms.render_critic_env(critic)
    for key in ("MODEL=", "MMPROJ=", "CTX='8192'", "PORT='8092'", "ALIAS='gemma-4-12b'", "EXTRA_ARGS="):
        assert key in env
    for line in env.splitlines():
        if not line or "=" not in line:
            continue
        k, _, v = line.partition("=")
        assert v.startswith("'") and v.endswith("'"), f"{k} value is not single-quoted: {v!r}"


def _tmp_critics(tmp_path):
    """The real critic definitions (name/alias/ctx/port/vram_gb), resolved against a
    throwaway tmp store with small dummy files in place of the real GGUFs.

    Tests must never depend on the real model files under arms.json's "store"
    (/mnt/nvme-apps/LinuxModels): they can be moved, archived to the HDD, or simply
    absent on a box that hasn't downloaded them, and none of that should fail a unit
    test. Only the JSON shape (paths, names, sizes-in-GB) is real."""
    real = json.loads((HERE / "benchmarks" / "arms.json").read_text())
    store = tmp_path / "store"
    arms_json = tmp_path / "arms.json"
    arms_json.write_text(json.dumps({"store": str(store), "critics": real["critics"]}))
    critics = arms.load_critics(arms_json)
    for critic in critics.values():
        for key in ("model_path", "mmproj_path"):
            p = critic.get(key)
            if not p:
                continue
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_bytes(b"dummy")
    return critics


def test_cmd_critic_use_refuses_when_vram_low(tmp_path, monkeypatch):
    """The whole point of Task 4: a critic must not be started when it cannot fit beside
    whatever else is already on the GPU. free_vram_gb() patched low must exit(2) and make
    no systemctl call and no env write."""
    critic = _tmp_critics(tmp_path)["gemma-4-12b"]
    monkeypatch.setattr(arms, "free_vram_gb", lambda: 1.0)
    calls: list[list[str]] = []
    monkeypatch.setattr(arms.subprocess, "run",
                         lambda cmd, **kw: (calls.append(list(cmd)), type("R", (), {"returncode": 0})())[1])
    env_path = tmp_path / "critic.env"
    with pytest.raises(SystemExit) as exc:
        arms.cmd_critic_use(critic, env_path=env_path)
    assert exc.value.code == 2
    assert not calls
    assert not env_path.exists()


def test_cmd_critic_use_proceeds_when_vram_high(tmp_path, monkeypatch, capsys):
    critic = _tmp_critics(tmp_path)["minicpm-v-4.6"]
    monkeypatch.setattr(arms, "free_vram_gb", lambda: 20.0)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(arms.subprocess, "run", fake_run)
    monkeypatch.setattr(arms, "_wait", lambda url, timeout: None)
    env_path = tmp_path / "critic.env"
    arms.cmd_critic_use(critic, env_path=env_path)
    assert calls == [["systemctl", "--user", "restart", "critic-server"]]
    assert "ALIAS='minicpm-v'" in env_path.read_text()
    out = capsys.readouterr().out
    assert "CAD_CRITIC_MODEL=local:minicpm-v" in out
    assert "CAD_CRITIC_URL=http://127.0.0.1:8092/v1/chat/completions" in out


def test_cmd_critic_off_stops_the_unit(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(arms.subprocess, "run",
                         lambda cmd, **kw: (calls.append(list(cmd)), type("R", (), {"returncode": 0})())[1])
    arms.cmd_critic_off()
    assert calls == [["systemctl", "--user", "stop", "critic-server"]]


def test_cmd_use_recovery_does_not_mask_the_original_failure(tmp_path, monkeypatch, capsys):
    """M8: a restore that itself fails must not become the only error anyone sees."""
    monkeypatch.setattr(arms, "apply_arm", lambda *a, **kw: None)
    monkeypatch.setattr(arms.subprocess, "run", lambda cmd, **kw: type("R", (), {"returncode": 0})())

    def failing_restore(*a, **kw):
        raise RuntimeError("resident unhealthy too")

    monkeypatch.setattr(arms, "cmd_restore", failing_restore)

    def dead_wait(url, timeout):
        raise SystemExit("maker never healthy")

    monkeypatch.setattr(arms, "_wait", dead_wait)
    model = tmp_path / "fake.gguf"; model.write_text("x")
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    with pytest.raises(SystemExit, match="maker never healthy"):
        arms.cmd_use({"name": "fake", "alias": "fake", "model_path": str(model)},
                     cad_json=cad_json, env_path=env_path)
    assert "resident unhealthy too" in capsys.readouterr().err


def test_cmd_restore_stops_the_critic_before_starting_the_resident(tmp_path, monkeypatch):
    """The resident wants ~23 GB of the 24 GB card. A critic left over from a card run is
    the difference between the resident loading and the resident OOMing, so restore must
    stop it, and stop it BEFORE the resident is started."""
    calls = []
    monkeypatch.setattr(arms.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or type("R", (), {"returncode": 0})())
    monkeypatch.setattr(arms, "_wait", lambda url, timeout: None)
    # Fresh tmp cad.json/maker.env with no pre-arm markers: exercises the plain-disable
    # branch (today's behaviour), same as it always has, off tmp paths instead of the
    # module's real ~/.openclaw defaults.
    arms.cmd_restore(cad_json=tmp_path / "cad.json", env_path=tmp_path / "maker.env")
    units = [(c[2], c[3]) for c in calls]
    assert ("stop", "critic-server") in units
    assert units.index(("stop", "critic-server")) < units.index(("start", "qwen38-server"))


def test_cmd_use_warns_when_the_critic_is_holding_vram(tmp_path, monkeypatch, capsys):
    model = tmp_path / "m.gguf"; model.write_text("x")
    monkeypatch.setattr(arms, "apply_arm", lambda *a, **kw: None)
    monkeypatch.setattr(arms, "unit_active", lambda unit: unit == "critic-server")
    monkeypatch.setattr(arms.subprocess, "run", lambda cmd, **kw: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(arms, "_wait", lambda url, timeout: None)
    arms.cmd_use({"name": "fake", "alias": "fake", "model_path": str(model)},
                 cad_json=tmp_path / "cad.json", env_path=tmp_path / "maker.env")
    err = capsys.readouterr().err
    assert "critic-server is active" in err and "critic off" in err


def test_cmd_use_is_silent_when_no_critic_is_running(tmp_path, monkeypatch, capsys):
    model = tmp_path / "m.gguf"; model.write_text("x")
    monkeypatch.setattr(arms, "apply_arm", lambda *a, **kw: None)
    monkeypatch.setattr(arms, "unit_active", lambda unit: False)
    monkeypatch.setattr(arms.subprocess, "run", lambda cmd, **kw: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(arms, "_wait", lambda url, timeout: None)
    arms.cmd_use({"name": "fake", "alias": "fake", "model_path": str(model)},
                 cad_json=tmp_path / "cad.json", env_path=tmp_path / "maker.env")
    assert "critic-server" not in capsys.readouterr().err


def test_unit_active_never_raises(monkeypatch):
    """A missing systemctl must read as "not active", not take an arm swap down with it."""
    def boom(*a, **kw):
        raise FileNotFoundError("systemctl")
    monkeypatch.setattr(arms.subprocess, "run", boom)
    assert arms.unit_active("critic-server") is False
    monkeypatch.setattr(arms.subprocess, "run",
                        lambda *a, **kw: type("R", (), {"stdout": "active\n"})())
    assert arms.unit_active("critic-server") is True
    monkeypatch.setattr(arms.subprocess, "run",
                        lambda *a, **kw: type("R", (), {"stdout": "inactive\n"})())
    assert arms.unit_active("critic-server") is False


# ---------------------------------------------------------------------------------------
# Task 1 (Phase 3): restore re-applies the pre-run maker block + maker.env, not just
# maker.enabled=false. The Phase 2 card trap: cad.json said gemma-4-31b while maker.env
# still pointed at the spike GGUF, because restore only ever touched cad.json.
# ---------------------------------------------------------------------------------------

def _fake_arm(tmp_path, name: str) -> dict:
    model = tmp_path / f"{name}.gguf"
    model.write_text("x")
    return {"name": name, "alias": name, "model_path": str(model), "mmproj_path": "",
            "ctx": 16384, "extra_args": ""}


def test_use_then_restore_leaves_cad_json_and_maker_env_byte_identical(tmp_path, monkeypatch):
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    original_cfg = {"code_model": "pinned",
                     "maker": {"enabled": False, "port": 8086, "alias": "qwen3.8-27b", "arm": "resident"}}
    cad_json.write_text(json.dumps(original_cfg, indent=2) + "\n")
    original_env = "MODEL='/old/model.gguf'\nALIAS='old-alias'\n"
    env_path.write_text(original_env)

    monkeypatch.setattr(arms.subprocess, "run", lambda cmd, **kw: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(arms, "_wait", lambda url, timeout: None)

    arms.cmd_use(_fake_arm(tmp_path, "new-arm"), cad_json=cad_json, env_path=env_path)
    assert json.loads(cad_json.read_text())["maker"]["alias"] == "new-arm"
    assert "new-arm" in env_path.read_text()

    arms.cmd_restore(cad_json=cad_json, env_path=env_path)
    assert json.loads(cad_json.read_text()) == original_cfg
    assert env_path.read_text() == original_env
    assert not (tmp_path / "cad.json.pre-arm").exists()
    assert not (tmp_path / "maker.env.pre-arm").exists()


def test_restore_without_markers_disables_maker_and_leaves_env_untouched(tmp_path, monkeypatch, capsys):
    cad_json = tmp_path / "cad.json"
    cad_json.write_text(json.dumps({"maker": {"enabled": True, "port": 8088,
                                               "alias": "some-arm", "arm": "some-arm"}}))
    env_path = tmp_path / "maker.env"
    env_path.write_text("ALIAS='some-arm'\n")

    monkeypatch.setattr(arms.subprocess, "run", lambda cmd, **kw: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(arms, "_wait", lambda url, timeout: None)

    arms.cmd_restore(cad_json=cad_json, env_path=env_path)
    cfg = json.loads(cad_json.read_text())
    assert cfg["maker"]["enabled"] is False
    assert cfg["maker"]["alias"] == "some-arm"            # today's behaviour: left as-is
    assert env_path.read_text() == "ALIAS='some-arm'\n"   # untouched: no marker existed
    assert "maker disabled" in capsys.readouterr().out


def test_restore_disable_flag_ignores_and_clears_the_marker(tmp_path, monkeypatch, capsys):
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    original_cfg = {"maker": {"enabled": False, "port": 8086, "alias": "qwen3.8-27b", "arm": "resident"}}
    cad_json.write_text(json.dumps(original_cfg))
    env_path.write_text("ALIAS='old'\n")

    monkeypatch.setattr(arms.subprocess, "run", lambda cmd, **kw: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(arms, "_wait", lambda url, timeout: None)

    arms.cmd_use(_fake_arm(tmp_path, "new-arm"), cad_json=cad_json, env_path=env_path)
    pre_cad = tmp_path / "cad.json.pre-arm"
    pre_env = tmp_path / "maker.env.pre-arm"
    assert pre_cad.exists() and pre_env.exists()

    arms.cmd_restore(cad_json=cad_json, env_path=env_path, disable=True)
    cfg = json.loads(cad_json.read_text())
    assert cfg["maker"]["enabled"] is False
    assert cfg["maker"]["alias"] == "new-arm"    # NOT rolled back: --disable ignores the marker
    assert not pre_cad.exists() and not pre_env.exists()
    assert "maker disabled" in capsys.readouterr().out


def test_nested_use_keeps_the_outermost_pre_arm_state(tmp_path, monkeypatch):
    """scripts/run_card.py calls cmd_use once per arm without restoring in between, so the
    FIRST call's snapshot is the one that matters -- it is the state from before the whole
    card run, and later calls in the same run must not clobber it."""
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    original_cfg = {"maker": {"enabled": False, "port": 8086, "alias": "qwen3.8-27b", "arm": "resident"}}
    cad_json.write_text(json.dumps(original_cfg))
    env_path.write_text("ALIAS='resident-shim'\n")

    monkeypatch.setattr(arms.subprocess, "run", lambda cmd, **kw: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(arms, "_wait", lambda url, timeout: None)

    arms.cmd_use(_fake_arm(tmp_path, "arm-a"), cad_json=cad_json, env_path=env_path)
    arms.cmd_use(_fake_arm(tmp_path, "arm-b"), cad_json=cad_json, env_path=env_path)
    assert json.loads(cad_json.read_text())["maker"]["alias"] == "arm-b"

    saved = json.loads((tmp_path / "cad.json.pre-arm").read_text())
    assert saved["maker"] == original_cfg["maker"]   # still the ORIGINAL, not arm-a's

    arms.cmd_restore(cad_json=cad_json, env_path=env_path)
    assert json.loads(cad_json.read_text()) == original_cfg
    assert env_path.read_text() == "ALIAS='resident-shim'\n"


def test_use_no_start_still_saves_the_pre_arm_marker(tmp_path, monkeypatch):
    """`use --no-start` must follow the same marker rule as a normal use: it still writes
    the pre-arm snapshot (and touches no systemctl unit)."""
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    original_cfg = {"maker": {"enabled": False, "port": 8086, "alias": "qwen3.8-27b", "arm": "resident"}}
    cad_json.write_text(json.dumps(original_cfg))

    calls: list[list[str]] = []
    monkeypatch.setattr(arms.subprocess, "run",
                        lambda cmd, **kw: calls.append(list(cmd)) or type("R", (), {"returncode": 0})())

    def must_not_be_called(url, timeout):
        raise AssertionError("no-start must not wait on the maker health endpoint")
    monkeypatch.setattr(arms, "_wait", must_not_be_called)

    arms.cmd_use(_fake_arm(tmp_path, "arm-x"), start=False, cad_json=cad_json, env_path=env_path)
    assert calls == []   # --no-start touches no systemctl unit at all
    assert (tmp_path / "cad.json.pre-arm").exists()
    assert (tmp_path / "maker.env.pre-arm").exists()
    assert json.loads(cad_json.read_text())["maker"]["alias"] == "arm-x"


# ---------------------------------------------------------------------------------------
# 2026-09-19: owner deleted the losing Phase 0 GGUFs to reclaim disk. Their arms.json
# entries stay (card results reference them by name) but gain "retired": true so `list`
# marks them and `use` refuses them instead of failing later with a confusing
# missing-model_path SystemExit.
# ---------------------------------------------------------------------------------------

def test_retired_arm_shown_in_list_and_refused_by_use(tmp_path, capsys):
    src = json.loads((HERE / "benchmarks" / "arms.json").read_text())
    store = tmp_path / "store"
    kept = {**src["arms"][0], "skip": False}          # a normal, non-retired arm
    retired = {
        "name": "retired-test-arm", "alias": "retired-test-alias", "role": "candidate",
        "gguf": "retired-test/model.gguf", "mmproj": None, "ctx": 16384, "extra_args": "",
        "hf": None, "notes": "test fixture",
        "retired": True,
        "retired_note": "GGUF deleted by the owner 2026-09-19; card results kept under benchmarks/results/card/",
    }
    arms_json = tmp_path / "arms.json"
    arms_json.write_text(json.dumps({"store": str(store), "arms": [kept, retired], "critics": []}))

    a = arms.load_arms(arms_json)
    assert a["retired-test-arm"]["retired"] is True
    assert a[kept["name"]].get("retired", False) is False   # a kept arm is not flagged

    arms.cmd_list(a)
    out = capsys.readouterr().out
    lines = {line.split()[1]: line for line in out.splitlines()}
    assert lines["retired-test-arm"].startswith("retired")
    assert not lines[kept["name"]].startswith("retired")

    with pytest.raises(SystemExit, match="retired-test-arm is retired"):
        arms.cmd_use(a["retired-test-arm"], cad_json=tmp_path / "cad.json", env_path=tmp_path / "maker.env")
    assert not (tmp_path / "cad.json").exists()   # refused before any write
