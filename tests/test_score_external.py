"""Tests for scripts/score_external.py, scoring CAD code written by an outside model
through the exact same one-shot path scripts/run_card.py scores the local arm with.

Offline: everything that would touch a model, the GPU, or the network is either genuinely
CPU-only (build123d execution via scripts/step + scripts/inspect, when the runtime is
available) or monkeypatched. engine._ollama is stubbed by score_external's own
_capture_prompt helper for the repair-prompt tests, same trick as scripts/compile_sft.py's
reconstruct_prompt, no network call is possible through it.

Run: python3 -m pytest tests/test_score_external.py tests/test_run_card.py -q
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import score_external as se  # noqa: E402  (plain import, scripts/ is on sys.path above,
                             # same as any other scripts/*.py importing its sibling modules)


@pytest.fixture(autouse=True)
def _clear_suite_cache():
    """score_external memoizes load_suite() results in a module-level dict, clear it so one
    test's fake suite never leaks into the next."""
    se._SUITE_CACHE.clear()
    yield
    se._SUITE_CACHE.clear()


def _fake_suite(monkeypatch, suite_name: str, specs: list[dict], acc: dict | None = None):
    """Point score_external.spec_and_crit() at an in-memory suite instead of a real
    benchmarks/<suite>/specs.json, so these tests need no on-disk suite fixtures."""
    table = {suite_name: (specs, acc or {})}
    monkeypatch.setattr(se.run_card, "load_suite", lambda name: table.get(name, ([], {})))


# ── row schema: a REAL build123d script through the real one-shot runners ──────────────


def test_row_schema_matches_run_row_keys_on_a_real_build(tmp_path):
    """A trivial real build123d script (a 20mm box), run through score_code + finish_row ,
    the same fluid_gen._materialize() call one-shot itself uses. Confirms both the row
    carries every key run_card.run_row() produces (so lift_report/card_report read it like
    any local-arm row) AND the extra fields this harness adds, and that a clean build scores
    ok with zero gate findings and no salvage/repair (score_code never repairs)."""
    pytest.importorskip("build123d")
    code_path = tmp_path / "cadprompt__unit-01.py"
    code_path.write_text("from build123d import *\nresult = Box(20, 20, 20)\n")
    build_dir = tmp_path / "build"
    spec = {"id": "unit-01", "tier": 0, "spec": "a 20mm cube"}
    crit = {"solids": 1}

    m, wall = se.score_code(code_path, build_dir, spec["spec"])
    row = se.finish_row("selfcheck", "cadprompt", spec, crit, m, wall, 123, build_dir)

    assert se.RUN_ROW_KEYS <= set(row)
    assert se.EXTRA_ROW_KEYS <= set(row)
    assert row["ok"] is True
    assert row["gate_hard"] == 0 and row["gate_spec"] == 0
    assert row["acc_passed"] == row["acc_total"] == 1        # crit={"solids": 1}, box has 1 solid
    assert row["mode"] == "oneshot" and row["subset"] == "claude-sub" and row["candidates"] == 1
    assert row["salvaged"] is False and row["no_salvage"] is True
    assert row["code_model"] == "claude-code-sub/selfcheck"
    assert row["tokens_out"] == 123
    assert row["error"] is None and row["stderr_tail"] == ""
    assert row["band"] is None                                # crit carries no reference_stl


def test_crashing_script_is_ok_false_with_no_automatic_repair(tmp_path):
    """score_code calls fluid_gen._materialize (not _materialize_with_salvage): a script that
    raises gets exactly one execution attempt, never an automatic fix."""
    pytest.importorskip("build123d")
    code_path = tmp_path / "cadprompt__unit-02.py"
    code_path.write_text("from build123d import *\nresult = this_name_does_not_exist\n")
    build_dir = tmp_path / "build"
    spec = {"id": "unit-02", "tier": 0, "spec": "irrelevant"}

    m, wall = se.score_code(code_path, build_dir, spec["spec"])
    row = se.finish_row("selfcheck", "cadprompt", spec, None, m, wall, None, build_dir)

    assert row["ok"] is False
    assert row["error"]
    assert "this_name_does_not_exist" in row["error"] or "NameError" in row["error"]
    assert row["gate_hard"] == 0 and row["gate_spec"] == 0     # never reached the gate
    assert row["stderr_tail"] != ""


# ── missing file: a row, never a skip ───────────────────────────────────────────────────


def test_missing_code_file_gives_ok_false_row_not_a_skip(tmp_path, monkeypatch):
    _fake_suite(monkeypatch, "unit-suite", [{"id": "01", "tier": 0, "spec": "a widget"}])
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps([["unit-suite", "01"]]))
    code_dir = tmp_path / "code"
    code_dir.mkdir()                                            # empty, no 01.py in it
    out = tmp_path / "card"

    args = _ns(arm="ext", code_dir=str(code_dir), ids=str(ids_file), out=str(out), tokens_json="")
    se.cmd_score(args)

    rows = se.read_rows(out / "rows.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["ok"] is False
    assert "missing code file" in row["error"]
    assert row["build_dir"] == ""
    assert se.RUN_ROW_KEYS <= set(row)


# ── resume: an (arm, suite, id) already in rows.jsonl is never rescored ────────────────


def test_resume_skips_done_rows(tmp_path, monkeypatch):
    _fake_suite(monkeypatch, "unit-suite", [
        {"id": "a", "tier": 0, "spec": "spec a"},
        {"id": "b", "tier": 0, "spec": "spec b"},
    ])
    out = tmp_path / "card"
    out.mkdir()
    done_row = se._build_row("ext", "unit-suite", {"id": "a", "tier": 0}, None, ok=True,
                             gate_hard=0, gate_spec=0, acc_passed=0, acc_total=0, band=None,
                             wall_s=1.0, tokens_out=None, build_dir="somewhere", error=None,
                             stderr_tail="")
    se.append_row(out / "rows.jsonl", done_row)

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "unit-suite__b.py").write_text("irrelevant, score_code is mocked below")
    calls = []

    def fake_score_code(code_path, build_dir, spec_text):
        calls.append(spec_text)
        return {"error": None, "facts": {"solids": 1}, "gate_hard": [], "gate_spec": []}, 0.5

    monkeypatch.setattr(se, "score_code", fake_score_code)

    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps([["unit-suite", "a"], ["unit-suite", "b"]]))
    args = _ns(arm="ext", code_dir=str(code_dir), ids=str(ids_file), out=str(out), tokens_json="")
    se.cmd_score(args)

    assert calls == ["spec b"]                     # "a" was skipped, never re-scored
    rows = se.read_rows(out / "rows.jsonl")
    assert len(rows) == 2
    assert {(r["suite"], r["id"]) for r in rows} == {("unit-suite", "a"), ("unit-suite", "b")}


# ── seed-baseline: copy one arm's rows for exactly the requested ids ───────────────────


def test_seed_baseline_filters_by_ids_and_reports_missing(tmp_path, capsys):
    src = tmp_path / "phase2_rows.jsonl"
    rows = [
        {"arm": "gemma-4-31b", "suite": "cadprompt", "id": "cp-01", "ok": True},
        {"arm": "gemma-4-31b", "suite": "cadprompt", "id": "cp-02", "ok": True},
        {"arm": "gemma-4-31b", "suite": "text2cadquery", "id": "t2cq-01", "ok": False},
        {"arm": "some-other-arm", "suite": "cadprompt", "id": "cp-01", "ok": True},
    ]
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ids_file = tmp_path / "ids.json"
    # cp-01 has a gemma-4-31b row, cp-99 does not, the seeder must report cp-99 missing and
    # still seed cp-01.
    ids_file.write_text(json.dumps([["cadprompt", "cp-01"], ["cadprompt", "cp-99"]]))
    out = tmp_path / "card"

    args = _ns(seed_baseline="gemma-4-31b", frm=str(src), ids=str(ids_file), out=str(out))
    se.cmd_seed_baseline(args)

    seeded = se.read_rows(out / "rows.jsonl")
    assert len(seeded) == 1
    assert seeded[0] == {"arm": "gemma-4-31b", "suite": "cadprompt", "id": "cp-01", "ok": True}
    err = capsys.readouterr().out
    assert "cp-99" in err

    # Resumable: seeding again does not duplicate the already-seeded row.
    se.cmd_seed_baseline(args)
    assert len(se.read_rows(out / "rows.jsonl")) == 1


# ── the keep rule ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("first_crashed,first_hard,first_spec,rep_error,rep_hard,rep_spec,expected", [
    (True,  0, 0, None,   3, 3, True),    # ran where the first crashed: improvement regardless
    (True,  0, 0, "boom", 0, 0, False),   # still crashes: no improvement
    (False, 2, 1, None,   1, 5, True),    # fewer hard findings, even though [spec] got worse
    (False, 1, 3, None,   1, 1, True),    # same hard, fewer [spec]
    (False, 1, 1, None,   1, 1, False),   # no change
    (False, 1, 1, None,   2, 0, False),   # more hard findings, even with zero [spec]
    (False, 0, 2, "boom", 0, 0, False),   # the repair attempt itself crashed
])
def test_improves_keep_rule(first_crashed, first_hard, first_spec, rep_error, rep_hard, rep_spec, expected):
    assert se.improves(first_crashed, first_hard, first_spec, rep_error, rep_hard, rep_spec) is expected


# ── emit-repairs: selects exactly the specs that need a repair prompt ──────────────────


def test_emit_repairs_selects_crash_and_gate_dirty_but_not_clean(tmp_path, monkeypatch):
    specs = [
        {"id": "clean", "tier": 0, "spec": "a clean part"},
        {"id": "crash", "tier": 0, "spec": "a part that crashes"},
        {"id": "gatedirty", "tier": 0, "spec": "a part with gate findings"},
        {"id": "nofile", "tier": 0, "spec": "a spec with no first-attempt code at all"},
    ]
    _fake_suite(monkeypatch, "unit-suite", specs)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    for sid in ("clean", "crash", "gatedirty"):
        (code_dir / f"unit-suite__{sid}.py").write_text("irrelevant, score_code is mocked")
    # deliberately no unit-suite__nofile.py

    canned = {
        "clean":     ({"error": None, "gate_hard": [], "gate_spec": []}, 0.1),
        "crash":     ({"error": "the script failed to run: NameError: boom"}, 0.1),
        "gatedirty": ({"error": None, "gate_hard": ["unfused bodies: 2 solids"],
                       "gate_spec": ["[spec] measured 40mm, spec said 80mm"]}, 0.1),
    }

    def fake_score_code(code_path, build_dir, spec_text):
        sid = code_path.stem.split("__", 1)[1]
        return canned[sid]

    monkeypatch.setattr(se, "score_code", fake_score_code)

    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps([["unit-suite", s["id"]] for s in specs]))
    out = tmp_path / "card"
    args = _ns(arm="ext", code_dir=str(code_dir), ids=str(ids_file), out=str(out))
    se.cmd_emit_repairs(args)

    repair_dir = out / "repair_prompts" / "ext"
    written = {p.stem for p in repair_dir.glob("*.json")}
    assert written == {"unit-suite__crash", "unit-suite__gatedirty"}

    crash_row = json.loads((repair_dir / "unit-suite__crash.json").read_text())
    assert crash_row["why"].startswith("crash:")
    assert crash_row["system"] and crash_row["prompt"]
    assert "boom" in crash_row["prompt"] or "NameError" in crash_row["prompt"]

    gate_row = json.loads((repair_dir / "unit-suite__gatedirty.json").read_text())
    assert gate_row["why"].startswith("gate:")
    assert "unfused bodies" in gate_row["prompt"]
    assert "measured 40mm" in gate_row["prompt"]

    # The captured prompt is the real engine._REVISE_SYSTEM text, not a stand-in, proof the
    # _ollama-stub trick actually ran engine.revise_script rather than being mocked away too.
    assert "debugging build123d" in crash_row["system"]


def test_emit_repairs_prompt_capture_makes_no_model_call(tmp_path, monkeypatch):
    """The _ollama stub must be restored even when revise_script is under test, this asserts
    engine._ollama is back to its real self after emit-repairs runs, and that the real
    function was never invoked (call count instrumented)."""
    _fake_suite(monkeypatch, "unit-suite", [{"id": "crash", "tier": 0, "spec": "x"}])
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "unit-suite__crash.py").write_text("irrelevant")
    monkeypatch.setattr(se, "score_code",
                        lambda *a, **k: ({"error": "boom: NameError"}, 0.1))

    calls = []
    real_ollama = se.engine._ollama

    def spy(*a, **k):
        calls.append(a)
        return real_ollama(*a, **k)

    monkeypatch.setattr(se.engine, "_ollama", spy)

    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps([["unit-suite", "crash"]]))
    out = tmp_path / "card"
    args = _ns(arm="ext", code_dir=str(code_dir), ids=str(ids_file), out=str(out))
    se.cmd_emit_repairs(args)

    assert calls == []                       # the real _ollama was never reached
    assert se.engine._ollama is spy          # score_external restored ITS stub, not our spy


# ── scoring the repaired code: keep-rule wiring end to end ─────────────────────────────


def test_score_repair_keeps_the_improved_build_and_carries_forward_otherwise(tmp_path, monkeypatch):
    _fake_suite(monkeypatch, "unit-suite", [
        {"id": "improves", "tier": 0, "spec": "x"},
        {"id": "worsens", "tier": 0, "spec": "y"},
        {"id": "norepairfile", "tier": 0, "spec": "z"},
    ])
    out = tmp_path / "card"
    out.mkdir()
    base_rows = [
        se._build_row("ext", "unit-suite", {"id": "improves", "tier": 0}, None, ok=False,
                      gate_hard=0, gate_spec=0, acc_passed=0, acc_total=1, band=None,
                      wall_s=1.0, tokens_out=None, build_dir="", error="the script failed to run: boom",
                      stderr_tail="boom"),
        se._build_row("ext", "unit-suite", {"id": "worsens", "tier": 0}, None, ok=True,
                      gate_hard=1, gate_spec=0, acc_passed=0, acc_total=1, band=None,
                      wall_s=1.0, tokens_out=None, build_dir="/orig/worsens", error=None,
                      stderr_tail=""),
        se._build_row("ext", "unit-suite", {"id": "norepairfile", "tier": 0}, None, ok=True,
                      gate_hard=1, gate_spec=0, acc_passed=0, acc_total=1, band=None,
                      wall_s=1.0, tokens_out=None, build_dir="/orig/norepairfile", error=None,
                      stderr_tail=""),
    ]
    for r in base_rows:
        se.append_row(out / "rows.jsonl", r)

    repair_dir = tmp_path / "repairs"
    repair_dir.mkdir()
    (repair_dir / "unit-suite__improves.py").write_text("irrelevant")
    (repair_dir / "unit-suite__worsens.py").write_text("irrelevant")
    # no unit-suite__norepairfile.py

    canned = {
        "improves": ({"error": None, "gate_hard": [], "gate_spec": []}, 2.0),   # now runs: improvement
        "worsens":  ({"error": None, "gate_hard": ["still unfused"], "gate_spec": []}, 2.0),  # same hard (1==1), not fewer
    }

    def fake_score_code(code_path, build_dir, spec_text):
        sid = code_path.stem.split("__", 1)[1]
        return canned[sid]

    monkeypatch.setattr(se, "score_code", fake_score_code)

    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps([["unit-suite", s] for s in
                                    ("improves", "worsens", "norepairfile")]))
    args = _ns(arm="ext+repair", code_dir=str(tmp_path / "code"), ids=str(ids_file),
              out=str(out), repair_dir=str(repair_dir), tokens_json="")
    se.cmd_score_repair(args)

    rows = {r["id"]: r for r in se.read_rows(out / "rows.jsonl") if r["arm"] == "ext+repair"}
    assert set(rows) == {"improves", "worsens", "norepairfile"}

    assert rows["improves"]["repaired"] is True
    assert rows["improves"]["ok"] is True
    assert rows["improves"]["no_salvage"] is False

    assert rows["worsens"]["repaired"] is False           # gate_hard stayed 1==1, kept the first attempt
    assert rows["worsens"]["gate_hard"] == 1               # carried the FIRST attempt's row forward
    assert rows["worsens"]["build_dir"] == "/orig/worsens"
    assert rows["worsens"]["no_salvage"] is False           # still marked repair-eligible

    assert rows["norepairfile"]["repaired"] is False       # no repair file at all: carried forward
    assert rows["norepairfile"]["build_dir"] == "/orig/norepairfile"


def _ns(**kw):
    """argparse.Namespace with every flag score_external.py's subcommands read, defaulted so
    a test only has to name the ones it cares about."""
    from argparse import Namespace
    base = dict(arm="", code_dir="", ids="", out="", tokens_json="",
               seed_baseline="", frm="", emit_repairs=False, repair_dir="")
    base.update(kw)
    return Namespace(**base)
