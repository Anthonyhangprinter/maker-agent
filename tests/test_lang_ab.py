"""Offline tests for benchmarks/lang-ab/ (the CAD-language A/B harness).

All tests here are offline: no network, no GPU, no systemctl, no request to a model
endpoint (:8085/:8086/:8088). Every model call goes through a fake `call_model_fn`; the
retrieval few-shot lookup (which would otherwise hit the local embed-server on :8089) is
monkeypatched to a stub returning [] wherever it would run, so the suite has zero
dependency on which system services happen to be up.

`run_cq.py`'s two real-subprocess tests need the isolated CadQuery venv
(benchmarks/lang-ab/.venv-cq, built by setup_cq_env.sh) and are skipped with a clear
reason when it is absent -- everything else needs only the system build123d/OCP install
this repo already depends on.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
LANG_AB_DIR = HERE / "benchmarks" / "lang-ab"
for _p in (str(HERE), str(HERE / "scripts"), str(LANG_AB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lang_ab  # noqa: E402
import report  # noqa: E402

CQ_VENV_PY = LANG_AB_DIR / ".venv-cq" / "bin" / "python"
RUN_CQ_PY = LANG_AB_DIR / "run_cq.py"


def _cq_venv_or_skip():
    if not CQ_VENV_PY.exists():
        pytest.skip(f"cadquery venv not found at {CQ_VENV_PY}; run "
                    f"benchmarks/lang-ab/setup_cq_env.sh first")


# ---------------------------------------------------------------------------
# Code-block extraction
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_plain_code_unchanged(self):
        assert lang_ab.extract_code("result = 1\n") == "result = 1"

    def test_python_fence_stripped(self):
        raw = "```python\nresult = 1\n```"
        assert lang_ab.extract_code(raw) == "result = 1"

    def test_bare_fence_stripped(self):
        raw = "```\nresult = 1\n```"
        assert lang_ab.extract_code(raw) == "result = 1"

    def test_language_tag_variants_stripped(self):
        for tag in ("cadquery", "openscad", "scad", "py"):
            raw = f"```{tag}\nresult = 1\n```"
            assert lang_ab.extract_code(raw) == "result = 1", tag

    def test_none_and_empty(self):
        assert lang_ab.extract_code(None) == ""
        assert lang_ab.extract_code("") == ""
        assert lang_ab.extract_code("   \n  ") == ""

    def test_whitespace_trimmed(self):
        raw = "\n\n  result = 1  \n\n"
        assert lang_ab.extract_code(raw) == "result = 1"


# ---------------------------------------------------------------------------
# run_cq.py, real subprocess
# ---------------------------------------------------------------------------

class TestRunCqReal:
    def test_good_script_exports_nonempty_step(self, tmp_path):
        _cq_venv_or_skip()
        code = tmp_path / "good.py"
        code.write_text(
            "import cadquery as cq\n"
            "w, d, h = 20.0, 15.0, 10.0\n"
            "result = cq.Workplane('XY').box(w, d, h)\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.step"
        p = subprocess.run(
            [str(CQ_VENV_PY), str(RUN_CQ_PY), str(code), str(out)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=60,
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
        )
        assert p.returncode == 0, p.stderr
        assert out.exists() and out.stat().st_size > 0

    def test_bad_script_nonzero_with_traceback(self, tmp_path):
        _cq_venv_or_skip()
        code = tmp_path / "bad.py"
        code.write_text(
            "import cadquery as cq\n"
            "result = cq.Workplane('XY').box(10, 10, 10).nonexistent_method_xyz()\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.step"
        p = subprocess.run(
            [str(CQ_VENV_PY), str(RUN_CQ_PY), str(code), str(out)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=60,
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
        )
        assert p.returncode != 0
        assert "AttributeError" in p.stderr
        assert "Traceback" in p.stderr
        assert not out.exists() or out.stat().st_size == 0


# ---------------------------------------------------------------------------
# Error classification, from real tracebacks
# ---------------------------------------------------------------------------

class TestClassifyError:
    def test_attribute_error_from_cadquery_is_api_misuse(self, tmp_path):
        _cq_venv_or_skip()
        code = tmp_path / "bad.py"
        code.write_text(
            "import cadquery as cq\n"
            "result = cq.Workplane('XY').box(10, 10, 10).nonexistent_method_xyz()\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.step"
        p = subprocess.run(
            [str(CQ_VENV_PY), str(RUN_CQ_PY), str(code), str(out)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=60,
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
        )
        assert "AttributeError" in p.stderr
        assert lang_ab.classify_error(p.stderr) == "api_misuse"

    def test_name_error_is_import_or_name(self):
        try:
            exec(compile("result = undefined_variable_xyz\n", "<t>", "exec"), {})
        except NameError:
            tb = traceback.format_exc()
        assert "NameError" in tb
        assert lang_ab.classify_error(tb) == "import_or_name"

    def test_syntax_error_is_syntax(self):
        try:
            compile("def f(:\n    pass\n", "<t>", "exec")
        except SyntaxError:
            tb = traceback.format_exc()
        assert "SyntaxError" in tb
        assert lang_ab.classify_error(tb) == "syntax"

    def test_kernel_signature_beats_attribute_error(self):
        tb = "Traceback...\nAttributeError: OCP.TopoDS.TopoDS_Shape has no attribute 'x'\n"
        assert lang_ab.classify_error(tb) == "kernel"

    def test_no_code_and_timeout_flags_win(self):
        assert lang_ab.classify_error("anything", no_code=True) == "no_code"
        assert lang_ab.classify_error("anything", timed_out=True) == "timeout"

    def test_empty_text_is_other(self):
        assert lang_ab.classify_error("") == "other"
        assert lang_ab.classify_error(None) == "other"


# ---------------------------------------------------------------------------
# Fixture suites (two tiny public-suite-shaped fixtures with real reference STLs)
# ---------------------------------------------------------------------------

def _write_box_stl(build123d_dims: tuple[float, float, float], path: Path) -> None:
    from build123d import Box, export_stl
    path.parent.mkdir(parents=True, exist_ok=True)
    export_stl(Box(*build123d_dims), str(path))


def _make_fixture_suites(root: Path) -> None:
    """Two tiny suites shaped like benchmarks/cadprompt and benchmarks/text2cadquery:
    specs.json (plain list), acceptance.json ({id: {reference_stl, normalized}}), refs/."""
    suites = {
        "cadprompt": [
            ("cpf-01", "A box 20x15x10mm"),
            ("cpf-02", "A box 30x20x8mm"),
        ],
        "text2cadquery": [
            ("t2f-01", "A box 25x12x6mm"),
            ("t2f-02", "A box 18x18x18mm"),
        ],
    }
    for suite, specs in suites.items():
        d = root / suite
        d.mkdir(parents=True, exist_ok=True)
        specs_json = [{"id": sid, "tier": 0, "spec": text} for sid, text in specs]
        (d / "specs.json").write_text(json.dumps(specs_json), encoding="utf-8")
        acc = {}
        for sid, text in specs:
            dims = _dims_from_text(text)
            ref = f"refs/{sid}.stl"
            _write_box_stl(dims, d / ref)
            acc[sid] = {"reference_stl": ref, "solids": 1, "normalized": False}
        (d / "acceptance.json").write_text(json.dumps(acc), encoding="utf-8")


_DIMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)")


def _dims_from_text(text: str) -> tuple[float, float, float]:
    m = _DIMS_RE.search(text)
    assert m, f"fixture spec text has no NxNxN dims: {text!r}"
    return tuple(float(x) for x in m.groups())


def fake_call_model_factory():
    """A fake call_model that returns a box in whichever language the system prompt
    names, sized from the NxNxN dims embedded in the user prompt's spec text -- and
    counts its own calls, keyed by a short fingerprint of the user prompt, so tests can
    assert on how many (and which) calls were actually made."""
    calls = []

    def fake_call_model(system: str, user: str, temperature: float = 0.2) -> str:
        calls.append({"system_head": system[:40], "user": user})
        l, w, h = _dims_from_text(user)
        sys_lower = system.lower()
        if "build123d" in sys_lower:
            return f"```python\nfrom build123d import *\nresult = Box({l}, {w}, {h})\n```"
        if "cadquery" in sys_lower:
            return f"```python\nimport cadquery as cq\nresult = cq.Workplane('XY').box({l}, {w}, {h})\n```"
        if "openscad" in sys_lower:
            return (f"/* [Dimensions] */\nwidth = {l}; // [1:1:500]\ndepth = {w}; // [1:1:500]\n"
                    f"height = {h}; // [1:1:500]\n$fn = 32; // [16:8:128]\n"
                    f"cube([width, depth, height], center=true);\n")
        raise AssertionError(f"unrecognised system prompt head: {system[:80]!r}")

    fake_call_model.calls = calls
    return fake_call_model


@pytest.fixture()
def fixture_suite_root(tmp_path):
    root = tmp_path / "benchmarks"
    _make_fixture_suites(root)
    return root


@pytest.fixture(autouse=True)
def _no_real_retrieval(monkeypatch):
    """Every test in this file is offline by contract (see the module docstring): stub out
    the retrieval few-shot lookup so a b123d/b123d-nofs codegen call never reaches the
    local embed-server, regardless of whether it happens to be up on this box."""
    monkeypatch.setattr(lang_ab.engine, "retrieval_notes_for", lambda *a, **k: [])


def _openscad_headless_ok() -> bool:
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            scad = Path(td) / "t.scad"
            stl = Path(td) / "t.stl"
            scad.write_text("cube([1,1,1]);\n", encoding="utf-8")
            import openscad_gen
            ok, _err = openscad_gen.compile_scad(scad, stl)
            return ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Deterministic, same-specs-for-every-arm selection
# ---------------------------------------------------------------------------

class TestSelectSpecs:
    def _suites(self):
        return {
            "a": [{"id": f"a{i}", "tier": 0, "spec": f"spec a{i}"} for i in range(5)],
            "b": [{"id": f"b{i}", "tier": 0, "spec": f"spec b{i}"} for i in range(5)],
        }

    def test_same_seed_same_selection(self):
        suites = self._suites()
        first = lang_ab.select_specs(suites, 4, seed=42)
        second = lang_ab.select_specs(suites, 4, seed=42)
        assert [s["id"] for s in first] == [s["id"] for s in second]

    def test_selection_spans_both_suites(self):
        suites = self._suites()
        chosen = lang_ab.select_specs(suites, 4, seed=42)
        assert len(chosen) == 4
        assert {s["suite"] for s in chosen} == {"a", "b"}

    def test_n_at_least_total_returns_everything(self):
        suites = self._suites()
        chosen = lang_ab.select_specs(suites, 100, seed=1)
        assert len(chosen) == 10

    def test_stratifies_by_suite_and_tier(self):
        suites = {
            "a": ([{"id": f"a{i}", "tier": 0, "spec": f"x{i}"} for i in range(6)]
                  + [{"id": f"a{i}", "tier": 1, "spec": f"x{i}"} for i in range(6, 9)]),
        }
        chosen = lang_ab.select_specs(suites, 3, seed=7)
        tiers = {s.get("tier", 0) for s in chosen}
        # 9 specs, 6 tier0 / 3 tier1, cap 3: ceil(3*6/9)=2 tier0, ceil(3*3/9)=1 tier1 -> both represented
        assert tiers == {0, 1}

    def test_same_specs_used_for_every_arm_end_to_end(self, fixture_suite_root, tmp_path, monkeypatch):
        """The property that actually matters: one selection, reused across every arm."""
        import argparse
        monkeypatch.setattr(lang_ab, "RESULTS_DIR", tmp_path / "results")
        args = argparse.Namespace(
            arms="b123d,cadquery", specs=3, seed=123, suite_root=str(fixture_suite_root),
            run_id="same-specs-test", arm=None, i_know_the_gpu_is_free=True,
        )
        fake = fake_call_model_factory()
        out_dir = lang_ab.run_harness(args, call_model_fn=fake)
        rows = [json.loads(l) for l in (out_dir / "rows.jsonl").read_text().splitlines()]
        ids_by_arm = {arm: sorted(r["spec_id"] for r in rows if r["arm"] == arm)
                      for arm in ("b123d", "cadquery")}
        assert ids_by_arm["b123d"] == ids_by_arm["cadquery"]
        assert len(ids_by_arm["b123d"]) == 3


# ---------------------------------------------------------------------------
# Resume skips done pairs
# ---------------------------------------------------------------------------

class TestResume:
    def test_rerun_with_same_run_id_skips_done_pairs(self, fixture_suite_root, tmp_path, monkeypatch):
        import argparse
        monkeypatch.setattr(lang_ab, "RESULTS_DIR", tmp_path / "results")
        args = argparse.Namespace(
            arms="b123d", specs=4, seed=1, suite_root=str(fixture_suite_root),
            run_id="resume-test", arm=None, i_know_the_gpu_is_free=True,
        )
        fake1 = fake_call_model_factory()
        out_dir = lang_ab.run_harness(args, call_model_fn=fake1)
        rows_after_first = (out_dir / "rows.jsonl").read_text().splitlines()
        assert len(rows_after_first) == 4
        assert len(fake1.calls) == 4

        fake2 = fake_call_model_factory()
        out_dir2 = lang_ab.run_harness(args, call_model_fn=fake2)
        assert out_dir2 == out_dir
        rows_after_second = (out_dir / "rows.jsonl").read_text().splitlines()
        assert len(rows_after_second) == 4          # no duplicate rows appended
        assert len(fake2.calls) == 0                 # nothing re-called: all 4 already done

    def test_resume_adds_a_new_arm_without_rebuilding_the_old_one(self, fixture_suite_root, tmp_path, monkeypatch):
        import argparse
        monkeypatch.setattr(lang_ab, "RESULTS_DIR", tmp_path / "results")
        args1 = argparse.Namespace(
            arms="b123d", specs=4, seed=1, suite_root=str(fixture_suite_root),
            run_id="resume-add-arm", arm=None, i_know_the_gpu_is_free=True,
        )
        lang_ab.run_harness(args1, call_model_fn=fake_call_model_factory())

        args2 = argparse.Namespace(
            arms="b123d,cadquery", specs=4, seed=1, suite_root=str(fixture_suite_root),
            run_id="resume-add-arm", arm=None, i_know_the_gpu_is_free=True,
        )
        fake2 = fake_call_model_factory()
        out_dir = lang_ab.run_harness(args2, call_model_fn=fake2)
        rows = [json.loads(l) for l in (out_dir / "rows.jsonl").read_text().splitlines()]
        assert len(rows) == 8
        assert len(fake2.calls) == 4  # only the new arm's 4 specs were actually called


# ---------------------------------------------------------------------------
# report.py maths, hand-built rows (paired flips exact)
# ---------------------------------------------------------------------------

class TestReportMaths:
    def _row(self, arm, suite, spec_id, tier=0, ok=True, band=None, error_class="none",
              codegen=1.0, build=1.0, tokens_out=100):
        return {
            "arm": arm, "suite": suite, "spec_id": spec_id, "tier": tier, "ok": ok,
            "band": band, "error_class": error_class, "traceback_tail": "" if ok else "Xception: boom",
            "seconds_codegen": codegen, "seconds_build": build, "tokens_out": tokens_out,
        }

    def test_summarize_basic_percentages(self):
        rows = [
            self._row("b123d", "s", "1", ok=True, band="match"),
            self._row("b123d", "s", "2", ok=True, band="valid"),
            self._row("b123d", "s", "3", ok=False, error_class="syntax"),
            self._row("b123d", "s", "4", ok=False, error_class="kernel"),
        ]
        summary = report.summarize(rows)
        s = summary["b123d"]
        assert s["n"] == 4
        assert s["built_pct"] == 50.0
        assert s["match_pct"] == 25.0
        assert s["match_or_valid_pct"] == 50.0
        assert s["error_class_pct"] == {"kernel": 25.0, "syntax": 25.0}

    def test_paired_flips_exact(self):
        # 4 shared specs. spec1: b123d fails, cadquery matches -> +improved (match, built).
        # spec2: b123d matches, cadquery fails -> -worsened (match, built).
        # spec3: both match -> no flip.
        # spec4: both fail -> no flip.
        rows = [
            self._row("b123d", "s", "1", ok=False, error_class="kernel"),
            self._row("cadquery", "s", "1", ok=True, band="match"),
            self._row("b123d", "s", "2", ok=True, band="match"),
            self._row("cadquery", "s", "2", ok=False, error_class="api_misuse"),
            self._row("b123d", "s", "3", ok=True, band="match"),
            self._row("cadquery", "s", "3", ok=True, band="match"),
            self._row("b123d", "s", "4", ok=False, error_class="syntax"),
            self._row("cadquery", "s", "4", ok=False, error_class="syntax"),
        ]
        paired = report.paired_vs_baseline(rows, baseline_arm="b123d")
        p = paired["cadquery"]
        assert p["n_pairs"] == 4
        assert p["match_flips"] == {"improved": 1, "worsened": 1}
        assert p["built_flips"] == {"improved": 1, "worsened": 1}

    def test_paired_excludes_specs_missing_from_either_arm(self):
        rows = [
            self._row("b123d", "s", "1", ok=True, band="match"),
            self._row("cadquery", "s", "1", ok=True, band="match"),
            self._row("b123d", "s", "2", ok=True, band="match"),
            # cadquery never attempted spec 2
            self._row("openscad", "s", "2", ok=True, band="match"),
        ]
        paired = report.paired_vs_baseline(rows, baseline_arm="b123d")
        assert paired["cadquery"]["n_pairs"] == 1
        assert paired["openscad"]["n_pairs"] == 1

    def test_by_tier_breakdown(self):
        rows = [
            self._row("b123d", "s", "1", tier=0, ok=False, error_class="kernel"),
            self._row("cadquery", "s", "1", tier=0, ok=True, band="match"),
            self._row("b123d", "s", "2", tier=1, ok=True, band="match"),
            self._row("cadquery", "s", "2", tier=1, ok=True, band="match"),
        ]
        paired = report.paired_vs_baseline(rows, baseline_arm="b123d")
        by_tier = paired["cadquery"]["by_tier"]
        assert by_tier[0] == {"n": 1, "match_improved": 1, "match_worsened": 0}
        assert by_tier[1] == {"n": 1, "match_improved": 0, "match_worsened": 0}

    def test_render_md_has_no_em_dash_and_mentions_arms(self):
        rows = [self._row("b123d", "s", "1", ok=True, band="match"),
                self._row("cadquery", "s", "1", ok=True, band="match")]
        summary = report.summarize(rows)
        paired = report.paired_vs_baseline(rows)
        md = report.render_md(summary, paired, {"run_id": "x", "n_specs": 1, "seed": 1, "arms": ["b123d", "cadquery"]})
        assert "—" not in md
        assert "b123d" in md and "cadquery" in md

    def test_write_report_round_trips_through_files(self, tmp_path):
        rows_path = tmp_path / "rows.jsonl"
        rows = [self._row("b123d", "s", "1", ok=True, band="match")]
        rows_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (tmp_path / "meta.json").write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
        md = report.write_report(tmp_path)
        assert (tmp_path / "REPORT.md").exists()
        assert (tmp_path / "report.json").exists()
        assert "b123d" in md
        loaded = json.loads((tmp_path / "report.json").read_text())
        assert loaded["summary"]["b123d"]["n"] == 1

    def test_write_report_on_empty_rows_does_not_raise(self, tmp_path):
        md = report.write_report(tmp_path)  # no rows.jsonl at all
        assert "lang-ab report" in md


# ---------------------------------------------------------------------------
# Dry integration run: fake model, 3 specs, all four arms, temp results dir
# ---------------------------------------------------------------------------

class TestDryIntegration:
    def test_full_dry_run_all_arms(self, fixture_suite_root, tmp_path, monkeypatch):
        import argparse
        monkeypatch.setattr(lang_ab, "RESULTS_DIR", tmp_path / "results")

        arms = ["b123d", "b123d-nofs", "cadquery"]
        cq_ok = CQ_VENV_PY.exists()
        if not cq_ok:
            arms.remove("cadquery")
            print("dry run: skipping cadquery arm, .venv-cq not built "
                  "(run benchmarks/lang-ab/setup_cq_env.sh)")
        if _openscad_headless_ok():
            arms.append("openscad")
        else:
            print("dry run: skipping openscad arm, the AppImage could not compile headless "
                  "in this environment")

        args = argparse.Namespace(
            arms=",".join(arms), specs=3, seed=99, suite_root=str(fixture_suite_root),
            run_id="dry-integration", arm=None, i_know_the_gpu_is_free=True,
        )
        fake = fake_call_model_factory()
        out_dir = lang_ab.run_harness(args, call_model_fn=fake)

        rows_path = out_dir / "rows.jsonl"
        assert rows_path.exists()
        rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
        assert len(rows) == 3 * len(arms)

        for arm in arms:
            arm_rows = [r for r in rows if r["arm"] == arm]
            assert len(arm_rows) == 3
            # every row has the full expected shape
            for r in arm_rows:
                for key in ("run_id", "arm", "spec_id", "suite", "tier", "ok", "error_class",
                            "traceback_tail", "band", "chamfer_mm", "facts", "tokens_in",
                            "tokens_out", "seconds_codegen", "seconds_build", "code"):
                    assert key in r, (arm, key)
            # the fake model always returns valid code for a box matching the reference
            # exactly, so every arm should build cleanly here (offline sanity check, not a
            # claim about a real model's real reliability).
            assert all(r["ok"] for r in arm_rows), [
                (r["spec_id"], r["error_class"], r["traceback_tail"][:200]) for r in arm_rows if not r["ok"]
            ]
            assert all(r["band"] in ("match", "valid") for r in arm_rows), [
                (r["spec_id"], r["band"]) for r in arm_rows
            ]

        assert (out_dir / "REPORT.md").exists()
        assert (out_dir / "report.json").exists()
        assert (out_dir / "meta.json").exists()
        report_json = json.loads((out_dir / "report.json").read_text())
        assert set(report_json["summary"]) == set(arms)
