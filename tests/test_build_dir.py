import importlib, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def test_new_build_dir_keeps_itself_and_prunes_oldest_by_mtime(tmp_path, monkeypatch):
    import cad_engine
    monkeypatch.setattr(cad_engine, "BUILDS_DIR", tmp_path)
    monkeypatch.setattr(cad_engine, "KEEP_BUILDS", 3)
    # names chosen so a name sort would put the NEW date-named dir first (the old bug)
    for i, name in enumerate(["fluid_a", "fluid_b", "fluid_c"]):
        d = tmp_path / name; d.mkdir(); os.utime(d, (1_000 + i, 1_000 + i))
    new = cad_engine._new_build_dir("A block")
    assert new.exists(), "the freshly created build dir must never be pruned"
    survivors = sorted(p.name for p in tmp_path.iterdir())
    assert new.name in survivors and len(survivors) == 3
    assert "fluid_a" not in survivors and "fluid_c" in survivors   # oldest by mtime went first
