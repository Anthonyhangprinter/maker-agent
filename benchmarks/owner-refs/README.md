# Owner references

18 hand-authored parts under `~/CAD/references/<name>/` (spec.txt + model.step/model.stl),
never used as training data (see the "18 owner references" hold-out in
`lab/compile_codefirst.py` and `lab/README.md`). This directory (`specs.json`,
`acceptance.json`, `reference/*.stl`) is a gitignored, on-demand copy of that data in the
same shape every other suite here uses (`scripts/run_card.py`'s `load_suite()`), so the
owner references can be carded through the unchanged harness: `--suites owner-refs`.

Regenerate with `python3 lab/eval_round.py --materialize-only` (CPU only, idempotent -- a
name whose `reference/<name>.stl` already exists is left alone). mm-scale, not normalised
(`acceptance.json`'s `normalized: false`): these are the real part dimensions, not
DeepCAD-style unit geometry like the public suites.
