"""cad_v5 — structured refactor of the build123d agentic CAD pipeline.

Same validated observe-edit loop, gate, escalation and learning logic as cad_engine.py (v4.3),
re-architected into focused modules with a pluggable OUTPUT TARGET layer. Default target is the
local CAD Viewer (earthtojake text-to-cad skill); Onshape upload is opt-in via --onshape.

Modules (as they exist today):
  config    paths, models, loop constants, config/creds loaders, logging
            (single source of truth — the v4 engine imports its constants from here)
  engine    importlib seam re-exporting the v4 pipeline under stable names
  targets   pluggable output: cad-viewer (default) | onshape | fstl | file
  loop      interactive describe -> build -> refine session loop
  cli       the `cad` command entry point

Planned extractions (currently still inside cad_engine.py, to be split out one at a
time behind engine.py's stable names once benchmark parity is provable):
  geometry / gate / brief / codegen / critic / learning

v4.3 (cad_engine.py) stays in place as the working rollback until v5 passes parity.
"""

VERSION = "5.0"
