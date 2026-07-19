"""v5 command-line entry — `cad "<spec>"`.

One command: build, then refine interactively (the default). Flags only tune speed and where the
result is shown; there is intentionally no separate one-shot subcommand — pass --once, or just type
`done` after the first build.
"""
import os
import sys
import argparse

from . import loop, targets, config, engine

# N6 — printpal-style prompting rubric, shown in --help so a good spec is written up front
# (reduces how often the N2 ambiguity gate even triggers). Kept in one place; USER_GUIDE.md and
# SKILL.md carry the same content in doc form, and Satine's /help carries a compact version.
_PROMPT_RUBRIC = """\
GOOD PROMPTS — the 4 S's
  Size      overall envelope in mm, e.g. "120x80x40mm"
  Specs     counts + diameters, e.g. "4x M3 through-holes", "a 20mm bore"
  Surfaces  which faces features live on, e.g. "holes in the floor", "a groove on the outside"
  Symmetry  patterns/spacing, e.g. "6 holes equally spaced on a 60mm bolt circle"

Clearances: push-fit 0.0-0.1mm  ·  slip-fit 0.2mm  ·  loose-fit 0.5-1.0mm
Named hardware is understood — say "M3", "608ZZ bearing", "2020 V-slot" and it just works.

Examples:
  cad "a 120x80x40mm enclosure, 2mm walls, 4x M3 mounting holes in the floor"
  cad "a flange: 80mm OD, 10mm thick, 30mm through-bore, 6x M6 bolt holes on a 60mm bolt circle"
  cad "a shaft 12mm dia x 100mm long with a 4mm cross-hole 20mm from one end"
"""


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # No-LLM parametric commands (E1): `cad params` / `cad regen wall=3 [--source PATH]`.
    if argv and argv[0] in ("params", "regen"):
        from . import params as params_mod
        rest = argv[1:]
        source = None
        if "--source" in rest:
            i = rest.index("--source")
            source = rest[i + 1] if i + 1 < len(rest) else None
            rest = rest[:i] + rest[i + 2:]
        if argv[0] == "params":
            raise SystemExit(params_mod.cmd_params(source))
        raise SystemExit(params_mod.cmd_regen(rest, source))

    p = argparse.ArgumentParser(
        prog="cad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="build123d CAD agent v5 — describe a part, build it, refine it by chatting.",
        epilog=_PROMPT_RUBRIC)
    p.add_argument("spec", nargs="*", help="what to build (omit to be prompted)")
    p.add_argument("--coder", choices=["auto", "fast", "strong", "cloud"], default="auto",
                   help="coder model strategy (default: auto — fast, escalate on failure)")
    p.add_argument("--target", default=None,
                   help="output target: cad-viewer (default) | onshape | freecad | print | fstl | file"
                        " ('print' slices the STL and dry-run-validates the gcode via OrcaSlicer)")
    p.add_argument("--onshape", action="store_true",
                   help="use Onshape as the output target (shortcut for --target onshape)")
    p.add_argument("--image", default=None, metavar="PATH",
                   help="reference photo/sketch (jpg/png/webp) — a local vision model analyzes "
                        "it to guide the build and judges every render against it. With a text "
                        "spec, proportions come from the image and absolute mm from your text; "
                        "ALONE (no spec), the image is the whole request and sensible sizes "
                        "are chosen for you (correct them by refining).")
    p.add_argument("--no-fewshots", action="store_true",
                   help="disable few-shot retrieval (A/B the learning lift)")
    p.add_argument("--candidates", type=int, default=None, metavar="N",
                   help="best-of-N first-turn sampling: draw N initial candidates at varied "
                        "temperatures, the deterministic gate picks the survivor (default: "
                        "cad.json `candidates` or 1)")
    p.add_argument("--once", action="store_true",
                   help="build once and exit — skip the interactive refine prompt")
    p.add_argument("--json", action="store_true",
                   help="with --once: print the result dict as one JSON line on stdout "
                        "(machine consumers — Satine, the benchmark runner)")
    p.add_argument("--ask", action="store_true",
                   help="N2 ambiguity gate: pre-check the spec for a missing critical dimension "
                        "or basic form before building. Interactive sessions always run this "
                        "check; with --json this flag is what turns it on, and a non-empty "
                        "result prints one JSON line {needs_clarification, questions, spec} and "
                        "exits instead of building.")
    a = p.parse_args(argv)

    if a.candidates is not None:
        # The engine resolves candidate count at build time via the env var, so the flag
        # works without threading a parameter through build()'s whole call chain.
        os.environ["CAD_CANDIDATES"] = str(max(1, a.candidates))

    target_name = "onshape" if a.onshape else (a.target or targets.default_target_name())
    spec = " ".join(a.spec).strip() or None

    if a.image:
        from pathlib import Path
        img = Path(a.image).expanduser()
        if not img.is_file():
            p.error(f"--image: file not found: {a.image}")
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            p.error(f"--image: unsupported type {img.suffix} (jpg/png/webp)")
        a.image = str(img)

    if a.json and not a.once:
        p.error("--json requires --once (interactive sessions have no single result)")
    if a.json:
        # Machine mode: the human progress output goes to stderr; stdout carries exactly
        # ONE JSON line — consumers parse it instead of scraping. NOTE: everything in this
        # branch before redirect_stdout runs UN-redirected — no prints besides the contract line.
        import contextlib
        import json as _json
        if a.ask and spec:
            # N2 in machine mode is opt-in (--ask) so the benchmark runner and any other --json
            # caller that doesn't pass it sees ZERO behavior change (load-bearing for scoring).
            # With a reference photo, triage sees the vision analysis too — the image often
            # answers "what basic form?" so the gate shouldn't ask (analysis is cached; the
            # build's own pre-pass reuses it for free).
            triage_spec = spec
            if a.image:
                addendum = engine.image_analysis_text(engine.analyze_reference_image(a.image))
                if addendum:
                    triage_spec = spec + "\n\n" + addendum
            questions = engine.triage_ambiguity(triage_spec)
            if questions:
                print(_json.dumps({"needs_clarification": True, "questions": questions,
                                   "spec": spec}))
                raise SystemExit(0)
        with contextlib.redirect_stdout(sys.stderr):
            result = loop.run(spec=spec, coder=a.coder, target_name=target_name,
                              use_fewshots=not a.no_fewshots, interactive=False,
                              image=a.image)
        print(_json.dumps(result or {"ok": False, "error": "build produced no result"}))
        raise SystemExit(0 if result else 1)

    print(f"CAD agent v{config.VERSION}  (engine v{engine.ENGINE_VERSION})"
          f" · output: {target_name} · coder: {a.coder}"
          + (f" · reference: {a.image}" if a.image else ""))
    loop.run(spec=spec, coder=a.coder, target_name=target_name,
             use_fewshots=not a.no_fewshots, interactive=not a.once, image=a.image)


if __name__ == "__main__":
    main()
