"""v5 command-line entry — `cad "<spec>"`.

One command: build, then refine interactively (the default). Flags only tune speed and where the
result is shown; there is intentionally no separate one-shot subcommand — pass --once, or just type
`done` after the first build.
"""
import sys
import argparse

from . import loop, targets, config, engine


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
        description="build123d CAD agent v5 — describe a part, build it, refine it by chatting.")
    p.add_argument("spec", nargs="*", help="what to build (omit to be prompted)")
    p.add_argument("--coder", choices=["auto", "fast", "strong"], default="auto",
                   help="coder model strategy (default: auto — fast, escalate on failure)")
    p.add_argument("--target", default=None,
                   help="output target: cad-viewer (default) | onshape | fstl | file")
    p.add_argument("--onshape", action="store_true",
                   help="use Onshape as the output target (shortcut for --target onshape)")
    p.add_argument("--no-fewshots", action="store_true",
                   help="disable few-shot retrieval (A/B the learning lift)")
    p.add_argument("--once", action="store_true",
                   help="build once and exit — skip the interactive refine prompt")
    a = p.parse_args(argv)

    target_name = "onshape" if a.onshape else (a.target or targets.default_target_name())
    spec = " ".join(a.spec).strip() or None

    print(f"CAD agent v{config.VERSION}  (engine v{engine.ENGINE_VERSION})"
          f" · output: {target_name} · coder: {a.coder}")
    loop.run(spec=spec, coder=a.coder, target_name=target_name,
             use_fewshots=not a.no_fewshots, interactive=not a.once)


if __name__ == "__main__":
    main()
