#!/usr/bin/env python3
"""lab/data.py -- render gate-verified SFT pairs into Gemma 4 prompt/completion rows.

Reads ~/.openclaw/cad-sft-train.jsonl and cad-sft-val.jsonl (ChatML `messages`: system,
user, assistant), extracts the verbatim user spec, drops rows whose spec collides with a
public or internal benchmark suite (exact key or unique near-duplicate slug, the same rule
scripts/run_card.py's contamination() uses), and renders the surviving rows through the
checkpoint's own Jinja chat template with thinking off. Output is
{"prompt", "completion", "id", "kind"} JSONL, not {"text"}, so the trainer can mask the
prompt tokens out of the loss and train only on the completion.

  python3 lab/data.py --src-train ~/.openclaw/cad-sft-train.jsonl \
                       --src-val ~/.openclaw/cad-sft-val.jsonl --out lab/data

Template: the checkpoint ships its own chat_template.jinja (Gemma 4's private
<|turn>role\\n...<turn|> framing). Rendering with that exact file, not a hand-rolled copy,
is the point: any drift between what this script renders and what the checkpoint's own
tokenizer.apply_chat_template() would render at inference is training on the wrong string.
A small built-in fallback template exists for test runs and for a machine that never
downloaded the checkpoint; it reproduces the exact framing this script depends on for a plain
system+user+assistant conversation with no tool calls, images, or thinking traces, which is
all the source data ever contains. It is NOT reached by accident: since fix round 3 (finding
9) a missing checkpoint template is a hard error unless --allow-fallback-template is passed,
because silently rendering a whole training set through a stand-in framing the checkpoint
never used is an expensive, invisible failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jinja2

HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))
import harvest_census as hc  # noqa: E402

DEFAULT_TEMPLATE = Path(
    "/mnt/nvme-apps/LinuxModels/gemma-4-31B-it-unsloth-bnb-4bit/chat_template.jinja")
DEFAULT_SRC_TRAIN = Path.home() / ".openclaw" / "cad-sft-train.jsonl"
DEFAULT_SRC_VAL = Path.home() / ".openclaw" / "cad-sft-val.jsonl"
DEFAULT_OUT = HERE / "lab" / "data"

# No `-` (whitespace-trim) markers here: every literal fragment below is a plain Python
# string concatenation with no accidental newlines or indentation between tags, so nothing
# needs trimming, and the `\n` characters written explicitly (the real framing) must survive.
FALLBACK_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages and messages[0]['role'] in ['system', 'developer'] %}"
    "<|turn>system\n{{ messages[0]['content'] | trim }}<turn|>\n"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "<|turn>{{ 'model' if message['role'] == 'assistant' else message['role'] }}\n"
    "{{ message['content'] | trim }}<turn|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|turn>model\n"
    "{% if not enable_thinking %}<|channel>thought\n<channel|>{% endif %}"
    "{% endif %}"
)

# cad_engine.py builds the coder-prompt user message in three different shapes depending on
# which prompt function wrote it (grep-confirmed against cad_engine.py on this branch):
#   - generate_code / generate_code_raw (cad_engine.py:1592, :1672): a block header
#     "USER REQUEST (verbatim ... AUTHORITATIVE...):" with the spec on the following
#     line(s), up to the first blank line.
#   - revise_script (cad_engine.py:1892, the GIFT-FAIL repair prompt): "Target part: <spec>"
#     inline, on the same line as the header.
#   - decide_or_edit (cad_engine.py:1934, the agent-loop revise turn): "User request: <spec>"
#     inline, on the same line as the header.
# Fix round 1 (2026-09-17): the first version of this function only recognised the first
# shape, so extract_spec() silently returned "" for every "Target part:"/"User request:"
# row (78 of 353 real training rows: all 45 fail-kind GIFT-FAIL pairs plus 33 good-kind
# revise-turn pairs) and render_pairs()'s `if spec:` guard then skipped the contamination
# check on those rows entirely, with no drop and no warning. Recognising all three headers
# fixes the miss; extract_spec() returning "" now means none of the three headers were
# found at all, which render_pairs() treats as a fail-closed drop, never a silent pass.
_SPEC_BLOCK_HEADER = "USER REQUEST"
_SPEC_INLINE_PREFIXES = ("Target part: ", "User request: ")


def extract_spec(user_content: str) -> str:
    """Pull the verbatim spec out of a coder prompt's user message, across all three header
    shapes cad_engine.py emits (see the module-level comment above). Returns "" if none of
    them are found."""
    lines = user_content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(_SPEC_BLOCK_HEADER):
            out = []
            for follow in lines[i + 1:]:
                if follow.strip() == "":
                    break
                out.append(follow)
            return "\n".join(out).strip()
        for prefix in _SPEC_INLINE_PREFIXES:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    return ""


def load_template(path: Path | None, allow_fallback: bool = False):
    """Return (jinja_template, from_checkpoint). `tojson` is registered because the real
    checkpoint template family uses it for structured tool-call arguments; harmless to
    register even when the specific template in use does not call it.

    Fix round 3 (finding 9): the checkpoint's own template is MANDATORY by default. A moved,
    renamed or never-downloaded checkpoint used to fall back to FALLBACK_TEMPLATE with a
    single printed line, so an unattended rerun could train on a framing the checkpoint never
    used. Callers that genuinely want the stand-in (tests, a machine without the checkpoint)
    pass allow_fallback=True, which is what --allow-fallback-template sets."""
    env = jinja2.Environment()
    env.filters["tojson"] = json.dumps
    if path and Path(path).exists():
        return env.from_string(Path(path).read_text()), True
    if not allow_fallback:
        raise SystemExit(
            f"chat template not found: {path}. Rendering training rows through the built-in "
            f"fallback framing instead of the checkpoint's own template is a silent, "
            f"expensive mistake, so it now has to be asked for: pass "
            f"--allow-fallback-template (or allow_fallback=True) if that is really what you "
            f"want."
        )
    return env.from_string(FALLBACK_TEMPLATE), False


def default_contamination_sets():
    """Exact keys and near-duplicate slugs for every card suite spec (internal + public).

    Mirrors scripts/run_card.py's contamination(): a slug is trusted as evidence of a
    match only when it identifies exactly one suite spec, since a public suite like
    text2cadquery collapses dozens of specs onto one truncated 40-char opening -- a slug
    shared by 28 specs is not evidence about any single one of them.

    Fix round 3 (finding 13): uniqueness is counted PER SUITE, through
    harvest_census.suite_slug_counts(), exactly as run_card.contamination() counts it. The
    earlier global count made a slug appearing once in each of two suites non-unique here
    while run_card still treated it as evidence in both, i.e. this guard was marginally more
    permissive than the one it claims to mirror."""
    keys = hc.suite_keys()
    unique_slugs: set[str] = set()
    for counts in hc.suite_slug_counts(40).values():
        unique_slugs |= {slug for slug, n in counts.items() if n == 1}
    return keys, unique_slugs


def _tally_reasons(dropped: list[tuple[str, str]]) -> dict[str, int]:
    """Drop reasons grouped by category (the part of the reason string before the first
    ':', or the whole reason when there is no ':') so a run's drop counts are auditable at
    a glance instead of only as a per-id list."""
    tally: dict[str, int] = {}
    for _rid, reason in dropped:
        cat = reason.split(":", 1)[0]
        tally[cat] = tally.get(cat, 0) + 1
    return tally


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return float(s[k])


def scan_special_tokens(src: Path) -> list[str]:
    """Row ids whose raw message text already contains the framing tokens this script is
    about to introduce. If any do, rendering would be ambiguous (the trained model could
    not tell a real turn boundary from one embedded in the data)."""
    hits = []
    if not src.exists():
        return hits
    with src.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for m in row.get("messages", []):
                content = m.get("content", "")
                if "<|turn>" in content or "<turn|>" in content:
                    hits.append(f"{src.name}#{i}")
                    break
    return hits


_TURN_END = "<turn|>"


def _make_completion(assistant_content: str) -> str:
    """assistant content, framed with exactly one closing <turn|> marker.

    Source rows are gate-verified build123d code and should never contain the literal
    framing token, but a naive `.strip() + "<turn|>\\n"` would double it up if one ever did
    (e.g. a stray copy-paste of a rendered example). Strip an existing trailing marker
    first, then append it once."""
    content = assistant_content.strip()
    if content.endswith(_TURN_END):
        content = content[: -len(_TURN_END)].rstrip()
    return content + _TURN_END + "\n"


def render_pairs(src: Path, out: Path, keys=None, slugs=None, template=None, tag="row"):
    """Render one source JSONL of ChatML SFT rows into Gemma-4 prompt/completion rows.

    `keys`/`slugs` are the contamination sets from default_contamination_sets(); passing
    them in explicitly (as tests do) lets a planted spec be checked against a small,
    controlled set instead of the real benchmark suites on disk.

    Returns (kept, dropped): kept is the list of row dicts written to `out`, dropped is a
    list of (id, reason) for every row skipped as contaminated or malformed."""
    if keys is None or slugs is None:
        d_keys, d_slugs = default_contamination_sets()
        keys = d_keys if keys is None else keys
        slugs = d_slugs if slugs is None else slugs
    if template is None:
        template, _ = load_template(DEFAULT_TEMPLATE)

    kept: list[dict] = []
    dropped: list[tuple[str, str]] = []
    with src.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = f"{tag}-{i:04d}"
            msgs = row.get("messages", [])
            system_msg = next((m for m in msgs if m["role"] == "system"), None)
            user_msg = next((m for m in msgs if m["role"] == "user"), None)
            assistant_msg = next((m for m in msgs if m["role"] == "assistant"), None)
            if not (system_msg and user_msg and assistant_msg):
                dropped.append((row_id, "missing a system/user/assistant message"))
                continue
            spec = extract_spec(user_msg["content"])
            if not spec:
                # Fail closed: none of the known headers matched, so contamination could
                # not be checked. Never keep a row unchecked (see the note above
                # extract_spec for the bug this replaced).
                dropped.append((row_id, "no-spec-header"))
                continue
            key = hc._key(spec)
            slug = hc._slug(spec, 40)
            if key in keys:
                dropped.append((row_id, f"exact suite match: {spec[:70]}"))
                continue
            if slug in slugs:
                dropped.append((row_id, f"near-duplicate suite slug: {spec[:70]}"))
                continue
            prompt = template.render(messages=[system_msg, user_msg], bos_token="",
                                      add_generation_prompt=True)
            completion = _make_completion(assistant_msg["content"])
            kept.append({"prompt": prompt, "completion": completion, "id": row_id,
                         "kind": row.get("kind", "unknown")})

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    return kept, dropped


def _print_length_stats(label: str, kept: list[dict], tokenizer) -> None:
    if not kept:
        print(f"  {label}: no rows")
        return
    plens = [len(r["prompt"]) for r in kept]
    flens = [len(r["prompt"]) + len(r["completion"]) for r in kept]
    print(f"  prompt chars p50/p95: {_pctl(plens, 50):.0f}/{_pctl(plens, 95):.0f}")
    print(f"  prompt+completion chars p50/p95: {_pctl(flens, 50):.0f}/{_pctl(flens, 95):.0f}")
    if tokenizer is None:
        print("  token stats: skipped (pass --tokenizer <checkpoint dir> with `tokenizers` "
              "installed to enable)")
        return
    tlens = [len(tokenizer.encode(r["prompt"]).ids) for r in kept]
    flens_t = [len(tokenizer.encode(r["prompt"] + r["completion"]).ids) for r in kept]
    print(f"  prompt tokens p50/p95: {_pctl(tlens, 50):.0f}/{_pctl(tlens, 95):.0f}")
    print(f"  prompt+completion tokens p50/p95: "
          f"{_pctl(flens_t, 50):.0f}/{_pctl(flens_t, 95):.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-train", type=Path, default=DEFAULT_SRC_TRAIN)
    ap.add_argument("--src-val", type=Path, default=DEFAULT_SRC_VAL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output directory; writes train.jsonl and val.jsonl under it")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="checkpoint chat_template.jinja to render with")
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="directory holding tokenizer.json; enables real token-count stats")
    ap.add_argument("--allow-fallback-template", action="store_true",
                    help="render with the built-in stand-in framing when the checkpoint's own "
                         "chat_template.jinja is missing (default: refuse)")
    a = ap.parse_args()

    keys, slugs = default_contamination_sets()
    template, from_checkpoint = load_template(a.template, allow_fallback=a.allow_fallback_template)
    if from_checkpoint:
        print(f"template: checkpoint ({a.template})")
    else:
        print(f"template: fallback ({a.template} not found; framing is a stand-in, not the "
              f"real checkpoint template)")

    tokenizer = None
    if a.tokenizer:
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(a.tokenizer / "tokenizer.json"))
        except Exception as e:
            print(f"token stats skipped: {e}")

    for src in (a.src_train, a.src_val):
        hits = scan_special_tokens(src)
        if hits:
            print(f"WARNING: {len(hits)} row(s) in {src} already contain <|turn> or "
                  f"<turn|>: {hits[:10]}")
        else:
            print(f"no pre-existing <|turn>/<turn|> tokens found in {src}")

    for tag, src, out_name in (("train", a.src_train, "train.jsonl"),
                                ("val", a.src_val, "val.jsonl")):
        out = a.out / out_name
        kept, dropped = render_pairs(src, out, keys=keys, slugs=slugs, template=template,
                                      tag=tag)
        print(f"\n{tag}: {len(kept)} rows kept, {len(dropped)} dropped -> {out}")
        for rid, reason in dropped:
            print(f"  dropped {rid}: {reason}")
        if dropped:
            tally = _tally_reasons(dropped)
            tally_str = ", ".join(f"{cat}={n}" for cat, n in sorted(tally.items()))
            print(f"  dropped reasons tallied: {tally_str}")
        _print_length_stats(tag, kept, tokenizer)


if __name__ == "__main__":
    main()
