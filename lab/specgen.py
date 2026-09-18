#!/usr/bin/env python3
"""lab/specgen.py -- local spec generation on the maker arm (Phase 3 Task 2).

Writes new CAD part specs by prompting the strong-rung coder (engine.CODE_MODEL_STRONG,
the maker arm on :8088 when cad.json's maker.enabled, else the resident on :8086) with the
same family/tier structure scripts/teacher_specgen.py used for its cloud teacher calls --
but entirely local. This module never imports or calls cad_engine._cloud_chat, and never
touches Ollama (retired 2026-09-19; cad_engine._ollama raises on anything but a local:/
cloud/ model tag).

Generated specs are admitted through lab.specbank.add_items(), which applies the same
contamination refusal (exact card-suite key + per-suite-unique-slug) and in-bank
duplicate check that import-teacher uses -- a spec that collides is silently dropped,
counted, never retried.

  python3 lab/specgen.py --once --group plate --n 20         # one family, one call (smoke)
  python3 lab/specgen.py --total 2500 --target-tier34 0.45   # full run (long; controller-launched)

Every invocation ends (success or failure) by running `scripts/arms.py restore`, which
undoes exactly whatever `scripts/arms.py use <arm>` at the start of the run changed --
see the module docstring on `main()` for the two circuit breakers and the wall-clock cap
that keep a bad run from holding the resident down far longer than a supervisor watching
a "few hours" window would expect.

Supervisor checklist (fix round 2, from an operator's point of view -- what to do to
stop this run early, and what to check after any hard kill):
- `kill <pid>` (SIGTERM) and a terminal hangup (SIGHUP) are both caught: they raise
  SpecgenAborted, which runs the same restore bookend as any other stop.
- `kill -9 <pid>` (SIGKILL) cannot be caught by any Python process. After one: the
  resident (qwen38-server) self-heals on its own within some bounded time (the next
  ordinary CAD build's own eviction/resume, or maker-server's 12 hour RuntimeMaxSec
  dead-man switch) -- but cad.json's `maker` block and any stray `.pre-arm` marker files
  do NOT self-heal on their own.
- After any hard kill, run `ls ~/.openclaw/cad.json.pre-arm ~/.openclaw/maker.env.pre-arm`;
  if either file exists, run `python3 scripts/arms.py restore` before trusting the box is
  back to normal, even if qwen38-server already looks healthy.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import cad_engine as engine  # noqa: E402
from lab import specbank  # noqa: E402

ARMS_PY = HERE / "scripts" / "arms.py"
SPECGEN_LOG = HERE / "lab" / "state" / "specgen_log.jsonl"

# Fallback only: used as the --arm default when cad.json has no maker block configured
# yet (a fresh box). Any box that has ever run a CAD build or a card already has a real
# maker.arm/alias on disk, which _default_arm() reads and uses instead -- this module
# never hardcodes which arm is the campaign's current standing pick (fix round 1).
DEFAULT_ARM_FALLBACK = "gemma-4-31b"

MAX_BATCHES_DEFAULT = 200
MAX_HOURS_DEFAULT = 4.0
MAX_CONSECUTIVE_CALL_FAILURES = 3
MAX_CONSECUTIVE_NO_PROGRESS = 8


class SpecgenAborted(RuntimeError):
    """Raised by run_total when a circuit breaker (consecutive call failures, a
    no-progress streak) or the wall-clock cap trips, and by the SIGTERM/SIGHUP handler
    below. main() catches this so every abort path still runs the arms.py restore
    bookend before the process exits non-zero, with the abort reason printed as the
    last line of the log so a supervisor can grep it."""


_CLEANUP_STARTED = False   # set by _mark_cleanup_started(); see _install_signal_handlers


def _signal_handler(signum, frame) -> None:
    """Raise SpecgenAborted so the existing try/finally bookend in main() runs the
    restore step, mirroring lab/gpu_window.sh's trap-based restore-on-signal pattern
    (Phase 2 fixed the same class of bug there: a foreground trap that ran the resident
    restore too early, before the GPU job it was supposed to wait for had actually
    stopped). Idempotent by design: once cleanup has begun (_mark_cleanup_started() was
    called), a second SIGTERM/SIGHUP is ignored here rather than raising again --
    raising a second time WHILE the restore subprocess call is itself blocked would
    interrupt that call via the same PEP 475 mechanism this handler relies on, which
    could abandon the restore mid-way and leave things in a worse state than either
    finishing it or never starting it."""
    if _CLEANUP_STARTED:
        return
    raise SpecgenAborted(f"terminated by signal {signum} ({signal.Signals(signum).name})")


def _install_signal_handlers() -> None:
    """SIGTERM (the default signal `kill <pid>` sends, and the one a detached
    `setsid nohup ...` process is stopped with) and SIGHUP (a terminal hangup) both
    otherwise terminate the process immediately with no exception raised and no
    `finally` executed -- confirmed empirically (Task 2 fix round 1 review), not textbook
    assumption: Python's default disposition for SIGTERM runs no cleanup at all, unlike
    SIGINT, which the runtime converts to KeyboardInterrupt by default. Installed at the
    very start of main(), before the `use` call, so even a signal arriving during the
    initial arm switch is caught (see main()'s should_restore handling for what that
    means for the bookend)."""
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)


def _mark_cleanup_started() -> None:
    global _CLEANUP_STARTED
    _CLEANUP_STARTED = True


# ---------------------------------------------------------------------------
# Families and tier guidance -- copied from scripts/teacher_specgen.py (2026-09-19: that
# file's own calls go through cad_engine._cloud_chat/cloud_config, which is forbidden in
# lab/; only the family tables and the system prompt travel here, retargeted below to the
# local engine._ollama call).
# ---------------------------------------------------------------------------
FAMILIES = [
    ("plate",     26, "tier 1-2: flat plates with holes, slots, chamfered edges, counterbores"),
    ("bracket",   26, "tier 2: L/T/U brackets, gussets, mounting tabs with bolt holes"),
    ("enclosure", 26, "tier 2-3: hollow boxes/cases with wall thickness, lids, bosses, cutouts"),
    ("shaft",     24, "tier 2-3: stepped shafts, keyways, circlip grooves, flats, cross-holes"),
    ("flange",    20, "tier 2-3: pipe/mounting flanges with bolt circles, raised faces, hubs"),
    ("surface",   34, "tier 3: lofted/swept/revolved forms, fillets, drafted walls, shells"),
    ("pattern",   20, "tier 2-3: polar/linear feature patterns, grids of holes, vent slots"),
    ("assembly",  20, "tier 3: 2-3 part assemblies as separate solids that visibly mate"),
]

HARD_FAMILIES = [
    ("engine",    30, "tier 3: ONE body composing 4+ features -- finned cylinders with a "
                      "central bore, head flange and stud holes; cooling-jacketed sleeves; "
                      "ribbed exhaust stubs with mounting flanges"),
    ("housing",   30, "tier 3: ONE body composing 4+ features -- bearing housings with bore, "
                      "bolt flange, ribs and grease port; gearbox end covers with recesses, "
                      "bosses and seal grooves; motor mounts with slots, gussets and pads"),
    ("machined",  30, "tier 3: ONE body composing 4+ features -- brackets/blocks combining "
                      "angled faces, counterbored holes, keyed bores, T-slots, dovetails, "
                      "chamfered pockets, cross-drilled passages"),
    ("manifold",  25, "tier 3: ONE body -- flanged manifolds/elbows/tees with through "
                      "passages, bolt circles on each port face, wall thickness stated"),
    ("mechanism", 40, "tier 4: 2-4 separate solids that genuinely assemble, ALWAYS "
                      "including the fastening elements as their own solids (clevis pins, "
                      "wrist pins, pivot pins, shoulder bolts) sized to their holes with "
                      "running clearance; correct centre distances and phasing"),
]

_TIER_BY_GROUP = {g: 3 for g, _, _ in HARD_FAMILIES}
_TIER_BY_GROUP["mechanism"] = 4
_TIER_BY_GROUP.update({
    "plate": 1, "bracket": 2, "enclosure": 2, "shaft": 2, "flange": 2,
    "surface": 3, "pattern": 2, "assembly": 3,
})

FAMILY_GUIDANCE = {g: guide for g, _, guide in (*FAMILIES, *HARD_FAMILIES)}
FAMILY_N = {g: n for g, n, _ in (*FAMILIES, *HARD_FAMILIES)}
ALL_GROUPS = [g for g, _, _ in FAMILIES] + [g for g, _, _ in HARD_FAMILIES]

_SYSTEM = """\
You write CAD part specifications for a text-to-CAD training corpus. Each spec is one
plain-English sentence a mechanical engineer might type, describing ONE buildable part
(or small assembly when asked) with CONCRETE millimetre dimensions for every major feature.

Rules:
- Every spec self-consistent and physically buildable; features must fit inside the part.
- Vary dimensions, feature counts, and phrasing across specs -- no two alike.
- Do NOT phrase anything as a bare catalog part name (avoid leading with exactly
  "spur gear ...", "hex bolt ...", "W-section I-beam ..." -- describe the geometry instead).
- Return ONLY a JSON array of objects: [{"tier": <1-4>, "spec": "<sentence>"}, ...].
  No markdown fences, no commentary."""

_REPAIR_SUFFIX = (
    "\n\nYour previous reply could not be parsed as a JSON array. Reply again with ONLY "
    "the JSON array, no markdown fences, no commentary, no explanation.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_array(raw: str) -> list:
    raw = re.sub(r"^```(json)?\s*\n?", "", raw.strip())
    raw = raw.rstrip("`").strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON array in reply: {raw[:120]!r}")
    return json.loads(raw[start:end + 1])


def _normalize_items(items: list, group: str) -> tuple[list[dict], dict[str, int]]:
    """Turn a parsed JSON array into normalized spec dicts. Never raises: a malformed
    item (not a dict, no spec, a tier that will not coerce to int) is skipped with a
    counted reason rather than crashing the whole batch on an unhandled AttributeError
    (fix round 1, finding 5 -- a temperature-0.8 reply drifting to a bare string array
    used to hit `x.get("tier")` on a str and blow up here). A missing tier (the key is
    simply absent) still defaults from the group's tier table, same as before -- only a
    PRESENT-but-uncoercible tier is a skip."""
    out: list[dict] = []
    skipped: dict[str, int] = {}
    for x in items:
        if not isinstance(x, dict):
            skipped["not-a-dict"] = skipped.get("not-a-dict", 0) + 1
            continue
        spec = str(x.get("spec") or "").strip()
        if not spec:
            skipped["missing-spec"] = skipped.get("missing-spec", 0) + 1
            continue
        tier_raw = x.get("tier")
        if tier_raw is None:
            tier = _TIER_BY_GROUP.get(group, 2)
        else:
            try:
                tier = int(tier_raw)
            except (TypeError, ValueError):
                skipped["bad-tier"] = skipped.get("bad-tier", 0) + 1
                continue
        out.append({"tier": tier, "group": group, "spec": spec})
    return out, skipped


def _seed_specs(group: str, n: int = 5) -> list[str]:
    """Up to `n` random bank specs from the same group as style exemplars. Falls back to
    any teacher-suite spec when the group has no bank rows yet (a brand-new group on an
    empty bank), and to an empty seed block (the prompt says so) when the bank itself is
    empty -- never crashes for want of examples."""
    bank = specbank.load_bank()
    pool = [r["spec"] for r in bank if r.get("group") == group]
    if not pool:
        pool = [r["spec"] for r in bank if r.get("source") == "teacher-suite"]
    if not pool:
        return []
    random.shuffle(pool)
    return pool[:n]


def gen_family(group: str, n: int, guidance: str, seeds: list[str],
               temperature: float = 0.8) -> tuple[list[dict], dict[str, int]]:
    """One local call to the strong rung, thinking left ON (no_think=False -- variety
    matters more than latency for spec writing, per the Phase 3 plan), with one
    JSON-repair retry on a parse failure. Returns (kept items, shape-skip counts) --
    item-level shape problems never raise (see _normalize_items)."""
    seed_block = "\n".join(f"- {s}" for s in seeds[:5]) or "(no seed examples yet)"
    prompt = (
        f"Part family: {group} -- {guidance}\n\n"
        f"Style exemplars (match their voice and specificity, NOT their dimensions or "
        f"feature mix):\n{seed_block}\n\n"
        f"Write {n} NEW specs for this family as the JSON array described.")
    raw = engine._ollama(engine.CODE_MODEL_STRONG, _SYSTEM, prompt,
                         no_think=False, temperature=temperature)
    try:
        items = _parse_json_array(raw)
    except Exception:
        raw2 = engine._ollama(engine.CODE_MODEL_STRONG, _SYSTEM, prompt + _REPAIR_SUFFIX,
                              no_think=False, temperature=temperature)
        items = _parse_json_array(raw2)
    return _normalize_items(items, group)


def _tally_reasons(refused: list[dict]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for item in refused:
        reason = item.get("reason", "unknown")
        cat = reason.split(":", 1)[0]
        tally[cat] = tally.get(cat, 0) + 1
    return tally


def _log_batch(entry: dict) -> None:
    """Append one JSON line to lab/state/specgen_log.jsonl under an exclusive flock, the
    same append-only-under-flock discipline as lab/specbank.py (fix round 1, finding 4:
    token usage and per-batch outcomes need a durable, greppable record before the real
    2500-spec run, not just a printed summary that scrolls off a supervisor's terminal)."""
    SPECGEN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SPECGEN_LOG, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_batch(group: str, n: int, dry_run: bool = False) -> dict:
    guidance = FAMILY_GUIDANCE.get(group)
    if guidance is None:
        raise SystemExit(f"unknown group {group!r}; known: {', '.join(ALL_GROUPS)}")
    seeds = _seed_specs(group)
    usage_before = dict(engine._USAGE_TOTAL)
    t0 = time.time()
    error: Exception | None = None
    items: list[dict] = []
    skip_counts: dict[str, int] = {}
    try:
        items, skip_counts = gen_family(group, n, guidance, seeds)
    except Exception as e:
        error = e
    elapsed = round(time.time() - t0, 1)
    usage_after = dict(engine._USAGE_TOTAL)
    # engine._USAGE_TOTAL accumulates for the whole process (never reset by this module),
    # so a before/after delta around the two _ollama calls gen_family can make is this
    # batch's own usage, not the running process total.
    prompt_tokens = usage_after.get("prompt_tokens", 0) - usage_before.get("prompt_tokens", 0)
    completion_tokens = (usage_after.get("completion_tokens", 0)
                        - usage_before.get("completion_tokens", 0))

    if error is not None:
        _log_batch({
            "ts": _now(), "group": group, "tier": _TIER_BY_GROUP.get(group),
            "requested": n, "generated": 0, "accepted": 0, "refused": 0, "reasons": {},
            "seconds": elapsed, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "error": str(error),
        })
        raise error

    result = specbank.add_items(items, source="specgen", group_default=group, dry_run=dry_run)
    reasons = _tally_reasons(result["refused"])
    for reason, count in skip_counts.items():
        reasons[reason] = reasons.get(reason, 0) + count
    result["group"] = group
    result["requested"] = n
    result["generated"] = len(items)
    result["seconds"] = elapsed
    result["prompt_tokens"] = prompt_tokens
    result["completion_tokens"] = completion_tokens
    _log_batch({
        "ts": _now(), "group": group, "tier": _TIER_BY_GROUP.get(group),
        "requested": n, "generated": len(items), "accepted": result["accepted"],
        "refused": len(result["refused"]), "reasons": reasons,
        "seconds": elapsed, "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    })
    return result


def run_total(total: int, target_tier34: float, batch_size: int = 20,
              max_batches: int = MAX_BATCHES_DEFAULT,
              max_hours: float = MAX_HOURS_DEFAULT) -> dict:
    """Keep generating batches -- tier 3-4 families first while the bank's tier34_share
    is under `target_tier34`, then round-robin every family -- until the bank has at
    least `total` rows AND at least `target_tier34` of them are tier 3-4.

    Three independent stops, none of which is "keep retrying up to max_batches while a
    dead arm eats the health-wait timeout every time" (fix round 1, findings 2/3):
      - `max_batches` (default 200, lowered from 500): the last-resort hard stop.
      - a wall-clock cap (`max_hours`, default 4h), checked once per batch before it
        starts, so a slow-but-technically-working run still bails on schedule.
      - a call-failure circuit breaker: MAX_CONSECUTIVE_CALL_FAILURES (3) batches in a
        row that RAISE (the model call itself failed -- health wait, connection
        refused, timeout) abort immediately, since every failed attempt means
        `_ensure_default_server` has stopped qwen38-server and started maker-server
        again, i.e. the resident stays down for the whole retry storm, not just the
        intended run.
      - a no-progress guard: MAX_CONSECUTIVE_NO_PROGRESS (8) batches in a row that
        complete WITHOUT raising but add zero new specs (every candidate a duplicate or
        contamination refusal) abort separately -- a healthy arm generating specs
        nobody wants is a different failure mode from a dead arm, so it gets its own
        counter, reset independently of the call-failure counter.
    Every abort raises SpecgenAborted with a clear reason; main() catches it and still
    runs the restore bookend."""
    tier34_groups = [g for g, _, _ in HARD_FAMILIES]
    tier12_groups = [g for g, _, _ in FAMILIES]
    all_groups = tier34_groups + tier12_groups
    batches_run = 0
    log: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    consecutive_call_failures = 0
    consecutive_no_progress = 0
    last_error = None
    start = time.time()
    while batches_run < max_batches:
        elapsed_h = (time.time() - start) / 3600.0
        if elapsed_h >= max_hours:
            raise SpecgenAborted(
                f"wall-clock cap reached after {batches_run} batches "
                f"({elapsed_h:.2f}h >= --max-hours {max_hours})")
        st = specbank.stats()
        if st["total"] >= total and st["tier34_share"] >= target_tier34:
            break
        pool = tier34_groups if st["tier34_share"] < target_tier34 else all_groups
        group = pool[batches_run % len(pool)]
        n = min(batch_size, FAMILY_N.get(group, batch_size))
        try:
            r = run_batch(group, n)
            call_failed = False
        except Exception as e:
            r = {"group": group, "error": str(e), "accepted": 0, "requested": n}
            call_failed = True
            last_error = str(e)
        log.append(r)
        batches_run += 1
        usage_total["prompt_tokens"] += r.get("prompt_tokens") or 0
        usage_total["completion_tokens"] += r.get("completion_tokens") or 0

        if call_failed:
            consecutive_call_failures += 1
        else:
            consecutive_call_failures = 0
        if consecutive_call_failures >= MAX_CONSECUTIVE_CALL_FAILURES:
            raise SpecgenAborted(
                f"circuit breaker: {consecutive_call_failures} consecutive batches failed "
                f"calling the model (last error: {last_error})")

        if not call_failed and r.get("accepted", 0) == 0:
            consecutive_no_progress += 1
        elif not call_failed:
            consecutive_no_progress = 0
        if consecutive_no_progress >= MAX_CONSECUTIVE_NO_PROGRESS:
            raise SpecgenAborted(
                f"no-progress guard: {consecutive_no_progress} consecutive batches added "
                f"zero new specs to the bank (duplicates/refusals saturating)")

    return {"batches": batches_run, "final_stats": specbank.stats(), "log": log,
            "usage_total": usage_total}


def _default_arm() -> str:
    """The arm name currently configured in cad.json's maker block -- so this module
    never hardcodes which arm is the campaign's standing pick (fix round 1, finding 4:
    `LOCK_ALIAS = "gemma-4-31b"` would have silently reset a future promoted arm back to
    gemma-4-31b on every specgen exit). Falls back to DEFAULT_ARM_FALLBACK only when
    cad.json has no maker block yet."""
    m = engine._load_config().get("cad", {}).get("maker") or {}
    return m.get("arm") or m.get("alias") or DEFAULT_ARM_FALLBACK


def _run_arms(*args: str) -> subprocess.CompletedProcess:
    """Run scripts/arms.py <args>, forwarding its stdout/stderr into ours so a
    supervisor's log still shows what it did, but returning the CompletedProcess so a
    non-zero exit is visible to the caller instead of being silently swallowed the way a
    bare `subprocess.run(..., check=False)` with no capture would leave it."""
    p = subprocess.run([sys.executable, str(ARMS_PY), *args], capture_output=True, text=True)
    if p.stdout:
        sys.stdout.write(p.stdout if p.stdout.endswith("\n") else p.stdout + "\n")
    if p.stderr:
        sys.stderr.write(p.stderr if p.stderr.endswith("\n") else p.stderr + "\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", help="single family/group name for a one-batch run")
    ap.add_argument("--n", type=int, default=20, help="specs requested in a one-batch run")
    ap.add_argument("--once", action="store_true",
                    help="run exactly one batch (--group required) -- the smoke-test shape")
    ap.add_argument("--total", type=int, default=0, help="full-run target bank size")
    ap.add_argument("--target-tier34", type=float, default=0.45)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-batches", type=int, default=MAX_BATCHES_DEFAULT)
    ap.add_argument("--max-hours", type=float, default=MAX_HOURS_DEFAULT,
                    help="wall-clock cap for a --total run, checked between batches")
    ap.add_argument("--arm", default=None,
                    help="maker arm for this run (default: whatever cad.json's maker "
                         "block already names, or gemma-4-31b if nothing is configured)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-relock", action="store_true",
                    help="skip the arms.py use/restore bookend entirely (debugging only "
                         "-- never pass this on a real run)")
    a = ap.parse_args()

    if not a.once and not a.total:
        ap.error("pass --once --group NAME (smoke) or --total N (full run)")
    if a.once and not a.group:
        ap.error("--once requires --group")

    arm = a.arm or _default_arm()
    _install_signal_handlers()   # before the `use` call: even a signal during the arm
                                  # switch itself must reach the bookend below.

    if a.no_relock:
        rc = 0
        reason: str | None = None
        try:
            if a.once:
                result = run_batch(a.group, a.n, dry_run=a.dry_run)
                print(json.dumps(result, indent=2))
            else:
                result = run_total(a.total, a.target_tier34, a.batch_size,
                                  max_batches=a.max_batches, max_hours=a.max_hours)
                print(json.dumps({"batches": result["batches"],
                                  "final_stats": result["final_stats"],
                                  "usage_total": result["usage_total"]}, indent=2))
        except SpecgenAborted as e:
            rc = 1
            reason = str(e)
        except Exception as e:
            rc = 1
            reason = f"unexpected error: {e}"
        if reason:
            print(f"specgen abort: {reason}", file=sys.stderr)
        return rc

    rc = 0
    reason = None
    should_restore = True   # default: restore whenever anything might have changed
    keep_maker_prev = os.environ.get("CAD_KEEP_MAKER")
    try:
        use_p = _run_arms("use", arm)
        if use_p.returncode != 0:
            # An ORDINARY failure return (no signal involved): scripts/arms.py use
            # raised/exited before doing anything (e.g. the arm's GGUF is missing), so
            # there is nothing to undo. Calling restore here would incorrectly disable
            # an unrelated pre-existing maker configuration via cmd_restore's no-marker
            # fallback -- see fix round 1's report for why this distinction matters.
            reason = f"scripts/arms.py use {arm} failed (exit {use_p.returncode})"
            rc = 1
            should_restore = False
        else:
            # A signal arriving anywhere AFTER this point raises SpecgenAborted, caught
            # below, with should_restore staying True: `use`'s own _save_pre_arm() call
            # is the very first thing cmd_use does (before any slow systemctl/health-wait
            # call), so by the time control has returned here a pre-arm marker already
            # exists on disk and a restore is always the correct, safe action.
            os.environ.setdefault("CAD_KEEP_MAKER", "1")   # one warm arm across this run
            if a.once:
                result = run_batch(a.group, a.n, dry_run=a.dry_run)
                print(json.dumps(result, indent=2))
            else:
                result = run_total(a.total, a.target_tier34, a.batch_size,
                                  max_batches=a.max_batches, max_hours=a.max_hours)
                print(json.dumps({"batches": result["batches"],
                                  "final_stats": result["final_stats"],
                                  "usage_total": result["usage_total"]}, indent=2))
    except SpecgenAborted as e:
        # Either a circuit breaker/wall-clock abort during generation (use already
        # succeeded, should_restore is already True), or a SIGTERM/SIGHUP caught while
        # `use` itself was still running (see the comment above: safe to restore either
        # way, per the fix round 2 review's own tracing of cmd_use's call order).
        rc = 1
        reason = str(e)
    except Exception as e:
        rc = 1
        reason = f"unexpected error: {e}"
    finally:
        _mark_cleanup_started()   # a second SIGTERM/SIGHUP from here on is ignored
        if should_restore:
            _run_arms("restore")
        # Restore CAD_KEEP_MAKER at the source rather than relying on a test fixture or
        # the caller's own environment to clean it up (fix round 2, finding 3): main()
        # is only ever invoked as this module's own dedicated process today, so the env
        # mutation currently dies with the process regardless -- but the fix belongs
        # here so a future in-process caller (e.g. a webui handler importing this module
        # instead of shelling out to it) is never silently affected by our leftover.
        if keep_maker_prev is None:
            os.environ.pop("CAD_KEEP_MAKER", None)
        else:
            os.environ["CAD_KEEP_MAKER"] = keep_maker_prev
    if reason:
        # Printed AFTER the restore bookend's own output (captured and re-emitted by
        # _run_arms above), so this line is the true last line of the log a supervisor
        # would grep for the abort reason (fix round 1, finding 2/3 requirement).
        print(f"specgen abort: {reason}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
