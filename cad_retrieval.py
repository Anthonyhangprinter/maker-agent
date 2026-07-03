#!/usr/bin/env python3
"""
cad_retrieval.py — semantic few-shot retrieval for the CAD agent.

Embeds specs with nomic-embed-text (local Ollama) and returns the most similar known-good
build123d examples from the corpus, so a small coder adapts a verified solution instead of
inventing from scratch. Pure stdlib; embeddings are cached on disk; falls back to word-overlap
if embedding is unavailable — it must never hard-fail a build.

Anti-vaporware contract: this is the CONSUMER. If retrieval is wired but shows no measurable lift
on the benchmark suite (run with --no-fewshots vs default), it should be removed, not left dormant.
"""
import hashlib
import json
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Optional

_OPENCLAW    = Path(os.path.expanduser("~/.openclaw"))
CORPUS_FILE  = _OPENCLAW / "cad-examples.jsonl"         # unified gold + rated + auto corpus
LESSONS_FILE = _OPENCLAW / "cad-lessons.jsonl"          # fail->fix lessons (Stage B)
EMB_CACHE    = _OPENCLAW / "cad-embeddings.json"        # {sha1(text): [floats]}
EMBED_MODEL = "nomic-embed-text"
EMBED_URL   = "http://localhost:11434/api/embeddings"
EMBED_TIMEOUT = 30


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(EMB_CACHE.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        EMB_CACHE.write_text(json.dumps(cache))
    except Exception:
        pass


def embed(text: str, cache: Optional[dict] = None) -> Optional[list[float]]:
    """Return the embedding vector for text, or None on any failure (caller falls back)."""
    key = _sha(text)
    if cache is not None and key in cache:
        return cache[key]
    try:
        payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
        req = urllib.request.Request(EMBED_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as r:
            vec = json.loads(r.read()).get("embedding")
        if vec and cache is not None:
            cache[key] = vec
        return vec
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def load_corpus() -> list[dict]:
    """Corpus entries that actually carry build123d code (the only ones worth retrieving)."""
    rows = []
    if CORPUS_FILE.exists():
        for line in CORPUS_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("code"):
                rows.append(row)
    return rows


def _word_overlap(spec: str, rows: list[dict], n: int) -> list[dict]:
    sw = set(re.findall(r"[a-z]{3,}", spec.lower()))
    scored = []
    for row in rows:
        rw = set(re.findall(r"[a-z]{3,}", row.get("spec", "").lower()))
        ov = len(sw & rw)
        if ov:
            scored.append((ov, row))
    scored.sort(key=lambda x: (-x[0], -x[1].get("rating", 0)))
    return [r for _, r in scored[:n]]


def retrieve(spec: str, n: int = 2, min_score: float = 0.55) -> list[dict]:
    """Top-n corpus examples for spec. Semantic (cosine) if embeddings work, else word-overlap.
    Each returned row gets a `_score` and `_how` ('cosine' or 'overlap') for transparency/logging."""
    rows = load_corpus()
    if not rows:
        return []
    cache = _load_cache()
    qvec = embed(spec, cache)
    if qvec is None:
        return _word_overlap(spec, rows, n)
    scored = []
    for row in rows:
        rvec = embed(row["spec"], cache)
        if rvec is None:
            continue
        scored.append((_cosine(qvec, rvec), row))
    _save_cache(cache)
    if not scored:
        return _word_overlap(spec, rows, n)
    scored.sort(key=lambda x: -x[0])
    out = []
    for score, row in scored[:n]:
        if score < min_score:
            break
        row = dict(row)
        row["_score"], row["_how"] = round(score, 3), "cosine"
        out.append(row)
    return out


# ── Stage B: fail->fix lessons ────────────────────────────────────────────────

def load_lessons() -> list[dict]:
    rows = []
    if LESSONS_FILE.exists():
        for line in LESSONS_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("lesson"):
                rows.append(row)
    return rows


DEDUP_SIM = 0.90   # cosine similarity above which two lessons teach the same thing


def _near_duplicates(lesson: str, rows: list[dict]) -> set[int]:
    """Indices of rows whose lesson is semantically the same point in different words
    (exact-text dedup alone let 4 rewordings of one flange-fillet lesson pile up and be
    retrieved together as noise). Graceful: no embeddings → no semantic dedup."""
    cache = _load_cache()
    qvec = embed(lesson, cache)
    if qvec is None:
        return set()
    dupes = set()
    for i, r in enumerate(rows):
        rvec = embed(r.get("lesson", ""), cache)
        if rvec is not None and _cosine(qvec, rvec) >= DEDUP_SIM:
            dupes.add(i)
    _save_cache(cache)
    return dupes


def store_lesson(spec: str, lesson: str, problem: str = "", cap: int = 100) -> None:
    """Append a concrete fail->fix lesson. Deduped exactly AND semantically (the newest
    wording of a repeated lesson replaces the old ones); capped."""
    rows = [r for r in load_lessons() if r.get("lesson", "").strip() != lesson.strip()]
    dupes = _near_duplicates(lesson, rows)
    rows = [r for i, r in enumerate(rows) if i not in dupes]
    rows.append({"spec": spec, "lesson": lesson, "problem": problem[:300],
                 "timestamp": __import__("datetime").datetime.now().isoformat()})
    rows = rows[-cap:]
    # Atomic replace — a crash mid-rewrite must not lose the lessons store.
    tmp = LESSONS_FILE.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    os.replace(tmp, LESSONS_FILE)


def retrieve_lessons(spec: str, n: int = 3, min_score: float = 0.5) -> list[str]:
    """Return up to n pitfalls (lesson strings) most relevant to spec. Semantic, graceful."""
    rows = load_lessons()
    if not rows:
        return []
    cache = _load_cache()
    qvec = embed(spec, cache)
    if qvec is None:
        return [r["lesson"] for r in rows[-n:]]      # no embeddings: most-recent
    scored = []
    for r in rows:
        rvec = embed(r["spec"], cache)
        if rvec is not None:
            scored.append((_cosine(qvec, rvec), r))
    _save_cache(cache)
    scored.sort(key=lambda x: -x[0])
    return [r["lesson"] for s, r in scored[:n] if s >= min_score]


if __name__ == "__main__":  # quick manual test: python3 cad_retrieval.py "a flange with bolt holes"
    import sys
    q = " ".join(sys.argv[1:]) or "a circular flange with a bolt circle"
    hits = retrieve(q, n=3, min_score=0.0)
    print(f"query: {q}\ncorpus: {len(load_corpus())} entries\n")
    for h in hits:
        print(f"  {h.get('_score')}  [{h.get('source')}]  {h['spec'][:70]}")
