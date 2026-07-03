#!/usr/bin/env python3
"""
RAG module for the CAD Agent v2 — three-tier knowledge retrieval.

Tier 1 — Feature primitives (highest trust, injected closest to output schema):
  capabilities/*.md hand-crafted cards + MCP builder BTM patterns

Tier 2 — Internet examples (medium trust, compositional inspiration):
  FsDoc library, Onshape API docs, GitHub raw files

Tier 3 — Feedback (low volume, high precision — capped at 1 per shape_type):
  cad-feedback.jsonl — best-rated example per shape class, orientation only

Injection order in prompt: Tier3 (top, orientation) → Tier2 → Tier1 (right before schema)

Requires: pip install chromadb --break-system-packages
Embeddings: nomic-embed-text via Ollama

Usage:
    python3 cad_rag.py index [--source capabilities|mcp|apidocs|github|feedback|all] [--verbose]
    python3 cad_rag.py query "threaded lid container"
"""

import os, sys, json, re, hashlib, urllib.request, urllib.error
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

OLLAMA_EMBED_URL    = "http://localhost:11434/api/embeddings"
EMBED_MODEL         = "nomic-embed-text"
EMBED_TIMEOUT_S     = 30

CHROMA_DIR          = Path.home() / ".openclaw" / "cad-rag"

COLLECTION_CAPS     = "cad_capabilities"   # Tier 1 — capability cards + MCP builders
COLLECTION_DOCS     = "cad_apidocs"        # Tier 2 — internet examples
COLLECTION_FEEDBACK = "cad_feedback"       # Tier 3 — capped past builds

CHUNK_MAX_CHARS     = 2000
CHUNK_MIN_CHARS     = 80
CHUNK_OVERLAP_CHARS = 50

# Tier limits: how many chunks to retrieve per tier
TIER1_TOP_K = 3
TIER2_TOP_K = 3
TIER3_TOP_K = 1   # one reference build max

_CAPS_DIR      = Path(__file__).parent / "capabilities"
_BUILDERS_DIR  = Path.home() / ".openclaw/skills/onshape-mcp/onshape_mcp/builders"
_FEEDBACK_FILE = Path.home() / ".openclaw/cad-feedback.jsonl"

# ── Static docs to scrape ──────────────────────────────────────────────────────

FSDOC_URLS = [
    "https://cad.onshape.com/FsDoc/library.html",
]

API_DOC_URLS = [
    "https://onshape-public.github.io/docs/api-intro/",
]

# GitHub raw files — known-good, static, no JS required
GITHUB_RAW_URLS = [
    "https://raw.githubusercontent.com/onshape-public/onshape-clients/master/python/README.md",
    "https://raw.githubusercontent.com/PTC-Education/Onshape-Integration-Guides/main/API_Intro.md",
]

# ── ChromaDB client (lazy singleton) ──────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _get_collection(name: str):
    return _get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

# ── Embedding ──────────────────────────────────────────────────────────────────

def embed(text: str) -> list:
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT_S) as r:
        resp = json.loads(r.read())
    vec = resp.get("embedding")
    if not vec:
        raise RuntimeError(f"embed: unexpected keys: {list(resp.keys())}")
    return vec

# ── Deduplication ──────────────────────────────────────────────────────────────

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _already_indexed(collection, doc_id: str) -> bool:
    result = collection.get(ids=[doc_id], include=[])
    return len(result["ids"]) > 0

# ── Chunking ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str,
                max_chars: int = CHUNK_MAX_CHARS,
                min_chars: int = CHUNK_MIN_CHARS) -> list:
    chunks = []
    for para in text.split("\n\n"):
        if len(para) <= max_chars:
            if len(para.strip()) >= min_chars:
                chunks.append(para.strip())
        else:
            buf = ""
            for line in para.split("\n"):
                if len(buf) + len(line) + 1 <= max_chars:
                    buf = (buf + "\n" + line).lstrip("\n")
                else:
                    if len(buf.strip()) >= min_chars:
                        chunks.append(buf.strip())
                    if len(line) > max_chars:
                        pos = 0
                        while pos < len(line):
                            piece = line[pos:pos + max_chars]
                            if len(piece.strip()) >= min_chars:
                                chunks.append(piece.strip())
                            pos += max_chars - CHUNK_OVERLAP_CHARS
                        buf = ""
                    else:
                        buf = line
            if len(buf.strip()) >= min_chars:
                chunks.append(buf.strip())
    return chunks


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, rep in [("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"')]:
        text = text.replace(ent, rep)
    return re.sub(r"\s{3,}", "\n\n", text).strip()


def _chunk_python_builder(source: str, filename: str) -> list:
    # Match class and def at ANY indentation — gives method-level regions
    boundaries = [m.start() for m in re.finditer(r"^[ \t]*(class |def )", source, re.MULTILINE)]
    boundaries.append(len(source))

    chunks = []
    for i in range(len(boundaries) - 1):
        region = source[boundaries[i]:boundaries[i + 1]]
        if '"btType"' not in region and "'btType'" not in region:
            continue
        text = f"# File: {filename}\n" + region
        if len(text) > CHUNK_MAX_CHARS:
            text = text[:CHUNK_MAX_CHARS] + "\n# [truncated]"
        if len(text.strip()) >= CHUNK_MIN_CHARS:
            chunks.append({"text": text, "source": filename})
    return chunks

# ── Tier 1 indexers: capability cards + MCP builders ──────────────────────────

def index_capabilities(verbose: bool = False) -> int:
    if not _CAPS_DIR.exists():
        if verbose:
            print(f"[RAG] capabilities dir not found: {_CAPS_DIR}")
        return 0

    col = _get_collection(COLLECTION_CAPS)
    new_count = 0

    for md_file in sorted(_CAPS_DIR.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            if verbose:
                print(f"[RAG] caps: could not read {md_file.name}: {e}")
            continue

        # Each capability card is a single chunk (they're already well-scoped)
        # but we split on ## sections for cards longer than CHUNK_MAX_CHARS
        chunks = _chunk_text(text, max_chars=CHUNK_MAX_CHARS)

        for chunk in chunks:
            # Prepend filename so the model knows the capability topic
            full = f"# Capability: {md_file.stem.replace('_', ' ')}\n{chunk}"
            doc_id = _content_hash(full)
            if _already_indexed(col, doc_id):
                continue
            try:
                vec = embed(full)
            except Exception as e:
                if verbose:
                    print(f"[RAG] embed failed for {md_file.name}: {e}")
                continue
            col.add(ids=[doc_id], embeddings=[vec], documents=[full],
                    metadatas={"source": "capability", "filename": md_file.name, "tier": 1})
            new_count += 1
            if verbose:
                print(f"[RAG] caps: indexed {md_file.name} ({len(full)} chars)")

    return new_count


def index_mcp_builders(verbose: bool = False) -> int:
    if not _BUILDERS_DIR.exists():
        if verbose:
            print(f"[RAG] builders dir not found: {_BUILDERS_DIR}")
        return 0

    col = _get_collection(COLLECTION_CAPS)   # Tier 1 alongside capability cards
    new_count = 0

    for py_file in sorted(_BUILDERS_DIR.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
        except Exception as e:
            if verbose:
                print(f"[RAG] mcp: could not read {py_file.name}: {e}")
            continue

        chunks = _chunk_python_builder(source, py_file.name)
        if verbose and not chunks:
            print(f"[RAG] mcp: {py_file.name} — no btType chunks found")

        for chunk in chunks:
            doc_id = _content_hash(chunk["text"])
            if _already_indexed(col, doc_id):
                continue
            try:
                vec = embed(chunk["text"])
            except Exception as e:
                if verbose:
                    print(f"[RAG] embed failed for {py_file.name}: {e}")
                continue
            col.add(ids=[doc_id], embeddings=[vec], documents=[chunk["text"]],
                    metadatas={"source": "mcp_builder", "filename": chunk["source"], "tier": 1})
            new_count += 1
            if verbose:
                print(f"[RAG] mcp: indexed chunk from {py_file.name} ({len(chunk['text'])} chars)")

    return new_count

# ── Tier 2 indexers: internet docs + GitHub ────────────────────────────────────

def _fetch_url(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CADAgentBot/1.0)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def index_apidocs(verbose: bool = False) -> int:
    col = _get_collection(COLLECTION_DOCS)
    new_count = 0

    all_urls = API_DOC_URLS + FSDOC_URLS

    for url in all_urls:
        try:
            raw = _fetch_url(url)
        except Exception as e:
            if verbose:
                print(f"[RAG] apidocs: failed to fetch {url}: {e}")
            continue

        # Detect if HTML or Markdown
        text = _strip_html(raw) if raw.strip().startswith("<") or "<html" in raw[:200].lower() else raw
        if len(text) < 500:
            if verbose:
                print(f"[RAG] apidocs: {url} appears JS-rendered (<500 chars) — skipping")
            continue

        chunks = _chunk_text(text)
        added = 0
        for chunk in chunks:
            doc_id = _content_hash(chunk)
            if _already_indexed(col, doc_id):
                continue
            try:
                vec = embed(chunk)
            except Exception as e:
                if verbose:
                    print(f"[RAG] embed failed for chunk: {e}")
                continue
            col.add(ids=[doc_id], embeddings=[vec], documents=[chunk],
                    metadatas={"source": "apidoc", "url": url, "tier": 2})
            new_count += 1
            added += 1

        if verbose:
            print(f"[RAG] apidocs: {url} → {len(chunks)} chunks, {added} new")

    return new_count


def index_github(verbose: bool = False) -> int:
    col = _get_collection(COLLECTION_DOCS)
    new_count = 0

    for url in GITHUB_RAW_URLS:
        try:
            raw = _fetch_url(url)
        except Exception as e:
            if verbose:
                print(f"[RAG] github: failed to fetch {url}: {e}")
            continue

        if len(raw.strip()) < CHUNK_MIN_CHARS:
            if verbose:
                print(f"[RAG] github: {url} returned empty content — skipping")
            continue

        chunks = _chunk_text(raw)
        added = 0
        for chunk in chunks:
            doc_id = _content_hash(chunk)
            if _already_indexed(col, doc_id):
                continue
            try:
                vec = embed(chunk)
            except Exception as e:
                if verbose:
                    print(f"[RAG] embed failed: {e}")
                continue
            col.add(ids=[doc_id], embeddings=[vec], documents=[chunk],
                    metadatas={"source": "github", "url": url, "tier": 2})
            new_count += 1
            added += 1

        if verbose:
            print(f"[RAG] github: {url} → {len(chunks)} chunks, {added} new")

    return new_count

# ── Tier 3 indexer: feedback — capped at 1 per shape_type ─────────────────────

def index_feedback(verbose: bool = False) -> int:
    if not _FEEDBACK_FILE.exists():
        if verbose:
            print("[RAG] cad-feedback.jsonl not found — skipping")
        return 0

    # Read all rows, group by shape_type, keep only best-rated per class
    best: dict = {}   # shape_type -> row with highest rating
    with open(_FEEDBACK_FILE) as f:
        for line in f:
            try:
                row = json.loads(line.strip())
            except Exception:
                continue
            if row.get("rating", 0) < 4:
                continue
            st = row.get("shape_type", "unknown")
            if st not in best or row["rating"] > best[st]["rating"]:
                best[st] = row

    col = _get_collection(COLLECTION_FEEDBACK)
    new_count = 0

    for shape_type, row in best.items():
        spec       = row.get("spec", "")
        plan_steps = row.get("plan_steps", [])
        text       = f"spec: {spec}\nsteps: {json.dumps(plan_steps)}"
        new_id     = _content_hash(text)

        # Check if a different (lower-rated) entry already exists for this shape_type
        existing = col.get(where={"shape_type": shape_type}, include=["metadatas"])
        for ex_id, ex_meta in zip(existing["ids"], existing["metadatas"]):
            if ex_id != new_id:
                old_rating = ex_meta.get("rating", 0)
                if row["rating"] >= old_rating:
                    col.delete(ids=[ex_id])
                    if verbose:
                        print(f"[RAG] feedback: evicted lower-rated {shape_type} entry (was {old_rating}★)")

        if _already_indexed(col, new_id):
            continue

        try:
            vec = embed(text)
        except Exception as e:
            if verbose:
                print(f"[RAG] embed failed for feedback row: {e}")
            continue

        col.add(ids=[new_id], embeddings=[vec], documents=[text],
                metadatas={"shape_type": shape_type, "rating": int(row["rating"]),
                           "source": "feedback", "url": row.get("url", ""), "tier": 3})
        new_count += 1
        if verbose:
            print(f"[RAG] feedback: indexed best {shape_type} ({row['rating']}★): {spec[:60]!r}")

    return new_count

# ── Tiered query ───────────────────────────────────────────────────────────────

def _query_collection(col_name: str, vec: list, top_k: int) -> list:
    try:
        col   = _get_collection(col_name)
        count = col.count()
        if count == 0:
            return []
        n = min(top_k, count)
        result = col.query(
            query_embeddings=[vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(result["documents"][0],
                                   result["metadatas"][0],
                                   result["distances"][0]):
            out.append({"text": doc, "distance": dist, "metadata": meta,
                        "tier": meta.get("tier", 2)})
        return out
    except Exception:
        return []


def rag_query_tiered(spec: str) -> dict:
    """
    Returns {"tier1": [...], "tier2": [...], "tier3": [...]}
    Each list is ordered by ascending distance (most similar first).
    Gracefully returns empty tiers on any failure.
    """
    try:
        vec = embed(spec)
    except Exception:
        return {"tier1": [], "tier2": [], "tier3": []}

    tier1 = _query_collection(COLLECTION_CAPS, vec, TIER1_TOP_K)
    tier2 = _query_collection(COLLECTION_DOCS, vec, TIER2_TOP_K)
    tier3 = _query_collection(COLLECTION_FEEDBACK, vec, TIER3_TOP_K)

    return {"tier1": tier1, "tier2": tier2, "tier3": tier3}


def rag_query(spec: str, top_k: int = 5) -> list:
    """Flat query for backward compatibility — merges all tiers, sorted by distance."""
    tiered = rag_query_tiered(spec)
    all_results = tiered["tier1"] + tiered["tier2"] + tiered["tier3"]
    seen = set()
    deduped = []
    for r in all_results:
        h = _content_hash(r["text"])
        if h not in seen:
            seen.add(h)
            deduped.append(r)
    deduped.sort(key=lambda x: x["distance"])
    return deduped[:top_k]

# ── Orchestration ──────────────────────────────────────────────────────────────

def index_all(sources: list, verbose: bool = False) -> dict:
    if "all" in sources:
        sources = ["capabilities", "mcp", "apidocs", "github", "feedback"]

    source_map = {
        "capabilities": index_capabilities,
        "mcp":          index_mcp_builders,
        "apidocs":      index_apidocs,
        "github":       index_github,
        "feedback":     index_feedback,
    }

    counts = {}
    total = 0
    for src in sources:
        fn = source_map.get(src)
        if fn is None:
            continue
        if verbose:
            print(f"\n[RAG] === Indexing: {src} ===")
        n = fn(verbose=verbose)
        counts[src] = n
        total += n
        if verbose:
            print(f"[RAG] {src}: {n} new chunks")

    counts["total"] = total
    return counts

# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli_index(argv):
    import argparse
    p = argparse.ArgumentParser(prog="cad_rag.py index")
    p.add_argument("--source", default="all",
                   choices=["capabilities", "mcp", "apidocs", "github", "feedback", "all"])
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    sources = ["capabilities", "mcp", "apidocs", "github", "feedback"] if a.source == "all" else [a.source]
    counts = index_all(sources, verbose=a.verbose)
    print(json.dumps({"ok": True, "indexed": counts}, indent=2))


def _cli_query(argv):
    if not argv:
        print("Usage: cad_rag.py query <spec text>")
        sys.exit(1)
    spec    = " ".join(argv)
    tiered  = rag_query_tiered(spec)

    labels = {1: "FEATURE PRIMITIVE", 2: "INTERNET EXAMPLE", 3: "PAST BUILD"}
    for tier_n in [1, 2, 3]:
        results = tiered[f"tier{tier_n}"]
        if not results:
            continue
        print(f"\n=== Tier {tier_n} — {labels[tier_n]} ===")
        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", "?")
            sim = round(1 - r["distance"], 3)
            print(f"  [{i}] source={src}  similarity={sim}")
            print(f"       {r['text'][:120]!r}")

    if not any(tiered.values()):
        print("No results — run: python3 cad_rag.py index --verbose")


if __name__ == "__main__":
    subcmds = {"index": _cli_index, "query": _cli_query}
    if len(sys.argv) >= 2 and sys.argv[1] in subcmds:
        subcmds[sys.argv[1]](sys.argv[2:])
    else:
        print(f"Usage: {sys.argv[0]} <{'|'.join(subcmds)}> [args...]")
        sys.exit(1)
