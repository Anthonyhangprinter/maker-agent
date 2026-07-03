# CAD Agent v5 — How It Thinks
### Building general CAD competence on top of models that can't reason in 3D

---

## The thesis (one line)

> A local LLM cannot reliably reason about 3D geometry. So we don't ask it to. We give it a
> **vocabulary of correct-by-construction features**, reduce its job to **recognise-and-call**, and
> **verify every result against the stated intent** — generically.

---

## 1 · The problem, concretely

Ask a local coder model for *"a piston, 80mm bore, three ring grooves, a 20mm wrist-pin bore through
the skirt."* Here's what `qwen3-coder:30b` wrote:

```python
piston = Cylinder(radius=40, height=60)
for i in range(3):
    piston -= Pos(0, 0, 15 + i*15) * Cylinder(radius=38, height=60)  # "grooves"
piston -= Pos(0, 0, 30) * Cylinder(radius=10, height=120)            # "wrist-pin bore"
```

It **reads** plausible. It is **geometrically wrong**:
- The three "grooves" are full-depth Ø76 bores that overlap into **one big hollow** → a thin cup.
- The "wrist-pin bore" is a vertical cylinder → it goes through the **floor**, not the **side**.

Result: a cup with a hole in the bottom. Not a piston.

**This is not a prompt bug. It's the model's nature.** Code models pattern-match syntax; they have
no internal 3D picture. They can't *see* that the grooves merged or that the bore points the wrong
way. A bigger model raises the floor a little — the 30B *is* the biggest local one, and it still
failed — but it does not fix the category of error.

---

## 2 · Two ways to respond

| ❌ Hope it reasons | ✅ Constrain & verify |
|---|---|
| "Use a bigger model / better prompt and trust the output." | "Assume it *can't* reason in 3D. Hand it correct building blocks and check the result." |
| Fails silently — you only find out when you look at the part. | Fails *loudly and deterministically* — the system rejects a wrong part and forces a fix. |
| Doesn't generalise — every part is a fresh gamble. | Generalises — the same machinery covers any part built from the same features. |

v5 takes the right-hand column. It's the same philosophy the deterministic **gate** already embodied
— just extended until it actually carries the weight.

---

## 3 · The architecture — four *general* stages

```
   your words
       │
       ▼
 ┌───────────┐   structured feature intent
 │  BRIEF    │   [{bore, Ø20, orientation: radial}, {groove ×3}]
 │ (qwen3:8b)│
 └─────┬─────┘
       ▼
 ┌───────────┐   composes correct-by-construction helpers
 │  CODEGEN  │   result -= Pos(0,0,22) * cross_bore(20, 200)
 │(qwen3-30b)│   for z in (...): result -= Pos(0,0,z) * ring_groove(40,3,3)
 └─────┬─────┘
       ▼
 ┌───────────┐   real OCCT kernel → exact STEP solid
 │   BUILD   │
 └─────┬─────┘
       ▼
 ┌───────────┐   "does the geometry exhibit each intended feature?"
 │   GATE     │  Ø20 bore present? yes. radial? no — it's axial → REJECT, force edit
 │(deterministic)│
 └─────┬─────┘
       ▼      ▲
   converged? │ no → back to CODEGEN with a specific fix
       │ yes
       ▼
   CAD Viewer + STEP/STL/DXF  →  you refine in plain English
```

Each stage is **general** — nothing in it knows what a "piston" is:

- **Brief**: natural language → a structured list of features. Just NL→JSON. Works for any part.
- **Codegen**: pick helpers and numbers. The model never computes a 3D transform.
- **Gate**: one loop — *for each feature the brief listed, is it present in the geometry?* The check
  `is there a radial Ø20 face?` is the **same code** for a piston, a shaft, or a pump body.
- **Loop**: iterate until every feature verifies.

---

## 4 · Why this is general competence, not whack-a-mole

The natural objection: *"Aren't `cross_bore` and a 'radial bore' check just piston-specific hacks?"*

**No — and the distinction is the whole point:**

- A **`cross_bore`** (radial hole through a wall) appears in shafts, axles, pulleys, hinges,
  manifolds, tubes — *anything* where a hole crosses a wall. So do `ring_groove`, `bolt_circle`,
  `counterbore`, `gusset`, `fillet`, `shell`. These are the **finite, reusable vocabulary of
  mechanical design** — a standard library. Growing it *is* growing general competence.
- The **special-casing** version would be a `wrist_pin_in_a_piston_skirt()` helper. *That* is the
  dead end. `cross_bore()` is a primitive.
- The **gate is not** `if piston: check bore`. It's *"verify each feature in the brief's intent."*
  The piston-ness lives in the brief's extracted intent, **never** in the gate's code.

> General competence = a good **feature vocabulary** + reliable **language→intent** mapping +
> one **general intent-vs-geometry** verifier + **iteration**. Not a pile of part rules.

---

## 5 · The proof: one gate, two unrelated parts

The exact same `verify_expected` gate, driven only by the brief's `feature_checks`:

| Part | Intent | Built right | Built wrong |
|---|---|---|---|
| **Piston** | radial Ø20 bore, 3 grooves | ✅ passes (`Ø20 radial`) | ❌ rejects the cup (`Ø20 axial → must be RADIAL, use cross_bore`) |
| **Shaft** | axial Ø10 bore, radial Ø5 cross-hole | ✅ passes (`Ø10 axial, Ø5 radial`) | ❌ rejects (`Ø5 came out axial → must be RADIAL`) |

No part-specific branch was added between those two rows. The shaft was never coded for. That's
generality.

---

## 6 · The honest limit

One stage still genuinely *reasons*: the **brief's language→intent mapping** — recognising that
*"wrist-pin bore through the skirt"* means a **radial** bore. If that mapping is wrong, everything
downstream faithfully builds the wrong thing.

But that's the right place for the risk to live, because:
- **It's tractable** — language→structured-JSON is something an 8B model does well, far better than
  3D geometry math.
- **It's inspectable** — you can see the structured feature list *before* it builds, and the
  interactive loop lets you correct it in one sentence.

So effort goes into the *language→intent* step and the *vocabulary*, **not** into hand-coding
geometry rules.

---

## 7 · Where it goes next

- **Grow the vocabulary** — each new general primitive (slots, ribs, lugs, keyways, threads-as-features)
  widens competence for free across all parts.
- **Richer intent checks** — promote the count-based checks (grooves, bolt circles) from advisory to
  deterministic as the detectors get more robust.
- **Auto-learn** — verified-correct builds become few-shot examples; recovered failures become
  pitfalls. The system gets better at *recognise-and-call* over time.

---

### TL;DR

The model is a recogniser, not a geometer. So: **constrain it with a feature vocabulary, verify it
against intent, and iterate.** The piston is just a test that the *general* machinery works — and it
does, unchanged, on parts it was never built for.
