# CADAM Reference — deep read of github.com/Adam-CAD/CADAM (adam.new)

Read 2026-08-02 from the clone at `~/repos/CADAM`. All paths below are relative to that root
unless absolute. CADAM is the open-source web app behind adam.new/cadam: chat → LLM emits
complete OpenSCAD → **the browser compiles it with openscad-wasm** → three.js preview +
regex-derived parameter sliders → STL/SCAD/DXF export. License: **GPL-3.0** (see §8).

---

## 1. CODEGEN — how the OpenSCAD gets written

**Where:** `src/server/aiChat.ts` (single server file: prompt, providers, streaming, billing).
Built on the **Vercel AI SDK** (`streamText` + typed tools from `shared/chatAi.ts`).

**Architecture in one line:** the LLM is an agent with exactly two tools —
`build_parametric_model` (input: `{title, version, code}` = the whole OpenSCAD file) and
`answer_user` (final chat text). There is **no server-side execution of the tool**: the
**browser** compiles the code (see §4), renders a 7-view inspection sheet, uploads it to
Supabase storage, and returns the tool result; the server's `toModelOutput` re-downloads that
sheet and attaches it **as an image** to the tool result the model sees. So the generating
model is its own multimodal critic.

### The system prompt (`PARAMETRIC_AGENT_PROMPT`, aiChat.ts:112)

Key verbatim excerpts (the whole thing is ~150 lines; these are the load-bearing parts):

> `You are Adam, an agentic AI CAD editor that creates and modifies OpenSCAD models. The user can see a live preview of the model on the right while you work.`

The write→inspect→rewrite loop is prompted explicitly:

> `After you call build_parametric_model, the browser compiles the OpenSCAD and returns a multi-view preview sheet covering isometric, front, back, left, right, top, and bottom views. Inspect every view against the user's request. If the code fails to compile, or any view shows missing, wrong, disconnected, non-printable, too-simple, hidden, or visually unclear geometry, call build_parametric_model again with a corrected complete script. Keep looping through write → multi-view screenshot inspection → rewrite until the model is good or you hit the turn limit. Do not stop after the first successful compile unless the preview sheet shows that the model satisfies the request from every view.`

> `Do not finalize just because OpenSCAD compiled. Finalize only because the views look right.`

Domain checklists (anti-lazy-model):

> `Multi-feature checklist before stopping:`
> `- Phone case → hollow phone pocket, wrap-over lip, camera cutout, charging-port opening, side button cutouts, printable wall thickness, all cuts visible.`
> `- Mug → body, hollow interior, rim, base, handle, printable wall thickness.`

Library steering (this is how they get real threads):

> `- For screws, bolts, nuts, threaded rods, or tapped/threaded holes, use BOSL2 instead of trying to build threads from cylinder(), linear_extrude(), or hand-rolled helices. Include <BOSL2/screws.scad> for screw(), screw_hole(), and nut() ... Prefer standard spec strings like "M6x1" or "#8-32" ... set $fn = 64; or higher so threads resolve.`
> `- For organic, curved, swept, or lofted shapes ... use BOSL2 ... <BOSL2/skin.scad> for path_sweep() and skin(), <BOSL2/beziers.scad> ... <BOSL2/rounding.scad> ...`

Parameter discipline (feeds §2 directly):

> `Parameters:`
> `- Declare every editable parameter as a top-of-file variable.`
> `- Use full descriptive snake_case names (e.g. wheel_radius, seat_offset) — never abbreviate ... Names render directly in the parameter panel, so they must read well to the user.`
> `- Annotate each variable with a trailing OpenSCAD Customizer comment so the UI can render the right widget:`
> `    width = 50;        // [10:1:200]    ← min:step:max for sliders`
> `    height = 25;       // [5:50]        ← min:max`
> `    style = "round";   // [round, square, hex]   ← enum options`
> `    enabled = true;    //                ← booleans render as switches`
> `    label = "Cup";     // 24             ← maxLength for free-form strings`
> `- Optionally put a "// Description of the parameter" comment on the line ABOVE the variable ...`
> `- Group related parameters with /* [Group Name] */ section markers.`

Color discipline (feeds §4 colors):

> `- Expose colors as string parameters (e.g. body_color = "SteelBlue"; then color(body_color) ...) so the user can tweak them from the parameter panel. Always name them *_color — the UI uses that suffix to render a color picker. Defaults must be CSS named colors or #RRGGBB hex.`

STL-attachment rule (user uploads a mesh to modify):

> `- You MUST use import("filename.stl") to include the user's original model — DO NOT recreate it from scratch.`
> `- Apply modifications (holes, cuts, extensions) AROUND the imported STL ... Create parameters ONLY for the modifications ...`
> `- ... Always expose rotation_x / rotation_y / rotation_z parameters so the user can fine-tune.`

The prompt ends with a full worked example: user says "a mug", and the prompt shows the
complete expected `code` — customizer-commented params, a `torus()` module, `difference()`
of body/handle/cavity. One worked example, in-domain, showing exactly the desired style.

### Models & providers

Picker list `PARAMETRIC_MODELS` in `src/lib/utils.ts:238`: Gemini 3.1 Pro / Gemini 3.6 Flash,
Claude Fable 5 / Opus 4.8 / Sonnet 5, GPT-5.6 Sol, Grok 4.5, Kimi K3, GLM 5.2. **All frontier
cloud models — there is no local/small-model path anywhere.** Routing (`aiChat.ts`):
Anthropic and Google hit their first-party AI-SDK providers directly; everything else goes
through **OpenRouter**. Anthropic Claude 5 / 4.6+ get adaptive thinking enabled
unconditionally. Prices per model are tabled in `MODEL_PRICES` and converted to billing
tokens at $0.01/token, with an intentionally-expensive fallback price for unknown ids.

### Loop mechanics & error feedback

- `streamText({... stopWhen: stepCountIs(60)})` for parametric — up to **60 agent steps** per
  turn (vs our MAX_TURNS=4). `maxOutputTokens: 64000`.
- **Step 0 is forced**: `prepareStep` pins `toolChoice: build_parametric_model` on the first
  step of a user turn, so the model cannot answer with chat text instead of building.
  Exception: Claude Fable/Mythos reject forced tool_choice → auto tool choice + prompt
  steering, with an `onFinish` logger that flags any parametric turn that ended without a
  build call. Anthropic also rejects forced tools while thinking is on → thinking disabled
  for that one step only.
- **Compile errors:** in `src/components/chat/ChatSession.tsx`, the client's `onToolCall`
  compiles via the worker; on throw it resolves the tool call as
  `output-error` with `errorText: "Compilation failed:\n<OpenSCADError message incl. stderr>"`
  (`OpenSCADError` carries the full `stdErr` array from the wasm run — `src/lib/OpenSCADError.ts`).
  `sendAutomaticallyWhen` then auto-resubmits, so the model sees its own compiler stderr and
  rewrites. Success path returns a message that repeats the inspection instruction:
  > `Compilation successful. Inspect the multi-view render in this tool result against the user request from every visible angle. If any required feature is missing, wrong, too simple, disconnected, non-printable, hidden from some view, or visually unclear, call build_parametric_model again ...`
- The 7-view sheet is generated client-side (`src/utils/meshUtils.ts`,
  `generateInspectionPreview`, views `ISO, FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM`, orthographic
  cameras, labeled panels) and uploaded to `images/<uid>/<conv>/inspection-preview-<toolCallId>`;
  the server (`parametricTools.toModelOutput`, aiChat.ts:912) downloads and inlines it base64.
- Side-calls on **Claude Haiku 4.5**: conversation title and 2 follow-up suggestion pills,
  each a structured-output `generateText` (aiChat.ts:731, 763).

**There is no deterministic geometry verification anywhere.** No inspect step, no measured
dimension check, no manifold/volume assertion beyond "it compiled". Acceptance = the model
looking at its own render sheet. (Compare our gate — §COMPARISON.)

---

## 2. PARAMETERS — the sliders (the feature we want)

**The scheme: parametric-by-construction, zero-LLM at edit time.** The model is told to emit
standard **OpenSCAD Customizer syntax**; the app then *regex-parses the source* to derive the
panel. There is no LLM-emitted parameter schema and no separate metadata channel — **the code
is the schema**. Comment in `shared/parseParameters.ts` explains why:

> "This is the single source of truth for 'what does this CAD model expose as a slider/input?'
> — the model only emits the OpenSCAD `code`, and we derive parameter metadata client-side
> from the variable declarations at the top of the file. That removes the divergent-UI problem
> we had when different models (Claude vs Gemini) produced different shaped parameter arrays
> for the same code: now the same source always renders the same `<ParameterSection>`,
> regardless of provider."

### Parsing (`shared/parseParameters.ts`, ~290 lines, regex — they TODO an AST)

- Scope: only the file **above the first `module`/`function` keyword** ("implementation, not
  API").
- Variable regex `^([a-z0-9A-Z_$]+)\s*=\s*([^;]+);[\t ]*(\/\/[^\n]*)?` per line; values typed
  as number / boolean / quoted string / homogeneous array. **Values referencing another
  variable (computed expressions) are skipped** — and once one is hit, the rest of that group
  section is skipped too (assumed derived).
- Trailing comment decodes: `[min:max]`, `[min:step:max]`, `[a, b:Label, c]` enums (value:Label
  pairs), bare number = step (numbers) or maxLength (strings).
- `// description` on the line immediately above becomes the tooltip/description.
- `/* [Group Name] */` markers start group sections (tracked by source offset).
- Display name: snake_case → Title Case; `$fn` → "Resolution". `number[]` values are
  flattened into per-component sliders with heuristic labels (len 3 + name contains
  size/dimension/body/... → Width/Depth/Height, else X/Y/Z).

### Range/step defaults when the model omits annotations (`src/utils/parameterUtils.ts`)

`calculateParameterRange`: magnitude-based nice-number ranges from the default value
(94 → 0–100, 2.7 → 0–10, 0.5 → 0–1; negative defaults get symmetric range).
`calculateParameterStep`: 1% of range rounded to 1/2/5×10^k. `isMeasurementParameter`
flags mm-ish names (width/height/…/radius/diameter/…) for unit display.

### Color params

`isColorParameter` (`parameterUtils.ts`): a string param whose value parses as a CSS color
(clever canvas `fillStyle` round-trip in `cssToHex`, memoized). Those get a `ColorPicker`
and are grouped into a collapsible **Colors** section under the main **Dimensions** list
(`src/components/parameter/ParameterSection.tsx`). The prompt's `*_color` naming rule makes
this reliable.

### Edit → rebuild path (no LLM)

`ParameterSection` debounces 200 ms → `updateParameter` (`parameterUtils.ts:231`)
**regex-substitutes the assignment in the source** (preserving the trailing customizer
comment) → openscad-wasm recompiles the edited code → viewer updates. Edited code is
persisted back onto the message's tool part (`EditorView.persistParameterEdit`), and the
model's *first-authored* source is stashed once in `metadata.originalCode` so Reset /
slider-home / auto-ranges stay anchored to the original defaults across reloads
(`shared/chatAi.ts` AppUIMessage metadata comment). The wasm CLI also supports true
`-Dname=value` defines (used in the export path, `src/worker/openSCAD.ts:exportFile`).

**Takeaway:** the whole feature is (a) a prompt contract ("annotate every top-of-file
variable"), (b) ~300 lines of parser, (c) a regex substituter + recompile. Model-agnostic,
deterministic, portable.

---

## 3. THE "MESH" BUTTON

`src/components/TextAreaChat.tsx:1646` — the `Mesh` pill toggles the **conversation type**
between `parametric` and `creative`. It is not a render mode: creative conversations run a
completely different pipeline:

- Chat side (`aiChat.ts`): model is **hardcoded to `anthropic/claude-sonnet-4.5`** with the
  short `CREATIVE_AGENT_PROMPT` ("You are Adam, a concise 3D mesh assistant. Use the
  create_mesh tool whenever the user asks for a generated, edited, or stylized 3D asset...")
  and one tool, `create_mesh` (max 5 steps).
- Server mesh pipeline (`src/server/mesh.ts`, ~1600 lines): **text → seed image → image-to-3D
  via fal.ai**, with three quality tiers (picker `CREATIVE_MODELS` in utils.ts):
  - `ultra` "Max Quality" → seed image → **Meshy v6 preview** image-to-3D (quad/tri topology,
    target polycount 200–300k, PBR).
  - `quality` "Draft" → **SAM-3 segmentation + SAM-3D objects** with a Moondream3 long-caption
    pre-pass; the caption is "genericized" by gemini-2.5-flash-lite with an anti-IP prompt
    ("Replace ALL character names ... 'Pikachu' -> 'yellow creature with pointed ears'") to
    dodge IP-name refusals in segmentation prompts.
  - `fast` "Textureless" → **Tripo v2.5** image-to-3D, 50k face cap.
  - Fast GLB previews for all tiers via **Hunyuan3D v2 mini turbo**.
- Seed image generation (`generateMeshImage`): **gpt-image-2** (OpenAI Responses API,
  multi-turn via image_generation_call_id threading) → fallback **Gemini 3 Pro Image** ("nano
  banana pro") → fallback **Flux (fal)**. All prompts are prefixed with `INSTRUCTIONS_3D`
  (`src/server/imageGen.ts:16`): "You are generating a fully textured and rendered 3D model.
  Output one centered 3D model ... Plain white background ... Make sure the description
  strongly impacts the form and shape of the 3D Model not just the surface texture".
- Results arrive by **fal webhook** (`src/server/falWebhook.ts`) into Supabase storage +
  realtime broadcast. Mesh chat costs a flat 30 billing tokens.
- Extra prompt-box controls in creative mode: Quads/Polys toggle + polygon-count slider
  (persisted per model in localStorage), passed through the tool schema
  (`meshTopology`, `polygonCount`).

So: "Mesh" = generative-mesh mode (image→mesh models), "parametric" = OpenSCAD. The two share
the conversation/branching UI but nothing in the generation path.

---

## 4. RENDER / EXPORT PIPELINE

**Compiler: openscad-wasm in a Web Worker, fully client-side.** Vendored build at
`src/vendor/openscad-wasm/` (2025.03.25 playground build). Wrapper
`src/worker/openSCAD.ts`; per-consumer worker instances (`src/hooks/useOpenSCAD.ts`)
because a shared worker leaked results across consumers. A fresh wasm instance per compile
(documented workaround for a crash bug). Fonts: bundled Geist TTF + minimal fontconfig.

- **Preview compile** flags: `--backend=manifold --enable=lazy-union --enable=roof`, and
  **two outputs in one run**: `-o /out.stl -o /out.off`. The OFF companion is the color
  channel: OpenSCAD's manifold backend appends per-face RGBA to OFF face lines. If stderr
  says "Current top level object is not a 3D object." it re-runs with `--export-format=svg`
  (2D sketches preview as SVG).
- **Export compile** (`exportFile`): `--export-format=binstl` or `dxf`, with real
  `-Dname=value` defines for current param values.
- **Libraries on demand:** if the code contains the token `BOSL2`/`BOSL`/`MCAD`, the worker
  fetches the corresponding zip from `public/libraries/`, unzips (zip.js), and writes it into
  the wasm FS under `/libraries/<name>` before `callMain`. User STL attachments are written
  into the worker FS so `import("file.stl")` resolves.
- **Colors → three.js:** `src/utils/offParser.ts` + `src/utils/coloredOffMesh.ts` parse the
  OFF, strip OpenSCAD's two default paint colors (`#F9D72C` model-yellow and manifold's
  `#9DCB51` cut-face green → fall through to a brand fallback), bucket faces by RGBA, and
  build one `MeshStandardMaterial` mesh per color into a `Group`. Alpha < 1 → transparent
  material. So `color()` per part in the SCAD = per-part materials in the viewer, colors
  editable live via the `*_color` string params.
- **Viewer:** three.js scene (`src/components/viewer/ThreeScene.tsx`, `OpenSCADViewer.tsx`),
  `city.hdr` environment, orthographic/perspective toggle, view gizmo, GIF orbit export.
  Preview/thumbnail renders (`src/utils/meshUtils.ts`) use orthographic cameras
  ("the clean technical-drawing look") — same code path builds the 1-view thumbnail and the
  7-view inspection sheet.
- **Exports** (parametric, `ParameterSection` + `src/utils/downloadUtils.ts`): **.STL**
  (binstl from wasm), **.SCAD** (the source itself), **.DXF** ("2D Projection to the (x,y)
  plane" — fresh wasm compile with dxf export at click time, post-processed by
  `normalizeOpenSCADDxf` in `src/utils/dxfUtils.ts`). Creative meshes: GLB/FBX/OBJ+MTL/GIF,
  with texture extraction utilities.

---

## 5. IMAGE INPUT

**Parametric mode: the raw image goes straight to the (vision) model.** No pre-pass, no
captioning. Uploads land in a private Supabase `images` bucket; at request time
`aiChat.ts` re-downloads and inlines them as base64 data-URL file parts
(media type sniffed from magic bytes because storage metadata lies). All parametric picker
models except GLM 5.2 are `supportsVision: true`. So "make a bracket like this photo" is
handled entirely by the frontier model's own multimodal attention while it writes OpenSCAD.
STL attachments instead become a text part: `[user attached STL "x.stl"] Model dimensions
(mm): width=..., height=..., depth=... Use import("x.stl") ... Use rotation_x = 90 to stand
it upright.`

**Creative mode:** images are reference material for the seed-image generation chain (§3),
with careful branch-aware threading of gpt-image-2's multi-turn continuity ids
(fresh uploads suppress the prior-image thread so edits anchor on the new photo).

---

## 6. UI ARCHITECTURE

- **Stack:** React 19 + Vite + TypeScript; TanStack Router (file-based `src/routes/`,
  generated `routeTree.gen.ts`) + TanStack Query; Tailwind + Radix/shadcn components;
  PostHog analytics; Supabase = auth + Postgres + storage + realtime; server code in
  `src/server/*` exposed as API routes under `src/routes/api/` (deployed on Vercel — note
  `@vercel/request-context` waitUntil usage in mesh.ts); billing is an external
  "adam-billing" service reached via `src/server/billingClient.ts`.
- **Layout** (`src/views/EditorView.tsx` + `Layout.tsx`): chat panel **left**
  (`ChatSession`), 3D viewer **center** (`OpenSCADPreview` or GLB preview), **Parameters
  panel right** (`ParameterSection`, `border-l`, Dimensions group + Colors group + big
  download split-button). Mobile collapses the panel into a sheet
  (`ParameterSheetContent`).
- **Message model — a tree, not a list.** `messages` rows carry `parent_message_id`;
  `conversations.current_message_leaf_id` points at the active leaf. Retries repoint the
  leaf to the parent user message and the new assistant becomes a **sibling** (branch nav
  arrows in bubbles). The server rebuilds model context **only** by walking leaf→root in
  the DB — the client's messages array on the wire is ignored ("rock-solid against
  chat.regenerate()-style truncation hacks", aiChat.ts:280).
- **Chat → regeneration flow:** `ChatSession.tsx` owns the AI-SDK `Chat`; parent
  `EditorView` owns DB writes. Client-side tool execution (`onToolCall`) compiles + uploads
  previews, persists resolved tool parts to the DB **before** `addToolOutput` triggers the
  auto-continuation (`sendAutomaticallyWhen`), because the server continues from the DB
  branch. Careful failure choreography: persist-failure pauses the loop; stuck
  `input-available` tool parts from a killed session are rewritten to `output-error` on load
  (`stuckToolRecovery.ts`); a race-window matrix (insert/update/skip) in the server's
  `onFinish` decides who owns the assistant row (`chatToolPersistence.ts`).
- Each new artifact auto-switches the preview pane (`findLatestPreview`), and
  `handleViewArtifact` re-derives the parameter panel from the artifact code on every view.
- Model switching is per-message: the picker sits in the prompt box; retry lets you re-roll
  a specific assistant turn **with a different model** (`MessageBubble` → `handleRetry`),
  and `metadata.model` records who wrote what. Legacy model ids are remapped
  (`shared/models.ts`).

---

## 7. BENCHMARKS (`benchmarks/`)

**A showcase, not an eval.** 13 models, each as `NN-name.md` (the exact prompt, parametric
control counts, "what it demonstrates", full `.scad` source inline) + `.scad` + orbiting
`.gif`. Range: twisted hex vase, knurled knob, **hex bolt & nut with real ISO threads via
BOSL2 `screw()`/`nut()` ("M12x1.75")**, honeycomb bracket, NACA 2412 wing from the real
equations, threaded jar+lid (mating threads), bevel gear pair, centrifugal impeller,
herringbone planetary gearbox, radial 9-cyl engine, turbofan, turbine blisk, full V8
(22 dims · 8 colors). No scoring harness, no pass/fail, no runner — the repo's own README
frames it as "a record of how well CADAM turns plain language into real, printable, fully
parametric CAD".

**Reusable for us:**
- `render.sh` — OpenSCAD-CLI orbit-GIF + `--sheet` 4-view contact-sheet harness (BOSL2 on
  OPENSCADPATH, colorscheme Tomorrow). Directly reusable for demo GIFs of our STL outputs.
- The 13 prompts are excellent **tier-3/4 spec candidates** for our suites and the M6′ v2
  teacher dataset (mechanism-heavy, exactly where our v1 dataset was thin).
- The `.scad` sources are reference geometry: compile with system `openscad` → STL → our
  `geom_bands.py` can band-score any future OpenSCAD-backend output against them.

---

## 8. NOTABLE / LICENSING

- **License is GPL-3.0.** Reading, learning, porting *ideas* and re-implementing is fine.
  **Copying code verbatim into cad-builder makes the derived work GPL** — acceptable for a
  personal local stack, but flag it if anything here is ever distributed. The vendored
  openscad-wasm is itself GPL (SOURCE-OFFER.txt in the vendor dir).
- **Multi-part "assemblies" are cosmetic:** distinct parts are just `color()`-wrapped
  subtrees compiled into one mesh (lazy-union). No part tree, no mating semantics, no
  per-part export. Our Compound/assembly gate work has no counterpart here.
- **Prompt tricks worth stealing:** forced tool_choice on step 0 (the model *cannot* skip
  building); "Never say you created ... a model unless you used build_parametric_model in
  that turn"; text parts are stripped from build messages so the artifact is the only
  channel; per-object feature checklists; the single worked mug example; "Finalize only
  because the views look right", not because it compiled.
- **Cost architecture:** compile + render on the *user's* machine (wasm) — the server pays
  only LLM tokens, metered per-model into billing tokens with cache-read/write-aware USD
  accounting (`usdCostFromUsage`).
- `scripts/load-prod-snapshot.mjs`, Supabase migrations/schemas document the full DB shape
  (conversations / messages / meshes / images / previews / prompts).

---

# COMPARISON — CADAM vs our cad-builder engine

| Axis | CADAM | Ours (cad_v5 / cad_engine.py) |
|---|---|---|
| Language / kernel | OpenSCAD (CSG) + BOSL2, mesh-first | build123d Python on OCCT, **STEP-first** (BREP) |
| Compile | openscad-wasm **in the user's browser** | server-side python → STEP → STL/render |
| Models | Frontier cloud only (Fable 5, GPT-5.6 Sol, Gemini 3.1 Pro, …), user-pickable, OpenRouter for the long tail | Local 7B (Ollama) + 30B strong rung + Claude cloud teacher rung |
| Verification | **None deterministic.** Model self-inspects a 7-view render sheet; "compiled + looks right" = done | Deterministic inspect/gate ([spec] dims, interference, part_gaps, fragmentation, bare-primitive) + gemma vision critic + human review queue |
| Loop budget | Up to 60 steps/turn, error stderr fed back verbatim | MAX_TURNS 4 + best-of-3 first turn + escalation ladder |
| Iteration trigger | Same model reads its own screenshots | Separate critic model + measured geometry facts |
| Parameters | **Customizer comments in the code, regex-parsed, slider edit = regex substitute + recompile, zero LLM** | Refine = another LLM round-trip; no parameter panel |
| Colors | `color()` → OFF per-face RGBA → per-color three.js materials, live-editable | Single-material render PNG; STEP carries no color |
| Images | Raw photo → frontier VLM directly | gemma pre-pass → text analysis → brief-less codegen + critic 2nd image |
| Mesh mode | fal.ai chain (gpt-image-2 → Meshy/SAM-3D/Tripo/Hunyuan) | None (out of scope for 8GB local) |
| Assemblies | Cosmetic (one mesh, colored parts) | Real Compound parts, per-part inspect, interference gate |
| UI | React/Supabase chat-left / viewer-center / **params-right**, message tree with branching, per-message model switch | FastAPI webui: prompt + log + render + downloads (no side chat, no params) |
| Honesty | Model's word + user's eyes | Gate verdict attached to artifact; converged flag never faked |

**Where they beat us:** post-generation interactivity (sliders, colors, instant re-scale with
no LLM), UI polish, multi-provider frontier access, zero-cost compile, branching chat.
**Where we beat them:** measured truth. CADAM would happily ship our "5 of 7 log-converged
builds were measurably wrong" class of failure — nothing in their pipeline measures a
dimension against the user's words. Their acceptance bar is our pre-gate 2026-07 state.
STEP output also keeps us CAD-editable (Onshape Part Studio, FreeCAD parametric, CNC/DXF
kerf work) where CADAM ends at a mesh.

### Why CADAM's OpenSCAD works when our M8 spike failed 0/6

The M8 scad spike (July, `legacy/scad-spike/`) had the **local 7B** writing pseudo-OpenSCAD —
invented syntax, no compile feedback loop, no library. CADAM differs on every axis that
matters:

1. **Frontier models.** OpenSCAD is heavily represented in Fable-5/GPT-5.6/Gemini-class
   training data; a 7B q4's OpenSCAD is not comparable. This is the dominant factor —
   CADAM has no small-model path at all, and the closest thing to our fast rung doesn't
   exist in their product.
2. **Compiler-in-the-loop with stderr.** Every syntax hallucination costs one cheap step,
   not a failed build: `output-error: "Compilation failed:\n<stderr>"` auto-resubmits.
   Our spike scored the first (only) attempt.
3. **BOSL2 instead of raw CSG.** Threads, sweeps, bezier lofts, roundings are library calls
   with spec strings — the same "correct-by-construction helper" philosophy as our
   `b123d/domain.py` + bd_warehouse, but prompted rather than routed.
4. **A style contract + one worked example**, exactly the pattern our brief-less
   `generate_code_raw` + retrieved idioms converged on independently.
5. **Customizer discipline** constrains the code shape (top-of-file constants, no magic
   numbers inline), which itself reduces degrees of freedom for the generator.

**Verdict implication:** the M8 "OpenSCAD rejected" result stands **for the local 7B rung**.
It does NOT generalize to the cloud rung — a Sonnet/Fable-backed OpenSCAD path with compile
feedback and BOSL2 would very likely clear it. Whether we want one is a product question
(mesh-first output vs our STEP-first identity), not a feasibility one anymore.

---

# ADOPTION SKETCH

## (a) OpenSCAD codegen backend + STL-first pipeline as a new engine target

Scope it as a **cloud-rung-only backend**, kept beside (not replacing) build123d:

1. **Compiler runner** `scripts/scad_build`: system `openscad` CLI (snap/apt, ≥2021; nightly
   for `--backend=manifold`) → `openscad in.scad -o out.stl -o out.off --backend=manifold
   --enable=lazy-union`, capture stderr. Vendor BOSL2/BOSL/MCAD zips (grab from CADAM's
   `public/libraries/`) into `~/.openclaw/scad-libraries/`, set `OPENSCADPATH`. ~1 day.
2. **Prompt** `_SCAD_SYSTEM`: port CADAM's rules nearly verbatim (BOSL2 steering, customizer
   annotations, `*_color` params, snake_case naming, worked mug example) + our [spec]
   language ("state every user dimension as a top-of-file parameter"). Route via the
   existing cloud block in `cad.json` (`teacher_gen.py` already knows how to call it).
3. **Loop reuse:** compile-error → feed stderr back (CADAM's trick — we already do this for
   python tracebacks). Render: our `scripts/render` doesn't apply (no STEP); render the STL
   via trimesh/pyrender or CADAM-style three of `render.sh`'s CLI calls
   (`openscad --render --camera ... -o png`) — the CLI can render PNGs directly, no wasm
   needed.
4. **Gate adaptation:** inspect on mesh, not BREP — trimesh gives watertight, volume, bbox,
   connected components, and hole counting via face topology is weaker. Keep: bbox [spec]
   dims, component count vs expected parts, volume sanity, interference (trimesh boolean),
   part_gaps (component AABB distances). Lose: cylindrical-face bore diameters (approximate
   via cylinder fitting on face clusters if ever needed). The critic (gemma) works unchanged
   on the PNG.
5. **Targets seam:** register `openscad` in `cad_v5/targets.py`-adjacent config; result JSON
   carries `scad` source + `stl` (no `step`). Downstream CAM: STL slices fine for print;
   DXF/CNC paths stay build123d-only.
6. **Measure before believing:** rerun the M8 six specs on the cloud rung with this pipeline;
   also run CADAM's 13 benchmark prompts and band-score vs their reference `.scad` geometry
   (compile theirs to STL, `geom_bands.py`). Ship only if it clears the build123d cloud rung
   on the same specs — otherwise it's a toy backend.

## (b) CADAM-style parameter recognition + sliders + side-chat in our web UI

This works **without OpenSCAD** — port the discipline to build123d source:

1. **Code contract** (add to `_CODE_SYSTEM` and teacher prompts): every generated script
   starts with a `# --- PARAMETERS ---` block of simple assignments,
   `wall_thickness = 2.0  # [0.5:0.5:6] mm wall thickness`, colors as
   `body_color = "#4A90D9"`, nothing computed in the block; body must reference these names.
   This is CADAM's customizer vocabulary with `#` instead of `//` — the value of the format
   is that it survives regex parsing, not the comment glyph.
2. **Parser** `cad_v5/params.py` (~150 lines): direct port of `parseParameters.ts` semantics
   — block-scoped, typed values, range/enum/step decode, description-line-above, group
   markers, and the auto-range/auto-step nice-number fallbacks from `parameterUtils.ts`
   (worth porting exactly; they're good UX math). Unit-test against the same shapes.
3. **Edit path** (the whole point — no LLM): webui endpoint `POST /param-edit
   {build_id, name, value}` → regex-substitute the assignment in the stored `.py`
   (CADAM's `updateParameter` port, preserving trailing comment) → re-exec via
   `scripts/step` → STL/GLB → viewer refresh. Runs the *existing verified script*, so it's
   deterministic and fast (~seconds, no GPU). Cache last N param-variants per build.
   Optionally re-run the [spec] gate on the result and badge the panel if an edit broke a
   spec'd dimension — something CADAM can't do.
4. **UI**: add a right-hand Parameters panel to `webui/` (Dimensions group, Colors group,
   Reset-all, debounce ~200ms), and a left chat column that posts refine messages to the
   existing engine refine path — CADAM's three-pane layout on our FastAPI app. Keep the
   original code as `original.py` per build so Reset/default anchoring works
   (their `originalCode` metadata trick).
5. **Retrofit**: the corpus + gold examples should be migrated to the PARAMETERS block style
   so retrieval teaches the format (CADAM gets format compliance from the prompt alone on
   frontier models; our 7B will need it in the few-shots).

This is the highest-value, lowest-risk piece: it is exactly the user's ask ("rescale the
model without consulting the LLM") and none of it depends on OpenSCAD or on adopting (a).

## (c) A Mesh-button equivalent

Honest options, in order of realism on our hardware:

1. **Mode toggle → image-only "fluid" build** (already built): a "Mesh-ish" button in the
   webui that switches the prompt box to image-first mode (photo → gemma analysis → build).
   Cheap, local, but output is still parametric CAD — different value proposition than
   CADAM's mesh mode, arguably the better one for us.
2. **Cloud mesh rung:** wire fal.ai exactly as CADAM does (Tripo v2.5 textureless is the
   cheap tier; Hunyuan3D mini turbo for previews) behind a `mesh` conversation type in the
   webui; STL comes back via webhook/polling. ~1–2 days, costs per call, needs an API key —
   only worth it if organic/figurine outputs are actually wanted.
3. **Local image-to-3D is not viable** on the RX 6600 8GB (TripoSR/Hunyuan need ≥12–16GB
   CUDA); note-and-move-on per the stagger-heavy-jobs rule.

If (c) is built at all, reuse CADAM's `INSTRUCTIONS_3D` seed-image prompt and the
tiered fallback-chain pattern — both are directly portable.

---

*Sources: all findings from the working tree at `~/repos/CADAM` (read 2026-08-02):
`src/server/aiChat.ts` (prompts, providers, loop), `shared/chatAi.ts` (tools),
`shared/parseParameters.ts` + `src/utils/parameterUtils.ts` (params),
`src/worker/openSCAD.ts` + `src/hooks/useOpenSCAD.ts` (wasm), `src/utils/coloredOffMesh.ts`
(colors), `src/components/chat/ChatSession.tsx` (client tool loop),
`src/views/EditorView.tsx` + `src/components/parameter/ParameterSection.tsx` (UI),
`src/server/mesh.ts` + `src/server/imageGen.ts` (mesh mode), `benchmarks/`, `LICENSE`.*

---

## LOCAL BOOT (feasibility probe, 2026-08-02, this box)

### Prerequisites checklist — all already present

- Node v22.23.1 / npm 10.9.8 — satisfies `engines` (`^20.19.0 || >=22.12.0`, npm ≥10). ✔
- Docker 29.6.2, user in `docker` group. ✔ (needed only for the Supabase local stack)
- Supabase CLI ships as a devDependency (`npx supabase` → 2.107.0) — no global install. ✔
- Disk: `npm ci` installs 961 packages (~32s); Supabase images would add ~3–6 GB (not pulled).
- ngrok — NOT needed unless the fal.ai mesh/image-gen webhook flows are used.

### What actually boots (verified)

`npm ci` completed clean. With a minimal `.env.local` containing only `ENVIRONMENT="local"`,
`npm run dev` starts Vite 8 + TanStack Start on **http://localhost:3000/cadam** (the bare
`/cadam` path 307s to `/cadam/`) and SSRs a graceful gate: *"Missing API Keys. Please copy
.env.local.template to .env.local and restart."* — `src/App.tsx` checks
`isSupabaseConfigMissing` from `src/lib/supabase.ts`, which only tests that
`VITE_SUPABASE_URL`/`VITE_SUPABASE_ANON_KEY` are non-empty. Adding placeholder values for
those two clears the gate and the full app shell serves (verified, HTTP 200); auth and data
calls then fail at runtime until a real Supabase answers at that URL. Nothing crashes at
boot on missing keys: **every server-side `requiredEnv()` is lazy** (called inside request
handlers — `src/server/env.ts` + call sites), so missing providers degrade per-feature with
500s instead of preventing startup.

The CAD engine itself is fully offline: `src/vendor/openscad-wasm/openscad.wasm` (9.6 MB)
is committed and served by a dev middleware in `vite.config.ts`; BOSL/BOSL2/MCAD live as
zips in `public/libraries/`. Geometry never leaves the browser.

### Full env inventory

Client (`import.meta.env`, build-time): `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`
(both required to pass the gate), plus optional `VITE_POSTHOG_PROJECT_KEY`/`VITE_POSTHOG_HOST`,
`VITE_SENTRY_DSN`/`VITE_SENTRY_ENVIRONMENT`/`VITE_SENTRY_TRACES_SAMPLE_RATE`,
`VITE_SSO_PROVIDER`, `VITE_ACCOUNT_URL`.

Server (`env()`/`requiredEnv()`, lazy): `ENVIRONMENT` (set `"local"` — this **bypasses
billing entirely**, `src/server/billingClient.ts:81`, so `BILLING_SERVICE_URL`/`KEY` are
never read), `ANTHROPIC_API_KEY` (chat, direct Anthropic), `ANTHROPIC_BASE_URL`
(**honored and `/v1`-normalized**, `src/server/aiChat.ts:340` — the seam for pointing chat
at any Anthropic-compatible endpoint, e.g. a local proxy), `GOOGLE_API_KEY` (Gemini chat +
mesh), `OPENROUTER_API_KEY` (every non-Anthropic/non-Google picker model), `OPENAI_API_KEY`
(mesh mode), `FAL_KEY` (image gen + image-to-3D), `SUPABASE_SERVICE_ROLE_KEY` (server-side
DB/storage writes), `WEBHOOK_BASE_URL` + `NGROK_URL` (fal webhook plumbing only),
`ADAM_URL`, `ACCOUNT_PURGE_SECRET` (unset ⇒ endpoint 503s, by design), `DEBUG_LOGS`,
`CADAM_BILLING_MULTIPLIER`. The `replicate` npm dep is vestigial — no instantiation or
`REPLICATE_*` env anywhere.

### The Supabase story

The repo carries a complete local stack definition: `supabase/config.toml` (project
`cadam`, three private storage buckets `meshes`/`images`/`previews`), declarative schemas
in `supabase/schemas/*.sql`, 7 migrations, and a `seed.sql` that creates a ready login
**test@adamcad.com / password**. `npx supabase start` would pull roughly 8–10 images
(postgres, kong, gotrue, postgrest, realtime, storage-api, imgproxy, postgres-meta, studio,
edge-runtime — ~3–6 GB) and after that one pull runs **fully local and offline**; local
GoTrue captures signup emails in its bundled mail viewer, and the seeded user sidesteps
even that. Two gotchas: (1) `config.toml` has `[auth.external.google]` with
`env(GOOGLE_CLIENT_ID)`/`env(GOOGLE_SECRET)` — export dummy values or flip
`enabled = false` before `supabase start`; (2) the README's `npx supabase functions serve`
step is **stale** — there is no `supabase/functions/` dir; all server logic is TanStack
Start routes served by `npm run dev` itself.

### Exact boot recipe (unverified only where marked)

```bash
cd ~/repos/CADAM
npm ci
cp .env.local.template .env.local          # then edit:
#   ENVIRONMENT="local"
#   VITE_SUPABASE_URL='http://127.0.0.1:54321'
#   VITE_SUPABASE_ANON_KEY=<anon key printed by `supabase start`>
#   SUPABASE_SERVICE_ROLE_KEY=<service key printed by `supabase start`>
#   ANTHROPIC_API_KEY=<from the ~/.openclaw/openclaw.json env block>
#   (delete/ignore every other placeholder — lazy reads, nothing crashes)
GOOGLE_CLIENT_ID=x GOOGLE_SECRET=x npx supabase start   # one-time ~3–6GB pull [not run yet]
npm run dev                                  # http://localhost:3000/cadam  [verified]
# sign in as test@adamcad.com / password     [seeded; untested until supabase runs]
```

### Blockers, ranked by effort

1. **Supabase stack — trivial.** Docker pull + dummy Google-OAuth env vars. Everything else
   is committed (schemas, migrations, seed user).
2. **Chat — trivial.** One `ANTHROPIC_API_KEY` line; `anthropic/*` picker models go direct
   to the API. (Never copy key values into files that could be committed — `.env.local` is
   gitignored.)
3. **Non-Anthropic models — small.** Need `OPENROUTER_API_KEY`/`GOOGLE_API_KEY`, or prune
   the picker to Anthropic-only in a fork.
4. **Image gen + image-to-3D — feature-scoped, effectively cloud-only.** Hard-wired to
   fal.ai (`flux-pro`, `meshy v6`, `hunyuan3d mini turbo`, `sam-3`) plus OpenAI/Gemini, and
   needs a public `WEBHOOK_BASE_URL` (ngrok) for fal callbacks. Not localizable on an
   RX 6600; leave disabled — the text→OpenSCAD core doesn't touch it.
5. **Telemetry — cosmetic.** Sentry/PostHog silently no-op without keys (the
   `jackson-pollock` route is just a PostHog reverse proxy); the Sentry Vite plugin runs
   without an auth token but phones telemetry home unless `telemetry: false` is added.

### Verdict

**Fork-and-run-locally is "a day", with the first interactive boot closer to 1–2 hours.**
The hard parts are already solved upstream: engines match this box, deps install clean, the
dev server boots today, the whole database story is one `supabase start` from working, and
the CAD engine is vendored WASM with zero cloud dependency. The only unavoidable external
service for the core loop is the chat LLM itself — and `ANTHROPIC_BASE_URL` is an
officially-supported seam for swapping that later. The remaining "day" is fork hygiene, not
plumbing: trim the model picker, hide the fal-backed mesh/image features, strip
Sentry/PostHog/billing residue, and decide whether ~5 GB of Supabase containers is an
acceptable price for auth + persistence (or whether the fork should eventually replace
Supabase with something thinner — the far bigger job, since conversations, meshes, images
and previews all live in its schema).

*Probe artifacts: minimal `.env.local` left in place (gitignored); `src/routeTree.gen.ts`
regeneration churn from the dev boot was reverted; no source edits; `node_modules/` kept.*
