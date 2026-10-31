# Backlog

Ordered by what to do next. Each item says why it matters and roughly how
big it is (S = an hour or two, M = a session, L = several sessions).
Tidied 2026-10-31: Ollama-era items that no longer apply were dropped;
finished work is under Done at the bottom.

## Next up

### 1. Custom benchmarks (L) — started 2026-10-31

Asked for 2026-10-31. Assay's direction is comparing and evaluating models,
so: write your own benchmarks, run them on local models, and place each
model on the Artificial Analysis Intelligence Index by fitting a line
through the models that have both your score and a public one.

- Native Go eval runner, one model at a time, through the engine (no
  Python harness). External tools (promptfoo, lm-evaluation-harness,
  Inspect) can still use Assay's `/v1` API.
- Benchmarks are plain files: a list of questions with a checker each
  (exact answer, number, multiple choice, regex, contains; later: code that
  runs, a judge model). A few small built-in sets to start.
- Results per model digest, per benchmark, with every answer kept so a
  run can be inspected.
- The estimate shows its fit quality (how many anchor models, R²) and is
  always labelled as an estimate, never mixed up with the published score.
- ✅ (2026-10-31) Benchmarks page, 9 built-in sets incl. Hard coding,
  Equations to code (both run against hidden tests) and Proofs (judge model).
- Open: show the estimate on Library rows, Compare and the model page;
  harder knowledge/logic sets (the easy ones saturate: a 2B gets 85–95%);
  re-judge with another judge; other languages than Python for code sets.

### 2. Scores and Compare follow-ups (M)

- Discover: sort by score; rank within the current filter ("#2 of 14 in
  5–12B").
- Model page: show the measured speed.
- "Add to compare" on Library and Discover rows; compare a quant that
  isn't downloaded yet (fit and score only).
- Model-card benchmarks and uploaders' KLD numbers as extra Compare rows.

### 3. Leftovers from the llama.cpp switch (S–M)

- RAG keyword (BM25) fallback, so project files work without an embedding
  model (a project chat about a PDF got nothing because no embedding model
  had been picked). (M)
- Try EAGLE3 / DFlash / DSpark drafts and a real Unsloth LoRA; both are
  wired but untested. (S)
- SearXNG against a real instance. (S)
- Terser compression summaries from small models. (S)
- Dead code: the voice talker (`voiceTurn`, `/api/voice?model=` warm-up)
  and the Ollama-era parts of `autoctx.go`. Careful: `pickContext` and
  `kvFactor` are still used by Compare, Discover, Setup and Presets. (S)

### 4. Windows follow-ups (S–M)

Windows build and zip exist (see ARCHITECTURE "Platforms").
- Test on a real Windows PC: load a model, chat, models folder move,
  second launch, Ollama import, linked models from other drives.
- An exe icon and version info (a `.syso` resource), maybe code signing to
  avoid the SmartScreen warning.
- Optional: a tray icon instead of the console window (needs cgo or a
  pure-Go tray lib; weigh against the single-binary rule).
- PDFs need pdftotext; a pure-Go PDF text extractor would avoid that.

## Later

- **"Ask another model"** side by side in a chat (experimental). (M)
- **Chat search** across saved chats. (M)
- **Family grouping on Discover.** Collapse `Qwen3-8B`,
  `Qwen3-8B-Instruct-2507`, etc. into one expandable row. (M)
- **Exact memory estimates before download.** Read layer count, KV heads
  and head size from the GGUF header with an HTTP range request instead of
  the rule of thumb; fixes over-estimates for sliding-window/hybrid models.
  Then better MoE detection from the expert count. (M)
- **Projects:** link citations [n] to their passage; per-chat attachments;
  EPUB / XLSX; an approximate index for very large projects. (M)
- **Read GGUF headers of linked models** to catch mislabelled quants. (S)
- **`config.toml`** for listen address, HF token, default context, instead
  of flags and env vars. (S)

## Parked

### Voice follow-ups

Voice is dictation + read aloud since 2026-10-21. Live mode items only
matter if it comes back.
- Phone over HTTPS: document or automate `tailscale serve --bg 8484`
  (and `-allow-host <machine>.<tailnet>.ts.net`). (S)
- Gemma 4 hears audio natively: try sending the WAV instead of whisper. (M)
- Better speech detection (Silero VAD via onnxruntime-web) if noise still
  causes false starts. (M)

## Done

- 2026-10-25 → 2026-10-31, in short (details in ARCHITECTURE.md):
  design D across the whole app (Inter, copper/indigo accents, view
  transitions, prerender, stat strip, sortable Library columns, Measure
  all); live tok/s and an always-visible context bar in the chat header;
  rename to Assay; first-run setup with a llama.cpp build choice; start on
  boot and releases; free-disk check before downloads; readable by-name
  model links; path pickers; multi-GPU; phone layout; background replies;
  resource limits; per-model vision switch; starred default model; MTP-head
  downloads; prompt pruning of superseded code at 60% context; the
  `edit_code` and `forget` chat tools.
- llama.cpp switch (2026-10-21 → 10-23): Ollama replaced by one
  `llama-server` per model (Vulkan default, ROCm here: Qwen3.5-9B Q4
  504/38.7 → 904/50.3 tok/s prompt/write vs Ollama 0.40.6), HF downloads,
  `/v1` API with presets as models, per-model settings, one-click install,
  models folder choice. Gotcha kept for reference: always pass `-np 1`.
- 2026-10-24: speculative decoding (Auto = MTP head + n-gram), LoRA, chat
  KV caches across unloads, structured output, MoE expert placement, tool
  use, context compression, linking models other apps downloaded, Windows
  build.
- Intelligence score research (2026-10-20), still how scores work: the
  Artificial Analysis index via OpenRouter's keyless model list (joined on
  `hugging_face_id`; a visible link to artificialanalysis.ai is required;
  don't commit fetched data), GGUF repo → base model via
  `base_model:quantized:` tags (a finetune doesn't inherit the score), v4
  scores are compressed (~0–50) so show rank or bar, "not rated" never 0.

- Projects and RAG (2026-10-20): projects group chats (sidebar folders,
  drag and drop, a picker on the chat page) and can have instructions, a
  default model or preset, and files. Files are indexed with an embedding
  model the user picks and downloads in Settings → Knowledge; chats in the
  project get the closest passages in their prompt and show them under the
  reply. Plain files, no database. Open: citations [n] aren't linked to
  their passage yet; no per-chat attachments; files in other formats
  (EPUB, XLSX, images via a vision model) aren't read; very large projects
  would want an approximate index.

- Draft models are labelled as drafts (2026-10-20): DSpark/DFlash/EAGLE
  files inside a repo are listed apart from the real quants (they used to
  show as the model's "Q8_0"/"BF16" with "Fits on GPU"), draft repos say
  "Draft model" on Discover with a filter to hide them and a banner on their
  page, and get no score. Also: every DeepSeek architecture since V2
  (deepseek4, deepseek41, …) counts as MoE.

- Polish and runtime settings (2026-10-20):
  - Rename chats from the sidebar (pencil on hover; `PATCH /api/chats/{id}`).
  - Library rows show how each model fits (measured if a speed test ran,
    else estimated at 8k context) with a bar, its measured speed, a Speed
    sort and a Compare button (`/compare?add=` adds to the last comparison,
    which the browser remembers; the sidebar's Compare reopens it).
  - The model switcher shows each model's fit or measured speed, and stars
    presets (a preset shows its base model's numbers).
  - The empty chat offers up to three presets, most recently used first.
  - Tune → GPU layers (`num_gpu`, 0 to the model's layer count). Setting it
    turns auto context off for that chat and stops the spill from being
    learned as "context too big".
  - Discover and model-page fit estimates count the server's K/V cache type
    and `OLLAMA_NUM_PARALLEL`.

- Compare page (2026-10-20): up to 4 installed models side by side (specs,
  quant quality, context that fits on the GPU, intelligence score) with a
  per-model speed test saved under `bench/`. Auto context now checks itself:
  a context that spills onto the CPU is remembered and the next smaller one
  used (Qwen3.5-9B Q4 went from 256k at 15 tok/s to 128k at 39 tok/s).
  Settings moved to the bottom of the sidebar with its own icon.

- Voice mode's ~800 MB of speech models downloads only after the user agrees
  (prompt in the voice bar on the first mic press), and Settings → Voice can
  remove all of it again (2026-10-20).

- The speech service is started by Assay when the mic is pressed (embedded
  script, `uv run`), and stopped after 10 idle minutes or when Assay exits;
  nothing runs in the background otherwise (2026-10-20).

- Settings → Voice (2026-10-20): filler model, voice + preview, speed,
  sentence pause, Whisper model, detection sensitivity, prose prompt; saved
  in `voice.json`. Also: the app bar fits half-width windows, and Flash
  attention reads On/Off.

- Voice mode (2026-10-20). Mic button in chat: hands-free conversation with
  live transcription, sentence-by-sentence speech, talking over the reply to
  interrupt, and a CPU talker model that fills the silence while the main
  model thinks (see ARCHITECTURE.md, "Voice mode"). Speech service:
  `uv run tools/voice/voice_server.py` (faster-whisper + Kokoro on the CPU).
  Also: Ollama's settings are read from the running process when possible
  (fixes the Settings page for a hand-started `ollama serve`); auto context
  counts `OLLAMA_NUM_PARALLEL`; the phone chat page fits the screen (the
  closed Tune panel used to stretch it and push the composer off-screen).

- Settings page, downloads, scores (2026-10-20):
  - `/settings` reads the Ollama service env via `systemctl show` (no root),
    lets you pick new values and generates a `sudo` drop-in snippet. Auto
    context scales the KV estimate by the server's cache type (q8_0 ≈ 0.53×).
  - Downloads moved from the sidebar to an app-bar button with a popover.
  - Intelligence score (Artificial Analysis via OpenRouter) on Discover.
  - Still open from #2: GPU layers (`num_gpu`) in Tune; scale Discover's
    fit estimates by KV type. Speculative decoding: 0.40.6 has `draft` on
    /api/create and a `draft_num_predict` option, apparently MLX-runner
    only; untested on ROCm.

- Chat polish (2026-10-20):
  - A model switcher card replaces the plain select. It is searchable and
    shows loaded state and size.
  - Message actions: copy, edit and resend (any user message), and
    regenerate. On the server this is `fromUser` on `/api/chat`.
  - The empty chat screen has suggested prompts and a load-state hint.
  - Sidebar chats have a + button, delete on hover, date groups, and a
    search box when there are more than 8.
  - Library is rows with a filter, sort, and a disk total.
- App bar on every page, with loaded models as pills (size, context, eject).
  Tune is restyled like GNOME/libadwaita, with collapsible Sampling and More
  settings groups. Code blocks get a copy button. Models that share a digest
  (an alias and its hf.co name) show as one row. Discover keeps its filters
  visible beside the panel. (2026-10-20)

- Auto context. When nothing sets `num_ctx`, Assay picks the largest context
  (8k–256k) that keeps the model fully on the GPU:
  - The cache size comes from the model's real layer layout (hybrid
    attention and sliding windows included).
  - The model's weight memory is measured from Ollama.
  - Free VRAM comes from the driver.
  - Tune shows "auto-picked", and each reply's stats show the context it
    ran with. Loading a model uses the same context, so chats don't reload it.
- Discover side panel: clicking a model opens its quants beside the list.
  `?open=owner/name` deep-links to it.

- UI polish:
  - 14px text with a CSS spacing scale.
  - The Send button lines up with the message box.
  - Sidebar downloads and loaded models, Library names and status, preset names and base
    models, the chat load chip, chat stats, and model-page author and tags are all clickable.
- Discover rework:
  - **Search:** every match is fetched through HF cursor paging, capped at 1000 and cached
    for 5 minutes. Sorting and filtering happen locally, so sort only reorders.
  - **Layout:** shows 20 rows plus "Show more", in a narrower column with simpler rows.
  - **Interactions:** tags apply filters, and filters are stored in the URL.
  - **Hidden models:** image, video and audio generators are left out.

- M1: Ollama client, Library, load/eject, streaming chat, light/dark theme.
- M2: HF search with filters, model page with quant picker and fit
  estimates, background downloads with progress and short names.
- M3: Tune panel, presets, publish presets to Ollama as `assay/<name>`.
