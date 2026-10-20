# Backlog

Ordered by what to do next. Each item says why it matters and roughly how
big it is (S = an hour or two, M = a session, L = several sessions).

## Next up

### 0. Voice mode follow-ups (S–M)

Voice mode shipped 2026-10-20 (see Done). Open:

- **Phone over HTTPS.** The mic needs a secure context. Document or automate
  `tailscale serve --bg 8484` (and `-allow-host <machine>.<tailnet>.ts.net`). (S)
- **Gemma 4 hears audio natively.** Ollama lists `audio` in its capabilities,
  so whisper could be skipped for it. Keep whisper for the live transcript;
  try sending the WAV as the message for better understanding of names. (M)
- **Talker quality.** Qwen3.5-2B sometimes states a wrong fact in the
  lead-in ("water scatters blue light"). Try a 4B, or tighter few-shot
  prompts. The filters in `goodLine` catch numbers and conclusions only. (S)
- **Overthinking.** Qwen3.5-9B still thinks 20–90 s on easy questions.
  Small talk is now routed to the talker; simple factual questions could be
  too ("answer without thinking"), or a think budget. Watch whether the
  one-line prose prompt slows Qwen down. (M)
- **Better speech detection.** Typing no longer triggers it (click filter,
  sustained-voice window, duck-then-confirm interrupts). Silero VAD
  (onnxruntime-web, ~2 MB) if other noise still causes false starts. (M)

### 1. Intelligence score and benchmarks (L)

The M4 feature from the design (Compare page, score on Discover rows and
model pages).

- **Research done (2026-10-20).** Findings:
  - **Score source:** the Artificial Analysis Intelligence Index (v4.x).
    - **Easiest feed:** OpenRouter's keyless `https://openrouter.ai/api/v1/models`. It embeds
      `benchmarks.artificial_analysis.intelligence_index`, and about 130 of its models carry a
      `hugging_face_id`, which makes the join trivial.
    - **Direct AA API:** `GET https://artificialanalysis.ai/api/v2/language/models` with an
      `x-api-key` header. The key is free and allows 100 requests a day. Names don't map to HF
      ids, so it can only fill gaps.
    - **Attribution:** a visible link to artificialanalysis.ai is required. Don't commit the
      fetched data to the repo.
  - **GGUF repo → base model:** use `?expand[]=baseModels` on the HF model API, or the
    `base_model:quantized:<org>/<name>` tag. Prefer `relation: "quantized"`; a finetune doesn't
    inherit the base model's score.
  - **Scale:** v4 scores are compressed (roughly 0–50; small models score 4–23). Show the score
    as a rank or bar within the visible set, not as "/100".
  - **No score:** show a muted "not rated", never 0. HF `evalResults` entries are too sparse
    and inconsistent to combine into a score, though they could appear as labelled chips.
  - **Not useful:**
    - The Open LLM Leaderboard has been frozen since March 2025.
    - LiveBench has no data endpoint.
    - Epoch's CSV URL returned a 404.
  - **Caching:** keep a `scores.json` that refreshes at most daily and serves stale data when a
    refresh fails.
- **Local speed bench:** built (Compare → Run speed test). Open: show
  measured speed on Library rows and the model page.
- **Score on Discover + model page (M):** sort by intelligence, show it on
  rows, rank within the current filter ("#2 of 14 in 5–12B").
- **Compare page:** built (see Done). Still open: model-card benchmarks
  and uploaders' KLD numbers as extra rows; an "add to compare" action on
  Library and Discover rows; comparing a not-yet-downloaded quant (fit and
  score only).

### 2. More runtime settings: KV cache type, GPU layers, speculative decoding (M)

Requested 2026-10-20. Each one depends on what Ollama exposes, so check the
installed version (0.40.x) before building UI for it.

- **GPU max utilization / overflow.** Cap how much of a model goes on the
  GPU so the rest spills to CPU/RAM instead of failing or evicting.
  Ollama's per-request `num_gpu` option (layers offloaded) does this today
  and can live in presets like any other option. Show it as "GPU layers"
  with an "auto" default, ideally with the layer count from `/api/show` so
  it can be a slider ("28 of 36 layers on GPU"). Possibly also a
  percentage-of-VRAM view that converts to layers. `OLLAMA_GPU_OVERHEAD`
  (server env) is the global alternative; mention it, don't manage it.
- **Server settings page.** Show Ollama's server-wide settings (read from
  `systemctl show ollama -p Environment`, no root needed): KV cache type,
  flash attention, default context, network binding. Offer a ready-to-run
  `sudo systemctl edit ollama` snippet rather than running sudo ourselves.
  Feed the KV type into auto-context (`kvBytesPerValue` in `hw/kv.go`):
  q8_0 halves the cache, q4_0 quarters it.
- **KV cache dtype (f16 / q8_0 / q4_0).** Historically server-wide only
  (`OLLAMA_KV_CACHE_TYPE`, needs `OLLAMA_FLASH_ATTENTION=1`), not per model.
  Research whether 0.40 allows it per request or per model. If not, show the
  current server setting read-only with instructions, and factor it into
  the memory/fit estimates (q8_0 halves KV size, so bigger contexts fit).
- **Speculative decoding (draft model).** Research whether Ollama 0.40
  supports a draft model and how it's configured (Modelfile? request
  option?). If supported: pick a draft model from installed ones of the
  same family, draft token count. If not, record that and drop it or
  revisit later.

### 3. Design ideas (remaining)

Most of the 2026-10-20 list is done (see Done). Still open:

1. **Polish on what shipped:** renaming chats from the sidebar; a GPU-fit
   bar on Library rows; "fits on GPU" hints and a ★ for presets in the model
   switcher; showing recent presets on the empty chat screen.

## Later

- **Check split quants and vision (`mmproj`) downloads.** A normal
  download works end to end (Qwen3.5-9B Q8_0 with its short name, 2026-10-20).
  Split and vision repos haven't been tried with `ollama pull hf.co/…`. (S)
- **Family grouping on Discover.** Collapse `Qwen3-8B`, `Qwen3-8B-Instruct-2507`,
  etc. into one expandable row. (M)
- **Exact memory estimates.** Read layer count, KV heads and head size from
  the GGUF header with an HTTP range request instead of the rule of thumb.
  Fixes over-estimates for sliding-window/hybrid models. (M)
- **"Ollama can run it" filter.** Flag architectures the installed Ollama
  version doesn't support before downloading. (M, needs research on how to
  detect support)
- **Download fallback** for repos `hf.co` pulls can't handle: fetch the
  GGUF directly, then `/api/blobs` + `/api/create`. (M)
- **Preset drift detection.** On startup, compare published presets with
  `ollamui/*` in Ollama and offer to re-sync. (S)
- **`config.toml`.** Listen address, Ollama URL, HF token, default context,
  instead of flags and env vars. (S)
- **Better MoE detection** for unfamiliar architecture names (e.g. read
  expert count from GGUF metadata, same range request as above). (S, after
  exact memory estimates)
- **Chat search** across saved chats. (M)

## Done

- Compare page (2026-10-20): up to 4 installed models side by side (specs,
  quant quality, context that fits on the GPU, intelligence score) with a
  per-model speed test saved under `bench/`. Auto context now checks itself:
  a context that spills onto the CPU is remembered and the next smaller one
  used (Qwen3.5-9B Q4 went from 256k at 15 tok/s to 128k at 39 tok/s).
  Settings moved to the bottom of the sidebar with its own icon.

- Voice mode's ~800 MB of speech models downloads only after the user agrees
  (prompt in the voice bar on the first mic press), and Settings → Voice can
  remove all of it again (2026-10-20).

- The speech service is started by ollamui when the mic is pressed (embedded
  script, `uv run`), and stopped after 10 idle minutes or when ollamui exits;
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

- Auto context. When nothing sets `num_ctx`, ollamui picks the largest context
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
- M3: Tune panel, presets, publish presets to Ollama as `ollamui/<name>`.
