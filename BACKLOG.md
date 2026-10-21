# Backlog

Ordered by what to do next. Each item says why it matters and roughly how
big it is (S = an hour or two, M = a session, L = several sessions).

## Next up

### Engine: llama.cpp replaces Ollama (L, several sessions)

Decided 2026-10-20: full replacement, Vulkan build. Measured here with
the same GGUF files and settings (all layers on GPU, flash attention,
q8_0 cache, 1k prompt, 256 tokens out):

| Model | Ollama 0.40.6 | llama.cpp Vulkan | llama.cpp ROCm |
|---|---|---|---|
| Qwen3.5-9B Q4 prompt / write | 504 / 38.7 | 881 / 52.2 | 904 / 50.3 |
| Gemma 4 12B prompt / write | 675 / 32.4 | 750 / 35.8 | 809 / 37.6 |

Ollama itself runs an older bundled `llama-server`; the gain is newer
llama.cpp. Gotcha: `llama-server` defaults to 4 parallel slots, which
overfilled 12 GB VRAM (5.9 tok/s); always pass `-np 1`.

**Progress (2026-10-21, mid stage 1, committed):** `internal/gguf` (GGUF
header reader, tested on real files), `internal/engine` (store + Ollama
import by hard link, one llama-server per model, chat/embed translation to
the ollama types, keep-alive/pinning/evicting, memory read from the
`-lv 4` log) and a `server.Backend` interface; `-engine llama.cpp` is the
default, `-engine ollama` the fallback; `-llama-server PATH` (else
~/.local/share/ollamui/llama.cpp/llama-server, else PATH). Verified: a chat
through ollamui on llama.cpp streams and saves. Local Vulkan build:
scratchpad `llama/build-vk/bin/` (not yet copied to
~/.local/share/ollamui/llama.cpp/; the user's :8484 server still runs the
previous binary on Ollama).
Next, in order:
- Context: `--fit` with no `-c` grows to the model max (262k) and writing
  drops 3x (2B: 15 vs 46 tok/s at 8k). Now `-c 32768` by default
  (`engine.DefaultCtx`); check what's best (8k/16k/32k) with the speed test,
  and let Tune's context override it.
- 2B still writes 46 tok/s vs Ollama's 76 at the same model: find out why
  (thread count? `-ub`? the mmproj loaded? `-lv 4` cost?) before switching
  the user over.
- Running() reports the full name, chats use the alias, so the reply's
  context stat is missing: remember the requested name per process.
- Then stage 2 (HF downloads), etc. Meanwhile Discover pulls still go
  through Ollama and get imported (engine re-imports every 10 s).
- RAG keyword (BM25) fallback, so project files work without an embedding
  model (user's thesis PDF is stuck "waiting").

Stages (the app keeps working after each):
1. **Engine core.** `internal/gguf` reads GGUF metadata (what Ollama's
   `/api/show` gave us: architecture, layers, context, chat template,
   parameter count, quant type). `internal/engine` keeps a model store
   (`~/.local/share/ollamui/models/`), imports Ollama's models by hard
   link (no re-download), runs one `llama-server` per loaded model (port,
   context, GPU layers, cache type, `--jinja`, `-np 1`, idle unload,
   pinning, evicting to make room), and speaks OpenAI chat/embeddings to
   it. The server talks to an engine interface instead of the Ollama
   client.
2. **Downloads** straight from Hugging Face (resume, split files, mmproj),
   replacing `ollama pull`; delete.
3. **API for editors:** ollamui serves `/v1/chat/completions`, `/v1/models`,
   `/v1/embeddings` (OpenAI-compatible, loads models on demand); presets
   show up as models there (replacing "publish to Ollama").
4. **Settings and tuning:** an engine section (llama.cpp version, cache
   type, flash attention, threads) replaces the Ollama server page; auto
   context uses real memory numbers from `llama-server` (and/or `--fit`);
   Compare's speed test, the voice talker (`-ngl 0 -t 4`) and RAG
   embeddings move over. Speculative decoding with draft models.
5. **Install and clean up:** a one-click llama.cpp (Vulkan) install from
   its GitHub releases, removal of the Ollama client, docs.

Also pending: keyword search (BM25) for RAG so project files work without
an embedding model (a user asked a project chat about a PDF and got
nothing because no embedding model had been picked).

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

- **GPU layers:** done (Tune → GPU layers, see Done).
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
- **Speculative decoding: not on this setup (checked 2026-10-20).** Ollama
  0.40.6 accepts `draft` on `/api/create` and returns success, but for GGUF
  models it stores no draft layer and nothing changes (Qwen3.5-9B Q4 with a
  2B draft: 35.2 tok/s both ways). `ollama create --draft-quantize` is
  safetensors-only, which suggests the MLX runner. `draft_num_predict` is a
  documented option (default 4, 0 = off). Revisit on a newer Ollama; if it
  lands, check `/api/show` lists the draft before showing any UI.

### 3. Design ideas (remaining)

All of the 2026-10-20 polish list is done (see Done).

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
