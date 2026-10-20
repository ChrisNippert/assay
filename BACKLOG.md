# Backlog

Ordered by what to do next. Each item says why it matters and roughly how
big it is (S = an hour or two, M = a session, L = several sessions).

## Next up

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
- **Local speed bench (M):** run a fixed prompt through Ollama on installed
  models; record prompt tok/s, generation tok/s, time to first token. Store
  under `~/.local/share/ollamui/bench/`. This is the most trustworthy number
  we can show.
- **Score on Discover + model page (M):** sort by intelligence, show it on
  rows, rank within the current filter ("#2 of 14 in 5–12B").
- **Compare page (M):** up to 4 models side by side: fit, measured speed,
  score, card benchmarks, quant quality. Every number labeled with its
  source.

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
