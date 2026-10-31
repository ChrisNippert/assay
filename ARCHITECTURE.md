# Assay architecture

A lightweight web UI for finding, downloading, tuning and chatting with
local models. It runs them itself with llama.cpp's `llama-server` (see
"Engine" below). It started on top of Ollama; since 2026-10-23 Ollama isn't
needed at all. Models already in `~/.ollama/models` are imported by hard
link, and `internal/ollama` survives only as the request/response types
the code passes around.

## Goals

- **Discover and download** GGUF quants from Hugging Face (Unsloth, bartowski,
  …) with a quant picker that shows whether each one fits your hardware.
- **Tune without friction**: per-chat settings, saved as presets, without
  hand-writing Modelfiles.
- **Chat** with streaming output, thinking blocks and speed stats.
- **Compare** models by size, speed on your machine and published benchmarks.
- **Voice**: dictate a message, and have replies read aloud.
- One static Go binary. No Docker, no database server, no Node toolchain.

## Non-goals

- Training or fine-tuning (Unsloth's other half).
- Writing an inference engine. llama.cpp does that; Assay starts,
  sizes and stops its servers.
- Multi-user accounts. It's a single-user tool, optionally reachable from
  your other devices.

## The core idea: presets are models in the API

A preset is a base model plus settings and a system prompt, edited like an
Unsloth config (sliders, save) and kept as a TOML file. Assay's
OpenAI-compatible API (`/v1`, see "Engine") lists every preset as a model
under its own name, so Zed and other clients use it like any model, with
those settings applied. Nothing is created or copied: the preset file is the
only copy. (Until 2026-10-23 presets were "published" into Ollama as
`assay/<name>` models; the `published` field in old preset files is
ignored.)

Per-chat tweaks are layered over the preset on each request and change
nothing until you choose "Save to preset".

## Components

```
browser, Zed ──HTTP──► Assay (Go) ──HTTP──► llama-server, one per loaded model
                          │        └──HTTP──► speech service :8485 (optional, voice)
                     │
                     ├── internal/engine   llama-server per model, model store, Ollama import, /v1 proxy
                     ├── internal/gguf     GGUF header reader (arch, context, size, template)
                     ├── internal/ollama   shared request/response types (ollamatest: a client for tests)
                     ├── internal/llamacpp install and update llama.cpp releases
                     ├── internal/hf       HF search, file tree, quant parsing, fit estimate   (M2)
                     ├── internal/presets  TOML presets, option validation                      (M3)
                     ├── internal/chats    one JSON file per chat
                     ├── internal/scores   intelligence index (Artificial Analysis via OpenRouter)
                     ├── internal/bench    saved speed tests, one JSON file per model digest
                     ├── internal/voice    client for an OpenAI-style audio API (STT + TTS)
                     ├── internal/pick     native folder/file chooser (zenity, osascript, PowerShell)
                     ├── internal/server   pages, JSON API, streaming, host/CSRF guards
                     └── web/              templates, CSS, small vanilla JS (embedded)
tools/voice/voice_server.py   the speech service: faster-whisper + Kokoro (embedded, run with uv on demand)
```

### Engine (internal/engine)

Assay runs one `llama-server` per loaded model, the way Ollama does
internally, and answers in the `internal/ollama` types so the rest of the
code doesn't care which engine runs (`server.Backend`).

- **Store**: `<data>/models/` holds `blobs/` (GGUF files) and one JSON
  manifest per model (name, aliases, files, projector, default params).
  Ollama's models are imported by hard link (used in place across disks), so
  nothing is copied and Ollama keeps working if you still use it.
- **Processes**: started on demand with `-np 1` (more slots overfill VRAM),
  `-c 32768` unless a context is asked for (letting `--fit` pick grew it to
  262k and writing dropped 3×), `--fit` lowers it if memory is short. Other
  GPU models are stopped first unless pinned. Placement (layers on GPU,
  VRAM/RAM, context) is read from the `-lv 4` log.
- **Which build**: `-llama-server`, else `<data>/llama.cpp/llama-server`,
  else PATH. A ROCm build gets `HSA_OVERRIDE_GFX_VERSION` set for chips
  that borrow their family's kernels (gfx1031 → 10.3.0).
- **Vulkan vs ROCm** on the RX 6700 XT: ROCm writes much faster on small
  models (Qwen3.5-2B 81 vs 49 tok/s; Vulkan has per-token overhead), about
  equal on 9–12B. So ROCm here; Vulkan is the portable default for others.
  Other apps holding VRAM can make the first reply after a load very slow
  (6 vs 37 tok/s) while the driver evicts; it's not the engine.
- **Settings per model**: a manifest's `settings` (context, GPU layers,
  `cache_type`, `flash_attn`, `keep_alive`, any sampling option) are the
  user's own, saved from Tune via `PUT /api/models/settings`. Each request
  layers model params (the repo's recommendations) < settings < the
  request's options, for both the sampling fields and how it loads; a
  change to how it loads restarts it on the next request. A quantized cache
  forces flash attention on. The `-cache-type` flag (q8_0) and flash
  attention on are the defaults for models without their own. Re-pulling
  keeps a model's settings and aliases.
- **OpenAI-compatible API (`/v1`).** `GET /v1/models`, `POST
  /v1/chat/completions` (streaming or not) and `POST /v1/embeddings` on
  Assay's own address, for editors like Zed. `model` is an installed model
  or a preset (installed names win); a preset becomes its base model, its
  options and its system prompt and think setting, added only where the
  request set none. `engine.Proxy` loads the model on demand, fills sampling
  defaults only for fields the body left out, rewrites `model`, and passes
  the rest through untouched (`tools`, `tool_choice`, `response_format`),
  streaming the reply back with a flush per chunk. An open response counts
  as busy for idle unload. Errors use OpenAI's shape (unknown model 404,
  embeddings with a chat model 400). Presets need no publishing on
  llama.cpp: they're simply listed. Settings → Models shows the base URL.
  Checked: Qwen3.5-2B returned a proper `tool_calls` for a one-function
  `tools` list.
- **Speculative decoding** (`internal/engine/speed.go`). Per-model
  `speedup` (auto | off | mtp) and `spec_ngram` (on | off | an `ngram-*`
  type) sit next to `draft_model` and `draft_max`; all are part of a loaded
  server's identity. `planSpeed` decides what runs; `args` passes `-md
  <file>` and `--spec-type <list>`. llama-server loads `-md` but ignores it
  without `--spec-type` (so `draft_model` silently did nothing before
  2026-10-24, and its old "1.55× on CPU" figure wasn't real speculation): a
  draft model gets `draft-simple`, a downloaded EAGLE3/DFlash/DSpark draft
  `draft-<kind>`, and an MTP head (Unsloth's separate `mtp-<model>.gguf`,
  fetched beside the quant by `Pull`, attached by cache imports, stored as
  `Model.MTP`) `draft-mtp`. **Auto = the model's MTP head if it has one, plus
  `ngram-mod`; never a small draft model.** Measured on the RX 6700 XT
  (tok/s, prose / code edit): Gemma 4 12B 40.8 / 39.5, with its MTP head
  58.9 / 98, with head + n-gram up to 70 / 128; n-gram alone is free on
  prose and took Qwen3.5-9B from 51 to 272 on repeated code; a 2B draft
  slowed Qwen3.5-9B to 29 on prose and Gemma E4B slowed the 12B to 24, so
  Auto skips them. A draft the user picks wins over the head. Draft models
  are manifests with `Draft` (kind) and `DraftFor`, hidden from
  `Engine.List`, offered by `Engine.Drafts` to the model they were made for.
  Installed models without a head get a "Download its MTP head" button in
  Tune. Untested: real EAGLE3/DFlash/DSpark drafts.
- **GPU layers and memory.** Auto (no `num_gpu`) passes no `-ngl`, so
  llama.cpp's `--fit` sizes the load to free memory, keeping layers on the
  GPU and moving MoE experts to RAM (Gemma 4 26B-A4B IQ4_NL, 14.8 GB on a
  12 GB card: 33 tok/s). A number set by hand turns `--fit` off ("n_gpu_layers
  already set by user"), and too many layers crash the load. So the engine
  watches the whole log for allocation failures (`oomRe`; the error's tail
  is usually only a backtrace) and, on one, loads again with Auto; the
  process keeps the asked-for options so the next request doesn't restart
  it. Tune's GPU layers row says Auto fits to memory and a number doesn't.
- **LoRA adapters** (`internal/engine/lora.go`). A GGUF with
  `general.type` "adapter" and `adapter.type` "lora" (`gguf.File.IsLoRA`)
  lives in the store beside the models (`Model.Adapter`, with its base
  architecture), left out of `Store.List`, the picker and `/v1`. Added from
  a path (hard link, else used in place) or found by the scan, which reads
  each file's header once (`Finder.markAdapters`). Only models of the same
  architecture offer it (`Engine.AdaptersFor`, `checkLoRA`). The per-model
  (and preset) setting `lora` is a list of `{name, scale}` (0–2, up to 8).
  The server starts with `--lora … --lora-init-without-apply` and sets
  scales through `/lora-adapters`, so a scale change needs no restart;
  adding or removing one restarts the model. Library lists adapters; Tune's
  MODEL group has a row per compatible one. Checked with a synthetic rank-4
  adapter on Qwen3.5-2B (loaded, scaled live, applied through a preset on
  `/v1`); not yet with a real Unsloth LoRA.
- **Reply format.** Tune's Reply format (This chat → Advanced) sets
  `reply_format` (text, json, schema, grammar) with `reply_schema` or
  `reply_grammar`, saved with the chat and presets. `engine.replyFields`
  maps them to llama-server's `response_format` (`json_object`, or
  `json_schema` with the schema under `json_schema.schema`; this build
  rejects other types) or `grammar`. `Engine.Proxy` applies a preset's
  format on `/v1` only when the body has neither. A schema must be a JSON
  object, texts are capped at 32 KB, schema/grammar mode without text is a
  400. A reply that is JSON shows as a highlighted JSON block (display only).
  Checked on Qwen3.5-2B: a one-value enum schema forced `{"answer":"alpha"}`.
- **Chat caches** (`internal/engine/kvcache.go`). Requests send
  `cache_prompt` and servers run with `--cache-reuse 256` (no measured
  effect on hybrid Qwen3.5; untested on a dense model). When a model
  unloads, the KV cache of the chat it served last is saved to
  `<data>/kvcache/<model key>/<chat id>.bin` (`--slot-save-path`) if the
  history is over 1k tokens; the first request to a fresh server restores
  it before sending, then the file goes. Editing/regenerating or deleting
  the chat drops its file (`Engine.ForgetChat`); `/v1` traffic is never
  saved. The key covers blob, context, cache type, flash attention and
  draft, so a server with other settings never reads a file it can't use.
  Capped at 2 GB, oldest first; files are big (~0.5 GB for 5k tokens at
  16k context). Qwen3.5-2B with a 5k-token history: reload to first token
  2.6 s → 1.4 s; after the chat page preloads it, 1.6 s → 0.3 s (one early
  preload run didn't restore, cause unknown). Bigger models should gain more.
- **MoE expert placement.** For MoE models (GGUF `expert_count` > 0),
  Tune → Memory has `n_cpu_moe` (Auto · All = `--cpu-moe` · 0..layers =
  `--n-cpu-moe N`) and `load_mode` (`--load-mode` mmap/none/mlock; Auto
  passes nothing). A hand-set `n_cpu_moe` passes `-ngl all`, which turns
  `--fit` off; an out-of-memory load falls back to Auto, including when the
  failure is logged before `/health` (the server would otherwise die on the
  first request). Measured on Gemma 4 26B-A4B IQ4_NL, 32k context: Auto's
  fit is fastest (~33 tok/s writing, ~1030 prompt); more experts in RAM
  cost ~1 tok/s per layer (all of them ~22); no load mode beat mmap (mmap
  is 5–10% faster on prompts), so llama.cpp's "consider --load-mode none"
  warning is ignored. Healthy loads never log the allocation-failure text.
- **CPU threads.** `num_thread` becomes `-t` (Tune → Memory). Only matters
  for layers on the CPU; 4 threads beat the default (all 16) on a CPU-only
  9B (3.7 vs 2.6 tok/s, one sample), so Auto may deserve a better default.
- **Readable names.** Blobs are named by hash (a shared file is stored
  once), so every reload syncs `by-name/` with relative symlinks such as
  `unsloth-Qwen3.5-2B-GGUF-Q4_K_M.gguf`, `.mmproj.gguf` and `.mtp.gguf`
  (`byname.go`). Only symlinks there are touched; none on Windows. A move
  rebuilds them in the new folder and clears the old ones.
- **Models folder.** Chosen by `-models`, else `<config>/models.json`, else
  `<data>/models`. `Store.Plan(dst)` validates a target (resolves symlinks;
  refuses itself, inside, containing, a file, or a non-empty folder that
  isn't a store) and says move or switch, same disk or not, bytes to copy.
  Same disk: `blobs/` and `manifests/` are renamed (hard links survive).
  Another disk: copied through `.part` files with progress, sizes checked,
  then the store swaps and the originals go; hard links become copies. A
  failure or cancel removes the copies. `Engine.MoveFolder` refuses loads,
  stops all models, moves, clears the GGUF cache and saves the choice, all
  live. A running download refuses a move. Ollama imports already in the
  store aren't re-imported; across disks new ones become symlinks.
  `/api/models/folder` (GET info and mounts, POST start, DELETE cancel,
  POST `/check`); Settings → Models has the row.
- **Installing llama.cpp** (`internal/llamacpp`). The binary is, in order:
  `-llama-server`, `<data>/llama.cpp/` (your own build, never written to),
  the release Assay installed in `<data>/llama.cpp-release/`, then PATH;
  `<data>/llamacpp.json` can put the release first. The installer reads
  GitHub's `/releases` list, not `/latest` (llama.cpp's builds are all
  pre-releases; "latest" is an old tag with no binaries), and takes the
  newest with the chosen build for this system. **Builds** (`release.go`,
  `specs`): CPU, Vulkan (the default), ROCm/HIP and CUDA where the release
  has them (Linux x64: all four; Linux arm64: Vulkan, CPU, CUDA; Windows x64
  likewise with ROCm; macOS: Metal on Apple silicon, CPU on Intel). Names
  with a version (`ubuntu-cuda-12.8-x64`, `ubuntu-rocm-10.0-x64`) take the
  lowest CUDA (older drivers) and highest ROCm on offer. A CUDA build also
  needs its separate `cudart-llama-…` file (Linux names carry the tag,
  Windows names don't); both download with one progress bar and the runtime
  is unpacked beside `llama-server` (files already there win), and a CUDA
  build without its cudart isn't offered. `Recommend` picks from the GPU
  names: AMD → ROCm, NVIDIA → CUDA, either falling back to Vulkan, mixed or
  other vendors → Vulkan, none → CPU; the choice is always the user's.
  `GET /api/llamacpp/check` lists the builds (id, label, who it is for,
  size, files, URL) and `recommended`; `POST /api/llamacpp/install` takes
  `build` (empty keeps the installed build, else the default); `release.json`
  records the build. Settings has a build menu beside the install button.
  It downloads (sha256
  when GitHub gives one), unpacks safely into `<dir>.new`, and swaps via
  `<dir>.old`. `Engine.SetServer` switches binaries live; running models
  keep their process. Updating the release in use with models loaded asks
  first, then stops them. A probe (`--version`, `--list-devices`) warns when
  the build sees no GPU; it isn't run on your own ROCm build.
- **Found on this computer** (`internal/engine/sources.go`). Library
  offers, never imports by itself, the models other apps already
  downloaded, linked like the Ollama import: hard link on the store's disk,
  and across disks the file is **used in place**: its absolute path goes in
  the manifest (`Store.Blob` passes absolute names through), as found, not
  resolved, so a split model's parts keep the names llama.cpp needs. No
  symlinks, so no admin rights or Developer Mode on Windows. All of a
  model's parts link or none do. `Store.Delete` never removes an absolute
  (in-place) file, only forgets it. Imports from before 2026-10-24 may be
  symlinks; they keep working. Never a copy; the other app's folder is only
  read. Checked: SmolLM3 from Unsloth Studio's cache on a second disk loaded
  in place (99 tok/s), and deleting it left Studio's file untouched.
  Sources: the Hugging Face hub cache (`HF_HUB_CACHE`, `HF_HOME/hub`,
  `~/.cache/huggingface/hub`; also where current llama.cpp `-hf` puts
  downloads, per its `common/hf-cache.cpp`), **Unsloth Studio**'s own caches
  (read from its download records, `~/.unsloth/studio/cache/hub-state/
  manifests/*/*.json` → `hub_cache`, since Studio can use a cache on another
  disk that no environment variable names), older llama.cpp's flat cache
  (`owner_repo_file.gguf` + `.json` sidecar), LM Studio
  (`~/.lmstudio/models/<owner>/<repo>`), and folders you add
  (`<config>/sources.json`, 4 levels deep). Names follow the downloader
  (`hf.co/<owner>/<repo>:<QUANT>` via `hf.Quants`: splits grouped, mmproj
  attached, drafts, `.incomplete` and dangling snapshot links skipped).
  Already in the library = same name, `sha256-<hash>.gguf` in the store
  (the HF blob name is the hash), or the same file (`os.SameFile`). The
  manifest records `source`, `link` and `origin`; Library says "linked
  from …", and a model whose file has gone shows as missing and fails to
  load with a message naming where it was. Scans are stat-and-name only,
  cached 30 s; `GET /api/models/sources`, `POST …/sources/import`,
  `POST`/`DELETE …/sources/folder`. Checked against this machine: Unsloth
  Studio's cache on a second disk offered 11 models (~83 GB).

### internal/ollama

Ollama's API shapes (`ChatRequest`, `ShowResponse`, `PullProgress`, …),
which the engine and server still pass around, plus `APIError`. The HTTP
client that once talked to Ollama lives on in `internal/ollama/ollamatest`,
only so the server tests can drive their fake backend through it. Pulls
accept only `hf.co/…` names; an Ollama library name gets a clear error.

### internal/hf (M2)

- Search: `huggingface.co/api/models?search=…&filter=gguf&author=…`.
- Files and sizes: `/api/models/{repo}/tree/main`. Quant names (`Q4_K_M`,
  `UD-Q4_K_XL`, `IQ2_XXS`, …) are parsed from the filenames.
- Fit estimate: weights + KV cache at the chosen `num_ctx`, compared against
  detected VRAM/RAM (`nvidia-smi`, `rocm-smi`, `/proc/meminfo`).
  Result: fits on GPU / tight / spills to RAM.
- Architecture from GGUF metadata is checked against what the local Ollama
  version can run. Unsupported models are flagged before download.

What Discover can filter on comes from the search API itself:
`expand[]=gguf` returns parameter count, architecture and trained context
length for every result, so filters run locally over the top 100 results
without a request per repo. MoE is inferred from the architecture name
(`*moe`, `mixtral`, `gptoss`, …) or an `A3B`-style suffix in the repo name.
Vision comes from the pipeline tag on search results and from `mmproj`
files on the model page.

The fit estimate is deliberately simple: file size + a per-token KV-cache
allowance scaled by parameter count (halved for MoE) + 600 MB. It ignores
hybrid/sliding-window attention, so it's pessimistic for those models.
Reading layer counts from the GGUF header via an HTTP range request would
make it exact; that's a later improvement.

### Takes → Makes (modalities)

`Repo.Modalities()` maps the pipeline tag to what a model takes and makes
(text, image, video, audio): `image-text-to-text` is text+image → text,
`automatic-speech-recognition` audio → text, `text-to-speech` text → audio,
and so on; tags like `vision`, `omni`, `audio` add inputs, only to
text-output models; no tag counts as text → text; embedding pipelines make
nothing. Discover's "Takes → Makes" chips match when inputs ⊇ Takes and
outputs ⊇ Makes (in the browser; `takes=`/`makes=` in the URL, old
`vision=1` reads as Takes: Image). A plain search's top results bury media
makers and speech models, so when Makes has media or Takes has audio/video,
`/api/hf/search` also asks HF once per matching `pipeline_tag`
(`hf.SearchPipelines`) and merges by downloads. `Repo.CantRun()` explains
what llama.cpp can't run: generators (FLUX, Wan, TTS; hidden unless Makes
asks for media) and whisper.cpp-format speech models (ASR with arch
`whisper` or none). They're badged "Can't run here" and their model page
shows the reason instead of quants. Known gaps: repos without useful tags
(ggml-org's Voxtral and Qwen3-Omni) read as text → text.

### Draft models (speculative decoding)

Repos increasingly ship small draft models (DSpark, DFlash, EAGLE3) for
speculative decoding, both as their own repos and as files next to the real
quants (unsloth's DeepSeek-V4.1-Flash-1017 has `dspark-…-Q8_0.gguf` and
`dspark/…-BF16.gguf` beside 200+ GB quants). They can't chat on their own,
so they must never pass for the model:

- **Files:** `hf.Quants(repo, files)` flags a file as a draft when its path,
  with the repo's own model name removed, matches dspark / dflash / eagleN /
  draft / drafter (so a full model called "…-DSpark-MTP" isn't one). Drafts
  are named "DSpark Q8_0", sorted after the real quants, and the model page
  lists them in their own "Draft models" section with no Download button
  (Ollama 0.40.6 ignores draft GGUFs; see BACKLOG).
- **Repos:** `Repo.IsDraft` trusts the GGUF architecture when HF knows it
  (real drafts report `dflash`, `deepseek4-dspark`, `…-dflash-draft`; full
  models with draft heads keep `qwen35`, `deepseek4`), then the
  `draft-model` / `speculative-decoding-draft` tags, then the name. Discover
  shows "Draft model" (and what it's for, from `base_model`) instead of a
  fit label, drafts get no intelligence score, a "Hide draft models" filter
  exists, and the model page opens with a banner saying it isn't a model to
  chat with.

### Downloads (M2)

`internal/pulls` runs downloads in the background and tracks progress; what
actually downloads is a `Puller`. With the llama.cpp engine, `hf.co/{repo}:{quant}`
names go to `engine.Pull`: it lists the repo, picks the quant's files (and
the F16 `mmproj` for vision models), and fetches each with HTTP ranges into
`blobs/<name>.partial`, resuming after a cancel, a dropped connection or a
minute-long stall, and checking the sha256 Hugging Face lists before renaming
it into place. Single files are named by hash (two names share them); split
models keep their file names in a folder, since llama.cpp finds the later
parts by name. Unsloth's `params` file becomes the model's default
settings. The short name is an alias in the manifest (`engine.Copy`). Other
names (the Ollama library) are refused with a hint to use an `hf.co/` name.

Before fetching, `Pull` and `PullMTP` check free space in the models folder
(`engine/space.go`): what's still missing (finished files count nothing, a
`.partial` counts what's left) plus 1 GB to spare must fit, after subtracting
what other downloads in progress still need (`Engine.reserved`). Otherwise
the download fails at once with the sizes and folder. The model page flags
quants that won't fit (`quantRow.NoRoom`) but still offers the button.

### Load and eject

Loading from the UI starts the model's llama-server and pins it
(`keep_alive: -1`) until you eject it, the way Unsloth and LM Studio behave;
otherwise a model unloads after its keep-alive (5 minutes by default, or its
own setting) without requests. Every request refreshes the timer, and a
pinned model stays pinned. The sidebar shows each loaded model's VRAM, how much of it is
on the GPU, and whether it stays until ejected or auto-unloads.

### Voice

Dictation plus reading aloud. Until 2026-10-21 this was a hands-free
conversation mode (it sent on each pause, talked over thinking with a
filler model, and could be interrupted); it was swapped for dictation
because it was less predictable. The server side of that mode
(`voice: true` on `/api/chat`, the talker) still exists but no page uses it.

```
mic ─► browser: energy VAD, 16 kHz WAV ─► /api/voice/transcribe ─► whisper
         each pause ends an utterance; its text is appended to the message box
         (live partials every ~0.9 s while you talk). Send ends dictation.
message ─► /api/chat {spoken: true if "Spoken style"} ─► model
reply sentences (if "Read aloud", or a reply's speaker button) ─► /api/voice/speak ─► Kokoro
```

- **Composer.** The mic button toggles dictation. Utterances are written
  down in order, each after whatever is in the box, so typing between them
  is kept. Pressing Send while dictating waits for the last utterance, then
  sends. The bar under the composer (shown once voice is set up) has two
  toggles remembered per browser: **Spoken style** (adds the one-line prose
  prompt below) and **Read aloud** (speaks each reply as it streams). Every
  reply also gets a "Read aloud" action.

- **Speech service.** A separate local process with OpenAI's audio API
  (`/v1/audio/transcriptions`, `/v1/audio/speech`). `tools/voice/voice_server.py`
  is the bundled one (faster-whisper `small.en` + Kokoro int8, both on the
  CPU, so the GPU stays free; Whisper, Kokoro and the talker each get their
  own share of the cores, since on a shared pool transcription slowed from
  1 s to 7.5 s mid-reply); `-voice URL` points Assay at it or at any
  compatible server (speaches, …).
- **Started on demand.** The script is embedded in the Assay binary. When
  the mic is pressed and nothing answers on the speech URL, Assay writes it
  to `~/.cache/assay/` and runs it with `uv run --script` in its own
  process group; it stops it after `-voice-idle` (10 min) without voice use
  and when Assay exits. The first start installs the Python packages
  (about a minute); later starts take seconds. A service someone else
  started is used and left alone. `-voice-launch=false` turns this off,
  `-voice ""` turns voice off. Nothing about voice is needed for the rest of
  the app.
- **Downloads only after a yes.** A fresh install has no voice files. Until
  the service has been downloaded (a marker in `~/.cache/assay/`, or the
  Kokoro model already present), `/api/voice` reports `needsDownload` and
  won't start it; the voice bar asks "Download / Not now" and only
  `?install=1` starts the first run (~800 MB: uv's environment, Kokoro,
  whisper small.en). Settings → Voice shows the space used (hard links
  counted once) and **Remove** deletes it all: the marker and script,
  `~/.cache/assay-voice`, the faster-whisper models in Hugging Face's cache
  plus the blobs their symlinks point to (newer hub versions share one
  `blobs/` folder), uv's `voice-server-*` environments, and the script's
  packages from uv's cache (`uv cache clean <pkgs>`, since environments are
  hard-linked to it).
- **Browser.** An AudioWorklet feeds 20 ms frames to a speech detector with
  an adaptive noise floor and 300 ms of pre-roll. A frame counts as voice
  when it's loud without being a click (peak/RMS under 7), and an utterance
  starts only when 11 of the last 15 frames were voice: typing is short
  bursts with gaps, speech is sustained. While a reply is being read aloud
  the threshold is 2.5× higher, against echo. Replies are cut into sentences as
  they stream and synthesized ahead of playback; markdown is stripped and
  code blocks are skipped. Clips are spaced 450 ms after a full stop, 200 ms
  after a comma (Kokoro trims silence off each clip). Mic access needs a secure context: localhost or
  HTTPS (e.g. `tailscale serve`).
- **Does it think?** Ollama's capability list misses some thinkers (Gemma 4
  thinks without listing `thinking`), so a model counts as thinking unless
  thinking is switched off or Assay has seen it answer without thinking.
- **The talker.** While the main model thinks, a second model writes one
  short first-person line at a time ("I'm checking how the angle of the sun
  changes things"): a lead-in at once, then a new line whenever the speech
  is about to run out and there's enough new thinking, with growing gaps on
  long thinks. It runs on the CPU (`num_gpu: 0`, 4k context, 4 threads): no VRAM, no GPU
  contention, and no need for Ollama to serve two requests on one model,
  which hybrid models like Qwen3.5 can't. Assay picks the largest
  installed chat model under 3 GB and loads it when voice mode starts.
  Without one, the main model says a lead-in before it starts thinking.
- **Guarding the talker.** Small models reading half-finished notes blurt
  out wrong intermediate results. Lines are dropped when cut off, near
  repeats, phrased as conclusions, or containing numbers the question
  didn't; dropped lines back off the next attempt.
- **Small talk skips the big model.** The talker's first call either
  answers ("REPLY: Hey! What can I do for you?") for greetings, thanks and
  small talk, which cancels the big model's request, or gives the lead-in
  ("WAIT: …"). So "hi" gets one reply in about a second, not a lead-in plus
  an answer. The reply is saved with the talker as its model.
- **Prose for the ear.** Spoken-style replies get one system line asking for plain
  prose (no markdown, LaTeX, code, tables or lists). On Gemma 4 12B that
  took markup from 144 marks to 0 at no extra thinking time. Keep it one
  line: a longer list of style rules made Qwen3.5-9B think 95 s instead of
  8 s over "what is 2 plus 2".
- **Numbers.** espeak-based engines read "1.67" as two sentences, so
  `voice.SpeechText` spells out decimals ("1 point 6 7"), money, percents,
  multipliers, ranges and a few abbreviations before synthesis.
- What was said aloud is saved with the reply (`spoken`) and shown above it.
- **Settings → Voice** (`voice.json`, so every device shares them): Kokoro
  voice with a preview, speed, pause after a sentence, Whisper model (the
  service swaps models on request and downloads on first use) and
  speech-detection sensitivity. The filler model and prose switch of the
  old mode are kept in the file but no longer shown. The bundled service lists its voices at `/v1/audio/voices`.

### Projects and RAG

A **project** is a folder of chats that can also carry instructions, a
default model or preset, and files. Leave those empty and it's just a
folder. Chats store their `project` ID; the sidebar lists projects
(collapsible, drag chats onto them) above the dated loose chats, and the
chat page has a project picker above the conversation (moving an existing
chat, or choosing where a new one goes).

Storage, per project (`~/.local/share/assay/projects/<id>/`):
`project.json` (settings + file list with status), `files/` (the originals)
and `index/` (`index.json` with the chunks + `vectors.f32`, normalised
float32). No database: search is a linear cosine scan (`internal/index`),
fine for tens of thousands of chunks. Deleting a project keeps its chats.

**Indexing** (`runIndexer`, one file at a time in the background): extract
text (`internal/rag`: text and code, HTML, PDF via `pdftotext`, DOCX, ODT;
20 MB max), split into ~1,500-character chunks with 200 of overlap
(headings and PDF pages tracked, code fences kept whole), embed 16 at a time
through the engine (llama-server's embeddings endpoint) with "file · heading" prefixed, then store.
Progress shows on the project page. Files wait (not fail) until an
embedding model is chosen; restarts pick up unfinished files.

**Embedding model** (Settings → Knowledge, `knowledge.json`): the user picks
and downloads one (suggestions: qwen3-embedding:0.6b, embeddinggemma,
nomic-embed-text, all as `hf.co/` GGUFs; any GGUF embedding model works). One index holds one
model's vectors; switching models re-indexes every project.

**Retrieval** (in `prepareTurn`): for a chat in a project with indexed
files, the latest user message (plus the previous one when it's short) is
embedded, the top-K chunks (default 6, cosine ≥ 0.25) go into the system
prompt after the project's instructions, numbered for citation. They're
streamed to the page (`sources` event) and saved on the reply, shown as a
folded "N passages from the project's files" list.

### Tool use

When a chat has tools on (its Tune switch, else the preset's, else on) and
the model's template has the "tools" capability, `prepareTurn` builds the
tool set from `internal/tools` per `tools.json` (Settings → Tools):
`web_search` (Off · SearXNG · DuckDuckGo scraping), `fetch_url`,
`calculator`, `search_chats`, `search_project` (in a project with indexed
files) and `current_time`. `chatLoop` sends the schemas, streams as usual,
and when the model returns `tool_calls` (the engine joins streamed pieces)
runs them, appends the assistant message and one `tool` message per result,
and asks again, up to 8 rounds (the 9th has no tools). `fetch_url` checks
every dialed address in the dialer's Control hook (after DNS, on each
redirect) and refuses loopback, private, link-local, CGNAT/Tailscale,
multicast and unspecified; 2 MB, 20 s, ~8k chars of readable text. The
calculator is a parser, not eval. Calls are saved on the reply
(`chats.Message.Calls`: args, truncated result, label, and what the model
said/thought before each) and shown as foldable rows between thinking and
text, live and when reopened; copy and read-aloud use only the final text.
History replays them as assistant/tool messages. One short system line
mentions tools. Compression counts the tool schemas too (estimated at 3
characters per token). Checked with Qwen3.5 9B: calculator, fetch_url of
example.com, search_chats and DuckDuckGo search all called correctly.

**Opt-in tools.** A chat's Tune row "Edit code blocks" (`Chat.EditCode`, sent as `editCode`, off by default) adds `edit_code` to the tools in `turnTools`; it lives in `internal/server/edit.go`, outside the Settings catalog. The model sends `{find, replace}`; `applyEdit` takes the latest fenced block (saved message or an earlier successful `edit_code` result) that contains `find`, requires exactly one occurrence, and returns the whole updated block. The saved messages are never changed: the block comes back as the call's result, which the chat shows open and the model sees in later prompts.

**Opt-in forget.** Tune's "Trim old messages" (`Chat.Forget`, sent as `forget`) adds `forget` (`internal/server/trim.go`). The model sends `message_ids` or `from`/`to` (numbers count the chat's messages from 1) and a `note`; the chat gets a `Trim` (`Chat.Trims`: indexes plus the note). `trimmedMessages` applies the trims wherever the history is built (`prepareTurn`, `turn.hist`), so each trimmed message reads `[trimmed: note]` in later prompts, without thinking, tool calls or attachments; the last 4 messages are never trimmed. Saved messages never change. The trim applies from the next prompt, not later steps of the same reply. The reply's call shows "trimmed N messages · undo", and `DELETE /api/chats/{id}/trims/{trim}` undoes it. Editing from an earlier message drops the trims of the messages it cuts (`DropTrimsFrom`).

### Context compression

Before compressing, `pruneMessages` (`internal/server/prune.go`) runs once the
prompt is at 60% of the context: fenced code blocks and attached files in older
messages that a later message repeats (same file name, or same language and
first line) become a short note. Only the prompt changes, never the saved chat;
the last 4 messages and the system prompt are kept, so short chats stay
cache-identical.

`internal/server/compress.go`. Before a reply the prompt is counted with
llama-server's `/tokenize` (+~12 tokens per message; a 3-chars-per-token
estimate without it; a reply's `prompt_n` only counts uncached tokens, so it
can't measure the history). Past 80% of the context, or with less than
max(num_predict, 1k) left for the reply, the oldest messages are summarized
by the same model (one request, thinking off, temperature 0.2, ≤512 tokens,
no reply format). The cut is the fewest older messages that leave the rest
within half the context (keeping at least the last 4, starting at a user
message). The summary is stored in the chat (`compression`: text, upTo,
model, saved) and sent inside the system message, so templates that reject
a second system message are fine, followed by the messages from `upTo`. It
is extended when more messages need folding (in runs of about half the
context) and dropped by an edit or regenerate reaching before `upTo`; any
change forgets the chat's saved KV cache. Thinking is never sent. Per chat:
Tune → "Compress automatically" (Default/On/Off), else the preset, else
Settings → Chats (`chats.json`). Manual: Tune's "Compress now" / "Undo"
(`POST /api/chats/{id}/compress`, `DELETE …/compression`). The page shows a
divider at `upTo` that folds open to the summary. Checked on Qwen3.5-2B at
4k context: compressed on turn 13 and still recalled a fact from turn 1.

### internal/chats

One JSON file per conversation. Writes go to a temp file and are renamed into
place, so a crash can't leave a half-written chat. Chat IDs are validated
against a strict pattern before they touch the filesystem.

### internal/server

- Server-rendered pages (`html/template`) plus a small vanilla JS file for
  streaming chat and markdown.
- Markdown (chat.js, no library): everything is escaped before any tag is
  added. Blocks: headings, paragraphs, nested and task lists, quotes,
  rules, GitHub tables (alignment from `:--:`, `\|` and code spans keep
  pipes, wrapped so wide tables scroll inside the reply), fenced code (hljs,
  Copy/Download) and KaTeX math. Inline: code, bold, italic, strikethrough,
  links (only http(s) and mailto become links; bare URLs too). The whole
  reply is re-rendered on each streamed chunk, so a table appears once its
  `|---|` row arrives.
- Chat streaming: the browser `POST`s to `/api/chat` and reads an NDJSON
  response (`chat`, `thinking`, `content`, `done`, `error` events) with
  `fetch` + `ReadableStream`. Stopping is just aborting the fetch. The same
  approach will carry pull progress in M2.
- Security:
  - Listens on `127.0.0.1` by default.
  - The `Host` header must be `localhost`, an IP literal or a name passed via
    `-allow-host`. This blocks DNS-rebinding attacks from websites you visit.
  - `http.CrossOriginProtection` rejects cross-origin state-changing requests
    (CSRF).
  - No login. For access from other devices, use Tailscale (bind to the
    tailnet IP or `tailscale serve`). Don't expose it to the internet.
  - The llama-servers listen on 127.0.0.1 only. Remote browsers and
    editors only talk to Assay.

## Look and navigation

The look is design "D, Calm numbers". `web/static/app.css` starts with the tokens in three blocks (light in `:root`, dark by system preference, dark by choice with `[data-theme="dark"]`). Existing names stay (`--bg --surface --border --border-strong --text --text-2 --muted --soft --accent --accent-soft --accent-text --accent-line --good --bad --bad-soft --warn --warn-soft --code-bg --code-text`, `--s1..--s5`, `--fs*`); new ones are `--panel --sunk --raised --track --on-accent --radius (10px) --radius-lg (12px) --dur-1 (120ms) --dur-2 (180ms) --ease --vgap --shadow-pop`. `--surface` is `--panel`; `--soft` is the hover fill. Full-height layouts subtract `var(--vgap)` (the panel's 16px of margin; 0 on phones) as well as `--appbar-h`.

- **Accent.** Two accents, copper (default) and indigo, chosen in Settings, Appearance and stored like the theme (`localStorage` keys `theme` and `accent`, `data-theme` and `data-accent` on `<html>`, applied by `theme.js` before first paint). Only `--accent` and `--on-accent` change per accent and theme; `--accent-text` is the accent itself, and `--accent-soft`/`--accent-line` are `color-mix` tints of it over `--panel`, derived once in `:root`. Copper is `#a5552a` light / `#e09a6b` dark, indigo `#4453c4` / `#8e9bf2`; all pass 4.5:1 as text on the panel, raised and soft backgrounds. Text colours (`--muted`, `--good`, `--warn`) also pass 4.5:1 on bg, panel, sunk and raised.
- **Shell.** The sidebar sits on the page background with no border; the content is a rounded panel (`.content`) with a hairline ring. Inter (variable, latin and latin-ext, `static/fonts`, OFL text beside it) is the font; monospace is only for quant names and code. The block at the end of `app.css` restyles the shared classes (cards, popovers, chips, segmented controls, fields) in one place.
- **Motion.** Hovers and presses use `--dur-1`, panels and popovers `--dur-2` with a short slide and fade (`pop-in`, `slide-in-right`). Only background, colour, box-shadow, transform and opacity are transitioned. `prefers-reduced-motion` sets both durations to 0.
- **Page changes.** `@view-transition { navigation: auto }` cross-fades the main panel in about 150 ms; the sidebar and app bar have fixed `view-transition-name`s so they stay put (the old sidebar snapshot is hidden, so the chat list does not flash). `layout.html` also restores the sidebar and chat-list scroll position from `sessionStorage` before first paint. Speculation rules in `layout.html` prerender same-origin links on hover (`eagerness: moderate`), skipping `/api/*`, `/v1/*`, `/static/*`, `/settings*`, `/setup*`, model pages (`/discover/*/*`) and anything marked `data-no-prerender`. Browsers without these features (Firefox) just navigate as before.
- **Prerender gating.** A prerendered page runs its scripts, so work that polls, loads, or reattaches to a running reply must wait until the page is shown: wrap it in `window.assayWhenVisible(fn)` (defined in `theme.js`; runs `fn` now, or on `prerenderingchange` if `document.prerendering`). It is used for the loaded-models refresh, the GPU sampler, the downloads poll and running-chat tracking in `app.js`, the stream reattach in `chat.js`, the search in `discover.js`, the scan in `found.js`, setup, and the startup loads on Settings and project pages. New page scripts that fetch or poll on load should do the same.

## Platforms and releases

`sh scripts/release.sh [version]` cross-compiles (no cgo) five platforms into
`dist/release/` (see Releases below). Platform bits:

- **Windows**: data in `%LOCALAPPDATA%\assay`, settings in
  `%APPDATA%\assay`; RAM from `GlobalMemoryStatusEx`, VRAM from the
  display adapters' registry keys (`HardwareInformation.qwMemorySize`, any
  vendor; nvidia-smi first), disk space from `GetDiskFreeSpaceExW`, drive
  letters as models-folder suggestions. `-open` (default on Windows) opens
  the browser; a second launch finds the port taken and just opens the
  page; an error waits for Enter so a double-clicked console doesn't
  vanish. Hard links aren't counted (no link count in `FileInfo`).
- Checked under Wine (2026-10-24): pages, the llama.cpp install
  (`win-vulkan` zip, found the GPU over Vulkan) and a model download work;
  `llama-server.exe` itself crashes at warm-up under Wine even when run by
  hand, so loading a model needs a real Windows machine to test.

## Storage

No database. Plain files under XDG directories:

```
~/.config/assay/
  config.toml            listen address, HF token (planned)          (M2)
  presets/<name>.toml    one preset per file                         (M3)
  voice.json             voice mode settings (Settings → Voice)
  knowledge.json         RAG settings: embedding model, excerpts per answer
~/.local/share/assay/
  chats/<id>.json        one conversation per file
  scores.json            cached intelligence scores, refreshed daily
  projects/<id>/         project.json, files/, index/ (RAG)
  bench/<digest>.json    speed tests from the Compare page
~/.cache/assay-voice/  Kokoro model files (the speech service's)
```

Presets are plain TOML, so they diff, hand-edit and sync well.
Full-text search across chats is the only feature that might justify SQLite
later (embedded, still not a service).

## UI

Design reference: direction **F** on the design canvas
(https://claude.ai/artifact/Ad3hWUHs96WFjh3TDHnjoG).

- Compact, mostly sans-serif (IBM Plex Sans). Monospace only for quant names
  and code. Light and dark themes.
- Left sidebar: Chat, Library, Discover, Compare, Presets, recent chats;
  Settings and the theme toggle at the bottom. On phones it's a drawer (hamburger, swipe right).
- App bar: loaded models as pills (VRAM, context; "CPU" for the talker),
  downloads, page actions.
- **Chat**: centered column. The Tune panel is closed by default and slides
  in from the right. A button in the top bar summarizes the current settings
  and how many differ from the preset. Tune has two parts: "This chat"
  (thinking, temperature, reply length, system prompt, and an Advanced
  expander with the rest of sampling, closed by default) and the model's
  part (context, GPU layers, keep loaded, and a Memory expander with cache
  type and flash attention), saved for the model as soon as they change.
  Code blocks are highlighted with vendored highlight.js (only fences that
  name a language; no auto-detect, since it re-runs on every streamed
  chunk), with Copy and Download (file name from a first-line comment like
  `// main.go`, else `snippet.<ext>`).
- **Install on a phone**: a web manifest, icons and a service worker that
  caches nothing (it's there so browsers offer "Install"; pages and
  streamed replies always come from the server). A full install needs
  HTTPS or localhost; over plain HTTP it's an "Add to Home screen" shortcut.
- **Settings**: a jump bar, then Models (llama.cpp build and backend, the
  install/update, defaults, models folder, the `/v1` URL, models with their
  own settings and a "Use defaults" button), Knowledge, Voice,
  This machine. Below 1180px wide Tune floats over the chat instead of
  docking beside it. A mic button next to Send dictates
  into the box; the bar under the composer shows listening / reading aloud
  and holds the Spoken style and Read aloud toggles.
- **Library** (design D, "Calm numbers"; `library.html`, `library.js`,
  `internal/server/library.go`, `measure.go`, `prettyname.go`): a stat strip
  (GPU memory, system memory, models on disk and free space; the first two
  open a popover of what's using it, from `/api/loaded` and `/api/gpu`, polled
  every 4 s; disk links to Settings → Models), then one row per model:
  readable name (`prettyModel`: "gemma-4-12b-it-qat:ud-q4_k_xl" → "Gemma 4
  12B" + QAT chip; the raw name stays the identifier, in the tooltip, and in
  `data-model`), quant from the tag (UD-Q4_K_XL), size, a green dot when
  loaded; Speed (measured tok/s, fastest in accent, "Measure" when missing),
  Memory (estimate at 8k, GPU / Tight / + RAM, amber bar when not fitting),
  Score. Every number is clickable: speed → Compare, memory → a fit
  explanation with a link to Tune (`/chat?model=…&tune=1`, which `app.js`
  honours by clicking the Tune button), score → a popover of similar-size
  scores. Column headers sort (Speed ↓, Memory ↑, Score ↓; missing values
  last), as do the Name/Recent/Size chips; the sort lives in `?sort=` and
  localStorage, and rows reorder with a FLIP animation (`--dur-2`, off under
  reduced motion). Chat and ⋯ (Load/Eject, Tune, Compare, Measure, Star,
  Delete) show on hover or focus-within.
  Scores: the page renders instantly with cached ones; `GET
  /api/library/scores` resolves the rest (Hugging Face `base_model:quantized`
  tag → Artificial Analysis score, cached 24 h per repo in memory, failures
  1 min) and `GET /api/library/nearby?name=` returns the model's score among
  rated models within about 0.65–1.55× its size, two above and two below,
  with the reason when unrated (local, no base model, not listed, offline).
  **Measure all**: `POST /api/library/measure {names?, evictPinned?}` (no
  names: every untested non-embedding model) starts a server-side job that
  runs `runBench` on each in turn, so it survives leaving the page;
  `GET` reports `{running, items[{name,title,state,reason,genTps}], done,
  total, current, waiting, keptPinned, cancelled}` and `DELETE` cancels. If a
  pinned model is loaded the job keeps it (`unloadFor` unloads only the
  unpinned) and skips models that can't fit next to it, and the pinned model
  itself, with the reason; the page offers "Eject the pinned model and
  measure these". It waits while a chat reply is being written, refuses to
  start during one, and when done unloads what it loaded and reloads what was
  loaded (unpinned) before. The progress line polls every 1.5 s and fills
  numbers in as they land.
- **Discover** (design D, "Discover 3"; `discover.html`, `discover.js`): three
  columns: filters | results | the model's detail panel (opens on row click).
  Filters are boxed groups with plain titles: "Can do" (switches: reads
  images, watches video, hears audio, makes images/audio/video, long memory
  128k+), "Size" (Any/Small/Mid/Large segments = max 9B, 9-35B, 35B+, exact
  B range inputs, "Fits my computer"), "Your library" (hide what I have),
  "Source" (Uploader drop-down, hide gated, hide draft helpers) and a
  collapsed "More filters" (Takes/Makes chips including text, dense/MoE,
  architectures with readable names and counts, has score). Takes/Makes are
  one state (`takes`, `makes` sets) shown by both the switches and the chips
  (`[data-io="takes:image"]`). Everything applies instantly; active filters
  also appear as removable chips above the results with "Clear all". Sort is
  a drop-down on the result-count line. Rows: readable name (wraps, never
  ellipsised; `prettyModel` names and tags), a facts line (uploader, dense/MoE
  size, abilities in words such as "reads images", "long memory"; each is a
  button that narrows the list), then on the right downloads (or likes,
  trending, updated, following the sort), the Q4 size in GB, the intelligence
  score or "not rated", and a fit pill (Fits, Fits tight, Uses RAM · slower,
  CPU only, Too big). The results pane is a CSS container: under 640 px (a
  small window, or the detail panel open) the numbers fold onto one line
  under the name. The detail panel (the model page fragment, `?partial=1`):
  title, 2x2 clickable stat tiles (score and downloads open outside; parameters
  and memory narrow the list in place via `data-facet`), "Pick a size" with a
  fit pill per quant, and a Download button for the best-fit quant. **Loading**:
  while a search runs the old rows stay (dimmed); if there are none, eight
  placeholder rows shaped like real ones appear, but only after 200 ms, so a
  fast search never flashes them, and nothing moves when the results replace
  them.
- **Drop-downs** (`dropdown.js`): `<select data-dd data-dd-label="Sort">` is
  kept as the state (value, options, `change`) and hidden; a button and a
  fixed-position menu (keyboard: arrows, Home/End, Esc) stand in for it.
  Setting `select.value`, `selectedIndex` or the options from script
  repaints it. Used by Discover, Compare, project and Settings; a page that
  builds a select later calls `window.assayDropdown(select, {label})`.
- **Model page**: readable title (and tags) over the repository name, quant
  rows (name, size, fit label with bar and a "best fit" chip, quality,
  Download or Chat; the "Not enough disk space" note stays under the row),
  recommended settings, specs and the intelligence score.
- **Compare** (`/compare?m=a&m=b`, up to 4 installed models): specs from
  Ollama (parameters, quant + rule-of-thumb quality, disk, architecture,
  longest context, capabilities), the context that fits on the GPU, the
  intelligence score, and speed measured here, as sections with a muted
  source. Values are in the text colour with muted units; the best value in
  a row is in the accent. Parameters and architecture link to Discover, disk
  to the Library, the fitting context to a chat, the score to Artificial
  Analysis. Not yet: card benchmarks, quant loss (KLD).
- **Presets, projects, Settings, Setup** share one boxed-row shape
  (`.boxed-list` and `.row`: title and muted detail on the left, control on
  the right, a faint rule between rows). Presets are Library-style rows (name
  and system prompt, readable base model, settings chips, Chat and Delete on
  hover). Checkboxes on these pages are the same styled box; switches are
  `input.switch`.

## Benchmarks (M4)

| Data | Source | Trust |
|---|---|---|
| Speed (prompt tok/s, gen tok/s, time to first token, load time) | measured locally (Compare → Run speed test), see below | high, your hardware |
| Speed estimate before download | size and memory bandwidth heuristic | rough |
| Intelligence score | external index where it covers the model, otherwise none | medium, see open questions |
| Card benchmarks | parsed from the HF README | low, authors' own setups |
| Quant loss (KLD) | the uploader's published numbers, when present | medium |

Every number in the UI is labeled with where it came from.

**The speed test** (`POST /api/bench`, one at a time) unloads every model
and waits until Ollama reports them gone (an unload returns before memory
is free), settles the context with `loadAuto`, unloads again, then times a
fresh load at that context, reads a fixed ~1,080-token prompt and writes
256 tokens with `think: false`, temperature 0. Prompt and generation speed
come from Ollama's eval counts and durations; time to first word is
wall-clock to the first streamed token. Results are keyed by digest, so a
model's aliases share one result.

**Context that really fits.** The auto-context estimate leaves out some of
Ollama's buffers that grow with context, so a big context can look like it
fits and spill onto the CPU (Qwen3.5-9B at 256k: 11.2 GiB, 79% on the GPU,
15 tok/s instead of 39). When an auto-sized model is alone on the GPU and
`/api/ps` shows it partly on the CPU, `noteOverflow` caps that model one
step below (`ctxCap`); `loadAuto` (Load button, speed test) retries at once,
chats pick the smaller context next message. A saved speed test that stayed
fully on the GPU seeds the cap after a restart.

**GPU layers by hand.** Tune's GPU layers sets `num_gpu` (Ollama counts the
model's blocks plus the output layer; `/api/models/defaults` reports it as
`layers`). A request with `num_gpu` gets no auto context (auto assumes the
whole model on the GPU), and `manualGPU` marks the model so its deliberate
spill isn't recorded by `noteOverflow`; `loadAuto` and chats without it
clear the mark.

**Installed-model fit** (Library, model switcher): `hw.Estimate` at 8k
context from the file size and parameter count, overridden by a speed test
when there is one (100% on the GPU → fits; less → spills).

### Your own benchmarks (internal/evals, /benchmarks)

Sets of questions are TOML files: six built in (`internal/evals/builtin`,
embedded) and your own in `<data>/evals/sets/<id>.toml`. A set has a name,
a short column label, a default check and `[[questions]]` with `prompt` and
`answer`. Checks: `number` (with `tolerance`; fractions work), `exact`
(case, spacing, end punctuation ignored; `accept` lists alternatives),
`choice` (lettered `choices`), `contains` and `regex` (on the whole reply;
`not` inverts). For number, exact and choice the question ends with
"End your reply with a final line Answer: …"; the grader reads the last
such line, else the last `\boxed{}`, else a guess from the end. Thinking
is stripped before grading. Built-in answers are checked by a test that
grades a model-style reply carrying each right answer, and the math and
code sets' answers were computed by running them.

A set's hash covers questions, answers and checks (not the name), so a
result from before an edit shows as stale (`*`) and leaves the overall.

**Code and judge checks.** `code`: the reply's last ```python (or
unlabelled) block is written to a temp folder with the question's `tests`
and `harness.py` (embedded), run as `python3 -I -B harness.py <token>`
under `unshare --user --net --map-root-user` when that works (no network),
with RLIMIT_AS 1 GB, FSIZE 10 MB, CPU 20 s and a 15 s wall clock, a clean
environment and HOME/TMPDIR in the folder. The harness execs the solution
then the tests in one namespace and prints the token (taken off argv
before the solution runs) only at the end, so `sys.exit(0)` in a
solution can't pass. Not a full sandbox: the code runs as the user.
`judge`: answers are saved `pending`; after the tested model has
answered the whole set the judge model (picked in the run dialog; the
biggest installed by default, remembered) is loaded and grades each
against the reference in `answer` with `evals.JudgeSystem`, thinking as it
likes, up to 8192 tokens; the last "Verdict: PASS/FAIL" decides, no
verdict counts as a fail. Without a judge the answers wait (the cell says
"judge") and the next run with one grades only those. A model judging
itself is shown as such. Built-in code tasks were checked when written:
each reference solution passes its tests and a plausible wrong one fails
(references aren't shipped).

**Lean proofs (internal/lean).** Asked for 2026-10-31 instead of judged
proofs (a 12B judge is the strongest this machine runs). `lean`
questions give a `statement` and optional `preamble` (definitions shown
to the model); the model writes `theorem goal : statement` with any
helper lemmas, in a ```lean block. The checked file is preamble, the
model's code inside `namespace AssayProof`, then `theorem assay_check :
statement := AssayProof.goal` (so goal must prove exactly the statement)
and `#print axioms assay_check`; only propext, Classical.choice and
Quot.sound may appear. A regex refuses sorry, admit, axiom,
native_decide, implemented_by, extern, unsafe, elab/macro_rules/run_cmd
and similar before Lean runs; 60 s limit. Lean is core only (v4.35, no
Mathlib: no ring/linarith, but omega, simp, decide and grind work).
`lean.Installer` downloads the release zip from GitHub (about 0.9 GB;
the .tar.zst is smaller but needs zstd, which Go's stdlib lacks) into
`<data>/lean` (3.2 GB unpacked); a `lean` on PATH is used when nothing is
installed. The 16 built-in theorems' reference proofs (in `answer`) are
checked by `TestBuiltinLeanReferences`; `TestCheck` covers the cheats.

**Running.** `POST /api/evals/run` takes models × sets, a think mode
(`""` = the model's default, `off`) and a token limit per reply (8192 with
thinking, 2048 without). One background job, one question at a time:
`loadAuto` once per model × set, then each question at temperature 0,
seed 1, saved after every answer to
`<data>/evals/results/<digest>/<set>[@nothink].json` (every reply and
thinking, clipped at 40k characters). A stopped run resumes where it
left off; finished pairs are skipped unless `redo`. Between questions the
job waits while a chat reply is being written or a speed test holds
`benchMu`. Models aren't unloaded first: speed doesn't matter here.

**The estimate.** Overall = the mean score over the highlighted sets
(the suite; default every built-in set; click a column header to change
it, remembered per browser). Anchors are models with an overall and a
published Intelligence Index (the same base-model lookup as the Library);
quants of one base model count once, at their average. `evals.FitLine` is
ordinary least squares, index on overall, needing 3 anchors; the page
shows R², the typical miss (residual standard error) and marks estimates
outside the fitted range with ≈. Estimates are never shown as the
published score.

**Lessons from the first runs (2026-10-31).** Qwen3.5 2B with thinking off
already scores 85–95% on the plain math, logic and format sets, so those
can't separate bigger models. Hard math (competition-style, brute-forced
answers) does: 2B 10/22 vs 9B 19/22, thinking off; the 9B also got
22/22 math, 18/18 logic, 14/16 format. Nearly every Hard math miss hit the
2048-token limit, so with thinking off that limit is part of the score.
On the code sets the 2B scored 3/15 (Hard coding) and 8/14 (Equations to
code), thinking off.

## Milestones

1. **M1: usable day one.** Ollama client, Library (list, delete), load and
   eject (from the Library, the chat top bar and the sidebar's "In memory"
   list), streaming chat with saved history, light/dark theme.
2. **M2: Discover.** HF search and filters, quant picker with fit estimates,
   pull with progress, aliasing, model page. *Built.* Not yet: the download
   fallback for repos `hf.co` pulls can't handle, an "Ollama can run it"
   filter.
3. **M3: Presets.** Tune panel, per-chat overrides, save to preset, publish
   to `assay/*`. *Built.* Not yet: drift detection, `config.toml` (flags
   and `HF_TOKEN` for now).
4. **M4: Compare.** Local benchmarks, intelligence score, compare page.
   *Built.* Not yet: card benchmarks and quant loss on Compare, sorting
   Discover by measured speed, the download fallback path.

## Open questions

- **Split GGUFs.** Does `ollama pull hf.co/…` handle multi-part files? If
  not, the fallback path has to merge them or such models are marked
  unsupported.
- **Intelligence score source.** Which index, its API terms, and how to
  match HF repo names to its model names.
- **Chat templates.** Some HF quants may need a template override in the
  preset.
- **Vision models.** How `mmproj` files are handled by the `hf.co` pull.

## Attachments

`POST /api/attachments` (multipart `file`, 25 MB each) classifies a file: png/jpg/webp/gif are `image`; anything `rag.Supported` is `doc`, its text extracted once with `rag.Extract`. Uploads wait in `<chats>/.pending/` (swept after 24 h) and return a token; sending with `attachments:[tokens]` moves them into `<chats>/<chatid>.files/` (`Store.Adopt`), a document's text beside it as `<file>.txt`. `chats.Message.Attachments` records name, kind, mime, size, stored file and `Cut`; deleting the chat deletes the folder. `GET /api/chats/{id}/files/{file}` serves them (strict name pattern, nosniff, sandbox CSP). `Server.withFiles` fills stored messages in for each request: document text into the content (documents share 60% of the context, newest first, cut with a note), images into `Message.Images` for models with the "vision" capability, a note otherwise. Compression and `promptSize` count it. Edit and regenerate keep attachments. Not yet: compare mode and `/v1`, and files of edited-away messages stay until the chat is deleted.

## Loading, memory chip and live GPU view

A load is detached from the request that asked for it (`context.WithoutCancel`), so leaving the page doesn't kill it; `Engine.Unload` cancels a load in progress (`pending`, `errEjected`). `Engine.Loading()` names it, and `/api/loaded` adds it with `loading:true`. The app bar shows one memory chip (`#mem-btn`, VRAM meter + label) that opens `#mem-pop`: live bars from `GET /api/gpu` (`hw.ReadLive`: AMD sysfs `mem_info_vram_*`/`gpu_busy_percent`, else `nvidia-smi`; RAM from `/proc/meminfo`; unknown → 0/-1 and the section hides) and a row per model with eject, a shimmer while loading and "Ejecting…". Sampled every 1 s while open, 5 s otherwise, paused while the tab is hidden.

## Stopped replies and the queue

A reply the user stops is saved with `Stopped`; `toOllama` tells the model where it was cut off and, if it was still thinking, what it had thought. Text said before a tool call that was stopped mid-run is kept. In the chat page, a message sent while a reply streams goes into a queue (button reads "Queue", Esc stops the reply) and is sent when the reply finishes; stopping or an error pauses the queue ("Send next").

## Tune layout, resizing, context meter

**Chat in design D.** The model chip (`#switcher-btn`) shows a state dot (ring when not loaded, `--good` when loaded), the readable name and the quant in muted text; each `<option>` carries `data-pretty` (`modelGroup.Pretty()`, from `prettyModel`) and `data-quant`, the option's text stays the raw name (the identifier, also the row's tooltip). The menu has a state row with Eject (`#eject-btn`); while the model isn't loaded a small `#load-btn` sits beside the chip (phones: in the "⋯" sheet). On this page the global memory chip shows only an icon and its VRAM meter (its label returns while loading), so the loaded model isn't named twice; Tune is a quiet button whose summary lives in its title (`#tune-summary` is hidden). The composer is one card on one line (`.composer-card`, a flex row; `.composer-row` is `display: contents` and its children are placed with `order`): attach and the project chip on the left (just a folder icon with an invisible `<select>` over it until a project is picked), the textarea in the middle growing upward with its text, mic and the round send/queue/stop button on the right. A new reply shows "Thought for N seconds" (measured in the browser; a reloaded chat says "Thought process", as the duration isn't stored). With the model not loaded the numbers strip says "Context auto" (or the known size) and "GPU not loaded" (click loads it); once loaded, the context max follows what `/api/loaded` reports. Chat CSS for this sits just before the shared-controls block in `app.css` (prefixed `.content-chat` where it must beat that block).

The Tune panel (`tune.js`, markup in `chat.html`) is four collapsible sections — Reply style, Sampling (extra samplers nested), Context & memory, Performance & loading (collapsed; cache/flash/threads/MoE nested) — each with a one-line summary; open state in localStorage `tune:sec:<id>`. `resize.js` adds drag/keyboard/double-click-reset handles to the sidebar (180–420px) and the docked Tune panel (300–720px), as `--sidebar-w`/`--tune-w`, stored under `width:sidebar`/`width:tune`. The chat's numbers (`#chat-numbers`, one quiet line under the composer, asked for 2026-10-31; phones leave out GPU to keep it one line) are always shown: speed (`#num-speed`, links to Compare), context (`#ctx-open` opens Tune focused on Context length via `assayTune.open('num_ctx')`; `#ctx-compress` appears past 80% and POSTs the existing `/api/chats/{id}/compress`) and GPU (`#num-gpu`: the loaded model's `sizeVram` of the card's total, from `/api/loaded` + `/api/gpu`; click opens the memory popover). Context is the last reply's prompt + output (seeded from the last `.stats[data-used]`, minus a manual compression's `saved` made after it; a chat with no stats is guessed from its text at ~3.8 chars/token), counted up while streaming, against the length the reply/loaded model reports, else Tune's `contextSize()`, else "auto" (auto-fit models have no number until loaded). Live tok/s is characters streamed over time (smoothed, counted from the first token, off for resumed replies), replaced by the server's figure on `done`. Everything is painted by `paintNumbers` at most once per frame; changed numbers ease in (`.bump`) except while live counting; tabular numerals and min-widths keep the layout still. Earlier causes of a missing meter: it was hidden until `ctxUsed` was non-zero (new chats, chats without stats, until the first reply finished) and was a separate, two-line row under the composer. Under each reply `.stats` shows tok/s, reading tok/s (`Stats.PromptPerSec`), tokens and seconds, with Copy / Retry text links. Streaming follows the bottom only until the user scrolls up by hand; Stop (or Esc) needs a second press within 3 s.

## Resource limits, draft on GPU, vision on CPU

Resource limits (`internal/limits`, `limits.json` in the config dir, `GET/PUT /api/limits`, Settings → Resource limits) are app-wide, read at each load and never part of `loadOpts`, so changing them doesn't restart running models. `Engine.fitTarget` sets `--fit-target` (only while `--fit` is in effect: no hand-set `num_gpu`/`n_cpu_moe`) to `max(1024, free VRAM − (max VRAM − other loaded models' VRAM))` MiB, plus the draft file + 512 MiB when the model's `draft_gpu` is "always" (which also passes `--spec-draft-ngl all`). Free VRAM comes from `hw.ReadLive` (`Engine.Live`, swappable in tests). The limit is approximate (ROCm context/compute memory isn't counted, ~0.2 GB over). Gentle mode adds `--prio -1`, `-ub 256` and half the CPU threads; it can't cap GPU %. Per-model `mmproj_cpu` passes `--no-mmproj-offload`, keeping the vision projector in RAM.

## Background replies

Chat replies run in the background (`internal/server/runs.go`): `POST /api/chat` starts a run — a goroutine on a request-detached context that records its NDJSON events — and the request is only its first follower, so leaving the page doesn't stop the reply. `GET /api/chats/{id}/stream` replays the buffered events then streams live (204 if nothing runs); `POST /api/chats/{id}/stop` cancels it (the partial reply is saved with `Stopped`, the stream ends with a `stopped` event). One run per chat (a second send gets 409). The reply is saved once, then the run leaves the registry. `GET /api/chats/running` and the sidebar `.running` class show a pulsing dot on chats mid-reply; a chat page opened while its chat runs (`data-running`) reattaches. Voice turns are still request-bound. A thinking stream that repeats a long line 8 times is stopped as a loop (`errLooping`). A new chat starts on the model picked last in this browser (`localStorage chat:lastModel`) unless the link names one.

## Phone layout

On phones (max-width 760px) the UI follows the ChatGPT/Claude mobile pattern: the chat bar is menu, a tappable model title and "⋯". The model picker (`#switcher-card`), the "⋯" options (`#more-sheet`), the memory popover (`#mem-pop`) and Tune (`#tune`) are bottom sheets (`.sheet`) coordinated by `window.assaySheets` in `app.js` (`track`/`register`; `html.sheet-open` locks the page and shows the backdrop; drag-down on `.sheet-grab`, tap-outside, Esc and Back close). The composer is a pill with a round send button whose arrow/stop/label follow `#send[data-state]`. All phone rules live in one block headed PHONE at the end of `app.css`; phone-only elements are `display: none` outside it. Layout heights use `dvh`; scrolling boxes use `overscroll-behavior: contain`.

## Multiple GPUs

`hw.Detect` lists every card (`Memory.Cards`: AMD each `/sys/class/drm/card*/device` with `mem_info_vram_total`, iGPU carve-outs under 1 GiB skipped when a real card exists; NVIDIA each `nvidia-smi` line); `Memory.VRAM` is the total and `Memory.GPU` reads "4 GPUs (3x Quadro P2000, GeForce GTX 1050 Ti)" (one card keeps its own name). `hw.ReadLive` sums used/total and reports `gpuBusy` as the busiest card (layer-split cards take turns, so an average would look idle), plus `cards` (name, used, total, busy) when there is more than one; the Memory popover draws one small bar per card under the VRAM total. `hw.Classify` subtracts `ExtraCardOverhead` (512 MiB, each extra card's compute buffer) from the total; a single card is unchanged. llama-server uses all GPUs by default (layer split); the Tune panel (Memory & loading, only with more than one device) adds `gpu_devices` ("GPUs to use": checkboxes, stored as a comma list of `--device` names like CUDA0,CUDA2; all ticked = no value) and `split_mode` ("Split across GPUs": Layers, or Rows = `--split-mode row`); flags are passed only when not default. `Engine.Devices` runs `llama-server --list-devices` once per binary and caches it (nil on failure hides the options; `/api/models/defaults` carries `gpus`); the list is only consulted when a subset is chosen, and names no longer present are dropped. The VRAM limit (`--fit-target`, a per-device margin) spreads the held-back memory over the cards in use; with a subset the free memory still counts every card, so the limit is looser. Fit labels ignore the subset.

Test-only: `ASSAY_FAKE_GPUS="name:totalMiB:usedMiB,..."` makes hw (and `Engine.Devices`, as CUDA0..) report made-up cards so the multi-GPU UI can be checked on one-GPU machines.

## First-run setup

`/setup` (`internal/server/setup.go`, `web/static/setup.js`) is a guided,
skippable four-step page: Computer, Engine, First model, Done. Principle:
every install is one explicit click that states what, from where, how big and
where it goes; nothing downloads on its own.

- **When it shows.** `/chat` (without a chat id) redirects to `/setup` while
  `needsSetup(done, haveEngine, models)` holds: setup isn't done and there is
  no llama-server resolved or no installed model. Only with the llama.cpp
  engine (a nil manager never redirects) and never when listing models
  fails. "Done" is the file `<config>/setup.json`, written by "Start chatting",
  "Skip setup entirely" and the Discover link; Settings → Models → "Run setup
  again" just opens `/setup`, which always renders. `POST /api/setup/done`
  takes `{done:false}` to clear it.
- **Computer.** `GET /api/setup` returns OS, each GPU with its VRAM, RAM, a
  plain-words sentence from `reach` (largest model size that runs wholly on
  the GPU, and at all, by `hw.Classify` on a ~4.8 bits/weight quant), the
  engine state (path, source, `Manager.Probe` version and devices, install
  folder, license, models folder) and installed model names.
- **Engine.** If one is found it is shown with its probe and "Keep this one".
  Otherwise the builds from `/api/llamacpp/check` are radio rows with the
  recommended one preselected, and rows for what (tag, build, size), from
  (GitHub release page and file names), goes to (`<data>/llama.cpp-release`)
  and license (MIT). The button names the build and size; progress uses the
  existing install job (`/api/llamacpp`, reloading the page resumes it). "Use
  a llama-server I already have" is a path input with Browse… (`assayPick`)
  wired to `PUT /api/llamacpp/custom`; the verify result is the probe.
- **First model.** `GET /api/setup/models` (cached ten minutes) checks
  `firstModels`, a short list of well-known small instruct models (Qwen3.5
  0.8B to 9B, Gemma 4 E2B/E4B/12B QAT, Unsloth repos), against Hugging Face's
  file lists (read only): repos that are gone or have no ~4-bit quant
  (`pickQuant`: UD-Q4_K_XL, Q4_K_M, Q4_0, any 4-bit) are dropped. `rankSuggestions`
  keeps up to four from what fits comfortably (GPU, or RAM without a GPU): the
  most capable, the best of the other family, a small quick one, each with a
  "why". Each row shows size, fit, the hf.co repo link and the models folder;
  its button posts to `/api/pulls`, the same machinery as Discover.
- **Done.** Summary of the engine and installed models, and "Start chatting".

## Path pickers

Every input that wants a path on the server (llama-server, models folder, LoRA adapter, "Add a folder…" in Library) has a "Browse…" button: `<button data-browse="input-id" data-kind="file|folder" data-title data-filter>` (or `assayBrowseButton(input, opts)` for script-built rows). A choice fills the input and fires `input` and `picked` on it; the page decides what follows (models folder runs Check, the Library folder adds). `window.assayPick({kind, title, start, filter})` in `app.js` resolves to a path or null. From the machine running Assay and with a dialog tool present (`GET /api/pick` → `{native}`), it asks `POST /api/pick`, which runs `internal/pick`: zenity, kdialog or yad on Linux (needs DISPLAY or WAYLAND_DISPLAY), osascript on macOS, PowerShell with System.Windows.Forms (-STA, hidden) on Windows. Exit 1 or empty output is "cancelled"; one dialog at a time (second gets 409); no timeout, but the process is killed when the request ends. It is refused (403) unless `RemoteAddr` is loopback, so a phone never pops a dialog on the server's screen. Otherwise (remote, or no tool) an in-app browser opens over `GET /api/fs?path=&kind=&filter=&hidden=`: folders first then files (kind=file, glob filter), natural case-insensitive order, dotfiles hidden unless asked, a missing path shows its nearest existing parent, unreadable folders return `error` inline; it is a centred dialog on desktop and a `.sheet` on the phone. Test-only: `ASSAY_PICK_TOOLS` is an allow list of dialog tool names; set empty it disables native dialogs so the in-app browser can be checked. Go tests use a fake `zenity` script on PATH and never open a real dialog.

## Start on boot

`assay service install [flags]`, `service uninstall` and `service status` (and Settings → This machine → "Start when the computer starts") manage `internal/service`. The subcommands are handled in `cmd/assay/service.go` before flag parsing, so existing flags are untouched; `version` and `--version` print the version. The only place the app's name lives is `service.Name`.

- **Linux**: a systemd user unit `~/.config/systemd/user/<name>.service`: `ExecStart` = the running binary (symlinks resolved) plus the given flags (quoted, `%` and `$` doubled), `Restart=on-failure`, `RestartSec=5`, `WantedBy=default.target`. Then `systemctl --user daemon-reload` and `enable --now`. `Environment=` lines carry PATH and the GPU/engine variables (`HSA_OVERRIDE_GFX_VERSION`, `*_VISIBLE_DEVICES`, `OLLAMA_MODELS`, `ASSAY_LLAMA_SERVER`, `ASSAY_VOICE`) only when set at install time (the service manager's own environment is nearly empty). Starting before login needs lingering; install prints `loginctl enable-linger $USER` when `loginctl show-user` says it's off, and never runs it.
- **macOS**: `~/Library/LaunchAgents/local.<name>.plist` (`RunAtLoad`, `KeepAlive`, same environment) loaded with `launchctl bootstrap gui/<uid>` (an older one is `bootout`ed first). Uninstall removes the plist and boots it out.
- **Windows**: a value under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` via `reg.exe` (add/query/delete; no new dependency), `"<exe>" <flags> -open=false`. No environment is carried and "running" isn't reported (no service manager); it starts at the next sign-in.
- **Settings page**: `GET/PUT /api/service` (`{enabled}`) is answered only for loopback requests (`fromLoopback`); remote requests get an empty status on GET and 403 on PUT, and the row is not rendered. The web switch installs with the flags this instance was started with (`os.Args[1:]`) and neither starts nor stops anything (this copy already holds the port); the CLI install does start it (`--now`).
- **Tests** inject `Manager.Run` (a recording `Runner`), `Home`, `Env`, `GOOS`, so no test touches the real systemctl/launchctl/reg or `~/.config`. Test-only: `ASSAY_SERVICE_DRYRUN=1` swaps in an in-memory `service.Fake` (CLI and web), so the Settings switch can be exercised without installing anything.

## Releases

Version: `var version = "dev"` in `cmd/assay`, set with `-ldflags "-X main.version=v1.2.3"`. Shown in Settings → This machine and by `assay --version` / `assay version`.

`scripts/release.sh [version]` (default `git describe`) builds with `CGO_ENABLED=0 -trimpath` for linux/amd64, linux/arm64, windows/amd64, darwin/amd64, darwin/arm64 into `dist/release/` (gitignored): a `.zip` (Windows, README with CRLF) or `.tar.gz` per platform holding the binary, `README.txt` (from `tools/release/README-*.txt`) and `LICENSE` if the repo has one, plus `SHA256SUMS`. `.github/workflows/release.yml` runs `go vet`, `go test`, the script and `gh release create` on a pushed `v*` tag. There is no Windows installer; the zip holds the exe.

Update notice (`internal/update`): `update.Repo` ("owner/name") is empty, which hides the feature and makes no network call. When set, `GET /api/update` asks `api.github.com/repos/<repo>/releases/latest` at most once a day (a failed ask counts too), caches the answer in `<config>/update.json`, and says `available` when the tag is a newer plain `vX.Y.Z` than this build (`dev` and git-describe versions are never older). Settings shows "Version X is available" linking the release page, and a "Check for updates" switch (`PUT /api/update {check}`, default on). Nothing is downloaded or installed.
