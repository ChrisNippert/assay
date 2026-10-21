# ollamui architecture

A lightweight web UI for finding, downloading, tuning and chatting with
local models. It started on top of Ollama and is moving to running models
itself with llama.cpp's `llama-server` (see "Engine" below and BACKLOG.md);
`-engine ollama` keeps the old path while that lands.

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
- Writing an inference engine. llama.cpp does that; ollamui starts,
  sizes and stops its servers.
- Multi-user accounts. It's a single-user tool, optionally reachable from
  your other devices.

## The core idea: presets are real Ollama models

Ollama stores weights by content hash. `POST /api/create` with
`from: <base>` plus `parameters` and `system` writes a tiny new manifest and
reuses the base model's weight blobs, so the extra disk cost is negligible.

So a preset is edited like an Unsloth config (sliders, save), and when
marked **published** ollamui upserts it as `ollamui/<preset-name>`. Zed and
other clients then see it as an ordinary model with those settings baked in.

- The preset file is the source of truth. Ollama holds a derived copy.
- ollamui only creates, updates or deletes models under `ollamui/`. It never
  modifies models it didn't create.
- On startup, published presets are compared with `/api/show` and any drift
  is flagged so you can re-sync.

Per-chat tweaks are sent as `options` on each `/api/chat` request and create
nothing in Ollama until you choose "Save to preset".

## Components

```
browser ──HTTP──► ollamui (Go) ──HTTP──► Ollama :11434 ◄── Zed, etc.
                     │        └──HTTP──► speech service :8485 (optional, voice mode)
                     │
                     ├── internal/engine   llama-server per model, model store, Ollama import
                     ├── internal/gguf     GGUF header reader (arch, context, size, template)
                     ├── internal/ollama   typed API client, NDJSON streaming, server env
                     ├── internal/hf       HF search, file tree, quant parsing, fit estimate   (M2)
                     ├── internal/presets  TOML presets + sync to ollamui/*                   (M3)
                     ├── internal/chats    one JSON file per chat
                     ├── internal/scores   intelligence index (Artificial Analysis via OpenRouter)
                     ├── internal/bench    saved speed tests, one JSON file per model digest
                     ├── internal/voice    client for an OpenAI-style audio API (STT + TTS)
                     ├── internal/server   pages, JSON API, streaming, host/CSRF guards
                     └── web/              templates, CSS, small vanilla JS (embedded)
tools/voice/voice_server.py   the speech service: faster-whisper + Kokoro (embedded, run with uv on demand)
```

### Engine (internal/engine)

ollamui runs one `llama-server` per loaded model, the way Ollama does
internally, and answers in the `internal/ollama` types so the rest of the
code doesn't care which engine runs (`server.Backend`).

- **Store**: `<data>/models/` holds `blobs/` (GGUF files) and one JSON
  manifest per model (name, aliases, files, projector, default params).
  Ollama's models are imported by hard link (symlink across disks), so
  nothing is copied and Ollama keeps working; re-imported every 10 s while
  pulls still go through Ollama.
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

### internal/ollama

A thin typed client for `/api/version`, `tags`, `ps`, `show`, `delete`,
`copy`, `pull`, `create` and `chat`. Streaming endpoints decode NDJSON one
object at a time and turn mid-stream `{"error": …}` lines into Go errors.
Cancelling the request context aborts generation or download in Ollama.

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
names (the Ollama library) still pull through Ollama and are imported.

### Load and eject

Ollama has no explicit load endpoint. A `/api/generate` request with no
prompt loads a model, and `keep_alive: 0` unloads it. Loading from the UI
pins the model (`keep_alive: -1`) until you eject it, the way Unsloth and
LM Studio behave. Every chat request resets Ollama's keep-alive timer, so
ollamui sends `keep_alive: -1` with chats to pinned models to keep them
pinned. A model counts as pinned when `/api/ps` reports an expiry more than
a year away. The sidebar shows each loaded model's VRAM, how much of it is
on the GPU, and whether it stays until ejected or auto-unloads.

### Server settings

Ollama's tuning that matters most (K/V cache type, flash attention,
parallel requests) is server-wide environment only. ollamui reads it from
the running `ollama serve` process (`/proc/<pid>/environ`, matched by port;
readable when it runs as the same user), falling back to the systemd
unit's configuration (`systemctl show`). `/settings` shows it and writes a
`sudo` drop-in snippet for changes; ollamui never runs sudo. Auto context
scales its cache estimate by the K/V type and by `OLLAMA_NUM_PARALLEL`,
since Ollama gives every parallel slot its own cache.

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
  1 s to 7.5 s mid-reply); `-voice URL` points ollamui at it or at any
  compatible server (speaches, …).
- **Started on demand.** The script is embedded in the ollamui binary. When
  the mic is pressed and nothing answers on the speech URL, ollamui writes it
  to `~/.cache/ollamui/` and runs it with `uv run --script` in its own
  process group; it stops it after `-voice-idle` (10 min) without voice use
  and when ollamui exits. The first start installs the Python packages
  (about a minute); later starts take seconds. A service someone else
  started is used and left alone. `-voice-launch=false` turns this off,
  `-voice ""` turns voice off. Nothing about voice is needed for the rest of
  the app.
- **Downloads only after a yes.** A fresh install has no voice files. Until
  the service has been downloaded (a marker in `~/.cache/ollamui/`, or the
  Kokoro model already present), `/api/voice` reports `needsDownload` and
  won't start it; the voice bar asks "Download / Not now" and only
  `?install=1` starts the first run (~800 MB: uv's environment, Kokoro,
  whisper small.en). Settings → Voice shows the space used (hard links
  counted once) and **Remove** deletes it all: the marker and script,
  `~/.cache/ollamui-voice`, the faster-whisper models in Hugging Face's cache
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
  thinking is switched off or ollamui has seen it answer without thinking.
- **The talker.** While the main model thinks, a second model writes one
  short first-person line at a time ("I'm checking how the angle of the sun
  changes things"): a lead-in at once, then a new line whenever the speech
  is about to run out and there's enough new thinking, with growing gaps on
  long thinks. It runs on the CPU (`num_gpu: 0`, 4k context, 4 threads): no VRAM, no GPU
  contention, and no need for Ollama to serve two requests on one model,
  which hybrid models like Qwen3.5 can't. ollamui picks the largest
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

Storage, per project (`~/.local/share/ollamui/projects/<id>/`):
`project.json` (settings + file list with status), `files/` (the originals)
and `index/` (`index.json` with the chunks + `vectors.f32`, normalised
float32). No database: search is a linear cosine scan (`internal/index`),
fine for tens of thousands of chunks. Deleting a project keeps its chats.

**Indexing** (`runIndexer`, one file at a time in the background): extract
text (`internal/rag`: text and code, HTML, PDF via `pdftotext`, DOCX, ODT;
20 MB max), split into ~1,500-character chunks with 200 of overlap
(headings and PDF pages tracked, code fences kept whole), embed 16 at a time
through Ollama's `/api/embed` with "file · heading" prefixed, then store.
Progress shows on the project page. Files wait (not fail) until an
embedding model is chosen; restarts pick up unfinished files.

**Embedding model** (Settings → Knowledge, `knowledge.json`): the user picks
and downloads one (suggestions: qwen3-embedding:0.6b, embeddinggemma,
nomic-embed-text, …; any Ollama embedding model works). One index holds one
model's vectors; switching models re-indexes every project.

**Retrieval** (in `prepareTurn`): for a chat in a project with indexed
files, the latest user message (plus the previous one when it's short) is
embedded, the top-K chunks (default 6, cosine ≥ 0.25) go into the system
prompt after the project's instructions, numbered for citation. They're
streamed to the page (`sources` event) and saved on the reply, shown as a
folded "N passages from the project's files" list.

### internal/chats

One JSON file per conversation. Writes go to a temp file and are renamed into
place, so a crash can't leave a half-written chat. Chat IDs are validated
against a strict pattern before they touch the filesystem.

### internal/server

- Server-rendered pages (`html/template`) plus a small vanilla JS file for
  streaming chat and markdown.
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
  - Ollama itself can stay on localhost. Remote browsers only talk to
    ollamui.

## Storage

No database. Plain files under XDG directories:

```
~/.config/ollamui/
  config.toml            listen address, Ollama URL, HF token        (M2)
  presets/<name>.toml    one preset per file                         (M3)
  voice.json             voice mode settings (Settings → Voice)
  knowledge.json         RAG settings: embedding model, excerpts per answer
~/.local/share/ollamui/
  chats/<id>.json        one conversation per file
  scores.json            cached intelligence scores, refreshed daily
  projects/<id>/         project.json, files/, index/ (RAG)
  bench/<digest>.json    speed tests from the Compare page
~/.cache/ollamui-voice/  Kokoro model files (the speech service's)
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
  defaults, models with their own settings and a "Use defaults" button;
  the Ollama server's variables with `-engine ollama`), Knowledge, Voice,
  This machine. A mic button next to Send dictates
  into the box; the bar under the composer shows listening / reading aloud
  and holds the Spoken style and Read aloud toggles.
- **Library**: installed models, size, quant, what's loaded and its VRAM,
  delete.
- **Discover**: filters (total params range, dense/MoE, architecture,
  capabilities, fits on GPU, Ollama can run it, author), sort (intelligence,
  speed on my GPU, size, newest, downloads). Results as rows, each with an
  intelligence bar, the best quant that fits and an add-to-compare checkbox.
- **Model page**: intelligence score and rank, quant picker rows (size,
  memory bar, quality, estimated speed), model-card benchmarks, specs,
  similar-size models.
- **Compare** (`/compare?m=a&m=b`, up to 4 installed models): specs from
  Ollama (parameters, quant + rule-of-thumb quality, disk, architecture,
  longest context, capabilities), the context that fits on the GPU, the
  intelligence score, and speed measured here. Each section names its
  source; the best value in a row is highlighted. Not yet: card benchmarks,
  quant loss (KLD).

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

## Milestones

1. **M1: usable day one.** Ollama client, Library (list, delete), load and
   eject (from the Library, the chat top bar and the sidebar's "In memory"
   list), streaming chat with saved history, light/dark theme.
2. **M2: Discover.** HF search and filters, quant picker with fit estimates,
   pull with progress, aliasing, model page. *Built.* Not yet: the download
   fallback for repos `hf.co` pulls can't handle, an "Ollama can run it"
   filter.
3. **M3: Presets.** Tune panel, per-chat overrides, save to preset, publish
   to `ollamui/*`. *Built.* Not yet: drift detection, `config.toml` (flags
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
