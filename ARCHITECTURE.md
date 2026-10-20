# ollamui architecture

A lightweight web UI on top of Ollama for finding, downloading, tuning and
chatting with local models. Ollama stays the only model runtime, so anything
else pointed at it (Zed, scripts, other clients) sees the same models.

## Goals

- **Discover and download** GGUF quants from Hugging Face (Unsloth, bartowski,
  …) with a quant picker that shows whether each one fits your hardware.
- **Tune without friction**: per-chat settings, saved as presets, without
  hand-writing Modelfiles.
- **Chat** with streaming output, thinking blocks and speed stats.
- **Compare** models by size, speed on your machine and published benchmarks.
- **Talk**: a hands-free voice mode that keeps talking while a model thinks.
- One static Go binary. No Docker, no database server, no Node toolchain.

## Non-goals

- Training or fine-tuning (Unsloth's other half).
- Running models with any engine other than Ollama.
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
                     ├── internal/ollama   typed API client, NDJSON streaming, server env
                     ├── internal/hf       HF search, file tree, quant parsing, fit estimate   (M2)
                     ├── internal/presets  TOML presets + sync to ollamui/*                   (M3)
                     ├── internal/chats    one JSON file per chat
                     ├── internal/scores   intelligence index (Artificial Analysis via OpenRouter)
                     ├── internal/voice    client for an OpenAI-style audio API (STT + TTS)
                     ├── internal/server   pages, JSON API, streaming, host/CSRF guards
                     └── web/              templates, CSS, small vanilla JS (embedded)
tools/voice/voice_server.py   the speech service: faster-whisper + Kokoro, run with uv
```

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

### Downloads (M2)

The main path is `POST /api/pull` with `hf.co/{repo}:{quant}`. Ollama handles
dedupe, resume and registration, and its progress stream is passed to the
browser. After a pull, ollamui offers a short alias (`/api/copy`). A fallback
path (download the GGUF ourselves, `/api/blobs` + `/api/create`) covers repos
the `hf.co` pull can't handle.

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

### Voice mode

The goal is a spoken conversation that never leaves you in silence, even
while a thinking model thinks for half a minute.

```
mic ─► browser: energy VAD, 16 kHz WAV ─► /api/voice/transcribe ─► whisper
                                           (live partials every ~0.9 s while you talk)
transcript ─► /api/chat {voice: true} ─► main model on the GPU (thinks, then answers)
                                    └─► talker: small model on the CPU, one short line
                                        at a time, from the thinking so far ("say" events)
"say" lines + answer sentences ─► /api/voice/speak ─► Kokoro ─► browser plays in order
```

- **Speech service.** A separate local process with OpenAI's audio API
  (`/v1/audio/transcriptions`, `/v1/audio/speech`). `tools/voice/voice_server.py`
  is the bundled one (faster-whisper `small.en` + Kokoro int8, both on the
  CPU, so the GPU stays free; Whisper, Kokoro and the talker each get their
  own share of the cores, since on a shared pool transcription slowed from
  1 s to 7.5 s mid-reply); `-voice URL` points ollamui at it or at any
  compatible server (speaches, …). It's the one exception to "single
  binary": optional, and Ollama is already a separate service too.
- **Browser.** An AudioWorklet feeds 20 ms frames to a speech detector with
  an adaptive noise floor and 300 ms of pre-roll. A frame counts as voice
  when it's loud without being a click (peak/RMS under 7), and an utterance
  starts only when 11 of the last 15 frames were voice: typing is short
  bursts with gaps, speech is sustained. Talking over the reply first only
  ducks it to 25%; it stops once a transcript shows two or more real words
  that aren't mostly what the assistant just said (echo). A false alarm
  restores the volume. Replies are cut into sentences as
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
- **Prose for the ear.** Spoken replies get one system line asking for plain
  prose (no markdown, LaTeX, code, tables or lists). On Gemma 4 12B that
  took markup from 144 marks to 0 at no extra thinking time. Keep it one
  line: a longer list of style rules made Qwen3.5-9B think 95 s instead of
  8 s over "what is 2 plus 2".
- **Numbers.** espeak-based engines read "1.67" as two sentences, so
  `voice.SpeechText` spells out decimals ("1 point 6 7"), money, percents,
  multipliers, ranges and a few abbreviations before synthesis.
- What was said aloud is saved with the reply (`spoken`) and shown above it.

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
~/.local/share/ollamui/
  chats/<id>.json        one conversation per file
  scores.json            cached intelligence scores, refreshed daily
~/.cache/ollamui-voice/  Kokoro model files (the speech service's)
  bench/<model>.json     local benchmark results                     (M4)
```

Presets are plain TOML, so they diff, hand-edit and sync well.
Full-text search across chats is the only feature that might justify SQLite
later (embedded, still not a service).

## UI

Design reference: direction **F** on the design canvas
(https://claude.ai/artifact/Ad3hWUHs96WFjh3TDHnjoG).

- Compact, mostly sans-serif (IBM Plex Sans). Monospace only for quant names
  and code. Light and dark themes.
- Left sidebar: Chat, Library, Discover, Compare, Presets, Settings, recent
  chats, theme toggle. On phones it's a drawer (hamburger, swipe right).
- App bar: loaded models as pills (VRAM, context; "CPU" for the talker),
  downloads, page actions.
- **Chat**: centered column. The Tune panel is closed by default and slides
  in from the right. A button in the top bar summarizes the current settings
  and how many differ from the preset. A mic button next to Send starts
  voice mode; a status line under the composer shows listening / thinking /
  speaking.
- **Library**: installed models, size, quant, what's loaded and its VRAM,
  delete.
- **Discover**: filters (total params range, dense/MoE, architecture,
  capabilities, fits on GPU, Ollama can run it, author), sort (intelligence,
  speed on my GPU, size, newest, downloads). Results as rows, each with an
  intelligence bar, the best quant that fits and an add-to-compare checkbox.
- **Model page**: intelligence score and rank, quant picker rows (size,
  memory bar, quality, estimated speed), model-card benchmarks, specs,
  similar-size models.
- **Compare**: side by side: fit on your machine, speed measured locally,
  card benchmarks, quant loss.

## Benchmarks (M4)

| Data | Source | Trust |
|---|---|---|
| Speed (prompt tok/s, gen tok/s, time to first token) | measured locally: a short fixed prompt through `/api/generate`, timings from the response | high, your hardware |
| Speed estimate before download | size and memory bandwidth heuristic | rough |
| Intelligence score | external index where it covers the model, otherwise none | medium, see open questions |
| Card benchmarks | parsed from the HF README | low, authors' own setups |
| Quant loss (KLD) | the uploader's published numbers, when present | medium |

Every number in the UI is labeled with where it came from.

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
4. **M4: Compare.** Local benchmarks, intelligence score, compare page,
   download fallback path.

## Open questions

- **Split GGUFs.** Does `ollama pull hf.co/…` handle multi-part files? If
  not, the fallback path has to merge them or such models are marked
  unsupported.
- **Intelligence score source.** Which index, its API terms, and how to
  match HF repo names to its model names.
- **Chat templates.** Some HF quants may need a template override in the
  preset.
- **Vision models.** How `mmproj` files are handled by the `hf.co` pull.
