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
                     │
                     ├── internal/ollama   typed API client, NDJSON streaming
                     ├── internal/hf       HF search, file tree, quant parsing, fit estimate   (M2)
                     ├── internal/presets  TOML presets + sync to ollamui/*                   (M3)
                     ├── internal/chats    one JSON file per chat
                     ├── internal/server   pages, JSON API, streaming, host/CSRF guards
                     └── web/              templates, CSS, small vanilla JS (embedded)
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
- Left sidebar: Chat, Library, Discover, Compare, Presets, recent chats,
  VRAM in use, theme toggle.
- **Chat**: centered column. The Tune panel is closed by default and slides
  in from the right. A button in the top bar summarizes the current settings
  and how many differ from the preset.
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
