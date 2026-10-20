# Backlog

Ordered by what to do next. Each item says why it matters and roughly how
big it is (S = an hour or two, M = a session, L = several sessions).

## Next up

### 1. UI polish pass (S–M)

Small fixes that make every screen nicer to use.

- **Send button misaligned with the message box.** The composer bottom-aligns
  a 36px button against a taller textarea. Make them the same height and
  align them properly.
- **Too dense.** Bump base text from 13px to 14px, loosen row padding and
  gaps, give sections more breathing room. Keep it compact, just less
  cramped. Define a small spacing scale in CSS so this stays consistent.
- **Make things clickable.** Anything that looks like information about a
  thing should take you to that thing:
  - Sidebar **Downloads** entries → that model's page; finished ones → chat
    with it.
  - Sidebar **In memory** entries → chat with that model.
  - **Library** model names → their Discover page (for `hf.co/…` models)
    or a model detail view.
  - **Tags** on Discover rows (architecture, MoE/dense, vision) → apply that
    filter.
  - **Load state** in the chat top bar → toggles load/eject (today only the
    button next to it does).
  - **Chat stats line** → opens Tune.
  - **Presets page** base model → its Library/Discover entry.
  - Do a full pass over every screen for text that should be a link or button.

### 2. Discover rework (M)

Two problems with one root cause: Discover shows a wide, flat list of up to
100 rows, and filters only apply to that top-100 slice.

- **Fetch the whole result set, then sort and filter locally.** HF search
  supports cursor paging (`Link: rel="next"`). Page through all matches for
  the search text and uploader (capped at ~1000), then do all sorting and
  filtering in the browser. Changing the sort then only reorders. It never
  adds or removes results, and filters see everything, not a slice.
  Cache results per query for a few minutes so filter tweaks are instant.
- **Narrower, easier to scan.** Cap the results column width. Show 20 rows
  with "Show more" instead of everything at once. Simplify each row to a
  name, one line of key facts (size · type · arch) and the fit badge. Move
  popularity and dates to the model page.
- **Group model families** (optional, after the above). Collapse
  `Qwen3-8B`, `Qwen3-8B-Instruct-2507`, etc. into one expandable row.

### 3. Intelligence score and benchmarks (L, start with research)

The M4 feature from the design (Compare page, score on Discover rows and
model pages).

- **Research spike first (S):** pick the score's source. Candidates: an
  external index (e.g. Artificial Analysis Intelligence Index; check API
  access, terms, and how its model names map to HF repos) vs averaging
  model-card benchmarks (works for any model, but every author tests
  differently). Decide what's shown when a model has no score.
- **Local speed bench (M):** run a fixed prompt through Ollama on installed
  models; record prompt tok/s, generation tok/s, time to first token. Store
  under `~/.local/share/ollamui/bench/`. This is the most trustworthy number
  we can show.
- **Score on Discover + model page (M):** sort by intelligence, show it on
  rows, rank within the current filter ("#2 of 14 in 5–12B").
- **Compare page (M):** up to 4 models side by side: fit, measured speed,
  score, card benchmarks, quant quality. Every number labeled with its
  source.

## Later

- **Verify a real download end to end.** Not yet done on a real model (HF
  was unreachable during testing). Includes checking how split quants and
  vision (`mmproj`) models behave with `ollama pull hf.co/…`. (S)
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

- M1: Ollama client, Library, load/eject, streaming chat, light/dark theme.
- M2: HF search with filters, model page with quant picker and fit
  estimates, background downloads with progress and short names.
- M3: Tune panel, presets, publish presets to Ollama as `ollamui/<name>`.
