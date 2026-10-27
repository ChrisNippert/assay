// Tune panel, in two parts.
// "This chat": thinking, tools, temperature, reply length, system prompt and (hidden
// until opened) the rest of sampling. These are the chat's own, layered over
// the preset (if any), the model's settings and its recommended defaults.
// The model part: context, GPU layers, how long it stays loaded and (hidden)
// memory options, saved for the model straight away, so every chat with
// it uses them.
// Chat.js reads the chat's settings via window.ollamuiTune when sending.
(() => {
  const chatEl = document.getElementById('chat');
  if (!chatEl) return;
  const { errorText, el } = window.ollamui;
  const $ = (id) => document.getElementById(id);

  const select = $('model');
  const panel = $('tune');
  const openBtn = $('tune-btn');
  const summary = $('tune-summary');
  const sourceEl = $('tune-source');
  const systemEl = $('tune-system');
  const saveBtn = $('tune-save');
  const saveNewBtn = $('tune-save-new');
  const resetBtn = $('tune-reset');
  const errorEl = $('tune-error');
  const modelStatus = $('tune-model-status');

  let presets = JSON.parse(chatEl.dataset.presets || 'null') || [];
  const initial = JSON.parse(chatEl.dataset.tune || '{}');
  // This chat's own changes. Anything not in here is inherited.
  const state = { options: { ...(initial.options || {}) }, system: initial.system || '', think: initial.think ?? null, compress: initial.compress ?? null, tools: initial.tools ?? null };
  // The chat's summary of its oldest messages (see server compress.go), and
  // how many messages it has on the page.
  const chatState = { compression: initial.compression || null, messages: initial.messages || 0 };
  const KEEP_LATEST = 4; // messages a summary never covers
  let defaults = { options: {}, system: '', contextLength: 0, thinking: false, tools: false };
  // The model's saved settings (llama.cpp only).
  let saved = {};
  const perModel = () => !!defaults.engine;

  const CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
  const gb = (n) => (n >= 1e9 ? (n / 1e9).toFixed(1) + ' GB' : Math.round(n / 1e6) + ' MB');
  const kib = (n) => (n >= 1024 ? Math.round(n / 1024) + 'k' : String(n));

  // Each setting is one row. kind: combo (context dropdown), think, choice
  // (dropdown of fixed values), spin (number with − +), scale (slider plus
  // number). scope 'model' rows belong to the model part.
  const CHAT = [
    { key: 'think', label: 'Thinking', kind: 'think' },
    { key: 'tools', label: 'Tools', kind: 'tools' },
    { key: 'temperature', label: 'Temperature', kind: 'scale', min: 0, max: 2, step: 0.05, short: 'temp',
      help: 'Randomness: low is focused and repeatable, high is more varied' },
    { key: 'num_predict', label: 'Max reply length', kind: 'spin', min: -1, max: 262144, step: 256, int: true,
      help: 'Most tokens in one reply; empty means no limit' },
  ];
  const SAMPLING = [
    { key: 'top_p', label: 'Top P', kind: 'scale', min: 0, max: 1, step: 0.01, short: 'top-p',
      help: 'Picks only among the likeliest words whose chances add up to this share; lower is safer' },
    { key: 'top_k', label: 'Top K', kind: 'scale', min: 0, max: 200, step: 1, int: true, short: 'top-k',
      help: 'Picks only among this many likeliest words; 0 means no limit' },
    { key: 'min_p', label: 'Min P', kind: 'scale', min: 0, max: 1, step: 0.01, short: 'min-p',
      help: 'Skips words far less likely than the best one; higher is safer' },
    { key: 'repeat_penalty', label: 'Repeat penalty', kind: 'scale', min: 0.8, max: 2, step: 0.01, short: 'repeat',
      help: 'Above 1 makes words used recently less likely; 1 is off' },
    { key: 'presence_penalty', label: 'Presence penalty', kind: 'scale', min: 0, max: 2, step: 0.05,
      help: 'Flat penalty on any word already used, so it moves to new topics' },
    { key: 'frequency_penalty', label: 'Frequency penalty', kind: 'scale', min: 0, max: 2, step: 0.05,
      help: 'Penalty that grows each time a word is reused' },
    { key: 'repeat_last_n', label: 'Repeat penalty window', kind: 'spin', min: -1, max: 4096, step: 16, int: true,
      help: 'How many recent tokens the penalties look back over; -1 is the whole chat' },
    { key: 'seed', label: 'Seed', kind: 'spin', min: 0, max: 2147483647, step: 1, int: true, placeholder: 'Random',
      help: 'Fixed number for repeatable replies; empty picks a new one each time' },
  ];
  const MODEL = [
    { key: 'num_ctx', label: 'Context length', kind: 'combo', int: true, scope: 'model',
      help: 'Tokens the model can hold in mind at once; the prompt, thinking and reply all share it' },
    { key: 'num_gpu', label: 'GPU layers', kind: 'scale', min: 0, max: 99, step: 1, int: true, placeholder: 'All', scope: 'model',
      help: 'How many of the model’s layers run on the GPU; the rest run on the CPU, slower. Setting this by hand turns automatic fitting off, so the VRAM limit in Settings can’t apply' },
    { key: 'keep_alive', label: 'Keep in memory', kind: 'choice', scope: 'model', engine: true,
      help: 'How long the model stays loaded after the last message',
      choices: [['5m', '5 minutes'], ['30m', '30 minutes'], ['1h', '1 hour'], ['4h', '4 hours'], ['-1', 'Until unloaded']] },
    // What drafts for the model. Auto: its MTP head if it has one. Measured
    // here, a small separate draft model was slower than none, so Auto
    // leaves those to the Draft model row.
    { key: 'speedup', label: 'Speculative decoding', kind: 'choice', scope: 'model', engine: true,
      none: () => (defaults.speed?.mtp ? 'Auto (MTP head)' : 'Auto (none for this model)'),
      help: 'A cheap guesser drafts tokens and the model checks them: same reply, faster',
      dyn: () => [['off', 'Off'], ...(defaults.speed?.mtp ? [['mtp', 'MTP head']] : [])] },
    { key: 'draft_max', label: 'MTP head: tokens ahead', kind: 'spin', min: 1, max: 64, step: 1, int: true, scope: 'model',
      placeholder: 'Auto', help: 'Tokens the MTP head (or draft model) predicts per step; llama.cpp uses 3. Higher isn’t always faster',
      off: () => {
        const sp = (state.options.speedup ?? saved.speedup) || 'auto';
        if (sp === 'off') return 'Speculative decoding is off';
        if (!speedPlan().some((p) => p === 'MTP head' || p === 'draft model')) return mtpGet.size && mtpGet.name === defaults.speed?.name ? 'This model has no MTP head. Download it above' : 'This model has no MTP head';
        return '';
      } },
    // The choices come from the server: installed models with the same
    // vocabulary, and downloaded drafts made for this model.
    { key: 'draft_model', label: 'Draft model', kind: 'choice', scope: 'model', engine: true, none: 'Off',
      help: 'A separate small model that drafts tokens, used instead of the MTP head',
      dyn: () => (defaults.drafts || []).map((d) => [d.name, `${d.name.split('/').pop().replace('-GGUF', '').replace(':', ' ')} (${d.kind ? d.kind + ', ' : ''}${gb(d.size)})`]),
      empty: 'No installed model is a match for drafting' },
    { key: 'spec_ngram', label: 'N-gram guessing', kind: 'choice', scope: 'model', engine: true,
      none: () => `Auto (${speedPlan().includes('n-gram guessing') ? 'on' : 'off'})`,
      help: 'Drafts by copying runs of text already in the chat; great for code edits, costs nothing',
      choices: [['on', 'On'], ['off', 'Off']] },
    { key: 'spec_ngram_max', label: 'N-gram: tokens ahead', kind: 'spin', min: 1, max: 128, step: 1, int: true, scope: 'model',
      placeholder: 'Auto', help: 'Most tokens copied from the chat per step; llama.cpp allows up to 64',
      off: () => (speedPlan().includes('n-gram guessing') ? '' : 'N-gram guessing is off') },
    { key: 'draft_gpu', label: 'Draft on GPU', kind: 'choice', scope: 'model', engine: true, none: 'Auto',
      help: 'Keeps the whole draft (MTP head or draft model) on the GPU and puts more of the main model in RAM; best when the main model doesn’t fit anyway',
      choices: [['always', 'Always']],
      off: () => (speedPlan().some((p) => p === 'MTP head' || p === 'draft model') ? '' : 'No MTP head or draft model is in use') },
    { key: 'vision', label: 'Image & audio input', kind: 'choice', scope: 'model', engine: true, none: 'On',
      when: () => !!defaults.vision,
      help: 'Off skips loading the multimodal projector (the part that reads images, and audio on models that hear): saves its VRAM and load time',
      choices: [['off', 'Off']] },
    { key: 'mmproj_cpu', label: 'Image & audio input on CPU', kind: 'choice', scope: 'model', engine: true, none: 'Off',
      when: () => !!defaults.vision,
      off: () => (effective('vision') === 'off' ? 'Image & audio input is off' : ''),
      help: 'Keeps the image reader in RAM instead of the GPU: saves about 1–4 GB of VRAM; image reading gets slower',
      choices: [['on', 'On']] },
  ];
  const MEMORY = [
    { key: 'cache_type', label: 'KV cache precision', kind: 'choice', scope: 'model', engine: true,
      help: 'How precisely the chat’s memory (KV cache) is stored; half fits twice the context at almost the same quality',
      choices: [['f16', 'Full (f16)'], ['q8_0', 'Half (q8_0)'], ['q4_0', 'Quarter (q4_0)']] },
    { key: 'flash_attn', label: 'Flash attention', kind: 'choice', scope: 'model', engine: true,
      help: 'A faster way to compute attention; a half or quarter KV cache needs it',
      choices: [['on', 'On'], ['off', 'Off'], ['auto', 'Auto']] },
    { key: 'num_thread', label: 'CPU threads', kind: 'choice', scope: 'model', engine: true, int: true, none: 'Auto',
      help: 'Processor threads used for layers that run on the CPU',
      dyn: () => Array.from({ length: defaults.cpus || navigator.hardwareConcurrency || 8 }, (_, i) => [String(i + 1), String(i + 1)]) },
    // Mixture-of-experts models only. Layers counts the output layer too.
    { key: 'n_cpu_moe', label: 'MoE experts in RAM', kind: 'choice', scope: 'model', engine: true, int: true, none: 'Auto',
      when: () => !!defaults.moe,
      help: 'Mixture-of-experts models: expert weights of this many layers stay in RAM instead of the GPU (slower, saves VRAM)',
      dyn: () => {
        const blocks = defaults.layers ? defaults.layers - 1 : 30;
        return [['-1', 'All experts'], ...Array.from({ length: blocks + 1 }, (_, n) => [String(n), n ? `${n} layers` : 'None'])];
      } },
    { key: 'load_mode', label: 'File loading', kind: 'choice', scope: 'model', engine: true, none: 'Auto',
      when: () => !!defaults.moe,
      help: 'How the model file is read into memory; mostly changes load time',
      choices: [['mmap', 'Memory-mapped'], ['none', 'Read into RAM'], ['mlock', 'Locked in RAM']] },
  ];
  // Fallbacks when nothing else sets a value, by engine.
  const OLLAMA_DEFAULTS = { temperature: 0.8, top_p: 0.9, top_k: 40, min_p: 0, repeat_penalty: 1.1, num_predict: -1,
    presence_penalty: 0, frequency_penalty: 0, repeat_last_n: 64 };
  const LLAMA_DEFAULTS = { temperature: 0.8, top_p: 0.95, top_k: 40, min_p: 0.05, repeat_penalty: 1, num_predict: -1,
    presence_penalty: 0, frequency_penalty: 0, repeat_last_n: 64 };

  // Reply format: a choice, plus the text it uses (a JSON schema or a GBNF
  // grammar). The text has its own keys, so switching modes keeps both.
  const FORMAT = { key: 'reply_format', label: 'Reply format', kind: 'format' };
  const FORMAT_HELP = {
    text: 'Plain replies',
    json: 'Any valid JSON object, shown highlighted',
    schema: 'Matches the JSON schema below, shown highlighted',
    grammar: 'Matches the GBNF grammar below (for experts)',
  };
  const MAX_REPLY_TEXT = 32 << 10; // as the server checks it, in bytes

  const currentPreset = () => {
    const v = select.value;
    return v.startsWith('preset:') ? presets.find((p) => p.name === v.slice(7)) || null : null;
  };
  const baseModel = () => select.selectedOptions[0]?.dataset.base || '';
  const savedHere = (def) => def.scope === 'model' && perModel();

  // The value that applies when the row's own value isn't set. For a
  // model row that's the model's defaults; for a chat row the preset's or
  // the model's.
  function inherited(key, def) {
    const p = currentPreset();
    if (!def || !savedHere(def)) {
      if (p?.options && key in p.options) return { value: p.options[key], from: 'preset' };
      if (key in saved) return { value: saved[key], from: 'saved' };
    }
    if (key in defaults.options) return { value: defaults.options[key], from: 'model' };
    if (defaults.engine && key in defaults.engine) return { value: defaults.engine[key], from: 'engine' };
    const fallbacks = perModel() ? LLAMA_DEFAULTS : OLLAMA_DEFAULTS;
    if (key in fallbacks) return { value: fallbacks[key], from: 'engine' };
    // Auto context keeps the whole model on the GPU, so it's off when GPU
    // layers are set by hand (the server skips it too).
    const gpuSet = key === 'num_ctx' && !perModel() && (state.options.num_gpu != null || p?.options?.num_gpu != null);
    if (key === 'num_ctx' && defaults.auto?.fit) return { value: defaults.auto.ctx || null, from: 'auto' };
    if (key === 'num_ctx' && defaults.auto?.ctx && !gpuSet) return { value: defaults.auto.ctx, from: 'auto' };
    if (key === 'num_ctx' && defaults.ollamaContext) return { value: defaults.ollamaContext, from: 'engine' };
    return { value: null, from: null };
  }

  // A row's own value: the chat's, or for a saved model row the model's.
  // A chat or preset can still override a model row ("over").
  function layers(def) {
    if (!savedHere(def)) return { own: state.options[def.key], inh: inherited(def.key, def) };
    const p = currentPreset();
    let over = null;
    if (def.key in state.options) over = { value: state.options[def.key], from: 'chat' };
    else if (p?.options && def.key in p.options) over = { value: p.options[def.key], from: 'preset' };
    return { own: saved[def.key], inh: inherited(def.key, def), over };
  }

  const fmt = (v) => (typeof v === 'number' ? String(+v.toFixed(3)) : String(v));
  const icon = (d) => `<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">${d}</svg>`;

  // ---- saving model settings ----
  let saveTimer = null;
  function setModelOption(def, v) {
    // Changing it here also drops this chat's own value, which would hide it.
    delete state.options[def.key];
    if (v === null || v === '' || Number.isNaN(v)) delete saved[def.key];
    else saved[def.key] = v;
    render();
    clearTimeout(saveTimer);
    modelStatus.textContent = 'Saving…';
    saveTimer = setTimeout(saveModel, 400);
  }
  async function saveModel() {
    const name = baseModel();
    try {
      const res = await fetch('/api/models/settings', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, options: saved }),
      });
      if (!res.ok) throw new Error(await errorText(res));
      modelStatus.textContent = 'Saved for this model. It reloads with them on the next message.';
    } catch (err) { modelStatus.textContent = `Couldn’t save: ${err.message}`; }
  }

  // ---- rows ----
  const rows = [];

  function rowShell(def) {
    const row = el('div', 'row');
    const text = el('div', 'row-text');
    const head = el('div', 'row-head');
    const title = el('label', 'row-title', def.label);
    title.htmlFor = 'tp-' + def.key;
    // The explanation sits behind an ⓘ (hover, focus or tap), so the line
    // under the name stays short: just where the value comes from.
    const info = el('button', 'info-btn', 'i');
    info.type = 'button';
    info.hidden = true;
    info.setAttribute('aria-label', `About ${def.label}`);
    const tip = el('div', 'row-tip');
    tip.setAttribute('role', 'tooltip');
    info.addEventListener('click', (e) => { e.stopPropagation(); row.classList.toggle('tip-open'); });
    info.addEventListener('blur', () => row.classList.remove('tip-open'));
    head.append(title, info);
    const sub = el('div', 'row-sub');
    text.append(head, sub, tip);
    const ctrl = el('div', 'row-ctrl');
    const reset = el('button', 'icon-btn row-reset');
    reset.type = 'button';
    reset.innerHTML = icon('<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>');
    reset.title = 'Back to the default';
    reset.setAttribute('aria-label', `Reset ${def.label}`);
    ctrl.append(reset);
    row.append(text, ctrl);
    return { row, sub, ctrl, reset };
  }

  function setOption(def, v) {
    if (savedHere(def)) return setModelOption(def, v);
    if (v === null || v === '' || Number.isNaN(v)) delete state.options[def.key];
    else state.options[def.key] = v;
    render();
  }
  function resetRow(def) {
    if (def.kind === 'think') { state.think = null; return render(); }
    if (def.kind === 'compress') { state.compress = null; return render(); }
    if (def.kind === 'tools') { state.tools = null; return render(); }
    // A chat's own value over a saved one goes first.
    if (savedHere(def) && def.key in state.options) { delete state.options[def.key]; return render(); }
    setOption(def, null);
  }
  const parse = (def, s) => (s === '' ? null : def.kind === 'choice' ? (def.int ? +s : s) : def.int ? Math.round(+s) : +s);
  const clamp = (def, v) => Math.min(def.max, Math.max(def.min, v));

  function spinButton(def, num, delta, label, path) {
    const b = el('button', 'spin-btn');
    b.type = 'button';
    b.tabIndex = -1;
    b.setAttribute('aria-label', label);
    b.innerHTML = icon(path);
    b.addEventListener('click', () => {
      const cur = num.value === '' ? (shown(def) ?? def.min) : +num.value;
      const v = clamp(def, +(cur + delta * def.step).toFixed(4));
      setOption(def, def.int ? Math.round(v) : v);
    });
    return b;
  }

  function buildRow(def, parent) {
    const { row, sub, ctrl, reset } = rowShell(def);
    const r = { def, row, sub, reset };

    if (def.kind === 'combo' || def.kind === 'think' || def.kind === 'tools' || def.kind === 'choice') {
      const sel = el('select', 'combo');
      sel.id = 'tp-' + def.key;
      sel.addEventListener('change', () => {
        if (def.kind === 'think') {
          state.think = sel.value === '' ? null : sel.value === 'on';
          render();
        } else if (def.kind === 'tools') {
          state.tools = sel.value === '' ? null : sel.value === 'on';
          render();
        } else {
          setOption(def, parse(def, sel.value));
        }
      });
      ctrl.append(sel);
      if (def.key === 'draft_model') row.classList.add('wide-ctrl'); // model names are long
      r.select = sel;
    } else {
      const spin = el('div', 'spin');
      const num = el('input', 'spin-input');
      Object.assign(num, { type: 'number', id: 'tp-' + def.key, step: def.step, min: def.min, max: def.max });
      num.inputMode = def.int ? 'numeric' : 'decimal';
      if (def.placeholder) num.placeholder = def.placeholder;
      num.addEventListener('change', () => setOption(def, parse(def, num.value)));
      spin.append(spinButton(def, num, -1, `Decrease ${def.label}`, '<path d="M5 12h14"/>'), num,
        spinButton(def, num, 1, `Increase ${def.label}`, '<path d="M12 5v14M5 12h14"/>'));
      ctrl.append(spin);
      r.num = num;
      if (def.kind === 'scale') {
        const range = el('input', 'scale');
        Object.assign(range, { type: 'range', min: def.min, max: def.max, step: def.step, tabIndex: -1 });
        range.setAttribute('aria-hidden', 'true'); // the number box is the accessible control
        range.addEventListener('input', () => setOption(def, parse(def, range.value)));
        row.classList.add('has-scale');
        row.append(range);
        r.range = range;
      }
    }
    reset.addEventListener('click', () => resetRow(def));
    rows.push(r);
    parent.append(row);
  }

  // An expander row (like GNOME's AdwExpanderRow) that hides a group of
  // rows. Closed until opened; it remembers being opened.
  function buildExpander(id, title, defs, parent) {
    const box = el('details', 'expander');
    box.id = id;
    const head = el('summary', 'row');
    const text = el('div', 'row-text');
    const sub = el('div', 'row-sub');
    text.append(el('div', 'row-title', title), sub);
    const chevron = el('span', 'chevron');
    chevron.innerHTML = icon('<path d="M6 9l6 6 6-6"/>');
    head.append(text, chevron);
    const body = el('div', 'expander-body');
    box.append(head, body);
    defs.forEach((d) => buildRow(d, body));
    try {
      box.open = localStorage.getItem('tune:' + id) === '1';
      box.addEventListener('toggle', () => localStorage.setItem('tune:' + id, box.open ? '1' : ''));
    } catch {}
    parent.append(box);
    return { box, sub };
  }

  // Sections (see chat.html): Reply style, Sampling, Context & memory,
  // Performance & loading. Each open/closed state is remembered.
  const byKey = (k) => CHAT.find((d) => d.key === k);
  buildRow(byKey('think'), $('tune-reply'));
  buildRow(byKey('tools'), $('tune-reply'));
  buildFormat($('tune-reply'));
  buildRow(byKey('num_predict'), $('tune-reply'));
  buildRow(byKey('temperature'), $('tune-sampling-box'));
  const sampling = buildExpander('tune-sampling', 'More sampling', SAMPLING, $('tune-sampling-box'));
  MODEL.forEach((d) => buildRow(d, $(d.key === 'num_ctx' ? 'tune-context' : 'tune-perf')));
  const secs = ['reply', 'sampling', 'context', 'perf'].map((id) => {
    const box = $('tsec-' + id);
    try {
      const v = localStorage.getItem('tune:sec:' + id);
      if (v !== null) box.open = v === '1';
      box.addEventListener('toggle', () => localStorage.setItem('tune:sec:' + id, box.open ? '1' : '0'));
    } catch {}
    return { id, box, sub: $('tsec-' + id + '-sub') };
  });

  // ---- automatic compression ----
  // "Compress automatically" is the chat's own choice, else its preset's, else
  // the default in Settings. Below it, the summary of the chat's oldest
  // messages, with buttons to make one now or to undo it. Both reload the page.
  {
    const def = { key: 'compress', label: 'Auto-compress chat', kind: 'compress', help: 'Summarize the oldest messages when a chat nears the context length, so it can go on' };
    const { row, sub, ctrl, reset } = rowShell(def);
    const sel = el('select', 'combo');
    sel.id = 'tp-compress';
    sel.append(new Option('Default', ''), new Option('On', 'on'), new Option('Off', 'off'));
    sel.addEventListener('change', () => { state.compress = sel.value === '' ? null : sel.value === 'on'; render(); });
    ctrl.append(sel);
    row.title = def.help;
    reset.addEventListener('click', () => resetRow(def));
    rows.push({ def, row, sub, reset, select: sel });
    $('tune-context').append(row);
  }
  const sumRow = el('div', 'row');
  const sumText = el('div', 'row-text');
  const sumSub = el('div', 'row-sub');
  sumText.append(el('div', 'row-title', 'Earlier messages'), sumSub);
  const sumCtrl = el('div', 'row-ctrl');
  const nowBtn = el('button', 'btn', 'Compress now');
  nowBtn.type = 'button';
  nowBtn.id = 'tune-compress-now';
  const undoBtn = el('button', 'btn subtle', 'Undo');
  undoBtn.type = 'button';
  undoBtn.id = 'tune-compress-undo';
  sumCtrl.append(nowBtn, undoBtn);
  sumRow.append(sumText, sumCtrl);
  $('tune-context').append(sumRow);

  function renderCompress(r) {
    const p = currentPreset();
    r.select.options[0].text = p?.compress != null ? `Preset (${p.compress ? 'on' : 'off'})` : 'Default';
    r.select.value = state.compress === null ? '' : state.compress ? 'on' : 'off';
    r.sub.textContent = 'Summarizes the oldest messages when a chat nears the context length';
    return state.compress !== null;
  }

  function renderSummary() {
    const { compression: c, messages: n } = chatState;
    const canMake = !!chatEl.dataset.chatId && n > KEEP_LATEST && (!c || n - KEEP_LATEST > c.upTo);
    const saved = c?.saved ? ` · saves about ${c.saved >= 1000 ? (c.saved / 1000).toFixed(1) + 'k' : c.saved} tokens` : '';
    sumSub.textContent = c ? `${c.upTo} messages summarized${saved}`
      : n > KEEP_LATEST ? `Nothing summarized. The last ${KEEP_LATEST} messages are always sent as they are`
        : 'Nothing to summarize yet';
    nowBtn.disabled = !canMake;
    undoBtn.hidden = !c;
  }

  async function chatAction(btn, busyText, url, init) {
    errorEl.textContent = '';
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = busyText;
    try {
      const res = await fetch(url, init);
      if (!res.ok) throw new Error(await errorText(res));
      location.reload();
    } catch (err) {
      errorEl.textContent = err.message;
      btn.textContent = label;
      btn.disabled = false;
    }
  }
  nowBtn.addEventListener('click', () => chatAction(nowBtn, 'Summarizing…',
    `/api/chats/${encodeURIComponent(chatEl.dataset.chatId)}/compress`, { method: 'POST' }));
  undoBtn.addEventListener('click', () => chatAction(undoBtn, 'Undoing…',
    `/api/chats/${encodeURIComponent(chatEl.dataset.chatId)}/compression`, { method: 'DELETE' }));

  // ---- LoRA adapters ----
  // The installed adapters for this model's architecture: each on or off,
  // with a scale from 0 to 2. Turning one on or off reloads the model; a
  // scale change applies to the running one.
  const lora = { box: el('div', 'lora-box'), list: el('div'), rows: new Map(), last: new Map() };
  const loraHead = el('div', 'row');
  const loraText = el('div', 'row-text');
  loraText.append(el('div', 'row-title', 'Adapters'),
    el('div', 'row-sub', 'Fine-tunes applied on top of the model. Turning one on or off reloads it; a scale change applies at once.'));
  loraHead.append(loraText);
  lora.box.append(loraHead, lora.list);
  lora.box.hidden = true;
  $('tune-perf').append(lora.box);
  const loraList = () => saved.lora || [];
  const setLoRA = (list) => setModelOption({ key: 'lora' }, list.length ? list : null);

  function loraRow(info) {
    const row = el('div', 'row has-scale');
    const text = el('div', 'row-text');
    const sub = el('div', 'row-sub');
    text.append(el('div', 'row-title', info.name), sub);
    const ctrl = el('div', 'row-ctrl');
    const check = el('input');
    check.type = 'checkbox';
    check.setAttribute('aria-label', `Use ${info.name}`);
    const spin = el('div', 'spin');
    const num = el('input', 'spin-input');
    Object.assign(num, { type: 'number', min: 0, max: 2, step: 0.05 });
    num.inputMode = 'decimal';
    num.placeholder = '1';
    num.setAttribute('aria-label', `Scale of ${info.name}`);
    spin.append(num);
    const range = el('input', 'scale');
    Object.assign(range, { type: 'range', min: 0, max: 2, step: 0.05, tabIndex: -1 });
    range.setAttribute('aria-hidden', 'true'); // the number box is the accessible control
    ctrl.append(check, spin);
    row.append(text, ctrl, range);
    check.addEventListener('change', () => {
      const rest = loraList().filter((u) => u.name !== info.name);
      if (check.checked) rest.push({ name: info.name, scale: lora.last.get(info.name) ?? 1 });
      setLoRA(rest);
    });
    const scaleTo = (v) => {
      if (!Number.isFinite(v)) return;
      const s = Math.round(Math.min(2, Math.max(0, v)) * 100) / 100;
      lora.last.set(info.name, s);
      setLoRA(loraList().map((u) => (u.name === info.name ? { name: u.name, scale: s } : u)));
    };
    num.addEventListener('change', () => scaleTo(parseFloat(num.value)));
    range.addEventListener('input', () => scaleTo(parseFloat(range.value)));
    return { info, row, sub, check, num, range };
  }

  // One row per compatible adapter, and one for each adapter the model
  // is set to load that's gone since (so it can be turned off).
  function buildLoRA() {
    lora.list.replaceChildren();
    lora.rows.clear();
    const infos = new Map((defaults.adapters || []).map((a) => [a.name, a]));
    for (const u of loraList()) if (!infos.has(u.name)) infos.set(u.name, { name: u.name, missing: true });
    for (const info of infos.values()) {
      const r = loraRow(info);
      lora.rows.set(info.name, r);
      lora.list.append(r.row);
    }
  }

  function renderLoRA() {
    const on = new Map(loraList().map((u) => [u.name, u.scale]));
    lora.box.hidden = !perModel() || !lora.rows.size;
    for (const [name, r] of lora.rows) {
      const scale = on.get(name);
      const active = scale !== undefined;
      const shown = active ? scale : lora.last.get(name) ?? 1;
      r.check.checked = active;
      r.num.disabled = r.range.disabled = !active;
      if (document.activeElement !== r.num) r.num.value = active ? scale : '';
      r.range.value = shown;
      r.range.style.setProperty('--pct', (shown / 2) * 100 + '%');
      const info = r.info;
      r.sub.textContent = info.missing
        ? 'Not installed any more. Turn it off to load without it.'
        : `LoRA for ${info.arch} models · ${info.size >= 1e6 ? gb(info.size) : Math.max(1, Math.round(info.size / 1e3)) + ' KB'}${active ? '' : ' · off'}`;
    }
  }

  const memory = buildExpander('tune-memory', 'Memory & loading', MEMORY, $('tune-perf'));

  // The reply format row sits under Reply style. Its text box shows for a
  // schema or a grammar.
  function buildFormat(parent) {
    const { row, sub, ctrl, reset } = rowShell(FORMAT);
    const sel = el('select', 'combo');
    sel.id = 'tp-reply_format';
    sel.replaceChildren(...[['text', 'Text'], ['json', 'JSON'], ['schema', 'JSON schema'], ['grammar', 'Grammar']].map(([v, t]) => new Option(t, v)));
    sel.addEventListener('change', () => {
      // Text is the default, but it must still override a preset's format.
      const inPreset = currentPreset()?.options?.reply_format != null;
      setOption(FORMAT, sel.value === 'text' && !inPreset ? null : sel.value);
    });
    ctrl.prepend(sel);
    const box = el('div', 'reply-box');
    const area = el('textarea', 'boxed-text reply-code');
    area.id = 'tp-reply_text';
    area.rows = 6;
    area.spellcheck = false;
    area.addEventListener('input', () => setOption(replyDef(), area.value === '' ? null : area.value));
    const err = el('p', 'tune-error small');
    box.append(area, err);
    row.classList.add('has-scale');
    row.append(box);
    reset.addEventListener('click', () => setOption(FORMAT, null));
    rows.push({ def: FORMAT, row, sub, reset, select: sel, box, area, err });
    parent.append(row);
  }
  const replyDef = () => (shown(FORMAT) === 'grammar' ? { key: 'reply_grammar' } : { key: 'reply_schema' });

  // What's wrong with a schema or grammar, checked as the server checks it.
  function replyProblem(mode, text) {
    if (!text) return mode === 'schema' ? 'Add a JSON schema, or replies aren’t constrained' : 'Add a grammar, or replies aren’t constrained';
    if (new TextEncoder().encode(text).length > MAX_REPLY_TEXT) return 'Over 32 KB';
    if (mode !== 'schema') return '';
    try {
      const v = JSON.parse(text);
      return v && typeof v === 'object' && !Array.isArray(v) ? '' : 'The schema must be a JSON object';
    } catch (e) {
      return `Not valid JSON: ${e.message}`;
    }
  }

  function renderFormat(r) {
    const mode = shown(FORMAT) ?? 'text';
    if (document.activeElement !== r.select) r.select.value = mode;
    const constrained = mode === 'schema' || mode === 'grammar';
    r.box.hidden = !constrained;
    if (constrained) {
      const text = shown(replyDef()) ?? '';
      if (document.activeElement !== r.area) r.area.value = text;
      r.area.placeholder = mode === 'grammar' ? 'root ::= "yes" | "no"' : '{"type": "object", "properties": {"name": {"type": "string"}}}';
      const problem = replyProblem(mode, text);
      r.err.textContent = problem;
      r.area.setAttribute('aria-invalid', String(!!problem));
    }
    r.sub.textContent = `${origin(FORMAT)} · ${FORMAT_HELP[mode]}`;
    return state.options.reply_format !== undefined;
  }

  // ---- rendering ----
  function changedCount() {
    return Object.keys(state.options).length + (state.system ? 1 : 0) + (state.think !== null ? 1 : 0) + (state.compress !== null ? 1 : 0) + (state.tools !== null ? 1 : 0);
  }

  function describe(inh) {
    switch (inh.from) {
      case 'preset': return 'From the preset';
      case 'saved': return 'Saved for the model';
      case 'model': return 'Model’s recommendation';
      case 'auto': return 'Auto';
      case 'engine': return perModel() ? 'llama.cpp default' : 'Ollama default';
      default: return 'Default';
    }
  }

  // What a row shows and where it comes from.
  function shown(def) {
    const { own, inh, over } = layers(def);
    return over ? over.value : own ?? inh.value;
  }
  function origin(def) {
    const { own, inh, over } = layers(def);
    if (over) return over.from === 'chat' ? 'Set by this chat, over the model’s' : 'Set by the preset';
    if (own != null) return savedHere(def) ? 'Saved' : 'Changed';
    return describe(inh);
  }

  function renderCombo(r) {
    const { def, select: sel } = r;
    const { own, inh, over } = layers(def);
    const mine = over?.from === 'chat' ? over.value : own;
    const max = defaults.contextLength || 262144;
    const steps = CTX_STEPS.filter((c) => c <= max);
    for (const v of [mine, over?.value]) if (v != null && !steps.includes(v)) steps.push(v);
    const first = inh.value != null ? `${describe(inh)} (${kib(inh.value)})` : describe(inh);
    const opts = [['', first], ...steps.sort((a, b) => a - b).map((c) => [String(c), kib(c)])];
    const same = sel.options.length === opts.length && [...sel.options].every((o, i) => o.value === opts[i][0] && o.text === opts[i][1]);
    if (!same) sel.replaceChildren(...opts.map(([v, t]) => new Option(t, v)));
    const val = over ? over.value : own;
    sel.value = val != null ? String(val) : '';
    let text = origin(def);
    if (val == null && inh.from === 'auto') {
      if (perModel()) text = inh.value ? `Auto: ${kib(inh.value)}, less if memory is short` : 'Auto: 32k, less if memory is short';
      else text = defaults.auto?.fit ? 'Largest that fits, picked by llama.cpp' : 'Largest that fits on your GPU';
    } else if (val == null && !perModel() && effective('num_gpu') != null) text = 'Not auto: GPU layers are set by hand';
    else if (val == null && inh.from === 'engine' && defaults.auto?.reason) text = `Not auto: ${defaults.auto.reason}`;
    if (defaults.contextLength) text += ` · up to ${kib(defaults.contextLength)}`;
    r.sub.textContent = text;
    r.row.title = def.help;
    return over?.from === 'chat' || own != null;
  }

  function renderThink(r) {
    const sel = r.select;
    if (!sel.options.length) sel.append(new Option('Default', ''), new Option('On', 'on'), new Option('Off', 'off'));
    const p = currentPreset();
    sel.options[0].text = p?.think != null ? `Preset (${p.think ? 'on' : 'off'})` : 'Default';
    sel.value = state.think === null ? '' : state.think ? 'on' : 'off';
    r.sub.textContent = defaults.thinking ? 'Reasons step by step before answering' : 'This model doesn’t say it can think';
    return state.think !== null;
  }

  function renderTools(r) {
    const sel = r.select;
    if (!sel.options.length) sel.append(new Option('Default', ''), new Option('On', 'on'), new Option('Off', 'off'));
    const p = currentPreset();
    sel.options[0].text = p?.tools != null ? `Preset (${p.tools ? 'on' : 'off'})` : 'Default (on)';
    sel.value = state.tools === null ? '' : state.tools ? 'on' : 'off';
    // A model whose chat template has no tool support can't call them.
    sel.disabled = !defaults.tools;
    r.row.title = 'Lets the model search the web, read pages, calculate and look through your chats. Choose which in Settings.';
    r.sub.textContent = defaults.tools
      ? (state.tools === false || (state.tools === null && p?.tools === false) ? 'Off: the model answers on its own' : 'The model can look things up and calculate')
      : 'This model’s template has no tool support';
    return state.tools !== null;
  }

  function renderChoice(r) {
    const { def, select: sel } = r;
    const { own, inh, over } = layers(def);
    const choices = def.dyn ? def.dyn() : def.choices;
    const label = (v) => choices.find(([c]) => c === String(v))?.[1] ?? String(v);
    const first = (typeof def.none === 'function' ? def.none() : def.none) || (inh.value != null ? `Default (${label(inh.value)})` : 'Default');
    const opts = [['', first], ...choices];
    const val = over ? over.value : own;
    if (val != null && !opts.some(([c]) => c === String(val))) opts.push([String(val), String(val)]);
    const same = sel.options.length === opts.length && [...sel.options].every((o, i) => o.value === opts[i][0] && o.text === opts[i][1]);
    if (!same) sel.replaceChildren(...opts.map(([v, t]) => new Option(t, v)));
    sel.value = val != null ? String(val) : '';
    const help = def.empty && !choices.length && val == null ? def.empty : def.help;
    r.sub.textContent = val != null ? `${origin(def)} · ${help}` : help;
    if (def.key === 'speedup') addMTPGet(r.sub);
    return over?.from === 'chat' || own != null;
  }

  // What the speculative decoding settings amount to, in words.
  function speedPlan() {
    const own = (k) => state.options[k] ?? saved[k];
    const sp = own('speedup') || 'auto';
    const draft = own('draft_model');
    const out = [];
    if (sp === 'mtp' || (sp === 'auto' && !draft)) { if (defaults.speed?.mtp) out.push('MTP head'); }
    else if (sp === 'auto' && draft) out.push('draft model');
    const ng = own('spec_ngram');
    if ((ng !== undefined && ng !== 'off') || (ng === undefined && sp === 'auto')) out.push('n-gram guessing');
    return out;
  }

  // A model downloaded without its repo's MTP head can fetch it.
  const mtpGet = { name: '', size: 0, busy: false };
  function addMTPGet(sub) {
    if (!mtpGet.size || mtpGet.name !== defaults.speed?.name) return;
    const b = el('button', 'btn subtle', mtpGet.busy ? 'Downloading…' : `Download its MTP head (${gb(mtpGet.size)})`);
    b.type = 'button';
    b.disabled = mtpGet.busy;
    b.addEventListener('click', downloadMTP);
    sub.append(' ', b);
  }
  async function downloadMTP() {
    mtpGet.busy = true;
    render();
    try {
      const res = await fetch('/api/pulls', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: 'mtp:' + mtpGet.name, alias: '' }) });
      if (!res.ok) throw new Error(await errorText(res));
      const job = await res.json();
      for (;;) {
        await new Promise((ok) => setTimeout(ok, 1500));
        const j = (await (await fetch('/api/pulls')).json()).find((x) => x.id === job.id);
        if (!j || j.state !== 'running') {
          if (j?.state === 'failed') throw new Error(j.error);
          break;
        }
      }
      mtpGet.size = 0;
      await loadDefaults();
    } catch (err) {
      mtpGet.busy = false;
      modelStatus.textContent = `Couldn’t download the MTP head: ${err.message}`;
      render();
    }
  }
  async function checkMTP() {
    const name = defaults.speed?.name;
    mtpGet.name = name || '';
    mtpGet.size = 0;
    mtpGet.busy = false;
    if (!name) return;
    try {
      const j = await (await fetch('/api/models/mtp?name=' + encodeURIComponent(name))).json();
      if (j.available && defaults.speed?.name === name) { mtpGet.size = j.size; render(); }
    } catch {}
  }

  function renderNumber(r) {
    const { def, num, range, sub } = r;
    if (def.key === 'num_gpu') return renderLayers(r);
    const { own, inh, over } = layers(def);
    const val = shown(def);
    const unlimited = def.key === 'num_predict' && val === -1;
    if (document.activeElement !== num) num.value = unlimited ? '' : val ?? '';
    if (def.key === 'num_predict') num.placeholder = 'No limit';
    if (range) {
      range.value = val ?? def.min;
      const pct = ((clamp(def, +range.value) - def.min) / (def.max - def.min)) * 100;
      range.style.setProperty('--pct', pct + '%');
    }
    const has = own !== undefined || over?.from === 'chat';
    // When changed, say what it was so it's easy to go back.
    const source = has && !over && inh.value != null && !savedHere(def) ? `${describe(inh)}: ${fmt(inh.value)}` : origin(def);
    sub.textContent = def.help ? `${source} · ${def.help}` : source;
    return has;
  }

  // GPU layers: unset means as many as fit, shown as all of them.
  function renderLayers(r) {
    const { def, num, range, sub } = r;
    const { own, over } = layers(def);
    const max = defaults.layers || 99;
    const val = shown(def);
    num.max = range.max = max;
    if (document.activeElement !== num) num.value = val ?? '';
    range.value = val ?? max;
    range.style.setProperty('--pct', (Math.min(+range.value, max) / max) * 100 + '%');
    const of = defaults.layers ? ` of ${max}` : '';
    const from = origin(def);
    // A number set by hand turns llama.cpp's fitting off (ollamui falls
    // back to Auto if it then runs out of memory), so Auto says what it does.
    const handSet = perModel() ? ' · fixed, so not fitted to memory' : ' · auto context off';
    if (val == null && perModel() && effective('n_cpu_moe') != null) sub.textContent = 'Auto: all layers on the GPU, since experts are set in Memory';
    else if (val == null) sub.textContent = perModel()
      ? `Auto: fits the model to free GPU memory${defaults.layers ? ` (${max} layers in all)` : ''}; MoE experts go to RAM when needed`
      : `Auto: all that fit${defaults.layers ? ` (${max} layers)` : ''}`;
    else if (val >= max) sub.textContent = `${from}: all ${max} on the GPU${perModel() ? handSet : ''}`;
    else sub.textContent = `${from}: ${val}${of} on the GPU, the rest on the CPU (slower)${handSet}`;
    return own !== undefined || over?.from === 'chat';
  }

  const effective = (key) => state.options[key] ?? inherited(key).value;

  function render() {
    const preset = currentPreset();
    for (const r of rows) {
      r.row.hidden = (!!r.def.engine && !perModel()) || (r.def.when && !r.def.when());
      const off = r.def.off ? r.def.off() : '';
      const k = r.def.kind;
      const changed = k === 'format' ? renderFormat(r) : k === 'combo' ? renderCombo(r) : k === 'think' ? renderThink(r) : k === 'compress' ? renderCompress(r) : k === 'tools' ? renderTools(r) : k === 'choice' ? renderChoice(r) : renderNumber(r);
      r.row.classList.toggle('changed', changed);
      r.reset.hidden = !changed;
      // A row that doesn't apply right now stays visible, greyed, with the reason.
      r.row.classList.toggle('off', !!off);
      r.row.querySelectorAll('input, select').forEach((c) => { if (off) c.disabled = true; else if (r.def.kind !== 'tools') c.disabled = false; });
      if (off) { r.sub.textContent = off; }
      // The help goes behind the ⓘ; the line keeps the rest.
      const help = r.def.help;
      const info = r.row.querySelector(':scope .info-btn');
      if (info) {
        const t = r.sub.textContent;
        if (help && !off && t.endsWith(help)) r.sub.textContent = t.slice(0, -help.length).replace(/\s*·\s*$/, '');
        // A status line too long for the row is shown whole in the tip too.
        const clipped = r.sub.scrollWidth > r.sub.clientWidth + 1 ? r.sub.textContent : '';
        const tip = [clipped, off ? '' : help].filter(Boolean).join('\n\n');
        info.hidden = !tip;
        r.row.querySelector(':scope .row-tip').textContent = tip;
        if (tip) r.row.removeAttribute('title');
      }
      if (r.sub.textContent && (!help || !info)) r.row.title = r.sub.textContent; // the line is clipped to one row
    }
    const samplingChanged = SAMPLING.filter((d) => d.key in state.options).length;
    sampling.sub.textContent = samplingChanged ? `${samplingChanged} changed` : 'Top P, top K, penalties, seed';
    memory.box.hidden = !perModel();
    memory.sub.textContent = MEMORY.slice(0, 2).map((d, i) => `${['KV cache', 'flash attention'][i]} ${shown(d) ?? 'default'}`).join(' · ');
    renderLoRA();

    const changedIn = (keys) => keys.filter((k) => k in state.options).length;
    const subs = {
      reply: [state.system ? 1 : 0, state.think !== null ? 1 : 0, state.tools !== null ? 1 : 0, changedIn(['num_predict', 'reply_format', 'reply_schema', 'reply_grammar'])].reduce((a, b) => a + b, 0),
      sampling: changedIn(['temperature', ...SAMPLING.map((d) => d.key)]),
    };
    const perm = perModel() ? 'saved for the model' : 'this chat';
    const t0 = effective('temperature');
    const c0 = shown(MODEL[0]);
    const text = {
      reply: subs.reply ? `${subs.reply} changed` : '',
      sampling: subs.sampling ? `${subs.sampling} changed` : t0 != null ? `temp ${fmt(t0)}` : '',
      context: [c0 != null ? `${kib(c0)} ctx` : '', perm].filter(Boolean).join(' · '),
      perf: perModel() ? `${shown(MODEL[1]) == null ? 'auto GPU' : shown(MODEL[1]) + ' GPU layers'} · ${perm}` : '',
    };
    for (const sc of secs) sc.sub.textContent = text[sc.id];

    if (document.activeElement !== systemEl) systemEl.value = state.system;
    systemEl.placeholder = preset?.system || defaults.system || 'None. Describe how the model should behave';

    sourceEl.textContent = preset ? `preset ${preset.name}` : '';

    saveBtn.hidden = !preset;
    if (preset) saveBtn.textContent = `Save to “${preset.name}”`;
    saveBtn.disabled = !changedCount();
    resetBtn.disabled = !changedCount();

    const parts = [];
    const t = effective('temperature');
    if (t != null) parts.push(`temp ${fmt(t)}`);
    const c = shown(MODEL[0]);
    if (c != null) parts.push(`${kib(c)} ctx`);
    renderSummary();
    const n = changedCount();
    summary.textContent = ['Tune', ...parts, n ? `${n} changed` : ''].filter(Boolean).join(' · ');
    openBtn.classList.toggle('has-changes', n > 0);
  }

  async function loadDefaults() {
    const base = baseModel();
    defaults = { options: {}, system: '', contextLength: 0, thinking: false, tools: false };
    if (base) {
      try {
        const res = await fetch('/api/models/defaults?name=' + encodeURIComponent(base));
        if (res.ok) defaults = await res.json();
      } catch {}
    }
    saved = { ...(defaults.saved || {}) };
    buildLoRA();
    modelStatus.textContent = '';
    render();
    checkMTP();
  }

  // ---- open / close ----
  function setOpen(open) {
    panel.classList.toggle('open', open);
    panel.inert = !open;
    openBtn.setAttribute('aria-expanded', String(open));
    try { localStorage.setItem('tuneOpen', open ? '1' : ''); } catch {}
    if (open) panel.querySelector('select, input, textarea')?.focus({ preventScroll: true });
  }
  openBtn.addEventListener('click', () => setOpen(!panel.classList.contains('open')));
  $('tune-close').addEventListener('click', () => { setOpen(false); openBtn.focus(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && panel.classList.contains('open')) { setOpen(false); openBtn.focus(); }
    if (e.key === '.' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); setOpen(!panel.classList.contains('open')); }
  });
  // Below 1180px the panel floats over the chat (see app.css), so it only
  // opens when asked, and a click outside it closes it.
  const floating = matchMedia('(max-width: 1180px)');
  try { if ((localStorage.getItem('tuneOpen') && !floating.matches) || location.hash === '#tune') setOpen(true); } catch {}
  document.addEventListener('click', (e) => {
    if (!floating.matches || !panel.classList.contains('open') || !e.isTrusted) return;
    const path = e.composedPath(); // still holds targets a handler re-rendered away
    if (path.includes(panel) || path.includes(openBtn) || e.target.closest?.('.stats')) return;
    setOpen(false);
  });

  // ---- inputs ----
  systemEl.addEventListener('input', () => { state.system = systemEl.value; render(); });
  select.addEventListener('change', loadDefaults);
  // Loading or ejecting changes what auto context can pick.
  document.addEventListener('loaded-changed', loadDefaults);
  resetBtn.addEventListener('click', () => {
    clearChanges();
    render();
  });

  // ---- presets ----
  async function putPreset(name, body) {
    const res = await fetch('/api/presets/' + encodeURIComponent(name), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await errorText(res));
    const saved = await res.json();
    presets = presets.filter((p) => p.name !== saved.name).concat(saved);
    return saved;
  }

  // Preset settings with this chat's changes applied on top.
  function mergedSettings() {
    const p = currentPreset();
    return {
      base: p ? p.base : baseModel(),
      options: { ...(p?.options || {}), ...state.options },
      system: state.system || p?.system || '',
      think: state.think ?? p?.think ?? null,
      compress: state.compress ?? p?.compress ?? null,
      tools: state.tools ?? p?.tools ?? null,
    };
  }

  function clearChanges() {
    state.options = {};
    state.system = '';
    state.think = null;
    state.compress = null;
    state.tools = null;
  }

  async function run(btn, fn) {
    errorEl.textContent = '';
    btn.disabled = true;
    try { await fn(); } catch (err) { errorEl.textContent = err.message; }
    btn.disabled = false;
    render();
  }

  saveBtn.addEventListener('click', () => run(saveBtn, async () => {
    const p = currentPreset();
    await putPreset(p.name, mergedSettings());
    clearChanges();
  }));

  saveNewBtn.addEventListener('click', () => run(saveNewBtn, async () => {
    const suggestion = baseModel().split('/').pop().split(':')[0].toLowerCase().replace(/[^a-z0-9._-]/g, '-');
    const name = prompt('Name for the new preset (lowercase letters, numbers, - . _):', suggestion);
    if (!name) return;
    if (presets.some((p) => p.name === name) && !confirm(`Replace the existing preset “${name}”?`)) return;
    const saved = await putPreset(name, mergedSettings());
    clearChanges();
    addPresetOption(saved);
  }));

  function addPresetOption(p) {
    let group = select.querySelector('optgroup[label="Presets"]');
    if (!group) {
      group = document.createElement('optgroup');
      group.label = 'Presets';
      select.prepend(group);
    }
    let opt = [...group.options].find((o) => o.value === 'preset:' + p.name);
    if (!opt) {
      opt = el('option', null, p.name);
      opt.value = 'preset:' + p.name;
      group.append(opt);
    }
    opt.dataset.base = p.base;
    select.value = opt.value;
    select.dispatchEvent(new Event('change'));
  }

  // ---- API for chat.js ----
  window.ollamuiTune = {
    // The context the model runs with, as far as Tune knows (0 if not).
    contextSize: () => +effective('num_ctx') || 0,
    selection() {
      const p = currentPreset();
      return p ? { preset: p.name } : { model: select.value };
    },
    settings() {
      return { options: state.options, system: state.system, think: state.think, compress: state.compress, tools: state.tools };
    },
    // chat.js reports the chat's summary and how many messages it shows.
    setChat(c) {
      Object.assign(chatState, c);
      render();
    },
    // The reply format in effect (the chat's, else the preset's), for display.
    replyFormat: () => shown(FORMAT) ?? 'text',
    baseModel,
    open() { setOpen(true); },
  };

  loadDefaults();
})();
