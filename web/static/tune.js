// Tune panel, in two parts.
// "This chat": thinking, temperature, reply length, system prompt and (hidden
// until opened) the rest of sampling. These are the chat's own, layered over
// the preset (if any), the model's settings and its recommended defaults.
// The model part: context, GPU layers, how long it stays loaded and (hidden)
// memory options. With llama.cpp these are saved for the model straight
// away, so every chat with it uses them. With Ollama they're per chat too.
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
  const publishRow = $('tune-publish-row');
  const publishBox = $('tune-publish');
  const publishName = $('tune-publish-name');
  const errorEl = $('tune-error');
  const modelTitle = $('tune-model-title');
  const modelSub = $('tune-model-sub');
  const modelStatus = $('tune-model-status');

  let presets = JSON.parse(chatEl.dataset.presets || 'null') || [];
  const initial = JSON.parse(chatEl.dataset.tune || '{}');
  // This chat's own changes. Anything not in here is inherited.
  const state = { options: { ...(initial.options || {}) }, system: initial.system || '', think: initial.think ?? null };
  let defaults = { options: {}, system: '', contextLength: 0, thinking: false };
  // The model's saved settings (llama.cpp only).
  let saved = {};
  const perModel = () => !!defaults.engine;

  const CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
  const kib = (n) => (n >= 1024 ? Math.round(n / 1024) + 'k' : String(n));

  // Each setting is one row. kind: combo (context dropdown), think, choice
  // (dropdown of fixed values), spin (number with − +), scale (slider plus
  // number). scope 'model' rows belong to the model part.
  const CHAT = [
    { key: 'think', label: 'Thinking', kind: 'think' },
    { key: 'temperature', label: 'Temperature', kind: 'scale', min: 0, max: 2, step: 0.05, short: 'temp',
      help: 'Lower stays focused, higher gets more varied' },
    { key: 'num_predict', label: 'Max reply length', kind: 'spin', min: -1, max: 262144, step: 256, int: true,
      help: 'Tokens' },
  ];
  const SAMPLING = [
    { key: 'top_p', label: 'Top P', kind: 'scale', min: 0, max: 1, step: 0.01, short: 'top-p',
      help: 'Picks only from the likeliest words that add up to this share' },
    { key: 'top_k', label: 'Top K', kind: 'scale', min: 0, max: 200, step: 1, int: true, short: 'top-k',
      help: 'Picks only from this many likeliest words' },
    { key: 'min_p', label: 'Min P', kind: 'scale', min: 0, max: 1, step: 0.01, short: 'min-p',
      help: 'Skips words much less likely than the best one' },
    { key: 'repeat_penalty', label: 'Repeat penalty', kind: 'scale', min: 0.8, max: 2, step: 0.01, short: 'repeat',
      help: 'Above 1 discourages repeating itself' },
    { key: 'presence_penalty', label: 'Presence penalty', kind: 'scale', min: 0, max: 2, step: 0.05,
      help: 'Nudges the model toward new topics' },
    { key: 'frequency_penalty', label: 'Frequency penalty', kind: 'scale', min: 0, max: 2, step: 0.05,
      help: 'Discourages repeating the same words' },
    { key: 'repeat_last_n', label: 'Repeat window', kind: 'spin', min: -1, max: 4096, step: 16, int: true,
      help: 'Tokens the repeat penalty looks back over' },
    { key: 'seed', label: 'Seed', kind: 'spin', min: 0, max: 2147483647, step: 1, int: true, placeholder: 'Random',
      help: 'Same seed and settings give the same reply' },
  ];
  const MODEL = [
    { key: 'num_ctx', label: 'Context length', kind: 'combo', int: true, scope: 'model',
      help: 'Prompt, thinking and reply all share it' },
    { key: 'num_gpu', label: 'GPU layers', kind: 'scale', min: 0, max: 99, step: 1, int: true, placeholder: 'All', scope: 'model' },
    { key: 'keep_alive', label: 'Keep loaded', kind: 'choice', scope: 'model', engine: true,
      help: 'After the last message',
      choices: [['5m', '5 minutes'], ['30m', '30 minutes'], ['1h', '1 hour'], ['4h', '4 hours'], ['-1', 'Until unloaded']] },
  ];
  const MEMORY = [
    { key: 'cache_type', label: 'Context cache', kind: 'choice', scope: 'model', engine: true,
      help: 'Half fits twice the context at almost the same quality',
      choices: [['f16', 'Full (f16)'], ['q8_0', 'Half (q8_0)'], ['q4_0', 'Quarter (q4_0)']] },
    { key: 'flash_attn', label: 'Flash attention', kind: 'choice', scope: 'model', engine: true,
      help: 'Faster; a half or quarter cache needs it',
      choices: [['on', 'On'], ['off', 'Off'], ['auto', 'Auto']] },
  ];
  // Fallbacks when nothing else sets a value, by engine.
  const OLLAMA_DEFAULTS = { temperature: 0.8, top_p: 0.9, top_k: 40, min_p: 0, repeat_penalty: 1.1, num_predict: -1,
    presence_penalty: 0, frequency_penalty: 0, repeat_last_n: 64 };
  const LLAMA_DEFAULTS = { temperature: 0.8, top_p: 0.95, top_k: 40, min_p: 0.05, repeat_penalty: 1, num_predict: -1,
    presence_penalty: 0, frequency_penalty: 0, repeat_last_n: 64 };

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
    const title = el('label', 'row-title', def.label);
    title.htmlFor = 'tp-' + def.key;
    const sub = el('div', 'row-sub');
    text.append(title, sub);
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
    // A chat's own value over a saved one goes first.
    if (savedHere(def) && def.key in state.options) { delete state.options[def.key]; return render(); }
    setOption(def, null);
  }
  const parse = (def, s) => (s === '' ? null : def.kind === 'choice' ? s : def.int ? Math.round(+s) : +s);
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

    if (def.kind === 'combo' || def.kind === 'think' || def.kind === 'choice') {
      const sel = el('select', 'combo');
      sel.id = 'tp-' + def.key;
      sel.addEventListener('change', () => {
        if (def.kind === 'think') {
          state.think = sel.value === '' ? null : sel.value === 'on';
          render();
        } else {
          setOption(def, parse(def, sel.value));
        }
      });
      ctrl.append(sel);
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

  CHAT.forEach((d) => buildRow(d, $('tune-chat')));
  const sampling = buildExpander('tune-sampling', 'Advanced', SAMPLING, $('tune-sampling-box'));
  MODEL.forEach((d) => buildRow(d, $('tune-model')));
  const memory = buildExpander('tune-memory', 'Memory', MEMORY, $('tune-memory-box'));

  // ---- rendering ----
  function changedCount() {
    return Object.keys(state.options).length + (state.system ? 1 : 0) + (state.think !== null ? 1 : 0);
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

  function renderChoice(r) {
    const { def, select: sel } = r;
    const { own, inh, over } = layers(def);
    const label = (v) => def.choices.find(([c]) => c === String(v))?.[1] ?? String(v);
    const first = inh.value != null ? `Default (${label(inh.value)})` : 'Default';
    const opts = [['', first], ...def.choices];
    const val = over ? over.value : own;
    if (val != null && !opts.some(([c]) => c === String(val))) opts.push([String(val), String(val)]);
    const same = sel.options.length === opts.length && [...sel.options].every((o, i) => o.value === opts[i][0] && o.text === opts[i][1]);
    if (!same) sel.replaceChildren(...opts.map(([v, t]) => new Option(t, v)));
    sel.value = val != null ? String(val) : '';
    r.sub.textContent = val != null ? `${origin(def)} · ${def.help}` : def.help;
    return over?.from === 'chat' || own != null;
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
    if (val == null) sub.textContent = `Auto: all that fit${defaults.layers ? ` (${max} layers)` : ''}`;
    else if (val >= max) sub.textContent = `${from}: all ${max} on the GPU`;
    else sub.textContent = `${from}: ${val}${of} on the GPU, the rest on the CPU (slower)${perModel() ? '' : ' · auto context off'}`;
    return own !== undefined || over?.from === 'chat';
  }

  const effective = (key) => state.options[key] ?? inherited(key).value;

  function render() {
    const preset = currentPreset();
    for (const r of rows) {
      r.row.hidden = !!r.def.engine && !perModel();
      const k = r.def.kind;
      const changed = k === 'combo' ? renderCombo(r) : k === 'think' ? renderThink(r) : k === 'choice' ? renderChoice(r) : renderNumber(r);
      r.row.classList.toggle('changed', changed);
      r.reset.hidden = !changed;
    }
    const samplingChanged = SAMPLING.filter((d) => d.key in state.options).length;
    sampling.sub.textContent = samplingChanged ? `${samplingChanged} changed` : 'Top P, top K, penalties, seed. The defaults suit most models.';
    memory.box.hidden = !perModel();
    memory.sub.textContent = MEMORY.map((d) => `${d.label.toLowerCase()} ${shown(d) ?? 'default'}`).join(' · ');

    modelTitle.textContent = baseModel() || 'Model';
    modelSub.textContent = perModel() ? 'Saved for the model, so every chat with it uses these.' : 'For this chat.';

    if (document.activeElement !== systemEl) systemEl.value = state.system;
    systemEl.placeholder = preset?.system || defaults.system || 'None. Describe how the model should behave';

    sourceEl.textContent = preset ? `preset ${preset.name}` : '';

    saveBtn.hidden = !preset;
    if (preset) saveBtn.textContent = `Save to “${preset.name}”`;
    saveBtn.disabled = !changedCount();
    resetBtn.disabled = !changedCount();
    publishRow.hidden = !preset || perModel(); // with llama.cpp every preset is in the /v1 API
    if (preset) {
      publishBox.checked = !!preset.published;
      publishName.textContent = 'ollamui/' + preset.name;
    }

    const parts = [];
    const t = effective('temperature');
    if (t != null) parts.push(`temp ${fmt(t)}`);
    const c = shown(MODEL[0]);
    if (c != null) parts.push(`${kib(c)} ctx`);
    const n = changedCount();
    summary.textContent = ['Tune', ...parts, n ? `${n} changed` : ''].filter(Boolean).join(' · ');
    openBtn.classList.toggle('has-changes', n > 0);
  }

  async function loadDefaults() {
    const base = baseModel();
    defaults = { options: {}, system: '', contextLength: 0, thinking: false };
    if (base) {
      try {
        const res = await fetch('/api/models/defaults?name=' + encodeURIComponent(base));
        if (res.ok) defaults = await res.json();
      } catch {}
    }
    saved = { ...(defaults.saved || {}) };
    modelStatus.textContent = '';
    render();
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
    };
  }

  function clearChanges() {
    state.options = {};
    state.system = '';
    state.think = null;
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
    await putPreset(p.name, { ...mergedSettings(), published: !!p.published });
    clearChanges();
  }));

  saveNewBtn.addEventListener('click', () => run(saveNewBtn, async () => {
    const suggestion = baseModel().split('/').pop().split(':')[0].toLowerCase().replace(/[^a-z0-9._-]/g, '-');
    const name = prompt('Name for the new preset (lowercase letters, numbers, - . _):', suggestion);
    if (!name) return;
    if (presets.some((p) => p.name === name) && !confirm(`Replace the existing preset “${name}”?`)) return;
    const saved = await putPreset(name, { ...mergedSettings(), published: false });
    clearChanges();
    addPresetOption(saved);
  }));

  publishBox.addEventListener('change', () => run(publishBox, async () => {
    const p = currentPreset();
    try {
      await putPreset(p.name, { ...p, published: publishBox.checked });
    } catch (err) {
      publishBox.checked = !publishBox.checked;
      throw err;
    }
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
    selection() {
      const p = currentPreset();
      return p ? { preset: p.name } : { model: select.value };
    },
    settings() {
      return { options: state.options, system: state.system, think: state.think };
    },
    baseModel,
    open() { setOpen(true); },
  };

  loadDefaults();
})();
