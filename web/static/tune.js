// Tune panel: per-chat settings layered over the preset (if any), the
// model's own defaults, ollamui's auto-picked context and Ollama's defaults.
// Chat.js reads the result via window.ollamuiTune when sending.
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

  let presets = JSON.parse(chatEl.dataset.presets || 'null') || [];
  const initial = JSON.parse(chatEl.dataset.tune || '{}');
  // This chat's own changes. Anything not in here is inherited.
  const state = { options: { ...(initial.options || {}) }, system: initial.system || '', think: initial.think ?? null };
  let defaults = { options: {}, system: '', contextLength: 0, thinking: false };

  const CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
  const kib = (n) => (n >= 1024 ? Math.round(n / 1024) + 'k' : String(n));

  // Each setting is one row. kind: combo (dropdown), think (dropdown),
  // spin (number with − +), scale (slider plus number).
  const MAIN = [
    { key: 'num_ctx', label: 'Context length', kind: 'combo', int: true,
      help: 'Prompt, thinking and reply all share it' },
    { key: 'num_gpu', label: 'GPU layers', kind: 'scale', min: 0, max: 99, step: 1, int: true, placeholder: 'All',
      help: 'Fewer leaves VRAM for other things but runs slower' },
    { key: 'think', label: 'Thinking', kind: 'think' },
    { key: 'num_predict', label: 'Max reply length', kind: 'spin', min: -1, max: 262144, step: 256, int: true,
      help: 'Tokens' },
  ];
  const SAMPLING = [
    { key: 'temperature', label: 'Temperature', kind: 'scale', min: 0, max: 2, step: 0.05, short: 'temp',
      help: 'Higher is more varied, lower more focused' },
    { key: 'top_p', label: 'Top P', kind: 'scale', min: 0, max: 1, step: 0.01, short: 'top-p' },
    { key: 'top_k', label: 'Top K', kind: 'scale', min: 0, max: 200, step: 1, int: true, short: 'top-k' },
    { key: 'min_p', label: 'Min P', kind: 'scale', min: 0, max: 1, step: 0.01, short: 'min-p' },
    { key: 'repeat_penalty', label: 'Repeat penalty', kind: 'scale', min: 0.8, max: 2, step: 0.01, short: 'repeat' },
  ];
  const ADVANCED = [
    { key: 'presence_penalty', label: 'Presence penalty', kind: 'scale', min: 0, max: 2, step: 0.05,
      help: 'Nudges the model toward new topics' },
    { key: 'frequency_penalty', label: 'Frequency penalty', kind: 'scale', min: 0, max: 2, step: 0.05,
      help: 'Discourages repeating the same words' },
    { key: 'repeat_last_n', label: 'Repeat window', kind: 'spin', min: -1, max: 4096, step: 16, int: true,
      help: 'Tokens the repeat penalty looks back over' },
    { key: 'seed', label: 'Seed', kind: 'spin', min: 0, max: 2147483647, step: 1, int: true, placeholder: 'Random',
      help: 'Same seed and settings give the same reply' },
  ];
  // Ollama's documented fallbacks, shown when neither preset nor model sets a value.
  const OLLAMA_DEFAULTS = { temperature: 0.8, top_p: 0.9, top_k: 40, min_p: 0, repeat_penalty: 1.1, num_predict: -1,
    presence_penalty: 0, frequency_penalty: 0, repeat_last_n: 64 };

  const currentPreset = () => {
    const v = select.value;
    return v.startsWith('preset:') ? presets.find((p) => p.name === v.slice(7)) || null : null;
  };
  const baseModel = () => select.selectedOptions[0]?.dataset.base || '';

  // The value that applies when this chat doesn't override it.
  function inherited(key) {
    const p = currentPreset();
    if (p?.options && key in p.options) return { value: p.options[key], from: 'preset' };
    if (key in defaults.options) return { value: defaults.options[key], from: 'model' };
    if (key in OLLAMA_DEFAULTS) return { value: OLLAMA_DEFAULTS[key], from: 'Ollama' };
    // Auto context keeps the whole model on the GPU, so it's off when GPU
    // layers are set by hand (the server skips it too).
    const gpuSet = key === 'num_ctx' && (state.options.num_gpu != null || p?.options?.num_gpu != null);
    if (key === 'num_ctx' && defaults.auto?.fit && !gpuSet) return { value: defaults.auto.ctx || null, from: 'auto' };
    if (key === 'num_ctx' && defaults.auto?.ctx && !gpuSet) return { value: defaults.auto.ctx, from: 'auto' };
    if (key === 'num_ctx' && defaults.ollamaContext) return { value: defaults.ollamaContext, from: 'Ollama' };
    return { value: null, from: null };
  }

  const fmt = (v) => (typeof v === 'number' ? String(+v.toFixed(3)) : String(v));
  const icon = (d) => `<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">${d}</svg>`;

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
    reset.title = 'Back to the inherited value';
    reset.setAttribute('aria-label', `Reset ${def.label}`);
    ctrl.append(reset);
    row.append(text, ctrl);
    return { row, sub, ctrl, reset };
  }

  function setOption(def, v) {
    if (v === null || v === '' || Number.isNaN(v)) delete state.options[def.key];
    else state.options[def.key] = v;
    render();
  }
  const parse = (def, s) => (s === '' ? null : def.int ? Math.round(+s) : +s);
  const clamp = (def, v) => Math.min(def.max, Math.max(def.min, v));

  function spinButton(def, num, delta, label, path) {
    const b = el('button', 'spin-btn');
    b.type = 'button';
    b.tabIndex = -1;
    b.setAttribute('aria-label', label);
    b.innerHTML = icon(path);
    b.addEventListener('click', () => {
      const cur = num.value === '' ? (inherited(def.key).value ?? def.min) : +num.value;
      const v = clamp(def, +(cur + delta * def.step).toFixed(4));
      setOption(def, def.int ? Math.round(v) : v);
    });
    return b;
  }

  function buildRow(def, parent) {
    const { row, sub, ctrl, reset } = rowShell(def);
    const r = { def, row, sub, reset };

    if (def.kind === 'combo' || def.kind === 'think') {
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
      reset.addEventListener('click', () => {
        if (def.kind === 'think') { state.think = null; render(); } else setOption(def, null);
      });
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
      reset.addEventListener('click', () => setOption(def, null));
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
    rows.push(r);
    parent.append(row);
  }

  // An expander row (like GNOME's AdwExpanderRow) that hides a group of rows.
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
    return sub;
  }

  MAIN.forEach((d) => buildRow(d, $('tune-main')));
  const samplingSub = buildExpander('tune-sampling', 'Sampling', SAMPLING, $('tune-sampling-box'));
  const advancedSub = buildExpander('tune-advanced', 'More settings', ADVANCED, $('tune-advanced-box'));

  // ---- rendering ----
  function changedCount() {
    return Object.keys(state.options).length + (state.system ? 1 : 0) + (state.think !== null ? 1 : 0);
  }

  function describe(inh) {
    switch (inh.from) {
      case 'preset': return 'From preset';
      case 'model': return 'Model default';
      case 'auto': return 'Auto';
      case 'Ollama': return 'Ollama default';
      default: return 'Default';
    }
  }

  function renderCombo(r) {
    const { def, select: sel } = r;
    const inh = inherited(def.key);
    const own = state.options[def.key];
    const max = defaults.contextLength || 262144;
    const steps = CTX_STEPS.filter((c) => c <= max);
    if (own != null && !steps.includes(own)) steps.push(own);
    const first = inh.value != null ? `${describe(inh)} (${kib(inh.value)})` : describe(inh);
    const opts = [['', first], ...steps.sort((a, b) => a - b).map((c) => [String(c), kib(c)])];
    const same = sel.options.length === opts.length && [...sel.options].every((o, i) => o.value === opts[i][0] && o.text === opts[i][1]);
    if (!same) sel.replaceChildren(...opts.map(([v, t]) => new Option(t, v)));
    sel.value = own != null ? String(own) : '';
    let text = own != null ? 'Changed' : describe(inh);
    if (own == null && inh.from === 'auto') text = defaults.auto?.fit ? (inh.value ? 'Largest that fits, picked by llama.cpp' : 'llama.cpp picks the largest that fits when it loads') : 'Largest that fits on your GPU';
    else if (own == null && effective('num_gpu') != null) text = 'Not auto: GPU layers are set by hand';
    else if (own == null && inh.from === 'Ollama' && defaults.auto?.reason) text = `Not auto: ${defaults.auto.reason}`;
    if (defaults.contextLength) text += ` · max ${kib(defaults.contextLength)}`;
    r.sub.textContent = text;
    r.row.title = def.help;
    return own != null;
  }

  function renderThink(r) {
    const sel = r.select;
    if (!sel.options.length) sel.append(new Option('Default', ''), new Option('On', 'on'), new Option('Off', 'off'));
    const p = currentPreset();
    sel.options[0].text = p?.think != null ? `Preset (${p.think ? 'on' : 'off'})` : 'Default';
    sel.value = state.think === null ? '' : state.think ? 'on' : 'off';
    r.sub.textContent = defaults.thinking ? 'Reason step by step before answering' : 'This model doesn’t report thinking support';
    return state.think !== null;
  }

  function renderNumber(r) {
    const { def, num, range, sub } = r;
    const own = state.options[def.key];
    const inh = inherited(def.key);
    const has = own !== undefined;
    const val = has ? own : inh.value;
    const unlimited = def.key === 'num_predict' && val === -1;
    if (document.activeElement !== num) num.value = unlimited ? '' : val ?? '';
    if (def.key === 'num_predict') num.placeholder = 'No limit';
    if (def.key === 'num_gpu') return renderLayers(r, own, inh);
    if (range) {
      range.value = val ?? def.min;
      const pct = ((clamp(def, +range.value) - def.min) / (def.max - def.min)) * 100;
      range.style.setProperty('--pct', pct + '%');
    }
    const source = has ? (inh.value != null ? `${describe(inh)}: ${fmt(inh.value)}` : 'Changed') : describe(inh);
    sub.textContent = def.help ? `${source} · ${def.help}` : source;
    if (unlimited) sub.textContent = `${source} · Tokens`;
    return has;
  }

  // GPU layers: unset means Ollama puts as many as fit, shown as all of them.
  function renderLayers(r, own, inh) {
    const { num, range, sub } = r;
    const max = defaults.layers || 99;
    const val = own ?? inh.value;
    num.max = range.max = max;
    if (document.activeElement !== num) num.value = val ?? '';
    range.value = val ?? max;
    range.style.setProperty('--pct', (Math.min(+range.value, max) / max) * 100 + '%');
    const of = defaults.layers ? ` of ${max}` : '';
    if (val == null) sub.textContent = `Auto: all that fit${defaults.layers ? ` (${max} layers)` : ''}`;
    else if (val >= max) sub.textContent = `${own != null ? 'Changed' : describe(inh)}: all ${max} on the GPU`;
    else sub.textContent = `${own != null ? 'Changed' : describe(inh)}: ${val}${of} on the GPU, the rest on the CPU (slower) · auto context off`;
    return own !== undefined;
  }

  const effective = (key) => state.options[key] ?? inherited(key).value;

  function render() {
    const preset = currentPreset();
    for (const r of rows) {
      const changed = r.def.kind === 'combo' ? renderCombo(r) : r.def.kind === 'think' ? renderThink(r) : renderNumber(r);
      r.row.classList.toggle('changed', changed);
      r.reset.hidden = !changed;
    }
    samplingSub.textContent = SAMPLING.map((d) => `${d.short} ${fmt(effective(d.key))}`).join(' · ');
    const advChanged = ADVANCED.filter((d) => d.key in state.options).length;
    advancedSub.textContent = advChanged ? `${advChanged} changed` : 'Penalties, seed';

    if (document.activeElement !== systemEl) systemEl.value = state.system;
    systemEl.placeholder = preset?.system || defaults.system || 'None. Describe how the model should behave';

    sourceEl.textContent = preset ? `preset ${preset.name} · ${preset.base}` : baseModel();

    saveBtn.hidden = !preset;
    if (preset) saveBtn.textContent = `Save to “${preset.name}”`;
    saveBtn.disabled = !changedCount();
    resetBtn.disabled = !changedCount();
    publishRow.hidden = !preset;
    if (preset) {
      publishBox.checked = !!preset.published;
      publishName.textContent = 'ollamui/' + preset.name;
    }

    const parts = [];
    const t = effective('temperature');
    if (t != null) parts.push(`temp ${fmt(t)}`);
    const c = effective('num_ctx');
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
  try { if (localStorage.getItem('tuneOpen') || location.hash === '#tune') setOpen(true); } catch {}

  // ---- inputs ----
  systemEl.addEventListener('input', () => { state.system = systemEl.value; render(); });
  select.addEventListener('change', loadDefaults);
  // Loading or ejecting changes what auto context can pick.
  document.addEventListener('loaded-changed', loadDefaults);
  resetBtn.addEventListener('click', () => {
    state.options = {};
    state.system = '';
    state.think = null;
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
