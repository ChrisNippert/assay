// Tune panel: per-chat settings layered over the preset (if any), the
// model's own defaults and Ollama's defaults. Chat.js reads the result via
// window.ollamuiTune when sending.
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
  const thinkSel = $('tune-think');
  const systemEl = $('tune-system');
  const noteEl = $('tune-note');
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

  const SAMPLING = [
    { key: 'temperature', label: 'Temperature', min: 0, max: 2, step: 0.05 },
    { key: 'top_p', label: 'Top P', min: 0, max: 1, step: 0.01 },
    { key: 'top_k', label: 'Top K', min: 0, max: 200, step: 1, int: true },
    { key: 'min_p', label: 'Min P', min: 0, max: 1, step: 0.01 },
    { key: 'repeat_penalty', label: 'Repeat penalty', min: 0.8, max: 2, step: 0.01 },
  ];
  const LENGTH = [
    { key: 'num_ctx', label: 'Context length', min: 2048, max: 131072, step: 1024, int: true },
    { key: 'num_predict', label: 'Max reply tokens', min: -1, max: 32768, step: 64, int: true, hint: '-1 = no limit' },
    { key: 'seed', label: 'Seed', min: 0, max: 2147483647, step: 1, int: true, noSlider: true, hint: 'blank = random' },
  ];
  // Ollama's documented fallbacks, shown when neither preset nor model sets a value.
  const OLLAMA_DEFAULTS = { temperature: 0.8, top_p: 0.9, top_k: 40, min_p: 0, repeat_penalty: 1.1, num_predict: -1 };

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
    return { value: null, from: null };
  }

  const fmt = (v) => (typeof v === 'number' ? String(+v.toFixed(3)) : String(v));

  // ---- parameter rows ----
  const rows = [];
  function buildRow(def, parent) {
    const row = el('div', 'param');
    const head = el('div', 'param-head');
    const label = el('label', null, def.label);
    label.htmlFor = 'tp-' + def.key;
    const num = el('input', 'num');
    Object.assign(num, { type: 'number', id: 'tp-' + def.key, step: def.step, min: def.min, max: def.max });
    const reset = el('button', 'icon-btn reset', '↺');
    reset.type = 'button';
    reset.title = 'Back to inherited value';
    reset.setAttribute('aria-label', `Reset ${def.label}`);
    head.append(label, num, reset);
    row.append(head);
    let range = null;
    if (!def.noSlider) {
      range = el('input');
      Object.assign(range, { type: 'range', min: def.min, max: def.max, step: def.step, tabIndex: -1 });
      range.setAttribute('aria-hidden', 'true'); // the number box is the accessible control
      row.append(range);
    }
    const hint = el('div', 'param-hint');
    row.append(hint);

    const set = (v) => {
      if (v === null || v === '' || Number.isNaN(v)) delete state.options[def.key];
      else state.options[def.key] = v;
      render();
    };
    const parse = (s) => (s === '' ? null : def.int ? Math.round(+s) : +s);
    num.addEventListener('change', () => set(parse(num.value)));
    range?.addEventListener('input', () => set(parse(range.value)));
    reset.addEventListener('click', () => set(null));
    rows.push({ def, row, num, range, hint, reset });
    parent.append(row);
  }
  SAMPLING.forEach((d) => buildRow(d, $('tune-sampling')));
  LENGTH.forEach((d) => buildRow(d, $('tune-length')));

  // ---- rendering ----
  function changedCount() {
    return Object.keys(state.options).length + (state.system ? 1 : 0) + (state.think !== null ? 1 : 0);
  }

  function render() {
    const preset = currentPreset();
    for (const { def, row, num, range, hint, reset } of rows) {
      const own = state.options[def.key];
      const inh = inherited(def.key);
      const has = own !== undefined;
      const val = has ? own : inh.value;
      row.classList.toggle('changed', has);
      if (document.activeElement !== num) num.value = val ?? '';
      if (range) {
        if (def.key === 'num_ctx' && defaults.contextLength) range.max = defaults.contextLength;
        range.value = val ?? def.min;
      }
      let text = has ? (inh.value != null ? `${inh.from} value: ${fmt(inh.value)}` : 'changed') : inh.from ? `from ${inh.from}` : 'Ollama decides';
      if (def.hint) text += ` · ${def.hint}`;
      hint.textContent = text;
      reset.hidden = !has;
    }

    thinkSel.value = state.think === null ? '' : state.think ? 'on' : 'off';
    thinkSel.closest('.tune-row').classList.toggle('changed', state.think !== null);
    if (document.activeElement !== systemEl) systemEl.value = state.system;
    systemEl.placeholder = preset?.system || defaults.system || 'None';

    const base = baseModel();
    sourceEl.textContent = preset ? `preset ${preset.name} · ${preset.base}` : base;
    noteEl.textContent = defaults.thinking ? '' : 'This model doesn’t report thinking support, so the Thinking setting may be ignored or rejected.';

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
    const show = (key, label) => {
      const v = state.options[key] ?? (inherited(key).from !== 'Ollama' ? inherited(key).value : null);
      if (v != null) parts.push(`${label} ${key === 'num_ctx' ? Math.round(v / 1024) + 'k' : fmt(v)}`);
    };
    show('temperature', 'temp');
    show('num_ctx', 'ctx');
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
    if (open) panel.querySelector('input, select, textarea')?.focus({ preventScroll: true });
  }
  openBtn.addEventListener('click', () => setOpen(!panel.classList.contains('open')));
  $('tune-close').addEventListener('click', () => { setOpen(false); openBtn.focus(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && panel.classList.contains('open')) { setOpen(false); openBtn.focus(); }
    if (e.key === '.' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); setOpen(!panel.classList.contains('open')); }
  });
  try { if (localStorage.getItem('tuneOpen')) setOpen(true); } catch {}

  // ---- inputs ----
  thinkSel.addEventListener('change', () => {
    state.think = thinkSel.value === '' ? null : thinkSel.value === 'on';
    render();
  });
  systemEl.addEventListener('input', () => { state.system = systemEl.value; render(); });
  select.addEventListener('change', loadDefaults);
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
  };

  loadDefaults();
})();
