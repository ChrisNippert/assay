// First-run setup (/setup): your computer, the llama.cpp engine, a first model.
// Every install is started by its own button and says what it fetches, from
// where, how big it is and where it goes.
(() => {
  const root = document.getElementById('setup');
  if (!root) return;
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const size = (n) => (n >= 1e9 ? (n / 1e9).toFixed(1) + ' GB' : Math.max(1, Math.round(n / 1e6)) + ' MB');
  async function api(path, opts) {
    const res = await fetch(path, opts);
    const body = res.status === 204 ? {} : await res.json().catch(() => ({}));
    if (!res.ok) { const e = new Error(body.error || res.statusText); e.status = res.status; e.body = body; throw e; }
    return body;
  }
  const send = (method, path, data) => api(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data || {}) });
  const link = (href, text) => { const a = el('a', null, text); a.href = href; a.target = '_blank'; a.rel = 'noopener'; return a; };
  // A row of a boxed list: title, lines of detail (text or nodes), controls.
  function row(title, subs, ctrl) {
    const r = el('div', 'row su-row');
    const t = el('div', 'row-text');
    if (title) t.append(el('span', 'row-title', title));
    for (const s of [].concat(subs || [])) {
      const line = el('span', 'row-sub');
      if (s instanceof Node) line.append(s); else line.textContent = s;
      t.append(line);
    }
    r.append(t);
    if (ctrl) { const c = el('div', 'row-ctrl'); c.append(...ctrl); r.append(c); }
    return r;
  }
  const list = (...rows) => { const l = el('div', 'boxed-list'); l.append(...rows); return l; };
  const btn = (text, cls, onclick) => { const b = el('button', 'btn' + (cls ? ' ' + cls : ''), text); b.type = 'button'; if (onclick) b.onclick = onclick; return b; };
  const bar = () => { const b = el('div', 'lc-bar'); b.append(document.createElement('span')); return b; };
  const sourceNames = { custom: 'a llama-server you picked', own: 'your own build in the data folder', release: 'a release ollamui downloaded', flag: 'the -llama-server option', path: 'your PATH' };

  let sys = null;           // GET /api/setup
  let step = 1;
  const visited = new Set();
  const state = { chk: null, chkErr: '', sel: '', wantInstall: false, job: null, note: '', noteBad: false, timer: null, models: null, modelsErr: '', pulls: [] };

  // ---- step 1: your computer ----
  function renderSystem() {
    const s = sys.system;
    const rows = [row('Operating system', `${s.os} (${s.arch})`)];
    if (s.gpus.length) for (const g of s.gpus) rows.push(row(s.gpus.length > 1 ? 'Graphics card' : 'Graphics card', g.name, [el('span', 'muted', size(g.vram) + ' video memory')]));
    else rows.push(row('Graphics card', 'None found. Models will run on the processor, which is slower.'));
    rows.push(row('Memory (RAM)', size(s.ram)));
    rows.push(row('What that means', s.advice));
    $('su-system').replaceChildren(...rows);
  }

  // ---- step 2: engine ----
  const eng = () => sys.engine;
  function probeLines(p) {
    if (!p) return [];
    if (p.error) return ['It didn’t start: ' + p.error];
    const out = [];
    if (p.version) out.push('Version: ' + p.version);
    if (p.devices && p.devices.length) out.push('Can use: ' + p.devices.join('; '));
    else if (p.listed) out.push('It sees no graphics card, so models would run on the processor.');
    return out;
  }
  function renderEngine() {
    const host = $('su-engine');
    const e = eng();
    const parts = [];
    if (!e.managed) {
      host.replaceChildren(list(row('Not managed here', 'llama.cpp isn’t the engine in use, so there is nothing to set up.')));
      return;
    }
    if (e.found) {
      const subs = ['Found at ' + e.path + (sourceNames[e.source] ? ' (' + sourceNames[e.source] + ')' : '')].concat(probeLines(e.probe));
      const keep = el('div', 'su-found');
      keep.append(list(row('llama.cpp is already installed', subs, [
        btn('Keep this one', 'primary', () => go(3)),
        state.wantInstall ? null : btn('Choose another…', null, () => { state.wantInstall = true; renderEngine(); loadBuilds(); }),
      ].filter(Boolean))));
      parts.push(keep);
      if (!state.wantInstall) { parts.push(customBlock()); host.replaceChildren(...parts); return; }
    } else {
      parts.push(list(row('No llama-server found', 'Models can’t run until llama.cpp is installed. Choose one of the options below.')));
    }
    parts.push(installBlock());
    parts.push(customBlock());
    host.replaceChildren(...parts);
  }

  function installBlock() {
    const wrap = el('div', 'su-block');
    wrap.append(el('h3', 'su-sub', 'Download llama.cpp'));
    const e = eng();
    if (!e.supported) {
      wrap.append(list(row('No ready-made download for this system', 'llama.cpp publishes no ready-made build for ' + sys.system.os + ' on ' + sys.system.arch + '. Install llama.cpp yourself (your package manager, or build it) and then pick it below.')));
      return wrap;
    }
    const job = state.job;
    if (job && job.state === 'running') { wrap.append(progressBlock(job)); return wrap; }
    if (state.chkErr) {
      wrap.append(list(row('Couldn’t reach GitHub', state.chkErr, [btn('Try again', null, () => loadBuilds(true))])));
      return wrap;
    }
    const c = state.chk;
    if (!c) { wrap.append(list(row(null, 'Asking GitHub which builds exist…'))); return wrap; }
    if (!c.supported || !c.builds || !c.builds.length) {
      wrap.append(list(row('No ready-made download for this system', 'Install llama.cpp yourself and pick it below.')));
      return wrap;
    }
    wrap.append(el('p', 'muted small', 'llama.cpp comes in builds for different graphics hardware. The one marked Recommended suits what was found in step 1; you can choose another.'));
    const builds = el('div', 'boxed-list su-builds');
    for (const b of c.builds) {
      const r = el('label', 'row su-build');
      const radio = el('input'); radio.type = 'radio'; radio.name = 'su-build'; radio.value = b.id; radio.checked = b.id === state.sel;
      radio.onchange = () => { state.sel = b.id; renderEngine(); };
      const t = el('div', 'row-text');
      const title = el('span', 'row-title', b.label);
      if (b.id === c.recommended) title.append(el('span', 'su-badge', 'Recommended'));
      t.append(title, el('span', 'row-sub', b.note));
      r.append(radio, t, el('span', 'muted small su-size', '~' + size(b.size)));
      builds.append(r);
    }
    wrap.append(builds);
    const b = c.builds.find((x) => x.id === state.sel) || c.builds[0];
    const files = el('span', 'mono su-files', b.files.join(', '));
    const details = list(
      row('What', `llama.cpp ${c.tag}, the ${b.label} build, ${b.files.length > 1 ? b.files.length + ' files' : 'one file'}, ~${size(b.size)} to download`, null),
      row('From', [link(c.releaseUrl, 'GitHub release ' + c.tag + ' ↗'), files]),
      row('Goes to', [el('span', 'mono', eng().installTo), 'Its own folder; nothing outside it is touched.']),
      row('License', [link('https://github.com/ggml-org/llama.cpp/blob/master/LICENSE', eng().license + ' license ↗'), 'Free and open source.']),
    );
    wrap.append(details);
    const note = el('p', 'small su-note' + (state.noteBad ? ' settings-warn' : ''), state.note);
    note.hidden = !state.note;
    const go = btn(`Download and install llama.cpp (${b.label}, ~${size(b.size)})`, 'primary', () => install(b.id));
    const act = el('div', 'su-actions');
    act.append(go);
    wrap.append(act, note);
    return wrap;
  }

  function progressBlock(job) {
    const pct = job.total > 0 ? Math.floor((100 * job.completed) / job.total) : 0;
    const b = bar();
    b.firstElementChild.style.width = (job.phase === 'Downloading' ? pct : 100) + '%';
    const sub = job.phase === 'Downloading' && job.total > 0
      ? `Downloading ${size(job.completed)} of ${size(job.total)} (${pct}%)` + (job.bytesPerSec > 0 ? ` · ${(job.bytesPerSec / 1e6).toFixed(1)} MB/s` : '')
      : (job.phase || 'Working') + '…';
    const r = row('Installing llama.cpp' + (job.tag ? ' ' + job.tag : ''), [sub], [btn('Cancel', null, async () => { try { await send('POST', '/api/llamacpp/cancel'); } catch {} })]);
    r.querySelector('.row-text').append(b);
    return list(r);
  }

  function customBlock() {
    const wrap = el('div', 'su-block');
    wrap.append(el('h3', 'su-sub', 'Use a llama-server I already have'));
    const input = el('input', 'mono'); input.type = 'text'; input.id = 'su-custom'; input.placeholder = '/path/to/llama-server'; input.spellcheck = false; input.autocomplete = 'off';
    input.setAttribute('aria-label', 'Path to llama-server');
    const browse = btn('Browse…', null);
    const use = btn('Use this one', null, async () => {
      const path = input.value.trim();
      if (!path) { input.focus(); return; }
      use.disabled = true;
      try {
        await send('PUT', '/api/llamacpp/custom', { path });
        state.wantInstall = false;
        await refresh();
      } catch (err) { msg.textContent = err.message; msg.hidden = false; }
      use.disabled = false;
    });
    const msg = el('p', 'small settings-warn'); msg.hidden = true;
    const r = row('Path to llama-server', ['A file, or the folder it’s in. Its libraries are loaded from the same folder. Nothing is downloaded or copied.'], [input, browse, use]);
    wrap.append(list(r), msg);
    if (window.ollamuiPick) browse.addEventListener('click', async () => {
      browse.disabled = true;
      try { const p = await window.ollamuiPick({ kind: 'file', title: 'Choose llama-server', start: input.value.trim() }); if (p) input.value = p; } finally { browse.disabled = false; }
    });
    return wrap;
  }

  async function loadBuilds(fresh) {
    state.chkErr = ''; state.chk = null; renderEngine();
    try {
      state.chk = await api('/api/llamacpp/check' + (fresh ? '?fresh=1' : ''));
      if (!state.sel || !(state.chk.builds || []).some((b) => b.id === state.sel)) state.sel = state.chk.recommended || (state.chk.builds && state.chk.builds[0] && state.chk.builds[0].id) || '';
    } catch (err) { state.chkErr = err.message; }
    renderEngine();
  }

  async function install(build) {
    state.note = ''; state.noteBad = false;
    try {
      state.job = await send('POST', '/api/llamacpp/install', { build, force: true });
    } catch (err) { state.note = 'Couldn’t start: ' + err.message; state.noteBad = true; renderEngine(); return; }
    renderEngine(); pollEngine();
  }

  function pollEngine() {
    clearTimeout(state.timer);
    state.timer = setTimeout(async () => {
      let st;
      try { st = await api('/api/llamacpp'); } catch { return pollEngine(); }
      state.job = st.job || null;
      if (st.job && st.job.state === 'running') { renderEngine(); return pollEngine(); }
      if (st.job && st.job.state === 'done') { state.wantInstall = false; state.job = null; await refresh(); return; }
      if (st.job && (st.job.state === 'failed' || st.job.state === 'canceled')) {
        state.note = st.job.state === 'canceled' ? 'The download was canceled. Nothing was changed.' : 'The install failed: ' + st.job.error + ' Nothing was changed.';
        state.noteBad = st.job.state === 'failed';
        state.job = null;
      }
      renderEngine();
    }, 600);
  }

  // ---- step 3: a first model ----
  async function loadModels(fresh) {
    state.modelsErr = ''; state.models = null; renderModels();
    try { state.models = await api('/api/setup/models' + (fresh ? '?fresh=1' : '')); } catch (err) { state.modelsErr = err.message; }
    renderModels();
    pollPulls();
  }
  const pullFor = (m) => state.pulls.find((j) => j.model === m.model && (j.state === 'running' || j.state === 'done' || Date.now() - Date.parse(j.finished || 0) < 5 * 60e3));
  function renderModels() {
    const host = $('su-models');
    if (state.modelsErr) { host.replaceChildren(list(row('Couldn’t get suggestions', state.modelsErr, [btn('Try again', null, () => loadModels(true))]))); return; }
    if (!state.models) { host.replaceChildren(list(row(null, 'Checking Hugging Face for models that fit…'))); return; }
    const dir = state.models.modelsDir;
    const rows = [];
    const noEngine = !eng().found && eng().managed;
    if (noEngine) rows.push(row('No engine yet', 'You can download a model now, but you need llama.cpp (step 2) before it can answer.', [btn('Back to the engine', null, () => go(2))]));
    for (const m of state.models.models) {
      const j = pullFor(m);
      const ctrl = [];
      const r = row(`${m.name} · ${m.quant}`, [
        m.why,
        `${size(m.size)} download · ${m.fit.level === 'gpu' ? 'fits on your graphics card' : m.fit.level === 'cpu' ? 'fits in your RAM' : m.fit.label.toLowerCase()}`,
        (() => { const s = el('span'); s.append('From ', link(m.url, 'huggingface.co/' + m.repo + ' ↗'), dir ? ' → ' : '', dir ? el('span', 'mono', dir) : ''); return s; })(),
      ], ctrl);
      if (j && j.state === 'running') {
        const pct = j.total > 0 ? Math.floor((100 * j.completed) / j.total) : 0;
        const b = bar(); b.firstElementChild.style.width = pct + '%';
        r.querySelector('.row-text').append(el('span', 'row-sub', j.total > 0 ? `Downloading ${size(j.completed)} of ${size(j.total)} (${pct}%)` + (j.bytesPerSec > 0 ? ` · ${(j.bytesPerSec / 1e6).toFixed(1)} MB/s` : '') : (j.status || 'Starting') + '…'), b);
        ctrl.push(btn('Cancel', null, async () => { try { await send('POST', `/api/pulls/${encodeURIComponent(j.id)}/cancel`); } catch {} pollPulls(); }));
      } else if (j && j.state === 'done') {
        ctrl.push(el('span', 'su-ok', 'Installed'));
      } else {
        const failed = j && (j.state === 'failed' || j.state === 'canceled');
        if (failed) r.querySelector('.row-text').append(el('span', 'row-sub settings-warn', j.state === 'canceled' ? 'Canceled. Partial data is kept so a retry resumes.' : 'Failed: ' + j.error));
        ctrl.push(btn((failed ? 'Retry ' : 'Download ') + size(m.size), 'primary', async (ev) => {
          ev.target.disabled = true;
          try { await send('POST', '/api/pulls', { model: m.model, alias: m.alias }); } catch (err) { alert('Couldn’t start the download: ' + err.message); }
          pollPulls();
        }));
      }
      r.querySelector('.row-ctrl').append(...ctrl);
      rows.push(r);
    }
    host.replaceChildren(list(...rows));
  }
  function pollPulls() {
    clearTimeout(state.pullTimer);
    state.pullTimer = setTimeout(async () => {
      try { state.pulls = await api('/api/pulls'); } catch { state.pulls = []; }
      if (step === 3) renderModels();
      if (state.pulls.some((j) => j.state === 'running')) pollPulls();
    }, 800);
  }

  // ---- step 4: done ----
  async function renderDone() {
    await refresh(false);
    const e = eng();
    const rows = [];
    if (!e.managed) rows.push(row('Engine', 'llama.cpp isn’t the engine in use.'));
    else if (e.found) rows.push(row('Engine: llama.cpp', ['In use: ' + e.path + (sourceNames[e.source] ? ' (' + sourceNames[e.source] + ')' : '')].concat(probeLines(e.probe))));
    else rows.push(row('Engine: llama.cpp', 'Not installed yet. Models can’t answer until it is; run setup again from Settings, or install it in Settings → Models.', [btn('Set it up', null, () => go(2))]));
    const names = sys.models || [];
    if (names.length) rows.push(row('Models', [names.join(', '), 'Stored in ' + (e.modelsDir || 'the models folder')]));
    else rows.push(row('Models', 'None yet. Find one in Discover, or go back and pick a suggestion.', [btn('Pick one', null, () => go(3)), (() => { const a = el('a', 'btn', 'Discover'); a.href = '/discover'; return a; })()]));
    $('su-summary').replaceChildren(...rows);
    $('su-done-lead').textContent = e.found && names.length ? 'You’re ready. Here is what is on this computer now.' : 'Here is what is on this computer now.';
  }

  // ---- navigation ----
  function go(n) {
    step = Math.min(4, Math.max(1, n));
    visited.add(step);
    root.querySelectorAll('.su-panel').forEach((p) => { p.hidden = Number(p.dataset.step) !== step; });
    root.querySelectorAll('.su-step').forEach((b) => {
      const k = Number(b.dataset.go);
      b.classList.toggle('current', k === step);
      b.classList.toggle('seen', k !== step && visited.has(k));
      b.setAttribute('aria-current', k === step ? 'step' : 'false');
    });
    $('su-back').hidden = step === 1;
    $('su-skip').hidden = step === 1 || step === 4;
    const next = $('su-next');
    next.textContent = step === 4 ? 'Start chatting' : step === 1 ? 'Next' : 'Next';
    if (step === 2) {
      renderEngine();
      if (eng().managed && eng().supported && (!eng().found || state.wantInstall) && !state.chk && !state.chkErr) loadBuilds();
    }
    if (step === 3 && !state.models && !state.modelsErr) loadModels();
    if (step === 3) { renderModels(); pollPulls(); }
    if (step === 4) renderDone();
    window.scrollTo({ top: 0 });
  }

  async function finish(to) {
    try { await send('POST', '/api/setup/done', { done: true }); } catch {}
    location.href = to;
  }

  async function refresh(render = true) {
    sys = await api('/api/setup');
    if (render) { renderSystem(); if (step === 2) renderEngine(); }
  }

  $('su-back').onclick = () => go(step - 1);
  $('su-skip').onclick = () => go(step + 1);
  $('su-next').onclick = () => (step === 4 ? finish('/chat') : go(step + 1));
  $('su-exit').onclick = () => finish('/chat');
  root.querySelectorAll('.su-step').forEach((b) => { b.onclick = () => go(Number(b.dataset.go)); });
  root.querySelectorAll('[data-skip-setup]').forEach((a) => a.addEventListener('click', (ev) => { ev.preventDefault(); finish(a.getAttribute('href')); }));

  (async () => {
    try { await refresh(); } catch (err) { $('su-system').replaceChildren(row('Couldn’t read this computer', err.message)); return; }
    // An install already under way (the page was reloaded) is picked up.
    try {
      const st = await api('/api/llamacpp');
      if (st.job && st.job.state === 'running') { state.job = st.job; pollEngine(); }
    } catch {}
    const want = Number(new URLSearchParams(location.search).get('step'));
    go(want >= 1 && want <= 4 ? want : 1);
  })();
})();
