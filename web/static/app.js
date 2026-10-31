// Phone bottom sheets (model picker, chat options, memory, Tune): one place
// for the page lock, the dimmed backdrop and drag-down-to-close. Each sheet
// opens and closes itself; it tells this file with track(), and registers how
// to close it (register) for the backdrop, the back button and the drag.
window.ollamuiSheets = (() => {
  const root = document.documentElement;
  const phone = matchMedia('(max-width: 760px)');
  const active = new Set();
  const closers = new WeakMap();
  const backdrop = document.createElement('div');
  backdrop.className = 'sheet-backdrop';
  document.body.append(backdrop);
  const sync = () => root.classList.toggle('sheet-open', active.size > 0);
  const closeOne = (s) => (closers.get(s) || (() => close(s)))();
  const closeAll = () => [...active].forEach(closeOne);
  const track = (s, on) => {
    if (on && !phone.matches) return;
    if (on) active.add(s); else active.delete(s);
    sync();
  };
  const register = (s, fn) => closers.set(s, fn);
  // For sheets that are only [hidden] plus .open (no component of their own).
  const open = (s) => {
    clearTimeout(s._hideT);
    s.hidden = false;
    s.inert = false;
    void s.offsetWidth;
    s.classList.add('open');
    track(s, true);
  };
  const close = (s) => {
    s.classList.remove('open');
    track(s, false);
    clearTimeout(s._hideT);
    s._hideT = setTimeout(() => { if (!s.classList.contains('open')) s.hidden = true; }, 280);
  };
  backdrop.addEventListener('click', closeAll);
  // The app bar sits above the backdrop while a sheet opens from it; its own
  // dimming layer is the bar itself.
  document.addEventListener('click', (e) => {
    if (e.target.classList?.contains('appbar') && active.size) closeAll();
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && active.size) closeAll(); });

  // Drag the grab handle down to dismiss; a short drag springs back.
  for (const grab of document.querySelectorAll('.sheet-grab')) {
    const sheet = grab.parentElement;
    let y0 = null, dy = 0, t0 = 0;
    grab.addEventListener('pointerdown', (e) => {
      y0 = e.clientY; dy = 0; t0 = Date.now();
      grab.setPointerCapture(e.pointerId);
      sheet.style.transition = 'none';
    });
    grab.addEventListener('pointermove', (e) => {
      if (y0 == null) return;
      dy = Math.max(0, e.clientY - y0);
      sheet.style.transform = `translateY(${dy}px)`;
    });
    const end = () => {
      if (y0 == null) return;
      const fast = dy / Math.max(1, Date.now() - t0) > 0.6;
      y0 = null;
      sheet.style.transition = '';
      sheet.style.transform = '';
      if (dy > 90 || (fast && dy > 24)) closeOne(sheet);
    };
    grab.addEventListener('pointerup', end);
    grab.addEventListener('pointercancel', end);
  }
  return { track, register, open, close, closeAll, phone };
})();

// Shared behavior: theme toggle, load/eject/delete buttons, the "In memory" list.
(() => {
  const root = document.documentElement;

  // ---- theme ----
  const themeBtn = document.getElementById('theme-toggle');
  const currentTheme = () =>
    root.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const syncThemeLabel = () => {
    if (themeBtn) themeBtn.querySelector('span').textContent = currentTheme() === 'dark' ? 'Light mode' : 'Dark mode';
  };
  themeBtn?.addEventListener('click', () => {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    try { localStorage.setItem('theme', next); } catch {}
    syncThemeLabel();
  });
  syncThemeLabel();

  // ---- helpers ----
  async function errorText(res) {
    try { return (await res.json()).error || res.statusText; } catch { return res.statusText; }
  }

  async function post(path, body) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    });
    if (!res.ok) throw new Error(await errorText(res));
    return res.status === 204 ? null : res.json();
  }

  function fmtBytes(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + ' GB';
    if (n >= 1e6) return Math.round(n / 1e6) + ' MB';
    return Math.round(n / 1e3) + ' KB';
  }

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  const ejectIcon = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5l7 9H5z"/><path d="M5 19h14"/></svg>';

  // ---- loaded models + live GPU view ----
  // One chip in the app bar (VRAM meter + summary); clicking it opens a
  // popover with the memory bars, a GPU sparkline and one row per model.
  const memWrap = document.getElementById('mem-wrap');
  const memBtn = document.getElementById('mem-btn');
  const memPop = document.getElementById('mem-pop');
  const memRows = document.getElementById('loaded-list');
  let memAll = [];        // last /api/loaded result
  let gpu = null;         // last /api/gpu result (null = unknown)
  const busyHist = [];    // GPU busy %, one per sample, newest last
  const segColors = ['var(--accent)', '#e5a50a', '#33b27a', '#c061cb', '#e66100', '#2ec7c9'];

  const shortName = (n) => n.replace(/^hf\.co\//, '').split('/').pop().replace(/:.*$/, '');
  function fmtIdle(ms) {
    const s = Math.max(0, Math.round(ms / 1000));
    return s >= 3600 ? Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm'
      : s >= 60 ? Math.floor(s / 60) + 'm' : s + 's';
  }
  function modelSub(m) {
    if (m.loading) return 'Loading…';
    const cpu = m.size > 0 && m.sizeVram === 0;
    const pct = m.size ? Math.floor((m.sizeVram * 100) / m.size) : 100;
    const parts = [fmtBytes(cpu ? m.size : m.sizeVram)];
    parts.push(cpu ? 'CPU only' : pct < 100 ? pct + '% on GPU' : '100% on GPU');
    if (m.context) parts.push(Math.round(m.context / 1024) + 'k context');
    if (m.pinned) parts.push('Stays loaded');
    else if (m.expiresAt) parts.push('Unloads in ' + fmtIdle(new Date(m.expiresAt) - Date.now()));
    return parts.join(' · ');
  }

  function renderChip() {
    if (!memBtn) return;
    const loading = memAll.find((m) => m.loading);
    const label = memBtn.querySelector('.mem-label');
    label.textContent = loading ? 'Loading ' + shortName(loading.name) + '…'
      : !memAll.length ? 'No model loaded'
      : memAll.length === 1 ? shortName(memAll[0].name) : memAll.length + ' models';
    memBtn.classList.toggle('loading', !!loading);
    memBtn.classList.toggle('empty', !memAll.length);
    const meter = memBtn.querySelector('.mem-meter');
    const fill = meter.firstElementChild;
    meter.hidden = !(gpu && gpu.vramTotal > 0);
    if (!meter.hidden) {
      const p = Math.min(100, (gpu.vramUsed * 100) / gpu.vramTotal);
      fill.style.width = p + '%';
      meter.classList.toggle('hot', p >= 90);
    }
    memBtn.title = gpu && gpu.vramTotal > 0 ? `VRAM ${fmtBytes(gpu.vramUsed)} of ${fmtBytes(gpu.vramTotal)}` : 'Models in memory';
  }

  function renderRows() {
    if (!memRows) return;
    const have = new Map();
    for (const r of memRows.querySelectorAll('.mem-row:not(.leaving)')) have.set(r.dataset.name, r);
    memRows.querySelector('.mem-none')?.remove();
    for (const m of memAll) {
      let row = have.get(m.name);
      if (!row) {
        row = el('div', 'mem-row enter');
        row.dataset.name = m.name;
        const main = el('div', 'mem-row-main');
        const link = el('a', 'row-link mem-name', m.name);
        link.href = '/chat?model=' + encodeURIComponent(m.name);
        link.title = 'Chat with ' + m.name;
        main.append(link, el('div', 'mem-sub'));
        const btn = el('button', 'icon-btn mem-eject');
        btn.type = 'button';
        btn.dataset.eject = m.name;
        btn.setAttribute('aria-label', 'Eject ' + m.name);
        btn.innerHTML = ejectIcon;
        const dot = el('span', 'mem-dot');
        dot.setAttribute('aria-hidden', 'true');
        row.append(dot, main, btn);
        memRows.append(row);
        requestAnimationFrame(() => requestAnimationFrame(() => row.classList.remove('enter')));
      }
      const cpu = m.size > 0 && m.sizeVram === 0;
      const pct = m.size ? Math.floor((m.sizeVram * 100) / m.size) : 100;
      row.classList.toggle('loading', !!m.loading);
      row.classList.toggle('partial', !m.loading && !cpu && pct < 100);
      row.classList.toggle('cpu', !m.loading && cpu);
      const ejecting = row.classList.contains('ejecting');
      row.querySelector('.mem-eject').title = (m.loading ? 'Stop loading ' : 'Eject ') + m.name;
      if (!ejecting) row.querySelector('.mem-sub').textContent = modelSub(m);
      let bar = row.querySelector('.mem-load');
      if (m.loading && !bar) {
        bar = el('div', 'mem-load bar');
        bar.append(el('span'));
        row.querySelector('.mem-row-main').append(bar);
      } else if (!m.loading && bar) bar.remove();
      have.delete(m.name);
    }
    for (const row of have.values()) {
      row.classList.add('leaving');
      setTimeout(() => row.remove(), 220);
    }
    if (!memAll.length && !memRows.querySelector('.mem-row')) memRows.append(el('div', 'mem-none', 'No model loaded'));
    else if (!memAll.length) setTimeout(() => {
      if (!memAll.length && !memRows.querySelector('.mem-row') && !memRows.querySelector('.mem-none')) memRows.append(el('div', 'mem-none', 'No model loaded'));
    }, 240);
  }

  const svgNS = 'http://www.w3.org/2000/svg';
  function renderGpu() {
    const box = document.getElementById('mem-gpu');
    if (!box) return;
    const hasV = gpu && gpu.vramTotal > 0;
    const hasR = gpu && gpu.ramTotal > 0;
    const hasB = gpu && gpu.gpuBusy >= 0 && hasV;
    box.hidden = !(hasV || hasR);
    if (box.hidden) return;
    const set = (id, fn) => { const n = document.getElementById(id); if (n) fn(n); };
    set('mem-vram', (n) => { n.hidden = !hasV; });
    set('mem-gpubusy', (n) => { n.hidden = !hasB; });
    set('mem-ram', (n) => { n.hidden = !hasR; });
    if (hasV) {
      set('mem-vram-text', (n) => { n.textContent = `${fmtBytes(gpu.vramUsed)} / ${fmtBytes(gpu.vramTotal)}`; });
      // Each loaded model's share of VRAM gets its own colour; what's left
      // of the used amount belongs to something else.
      const bar = document.getElementById('mem-vram-bar');
      const segs = memAll.filter((m) => m.sizeVram > 0).map((m) => ({ key: m.name, v: m.sizeVram }));
      const known = segs.reduce((a, s) => a + s.v, 0);
      const other = Math.max(0, gpu.vramUsed - known);
      const scale = known > gpu.vramUsed ? gpu.vramTotal * known / gpu.vramUsed : gpu.vramTotal;
      if (other > 0) segs.push({ key: '', v: other });
      const have = new Map([...bar.children].map((c) => [c.dataset.key, c]));
      segs.forEach((s, i) => {
        let c = have.get(s.key);
        if (!c) { c = el('span', 'seg'); c.dataset.key = s.key; bar.append(c); }
        have.delete(s.key);
        c.style.width = Math.min(100, (s.v * 100) / scale) + '%';
        c.style.background = s.key ? segColors[i % segColors.length] : 'var(--muted)';
        c.title = (s.key || 'Other apps') + ' · ' + fmtBytes(s.v);
      });
      for (const c of have.values()) c.remove();
    }
    // With several cards, a small bar for each under the total.
    set('mem-cards', (n) => {
      const cards = hasV && gpu.cards && gpu.cards.length > 1 ? gpu.cards : [];
      n.hidden = !cards.length;
      const rowsOld = [...n.children];
      cards.forEach((c, i) => {
        let row = rowsOld[i];
        if (!row) {
          row = el('div', 'mem-card');
          row.append(el('div', 'mem-card-head'), el('div', 'bar'));
          row.lastChild.append(el('span'));
          n.append(row);
        }
        const [head, bar] = row.children;
        const pct = c.total > 0 ? Math.min(100, (c.used * 100) / c.total) : 0;
        head.replaceChildren(el('span', '', c.name), el('span', '', `${fmtBytes(c.used)} / ${fmtBytes(c.total)}`));
        bar.firstChild.style.width = pct + '%';
        row.title = `GPU ${i}: ${c.name}${c.busy >= 0 ? ' · ' + c.busy + '% busy' : ''}`;
      });
      rowsOld.slice(cards.length).forEach((r) => r.remove());
    });
    if (hasB) {
      set('mem-gpu-text', (n) => { n.textContent = gpu.gpuBusy + '%' + (gpu.cards && gpu.cards.length > 1 ? ' (busiest)' : ''); });
      const svg = document.getElementById('mem-spark');
      const pts = busyHist.map((v, i) => `${((60 - busyHist.length + i) * 120) / 59},${26 - (v * 24) / 100}`);
      svg.replaceChildren();
      if (pts.length > 1) {
        const area = document.createElementNS(svgNS, 'polygon');
        area.setAttribute('points', `${pts[0].split(',')[0]},27 ${pts.join(' ')} ${pts[pts.length - 1].split(',')[0]},27`);
        area.setAttribute('class', 'spark-area');
        const line = document.createElementNS(svgNS, 'polyline');
        line.setAttribute('points', pts.join(' '));
        line.setAttribute('class', 'spark-line');
        svg.append(area, line);
      }
    }
    if (hasR) {
      set('mem-ram-text', (n) => { n.textContent = `${fmtBytes(gpu.ramUsed)} / ${fmtBytes(gpu.ramTotal)}`; });
      document.getElementById('mem-ram-bar').firstElementChild.style.width = Math.min(100, (gpu.ramUsed * 100) / gpu.ramTotal) + '%';
    }
  }

  function renderLoaded(models) {
    memAll = models;
    renderChip();
    renderRows();
    renderGpu();
  }

  // Opening and closing fade and slide; [hidden] is set once the exit ends.
  let memCloseT = null;
  function setMemOpen(open) {
    if (!memPop) return;
    clearTimeout(memCloseT);
    memBtn.setAttribute('aria-expanded', String(open));
    if (open) {
      memPop.hidden = false;
      void memPop.offsetWidth;
      memPop.classList.add('open');
      window.ollamuiSheets.track(memPop, true);
      sampleGpu();
      scheduleSample();
    } else {
      memPop.classList.remove('open');
      window.ollamuiSheets.track(memPop, false);
      memCloseT = setTimeout(() => { memPop.hidden = true; }, 280);
      scheduleSample();
    }
  }
  if (memBtn) {
    window.ollamuiSheets.register(memPop, () => setMemOpen(false));
    memBtn.addEventListener('click', () => setMemOpen(!memPop.classList.contains('open')));
    document.addEventListener('pointerdown', (e) => {
      if (memPop.classList.contains('open') && !memWrap.contains(e.target)) setMemOpen(false);
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && memPop.classList.contains('open')) { setMemOpen(false); memBtn.focus(); }
    });
  }

  // GPU sampling: every second while the popover is open, every 5s
  // otherwise, never while the tab is hidden.
  let sampleT = null;
  async function sampleGpu() {
    try {
      const res = await fetch('/api/gpu');
      if (res.ok) {
        gpu = await res.json();
        if (gpu.gpuBusy >= 0) { busyHist.push(gpu.gpuBusy); if (busyHist.length > 60) busyHist.shift(); }
        if (!(gpu.vramTotal > 0) && !(gpu.ramTotal > 0)) gpu = null;
      }
    } catch { /* keep the last reading */ }
    renderChip();
    renderGpu();
    if (memPop && memPop.classList.contains('open')) renderRows(); // idle countdowns
  }
  function scheduleSample() {
    clearTimeout(sampleT);
    if (document.hidden) return;
    const open = memPop && memPop.classList.contains('open');
    sampleT = setTimeout(async () => {
      await sampleGpu();
      refreshLoaded(true); // models can come and go on their own
      scheduleSample();
    }, open ? 1000 : 5000);
  }
  if (memBtn) {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) clearTimeout(sampleT);
      else { sampleGpu(); refreshLoaded(true); scheduleSample(); }
    });
    sampleGpu();
    scheduleSample();
  }

  // While a model loads, the list is polled until it's done; listeners
  // hear only about loaded models, and during polling only on a change.
  let loadPoll = null;
  let lastLoaded = null;
  async function refreshLoaded(polling) {
    clearTimeout(loadPoll);
    try {
      const res = await fetch('/api/loaded');
      if (!res.ok) return null;
      const all = await res.json();
      renderLoaded(all);
      const models = all.filter((m) => !m.loading);
      const key = models.map((m) => m.name).join(' ');
      if (polling !== true || key !== lastLoaded) {
        document.dispatchEvent(new CustomEvent('loaded-changed', { detail: models }));
      }
      lastLoaded = key;
      if (all.some((m) => m.loading)) loadPoll = setTimeout(() => refreshLoaded(true), 1500);
      return models;
    } catch {
      return null;
    }
  }
  // A page opened while a model loads picks it up (the page's own list was
  // rendered by the server without it).
  if (document.getElementById('loaded-list')) refreshLoaded(true);

  // Makes a text button act as Load or Eject for a model.
  function setLoadButton(btn, name, loaded) {
    if (!btn) return;
    delete btn.dataset.load;
    delete btn.dataset.eject;
    if (loaded) btn.dataset.eject = name; else btn.dataset.load = name;
    btn.textContent = loaded ? 'Eject' : 'Load';
    btn.disabled = false;
  }

  // ---- starring: one model is the default for new chats ----
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-star]');
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    const name = btn.getAttribute('aria-pressed') === 'true' ? '' : btn.dataset.star;
    try {
      await post('/api/models/star', { name });
    } catch (err) {
      alert(`Couldn't star ${btn.dataset.star}: ${err.message}`);
      return;
    }
    for (const b of document.querySelectorAll('[data-star]')) {
      const on = !!name && b.dataset.star === name;
      b.setAttribute('aria-pressed', String(on));
      b.title = on ? 'Default model for new chats. Click to unstar' : 'Star: use as the default model when none is loaded';
    }
    document.dispatchEvent(new CustomEvent('star-changed', { detail: name }));
  }, true);

  // ---- click handling for any [data-load], [data-eject], [data-delete] ----
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-load], [data-eject], [data-delete]');
    if (!btn || btn.disabled) return;

    if (btn.dataset.delete) {
      const name = btn.dataset.delete;
      const all = (btn.closest('[data-names]')?.dataset.names || name).split(' ');
      const label = all.length > 1 ? `${name}\n(also ${all.slice(1).join(', ')})` : name;
      if (!confirm(`Delete ${label} from disk?`)) return;
      btn.disabled = true;
      try {
        await post('/api/models/delete', { names: all });
        btn.closest('[data-names]')?.remove();
        refreshLoaded();
      } catch (err) {
        alert(`Couldn't delete ${name}: ${err.message}`);
        btn.disabled = false;
      }
      return;
    }

    const loading = !!btn.dataset.load;
    const name = btn.dataset.load || btn.dataset.eject;
    const isText = !btn.classList.contains('icon-btn') && !btn.classList.contains('keep-text');
    btn.disabled = true;
    if (isText) btn.textContent = loading ? 'Loading…' : 'Ejecting…';
    // Ejecting can take a few seconds (the chat's cache is saved first).
    const pill = btn.closest('.mem-row');
    if (pill && !loading) {
      pill.classList.add('ejecting');
      pill.title = 'Ejecting…';
      const detail = pill.querySelector('.mem-sub');
      if (detail) detail.textContent = 'Ejecting…';
    }
    // Loads show in the memory list while they run.
    const poll = loading && setTimeout(() => refreshLoaded(true), 400);
    try {
      await post(loading ? '/api/models/load' : '/api/models/eject', { name });
    } catch (err) {
      clearTimeout(poll);
      if (/ejected while loading/.test(err.message)) { btn.disabled = false; await refreshLoaded(); return; }
      alert(`Couldn't ${loading ? 'load' : 'eject'} ${name}: ${err.message}`);
    }
    btn.disabled = false;
    await refreshLoaded();
  });

  // Library rows follow load/eject changes made anywhere on the page.
  document.addEventListener('loaded-changed', (e) => {
    const names = new Set(e.detail.map((m) => m.name));
    for (const row of document.querySelectorAll('.lib-row[data-model]')) {
      if (!row.querySelector('.status')) continue;
      const rowNames = (row.dataset.names || row.dataset.model).split(' ');
      const loadedAs = rowNames.find((n) => names.has(n));
      const loaded = !!loadedAs;
      const name = loadedAs || row.dataset.model;
      const status = row.querySelector('.status');
      if (loaded) {
        const sb = el('button', 'status-btn tag good', '● Loaded');
        sb.type = 'button';
        sb.title = 'Loaded. Click to eject';
        sb.dataset.eject = name;
        status.replaceChildren(sb);
      } else {
        status.replaceChildren();
      }
      setLoadButton(row.querySelector('.actions [data-load], .actions [data-eject]'), name, loaded);
    }
  });

  // ---- downloads ----
  let pollTimer = null;
  const seenDone = new Set();

  function jobLabel(j) {
    if (j.model.startsWith('mtp:')) return 'MTP head · ' + j.model.slice(4).replace(/^hf\.co\//, '');
    return j.alias || j.model.replace(/^hf\.co\//, '');
  }

  // "hf.co/owner/repo:quant" -> "owner/repo" ('' for anything else)
  function hfRepo(model) {
    const m = /^hf\.co\/([^/:]+\/[^/:]+)(?::.*)?$/.exec(model || '');
    return m ? m[1] : '';
  }

  const dlWrap = document.getElementById('downloads');
  const dlBtn = document.getElementById('dl-btn');
  const dlPop = document.getElementById('dl-pop');
  let dlAutoClose = null;
  let knownJobs = null; // ids seen so far; null until the first poll

  function setDlOpen(open) {
    if (!dlPop) return;
    clearTimeout(dlAutoClose);
    dlPop.hidden = !open;
    dlBtn.setAttribute('aria-expanded', String(open));
  }
  if (dlBtn) {
    dlBtn.addEventListener('click', () => setDlOpen(dlPop.hidden));
    document.addEventListener('pointerdown', (e) => {
      if (!dlPop.hidden && !dlWrap.contains(e.target)) setDlOpen(false);
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !dlPop.hidden) { setDlOpen(false); dlBtn.focus(); }
    });
    dlPop.addEventListener('pointerdown', () => clearTimeout(dlAutoClose));
  }

  // Briefly shows a newly started download.
  function announceDownload() {
    dlBtn.classList.remove('pulse');
    void dlBtn.offsetWidth;
    dlBtn.classList.add('pulse');
    setDlOpen(true);
    dlAutoClose = setTimeout(() => setDlOpen(false), 4000);
  }

  function renderDownloads(jobs) {
    const list = document.getElementById('download-list');
    if (!dlWrap || !list) return;
    dlWrap.hidden = jobs.length === 0;
    if (jobs.length === 0) setDlOpen(false);
    const running = jobs.filter((j) => j.state === 'running');
    const count = document.getElementById('dl-count');
    count.hidden = jobs.length < 2;
    count.textContent = jobs.length;
    dlWrap.classList.toggle('idle', running.length === 0);
    const total = running.reduce((n, j) => n + j.total, 0);
    const done = running.reduce((n, j) => n + j.completed, 0);
    document.querySelector('#dl-mini > span').style.width = (total ? Math.floor((done * 100) / total) : 0) + '%';
    dlBtn.title = running.length ? `${running.length} downloading` : 'Downloads';
    list.replaceChildren();
    for (const j of jobs) {
      const row = el('div', 'dl-row dl-' + j.state);
      const name = el('div', 'dl-name');
      const repo = hfRepo(j.model);
      if (repo) {
        const a = el('a', 'row-link', jobLabel(j));
        a.href = '/discover/' + repo.split('/').map(encodeURIComponent).join('/');
        a.title = j.model + ' · open on Discover';
        name.append(a);
      } else {
        const a = el('a', 'row-link', jobLabel(j));
        a.href = '/library';
        a.title = j.model + ' · open Library';
        name.append(a);
      }
      row.append(name);
      if (j.state === 'running') {
        const pct = j.total ? Math.floor((j.completed * 100) / j.total) : 0;
        const bar = el('div', 'bar');
        const fill = el('span');
        fill.style.width = pct + '%';
        bar.append(fill);
        const detail = j.total
          ? `${fmtBytes(j.completed)} of ${fmtBytes(j.total)}${j.bytesPerSec > 0 ? ` · ${fmtBytes(j.bytesPerSec)}/s` : ''}`
          : j.status;
        const cancel = el('button', 'link-btn small', 'Cancel');
        cancel.type = 'button';
        cancel.dataset.cancelPull = j.id;
        const foot = el('div', 'dl-foot');
        foot.append(el('div', 'dl-detail', detail), cancel);
        row.append(bar, foot);
      } else if (j.state === 'done') {
        const chat = el('a', 'row-link small', 'Chat');
        chat.href = '/chat?model=' + encodeURIComponent(j.alias || j.model);
        const foot = el('div', 'dl-foot');
        foot.append(el('div', 'dl-detail good', j.warning || 'Downloaded'), chat);
        row.append(foot);
      } else if (j.state === 'failed') {
        row.append(el('div', 'dl-detail bad', j.error));
      } else {
        row.append(el('div', 'dl-detail', 'Canceled · start again to resume'));
      }
      list.append(row);
    }
    // A job we haven't seen before (after the first poll) means a download just started.
    const ids = new Set(jobs.map((j) => j.id));
    if (knownJobs && jobs.some((j) => j.state === 'running' && !knownJobs.has(j.id))) announceDownload();
    knownJobs = ids;
  }

  // Keeps quant rows on a model page in step with their downloads.
  function updatePullButtons(jobs) {
    for (const j of jobs) {
      const cell = document.querySelector(`.quant-row[data-tag="${CSS.escape(j.model)}"] .quant-action`);
      if (!cell) continue;
      if (j.state === 'running') {
        const pct = j.total ? Math.floor((j.completed * 100) / j.total) + '%' : '…';
        cell.replaceChildren(el('span', 'small muted', `Downloading ${pct}`));
      } else if (j.state === 'done') {
        const a = el('a', 'btn', 'Installed · Chat');
        a.href = '/chat?model=' + encodeURIComponent(j.alias || j.model);
        cell.replaceChildren(a);
      } else {
        const btn = el('button', 'btn', j.state === 'failed' ? 'Retry' : 'Resume');
        btn.type = 'button';
        btn.dataset.pull = j.model;
        if (j.alias) btn.dataset.alias = j.alias;
        cell.replaceChildren(btn);
      }
    }
  }

  async function pollDownloads() {
    clearTimeout(pollTimer);
    let jobs = [];
    try {
      const res = await fetch('/api/pulls');
      if (res.ok) jobs = await res.json();
    } catch {}
    renderDownloads(jobs);
    updatePullButtons(jobs);
    for (const j of jobs) {
      if (j.state === 'done' && !seenDone.has(j.id)) {
        seenDone.add(j.id);
        document.dispatchEvent(new CustomEvent('models-changed'));
      }
    }
    if (jobs.some((j) => j.state === 'running')) pollTimer = setTimeout(pollDownloads, 1000);
  }

  document.addEventListener('click', async (e) => {
    const pull = e.target.closest('[data-pull]');
    if (pull && !pull.disabled) {
      const addAlias = document.getElementById('add-alias');
      const alias = addAlias && !addAlias.checked ? '' : pull.dataset.alias || '';
      pull.disabled = true;
      pull.textContent = 'Starting…';
      try {
        await post('/api/pulls', { model: pull.dataset.pull, alias });
      } catch (err) {
        alert(`Couldn't start the download: ${err.message}`);
        pull.disabled = false;
        pull.textContent = 'Download';
      }
      pollDownloads();
      return;
    }
    const cancel = e.target.closest('[data-cancel-pull]');
    if (cancel) {
      cancel.disabled = true;
      try { await post(`/api/pulls/${encodeURIComponent(cancel.dataset.cancelPull)}/cancel`); } catch {}
      pollDownloads();
    }
  });
  // Already-running downloads (e.g. started on another page) show up right away.
  pollDownloads();

  // ---- presets page ----
  document.addEventListener('click', async (e) => {
    const del = e.target.closest('[data-delete-preset]');
    if (!del) return;
    const name = del.dataset.deletePreset;
    if (!confirm(`Delete the preset “${name}”? Apps that use it through the API lose it too.`)) return;
    const res = await fetch('/api/presets/' + encodeURIComponent(name), { method: 'DELETE' });
    if (res.ok) del.closest('tr')?.remove();
    else alert(`Couldn't delete: ${await errorText(res)}`);
  });

  // ---- sidebar chats ----
  const trashIcon = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>';
  const pencilIcon = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/></svg>';
  const chatListEl = document.getElementById('chat-list');
  const chatFilter = document.getElementById('chat-filter');

  const projectListEl = document.getElementById('project-list');
  const looseEl = document.getElementById('loose-chats');

  // Which projects are expanded, remembered across pages.
  let openProjects = new Set();
  try { openProjects = new Set(JSON.parse(localStorage.getItem('ollamui.openProjects') || '[]')); } catch { /* ignore */ }
  function saveOpenProjects() {
    try { localStorage.setItem('ollamui.openProjects', JSON.stringify([...openProjects])); } catch { /* ignore */ }
  }
  function setProjectOpen(proj, open, persist = true) {
    proj.querySelector('.project-chats').hidden = !open;
    proj.classList.toggle('open', open);
    proj.querySelector('.project-toggle').setAttribute('aria-expanded', open ? 'true' : 'false');
    if (!persist) return;
    if (open) openProjects.add(proj.dataset.project); else openProjects.delete(proj.dataset.project);
    saveOpenProjects();
  }
  function updateProjectCount(proj) {
    proj.querySelector('.project-count').textContent = proj.querySelectorAll('.project-chats .chat-row').length;
  }
  function initProjects() {
    if (!projectListEl) return;
    for (const proj of projectListEl.querySelectorAll('.project')) {
      const active = proj.dataset.active === '1' || proj.querySelector('.chat-link.active');
      setProjectOpen(proj, !!active || openProjects.has(proj.dataset.project), false);
    }
  }

  function applyChatFilter() {
    if (!chatListEl) return;
    const q = (chatFilter?.value || '').trim().toLowerCase();
    for (const row of chatListEl.querySelectorAll('.chat-row')) {
      row.hidden = !!q && !row.querySelector('.chat-link').textContent.toLowerCase().includes(q);
    }
    for (const g of chatListEl.querySelectorAll('.chat-group')) {
      g.hidden = !g.querySelector('.chat-row:not([hidden])');
    }
    for (const proj of chatListEl.querySelectorAll('.project')) {
      const box = proj.querySelector('.project-chats');
      if (!q) { proj.hidden = false; box.hidden = !proj.classList.contains('open'); continue; }
      const nameHit = proj.querySelector('.project-name').textContent.toLowerCase().includes(q);
      const hit = !!box.querySelector('.chat-row:not([hidden])');
      proj.hidden = !hit && !nameHit;
      box.hidden = !hit;
    }
  }
  chatFilter?.addEventListener('input', applyChatFilter);

  // Chats with a reply in progress show a dot. While any does, ask the
  // server every few seconds which still run.
  let runPoll = null;
  function setRunning(ids) {
    for (const row of document.querySelectorAll('.chat-row[data-chat]')) row.classList.toggle('running', ids.includes(row.dataset.chat));
  }
  function trackRunning(id, on = true) {
    if (id) for (const row of document.querySelectorAll(`.chat-row[data-chat="${id}"]`)) row.classList.toggle('running', on);
    if (runPoll || !document.querySelector('.chat-row.running')) return;
    runPoll = setInterval(async () => {
      try {
        const res = await fetch('/api/chats/running');
        if (res.ok) setRunning((await res.json()).running || []);
      } catch { /* try again next time */ }
      if (!document.querySelector('.chat-row.running')) { clearInterval(runPoll); runPoll = null; }
    }, 3000);
  }
  trackRunning();

  // Adds a newly created chat to the top of the Today group (creating it if missing).
  function addSidebarChat(id, title, projectId) {
    if (!chatListEl) return;
    for (const a of chatListEl.querySelectorAll('.active')) a.classList.remove('active');
    const row = el('div', 'chat-row');
    row.draggable = true;
    row.dataset.chat = id;
    const a = el('a', 'chat-link active', title.split('\n')[0].slice(0, 60));
    a.href = `/chat/${id}`;
    a.title = a.textContent;
    const ren = el('button', 'icon-btn chat-rename');
    ren.type = 'button';
    ren.dataset.chatRename = id;
    ren.title = 'Rename chat';
    ren.setAttribute('aria-label', 'Rename chat');
    ren.innerHTML = pencilIcon;
    const del = el('button', 'icon-btn chat-del');
    del.type = 'button';
    del.dataset.chatDel = id;
    del.title = 'Delete chat';
    del.setAttribute('aria-label', 'Delete chat');
    del.innerHTML = trashIcon;
    row.append(a, ren, del);
    const proj = projectId && projectListEl ? [...projectListEl.querySelectorAll('.project')].find((p) => p.dataset.project === projectId) : null;
    if (proj) {
      proj.querySelector('.project-chats').prepend(row);
      updateProjectCount(proj);
      setProjectOpen(proj, true, false);
    } else {
      const loose = looseEl || chatListEl;
      let group = loose.querySelector('.chat-group');
      if (group?.querySelector('.chat-group-label')?.textContent !== 'Today') {
        group = el('div', 'chat-group');
        group.append(el('div', 'chat-group-label', 'Today'));
        loose.prepend(group);
      }
      group.querySelector('.chat-group-label').after(row);
      group.hidden = false;
    }
    if (chatFilter && chatListEl.querySelectorAll('.chat-row').length > 8) chatFilter.hidden = false;
  }

  // Swaps a row's link for a text input; Enter or blur saves, Escape cancels.
  function startRename(row) {
    if (row.classList.contains('renaming')) return;
    const a = row.querySelector('.chat-link');
    const id = row.querySelector('[data-chat-rename]').dataset.chatRename;
    const input = el('input', 'chat-rename-input');
    input.type = 'text';
    input.value = a.textContent;
    input.maxLength = 200;
    input.setAttribute('aria-label', 'Chat title');
    row.classList.add('renaming');
    a.hidden = true;
    a.after(input);
    input.focus();
    input.select();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const title = input.value.trim();
      if (save && title && title !== a.textContent) {
        const res = await fetch(`/api/chats/${encodeURIComponent(id)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title }),
        });
        if (res.ok) {
          const saved = (await res.json()).title;
          a.textContent = saved;
          a.title = saved;
          if (a.classList.contains('active')) document.title = `${saved} · ollamui`;
          applyChatFilter();
        } else {
          alert(`Couldn't rename chat: ${await errorText(res)}`);
        }
      }
      input.remove();
      a.hidden = false;
      row.classList.remove('renaming');
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
    });
    input.addEventListener('blur', () => finish(true));
  }

  chatListEl?.addEventListener('click', async (e) => {
    const tog = e.target.closest('.project-toggle');
    if (tog) {
      const proj = tog.closest('.project');
      setProjectOpen(proj, !proj.classList.contains('open'));
      applyChatFilter();
      return;
    }
    const ren = e.target.closest('[data-chat-rename]');
    if (ren) {
      e.preventDefault();
      e.stopPropagation();
      startRename(ren.closest('.chat-row'));
      return;
    }
    const btn = e.target.closest('[data-chat-del]');
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    const id = btn.dataset.chatDel;
    const row = btn.closest('.chat-row');
    btn.disabled = true;
    const res = await fetch(`/api/chats/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (!res.ok) {
      btn.disabled = false;
      alert(`Couldn't delete chat: ${await errorText(res)}`);
      return;
    }
    if (document.getElementById('chat')?.dataset.chatId === id || row.querySelector('.chat-link.active')) {
      location.href = '/chat';
      return;
    }
    row.classList.add('removing');
    setTimeout(() => {
      const group = row.closest('.chat-group');
      const proj = row.closest('.project');
      row.remove();
      if (group && !group.querySelector('.chat-row')) group.remove();
      if (proj) updateProjectCount(proj);
      applyChatFilter();
    }, 200);
  });

  // ---- sidebar projects ----
  initProjects();
  applyChatFilter();

  document.getElementById('new-project')?.addEventListener('click', () => {
    if (!projectListEl) return;
    const existing = projectListEl.querySelector('.project-new-row');
    if (existing) { existing.querySelector('input').focus(); return; }
    const wrap = el('div', 'project-new-row');
    wrap.innerHTML = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>';
    const input = el('input', 'chat-rename-input');
    input.type = 'text';
    input.placeholder = 'Project name';
    input.maxLength = 100;
    input.setAttribute('aria-label', 'Project name');
    wrap.append(input);
    projectListEl.prepend(wrap);
    input.focus();
    let done = false;
    input.addEventListener('keydown', async (e) => {
      if (e.key === 'Escape') { e.preventDefault(); done = true; wrap.remove(); }
      else if (e.key === 'Enter') {
        e.preventDefault();
        const name = input.value.trim();
        if (!name || done) return;
        done = true;
        input.disabled = true;
        const res = await fetch('/api/projects', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
        if (res.ok) { location.href = '/projects/' + encodeURIComponent((await res.json()).id); return; }
        alert(`Couldn't create project: ${await errorText(res)}`);
        wrap.remove();
      }
    });
    input.addEventListener('blur', () => { if (!done && !input.value.trim()) { done = true; wrap.remove(); } });
  });

  // Drag a chat onto a project to file it, or onto "Chats" / the loose list to take it out.
  let dragRow = null;
  function dropTargetFor(t) {
    if (!(t instanceof Element)) return null;
    const proj = t.closest('.project');
    if (proj) return { el: proj.querySelector('.project-row'), project: proj.dataset.project, proj };
    const head = t.closest('.chats-head');
    if (head) return { el: head, project: '' };
    const loose = t.closest('#loose-chats');
    if (loose) return { el: loose, project: '' };
    return null;
  }
  function clearDrop() {
    for (const n of document.querySelectorAll('.drop-target')) n.classList.remove('drop-target');
  }
  chatListEl?.addEventListener('dragstart', (e) => {
    const row = e.target.closest?.('.chat-row');
    if (!row || row.classList.contains('renaming')) return;
    dragRow = row;
    row.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', row.dataset.chat);
  });
  document.getElementById('sidebar')?.addEventListener('dragover', (e) => {
    if (!dragRow) return;
    const t = dropTargetFor(e.target);
    clearDrop();
    if (!t) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    t.el.classList.add('drop-target');
  });
  document.getElementById('sidebar')?.addEventListener('dragleave', (e) => {
    if (!e.relatedTarget || !document.getElementById('sidebar').contains(e.relatedTarget)) clearDrop();
  });
  document.addEventListener('dragend', () => { dragRow?.classList.remove('dragging'); dragRow = null; clearDrop(); });
  document.getElementById('sidebar')?.addEventListener('drop', async (e) => {
    if (!dragRow) return;
    const t = dropTargetFor(e.target);
    clearDrop();
    if (!t) return;
    e.preventDefault();
    const row = dragRow;
    dragRow = null;
    row.classList.remove('dragging');
    const from = row.closest('.project');
    if ((from?.dataset.project || '') === t.project) return;
    const res = await fetch(`/api/chats/${encodeURIComponent(row.dataset.chat)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project: t.project }),
    });
    if (!res.ok) { alert(`Couldn't move chat: ${await errorText(res)}`); return; }
    if (!t.proj) { location.reload(); return; } // back to the dated list: easiest to re-render
    const group = row.closest('.chat-group');
    t.proj.querySelector('.project-chats').append(row);
    if (group && !group.querySelector('.chat-row')) group.remove();
    if (from) updateProjectCount(from);
    updateProjectCount(t.proj);
    setProjectOpen(t.proj, true);
    applyChatFilter();
  });

  window.ollamui = { addSidebarChat, trackRunning, setRunning, post, errorText, refreshLoaded, setLoadButton, el, fmtBytes, pollDownloads };
})();

// Phone layout: the sidebar is a drawer opened from the app bar.
(() => {
  const root = document.documentElement;
  const btn = document.getElementById('nav-toggle');
  const set = (open) => { root.classList.toggle('nav-open', open); btn?.setAttribute('aria-expanded', String(open)); };
  btn?.addEventListener('click', () => set(!root.classList.contains('nav-open')));
  document.getElementById('nav-backdrop')?.addEventListener('click', () => set(false));
  document.getElementById('sidebar')?.addEventListener('click', (e) => { if (e.target.closest('a')) set(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') set(false); });

  const phone = matchMedia('(max-width: 760px)');
  const tune = document.getElementById('tune');

  // On phones the menu and Tune act like pages: opening one adds a history
  // entry, so the back button closes it instead of leaving.
  const overlayOpen = () => root.classList.contains('nav-open') || root.classList.contains('sheet-open') || !!tune?.classList.contains('open');
  let popping = false;
  const syncHistory = () => {
    if (!phone.matches || popping) return;
    if (overlayOpen() && !history.state?.overlay) history.pushState({ ...history.state, overlay: true }, '');
    else if (!overlayOpen() && history.state?.overlay) history.back();
  };
  new MutationObserver(syncHistory).observe(root, { attributes: true, attributeFilter: ['class'] });
  if (tune) new MutationObserver(syncHistory).observe(tune, { attributes: true, attributeFilter: ['class'] });
  addEventListener('popstate', () => {
    if (history.state?.overlay || !overlayOpen()) return;
    popping = true;
    set(false);
    window.ollamuiSheets.closeAll();
    if (tune?.classList.contains('open')) document.getElementById('tune-close')?.click();
    setTimeout(() => { popping = false; });
  });

  // Swipes: right opens the menu, left closes it. Ignored on fields and
  // scrollable code. (Tune is a bottom sheet: dragged down, not swiped.)
  let start = null;
  document.addEventListener('touchstart', (e) => {
    const t = e.touches[0];
    start = e.touches.length === 1 && !e.target.closest('input, textarea, select, pre, .code, .katex-display')
      ? { x: t.clientX, y: t.clientY, at: Date.now() } : null;
  }, { passive: true });
  document.addEventListener('touchend', (e) => {
    if (!start) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - start.x, dy = t.clientY - start.y;
    const quick = Date.now() - start.at < 700;
    start = null;
    if (!quick || Math.abs(dx) < 60 || Math.abs(dx) < Math.abs(dy) * 1.5) return;
    const navOpen = root.classList.contains('nav-open');
    if (root.classList.contains('sheet-open')) return;
    if (dx > 0) { if (!navOpen && phone.matches) set(true); }
    else if (navOpen) set(false);
  }, { passive: true });
})();

// The API address on the presets page.
document.querySelectorAll('[data-api-url]').forEach((el) => { el.textContent = location.origin + '/v1'; });

// Path pickers. ollamuiPick({kind: 'file'|'folder', title, start, filter})
// resolves to a path on the server, or null if the user cancelled. From the
// machine running ollamui it opens that machine's own chooser (POST
// /api/pick); from anywhere else, or without a dialog tool there, it shows
// an in-app browser over GET /api/fs (a bottom sheet on the phone).
window.ollamuiPick = (() => {
  let nativeP = null;
  const isNative = () => nativeP ||= fetch('/api/pick').then((r) => (r.ok ? r.json() : {})).then((j) => !!j.native).catch(() => false);
  const ICON = {
    folder: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>',
    file: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>',
    up: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V6M6 11l6-6 6 6"/></svg>',
  };
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };

  function browse(o) {
    const kind = o.kind === 'file' ? 'file' : 'folder';
    const sheets = window.ollamuiSheets;
    return new Promise((resolve) => {
      const scrim = el('div', 'pk-scrim');
      const box = el('div', 'pk');
      box.setAttribute('role', 'dialog');
      box.setAttribute('aria-modal', 'true');
      box.setAttribute('aria-label', o.title || 'Choose');
      const title = el('h2', 'pk-title sheet-title', o.title || (kind === 'file' ? 'Choose a file' : 'Choose a folder'));
      const bar = el('div', 'pk-bar');
      const upBtn = el('button', 'btn pk-up'); upBtn.type = 'button'; upBtn.innerHTML = ICON.up; upBtn.title = 'Parent folder'; upBtn.setAttribute('aria-label', 'Parent folder');
      const pathIn = el('input', 'pk-path mono'); pathIn.type = 'text'; pathIn.spellcheck = false; pathIn.autocomplete = 'off'; pathIn.setAttribute('aria-label', 'Path');
      bar.append(upBtn, pathIn);
      const roots = el('div', 'pk-roots');
      const list = el('div', 'boxed-list pk-list');
      const note = el('div', 'pk-note muted small');
      const foot = el('div', 'pk-foot');
      const hid = el('label', 'pk-hidden small');
      const hidBox = document.createElement('input'); hidBox.type = 'checkbox';
      hid.append(hidBox, ' Show hidden');
      const cancel = el('button', 'btn', 'Cancel'); cancel.type = 'button';
      const choose = el('button', 'btn primary', 'Choose this folder'); choose.type = 'button';
      foot.append(hid, el('span', 'pk-spacer'), cancel);
      if (kind === 'folder') foot.append(choose);
      box.append(el('div', 'sheet-grab'), title, bar, roots, list, note, foot);

      const phone = sheets?.phone.matches;
      const prevFocus = document.activeElement;
      let cur = '', done = false, seq = 0;

      const finish = (val) => {
        if (done) return;
        done = true;
        document.removeEventListener('keydown', onKey, true);
        if (phone) {
          sheets.track(box, false);
          box.classList.remove('open');
          setTimeout(() => box.remove(), 280);
        } else {
          box.remove();
          scrim.remove();
        }
        prevFocus?.focus?.({ preventScroll: true });
        resolve(val);
      };
      const onKey = (e) => {
        if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(null); }
        else if (e.key === 'Tab') { // keep focus inside
          const f = [...box.querySelectorAll('button:not([disabled]), input')];
          if (!f.length) return;
          const first = f[0], last = f[f.length - 1];
          if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
          else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
        } else if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && document.activeElement !== pathIn) {
          const rows = [...list.querySelectorAll('.pk-row')];
          const i = rows.indexOf(document.activeElement);
          const next = rows[e.key === 'ArrowDown' ? Math.min(i + 1, rows.length - 1) : Math.max(i - 1, 0)];
          if (next) { e.preventDefault(); next.focus(); }
        }
      };

      async function go(path) {
        const mine = ++seq;
        const qs = new URLSearchParams({ path: path || '', kind });
        if (o.filter) qs.set('filter', o.filter);
        if (hidBox.checked) qs.set('hidden', '1');
        let d;
        try {
          const res = await fetch('/api/fs?' + qs);
          d = await res.json();
          if (!res.ok) throw new Error(d.error || res.statusText);
        } catch (err) {
          if (mine === seq) { note.classList.add('bad'); note.textContent = 'Couldn’t read that: ' + err.message; }
          return;
        }
        if (mine !== seq) return;
        cur = d.path;
        pathIn.value = cur;
        upBtn.disabled = !d.parent;
        upBtn.onclick = () => go(d.parent);
        roots.replaceChildren();
        for (const r of d.roots || []) {
          const b = el('button', 'pk-root' + (r.path === cur ? ' on' : ''), r.name); b.type = 'button';
          b.addEventListener('click', () => go(r.path));
          roots.append(b);
        }
        list.replaceChildren();
        for (const e of d.entries || []) {
          const r = el('button', 'pk-row'); r.type = 'button';
          const ic = el('span', 'pk-icon' + (e.dir ? '' : ' file')); ic.innerHTML = e.dir ? ICON.folder : ICON.file;
          r.append(ic, el('span', 'pk-name', e.name));
          r.addEventListener('click', () => (e.dir ? go(e.path) : finish(e.path)));
          list.append(r);
        }
        note.classList.toggle('bad', !!d.error);
        if (d.error) note.textContent = 'Couldn’t open this folder: ' + d.error;
        else if (!d.entries?.length) note.textContent = kind === 'file' ? 'No matching files here.' : 'No folders here.';
        else note.textContent = '';
        choose.textContent = 'Choose “' + (cur.split(/[\\/]/).filter(Boolean).pop() || cur) + '”';
        list.scrollTop = 0;
      }

      pathIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); go(pathIn.value.trim()); } });
      hidBox.addEventListener('change', () => go(cur));
      cancel.addEventListener('click', () => finish(null));
      choose.addEventListener('click', () => finish(cur));
      scrim.addEventListener('click', () => finish(null));
      document.addEventListener('keydown', onKey, true);

      if (phone) {
        box.classList.add('sheet');
        document.body.append(box);
        sheets.register(box, () => finish(null));
        sheets.track(box, true);
        void box.offsetWidth;
        box.classList.add('open');
      } else {
        document.body.append(scrim, box);
      }
      // Start in the current value's folder; a file value opens beside it.
      go(o.start || '').then(() => (phone ? box : pathIn).focus({ preventScroll: true }));
    });
  }

  return async (o) => {
    if (await isNative()) {
      try {
        const res = await fetch('/api/pick', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ kind: o.kind, title: o.title || '', start: o.start || '', filter: o.filter || '' }),
        });
        if (res.ok) { const j = await res.json(); return j.cancelled ? null : j.path || null; }
        if (res.status === 409) return null; // a dialog is already open
      } catch { /* fall back to the in-app browser */ }
    }
    return browse(o);
  };
})();

// Browse… buttons: <button data-browse="input-id" data-kind="file|folder"
// data-title data-filter>. A choice fills the input and fires "input" and
// "picked" on it; the page decides what else to do. found.js, which builds
// its rows in script, makes its button with ollamuiBrowseButton.
(() => {
  const wire = (b) => b.addEventListener('click', async () => {
    const input = document.getElementById(b.dataset.browse);
    if (!input) return;
    b.disabled = true;
    try {
      const p = await window.ollamuiPick({ kind: b.dataset.kind, title: b.dataset.title, filter: b.dataset.filter, start: input.value.trim() });
      if (p) {
        input.value = p;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('picked'));
      }
    } finally { b.disabled = false; }
  });
  document.querySelectorAll('[data-browse]').forEach(wire);
  window.ollamuiBrowseButton = (input, { kind, title, filter }) => {
    if (!input.id) input.id = 'pk-' + Math.random().toString(36).slice(2, 8);
    const b = document.createElement('button');
    b.className = 'btn';
    b.textContent = 'Browse…';
    b.type = 'button';
    Object.assign(b.dataset, { browse: input.id, kind, title: title || '', filter: filter || '' });
    wire(b);
    input.after(b);
    return b;
  };
})();
