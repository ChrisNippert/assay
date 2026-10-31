// Library: sorting with a quick reorder animation, filter, row menus,
// popovers behind every number, live memory stats and "Measure all".
(() => {
  const list = document.getElementById('lib-list');
  if (!list) return;
  const pop = document.getElementById('lib-pop');
  const none = document.getElementById('lib-none');
  const filter = document.getElementById('lib-filter');
  const status = document.getElementById('measure-status');
  const measureAll = document.getElementById('measure-all');
  const rows = () => [...list.querySelectorAll('.lr')];
  const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
  const gb = (n) => (n / 1e9).toFixed(1);
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch {} },
  };
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  async function api(method, url, body) {
    const res = await fetch(url, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    let data = null;
    try { data = await res.json(); } catch {}
    if (!res.ok) throw new Error((data && data.error) || res.statusText);
    return data;
  }

  // ---- sorting ----
  // Missing values (untested speed, unrated score) always sink to the end.
  const num = (r, k) => { const v = parseFloat(r.dataset[k]); return Number.isFinite(v) && v > 0 ? v : null; };
  const sorters = {
    name: [(a, b) => a.dataset.title.localeCompare(b.dataset.title, undefined, { numeric: true }), 'Name'],
    recent: [(a, b) => b.dataset.mod - a.dataset.mod, 'Recent'],
    size: [(a, b) => b.dataset.size - a.dataset.size, 'Size'],
    speed: [(a, b) => cmpMissing(num(a, 'speed'), num(b, 'speed'), -1), 'Speed'],
    memory: [(a, b) => cmpMissing(num(a, 'mem'), num(b, 'mem'), 1), 'Memory'],
    score: [(a, b) => cmpMissing(num(a, 'score'), num(b, 'score'), -1), 'Score'],
  };
  function cmpMissing(x, y, dir) {
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return (x - y) * dir;
  }
  const arrows = { speed: ' ↓', score: ' ↓', memory: ' ↑' };
  let sort = new URLSearchParams(location.search).get('sort') || store.get('assay.lib.sort') || 'name';
  if (!sorters[sort]) sort = 'name';

  function paintSort() {
    for (const b of document.querySelectorAll('[data-sort]')) {
      const on = b.dataset.sort === sort;
      b.classList.toggle('on', on);
      if (b.classList.contains('lib-th')) {
        const base = b.dataset.sort === 'name' ? 'Model' : sorters[b.dataset.sort][1];
        b.textContent = base + (on ? arrows[sort] || '' : '');
        b.setAttribute('aria-sort', on ? (sort === 'memory' ? 'ascending' : 'descending') : 'none');
      } else {
        b.setAttribute('aria-pressed', String(on));
      }
    }
  }

  function order(animate) {
    const items = rows();
    const first = new Map(items.map((r) => [r, r.getBoundingClientRect().top]));
    items.sort(sorters[sort][0]);
    list.append(...items);
    if (!animate || reduced()) return;
    const dur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--dur-2')) || 180;
    for (const r of items) {
      if (r.hidden) continue;
      const dy = first.get(r) - r.getBoundingClientRect().top;
      if (!dy) continue;
      r.animate([{ transform: `translateY(${dy}px)` }, { transform: 'none' }], { duration: dur, easing: 'cubic-bezier(.2,.7,.2,1)' });
    }
  }

  function setSort(key, animate = true) {
    sort = key;
    store.set('assay.lib.sort', key);
    const u = new URL(location.href);
    u.searchParams.set('sort', key);
    history.replaceState(null, '', u);
    paintSort();
    order(animate);
  }
  for (const b of document.querySelectorAll('[data-sort]')) b.addEventListener('click', () => setSort(b.dataset.sort));

  // ---- filter ----
  function applyFilter() {
    const q = filter.value.trim().toLowerCase();
    let shown = 0;
    for (const r of rows()) {
      const hit = !q || (r.dataset.names + ' ' + r.dataset.title).toLowerCase().includes(q);
      r.hidden = !hit;
      if (hit) shown++;
    }
    none.hidden = shown > 0 || rows().length === 0;
  }
  filter.addEventListener('input', applyFilter);
  new MutationObserver(() => {
    applyFilter();
    const h = document.querySelector('.appbar-title + .muted');
    if (h) h.textContent = `${rows().length} installed`;
  }).observe(list, { childList: true });

  // ---- row: whole row opens the model page; menu ----
  list.addEventListener('click', (e) => {
    const row = e.target.closest('.lr');
    if (!row || e.target.closest('a, button, .lr-menu, input')) return;
    if (getSelection().toString()) return;
    location.href = row.querySelector('.lr-name').href;
  });

  let openMenu = null;
  function closeMenu() {
    if (!openMenu) return;
    openMenu.menu.hidden = true;
    openMenu.btn.setAttribute('aria-expanded', 'false');
    openMenu.row.classList.remove('menu-open');
    openMenu = null;
  }
  list.addEventListener('click', (e) => {
    const btn = e.target.closest('.lr-more');
    if (!btn) return;
    const row = btn.closest('.lr');
    const was = openMenu && openMenu.row === row;
    closePop();
    closeMenu();
    if (was) return;
    const menu = row.querySelector('.lr-menu');
    menu.hidden = false;
    btn.setAttribute('aria-expanded', 'true');
    row.classList.add('menu-open');
    openMenu = { row, menu, btn };
    menu.querySelector('button, a')?.focus();
  });
  list.addEventListener('keydown', (e) => {
    if (!openMenu || !e.target.closest('.lr-menu')) return;
    const items = [...openMenu.menu.querySelectorAll('button, a')];
    const i = items.indexOf(document.activeElement);
    if (e.key === 'ArrowDown') { items[(i + 1) % items.length].focus(); e.preventDefault(); }
    if (e.key === 'ArrowUp') { items[(i - 1 + items.length) % items.length].focus(); e.preventDefault(); }
  });
  // Choosing something closes the menu (after the click has been handled).
  list.addEventListener('click', (e) => { if (e.target.closest('.lr-menu [role=menuitem]')) setTimeout(closeMenu, 0); });

  document.addEventListener('click', (e) => {
    if (openMenu && !e.target.closest('.lr-menu, .lr-more')) closeMenu();
    if (!pop.hidden && !e.target.closest('#lib-pop, [data-pop]')) closePop();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (openMenu) { const b = openMenu.btn; closeMenu(); b.focus(); }
    else if (!pop.hidden) { const a = popAnchor; closePop(); a?.focus(); }
  });

  // Load/Eject in the menu are handled by app.js through data-load/data-eject.
  // Follow load state changes made anywhere.
  document.addEventListener('loaded-changed', (e) => {
    const names = new Set(e.detail.map((m) => m.name));
    for (const r of rows()) {
      const all = (r.dataset.names || r.dataset.model).split(' ');
      const loadedAs = all.find((n) => names.has(n)) || '';
      r.dataset.loaded = loadedAs;
      r.querySelector('.lr-dot').classList.toggle('off', !loadedAs);
      const t = r.querySelector('[data-act=toggle]');
      delete t.dataset.load; delete t.dataset.eject;
      if (loadedAs) t.dataset.eject = loadedAs; else t.dataset.load = r.dataset.model;
      t.textContent = loadedAs ? 'Eject' : 'Load';
      t.disabled = false;
    }
    if (popKind === 'gpu' || popKind === 'ram') sampleStats();
  });

  // ---- live memory stats ----
  const stat = { vram: 0, ram: 0 };
  function paintStat(id, used, total, noGPU) {
    const v = document.getElementById('st-' + id);
    const bar = document.getElementById('st-' + id + '-bar');
    if (!total) return;
    v.replaceChildren(document.createTextNode(gb(used) + ' '), el('span', 'lib-unit', `of ${gb(total)} GB`));
    bar.style.width = Math.min(100, used * 100 / total) + '%';
  }
  async function sampleStats() {
    try {
      const g = await api('GET', '/api/gpu');
      stat.gpu = g;
      if (g.vramTotal) paintStat('gpu', g.vramUsed, g.vramTotal);
      if (g.ramTotal) paintStat('ram', g.ramUsed, g.ramTotal);
    } catch {}
  }
  sampleStats();
  setInterval(() => { if (!document.hidden) sampleStats(); }, 4000);

  // ---- popovers ----
  let popAnchor = null, popKind = '';
  function closePop() {
    pop.hidden = true;
    pop.replaceChildren();
    popAnchor = null; popKind = '';
  }
  function placePop(anchor) {
    pop.hidden = false;
    const a = anchor.getBoundingClientRect();
    const w = Math.min(340, innerWidth - 24);
    pop.style.width = w + 'px';
    const h = pop.offsetHeight;
    let left = Math.min(Math.max(12, a.left), innerWidth - w - 12);
    let top = a.bottom + 8;
    if (top + h > innerHeight - 12 && a.top - h - 8 > 12) top = a.top - h - 8;
    pop.style.left = left + 'px';
    pop.style.top = Math.max(12, top) + 'px';
  }
  function popHead(title, sub) {
    const h = el('div', 'pop-head');
    h.append(el('div', 'pop-title', title));
    if (sub) h.append(el('div', 'pop-sub', sub));
    return h;
  }
  function popRow(label, value, cls) {
    const r = el('div', 'pop-row' + (cls ? ' ' + cls : ''));
    const l = el('span', 'pop-label');
    if (typeof label === 'string') l.textContent = label; else l.append(label);
    r.append(l, el('span', 'pop-val', value));
    return r;
  }
  function popLink(text, href) {
    const a = el('a', 'pop-link', text);
    a.href = href;
    return a;
  }
  async function openPop(anchor, kind, build) {
    if (!pop.hidden && popAnchor === anchor) { closePop(); return; }
    closeMenu();
    popAnchor = anchor; popKind = kind;
    pop.replaceChildren(el('div', 'pop-sub', 'Loading…'));
    placePop(anchor);
    try {
      const nodes = await build();
      if (popAnchor !== anchor) return;
      pop.replaceChildren(...nodes);
    } catch (err) {
      pop.replaceChildren(el('div', 'pop-sub', err.message));
    }
    placePop(anchor);
  }

  const titleOfLoaded = (n) => {
    const r = rows().find((x) => (x.dataset.names || '').split(' ').includes(n));
    return r ? r.dataset.title : n;
  };

  async function memoryPop(kind) {
    const [loaded, live] = await Promise.all([api('GET', '/api/loaded'), api('GET', '/api/gpu')]);
    const gpu = kind === 'gpu';
    const used = gpu ? live.vramUsed : live.ramUsed;
    const total = gpu ? live.vramTotal : live.ramTotal;
    const nodes = [popHead(gpu ? 'GPU memory' : 'System memory', total ? `${gb(used)} of ${gb(total)} GB in use` : 'Not available on this computer')];
    let ours = 0;
    for (const m of loaded) {
      const amount = gpu ? m.sizeVram : Math.max(0, m.size - m.sizeVram);
      if (m.loading) { nodes.push(popRow(titleOfLoaded(m.name), 'loading…', 'muted')); continue; }
      if (!amount) continue;
      ours += amount;
      const row = popRow(titleOfLoaded(m.name), gb(amount) + ' GB');
      const eject = el('button', 'pop-mini keep-text', 'Eject');
      eject.type = 'button';
      eject.dataset.eject = m.name;
      row.querySelector('.pop-label').title = m.name + (m.pinned ? ' (stays loaded until ejected)' : '');
      row.append(eject);
      nodes.push(row);
    }
    if (!loaded.some((m) => (gpu ? m.sizeVram : m.size - m.sizeVram) > 0)) {
      nodes.push(el('div', 'pop-sub', gpu ? 'No model is on the GPU right now.' : 'No model is using system memory right now.'));
    }
    if (total && used > ours) nodes.push(popRow(gpu ? 'Everything else (desktop, browser…)' : 'Everything else', gb(used - ours) + ' GB', 'muted'));
    if (gpu && live.cards && live.cards.length > 1) {
      const sec = el('div', 'pop-sec');
      sec.append(el('div', 'pop-sub', 'Per card'));
      for (const c of live.cards) {
        const r = popRow(c.name, `${gb(c.used)} of ${gb(c.total)} GB`);
        const t = el('span', 'lib-track thin');
        t.append(Object.assign(el('span', 'lib-fill accent'), { style: `width:${Math.min(100, c.used * 100 / c.total)}%` }));
        sec.append(r, t);
      }
      nodes.push(sec);
    }
    return nodes;
  }

  function memoryFitPop(row) {
    const need = +row.dataset.mem, size = +row.dataset.size;
    const vram = +document.getElementById('lib-stats').dataset.vram;
    const ram = +document.getElementById('lib-stats').dataset.ram;
    const lvl = row.dataset.fit;
    const nodes = [popHead(row.dataset.title, 'Memory this model needs')];
    nodes.push(popRow('Weights', gb(size) + ' GB'));
    nodes.push(popRow('Cache and buffers at 8k context', '~' + gb(Math.max(0, need - size)) + ' GB'));
    nodes.push(popRow('Total', gb(need) + ' GB', 'strong'));
    if (vram) nodes.push(popRow('Your GPU', gb(vram) + ' GB'));
    const say = {
      gpu: 'Fits on the GPU with room to spare. Bigger contexts also fit; Assay picks the largest that does.',
      tight: 'Fits on the GPU, but with little to spare. Long contexts may need to spill into RAM.',
      partial: `Doesn’t fit on the GPU alone, so some layers run from system RAM (${gb(ram)} GB here). It works, but slower.`,
      cpu: 'No GPU found, so it runs from system RAM.',
      none: 'Too big for the GPU and RAM together.',
    }[lvl];
    const measured = row.dataset.benchCtx && +row.dataset.benchCtx > 0;
    const ctxK = measured ? Math.round(row.dataset.benchCtx / 1024) + 'k' : '';
    if (measured && lvl === 'partial') {
      nodes.push(el('p', 'pop-note', `In the speed test only ${row.dataset.benchGpu}% of it fit on the GPU at a ${ctxK} context; the rest ran from system RAM. A smaller context, or fewer GPU layers, changes that.`));
    } else {
      if (say) nodes.push(el('p', 'pop-note', say));
      if (measured) nodes.push(el('p', 'pop-note', `Measured in the speed test: ${row.dataset.benchGpu}% on the GPU at a ${ctxK} context.`));
    }
    nodes.push(popLink('Tune context and GPU layers →', `/chat?model=${encodeURIComponent(row.dataset.model)}&tune=1`));
    return nodes;
  }

  const reasons = {
    local: 'This model didn’t come from Hugging Face here, so there’s no way to tell which model it was made from.',
    'no-base': 'Hugging Face doesn’t list a base model for this repo. Fine-tunes and merges don’t inherit their base model’s score.',
    offline: 'Couldn’t reach Hugging Face to look it up. Try again in a moment.',
  };
  async function scorePop(row) {
    const d = await api('GET', '/api/library/nearby?name=' + encodeURIComponent(row.dataset.model));
    const src = el('p', 'pop-note');
    src.append('Source: ', Object.assign(el('a', null, 'Artificial Analysis'), { href: d.url, target: '_blank', rel: 'noopener' }), ' Intelligence Index.');
    if (!d.rated) {
      const nodes = [popHead(row.dataset.title, 'Not rated')];
      let why = reasons[d.reason];
      if (d.reason === 'unlisted') why = `Artificial Analysis hasn’t rated ${d.of ? 'its base model, ' + d.of : 'the base model'} (yet).`;
      nodes.push(el('p', 'pop-note', why || 'No score is known for this model.'));
      nodes.push(popLink('Find it on Discover →', d.repo ? '/discover/' + d.repo : '/discover?q=' + encodeURIComponent(row.dataset.title)));
      nodes.push(src);
      return nodes;
    }
    const nodes = [popHead(`${Math.round(d.score)} · ${row.dataset.title}`, `Score of ${d.of}, the base model`)];
    if (d.near && d.near.length > 1) {
      nodes.push(el('p', 'pop-note', `#${d.rank} of ${d.count} rated models around ${d.params ? Math.round(d.params) + 'B' : 'this size'}.`));
      const top = Math.max(...d.near.map((n) => n.score));
      for (const n of d.near) {
        const r = popRow(n.model, Math.round(n.score) + '', n.you ? 'you' : '');
        if (n.installed) r.querySelector('.pop-label').append(el('span', 'pop-inst', ' installed'));
        const t = el('span', 'lib-track thin');
        t.append(Object.assign(el('span', 'lib-fill' + (n.you ? ' accent' : '')), { style: `width:${n.score / top * 100}%` }));
        nodes.push(r, t);
      }
    } else {
      nodes.push(el('p', 'pop-note', 'No other rated models of a similar size to compare with.'));
    }
    nodes.push(src);
    return nodes;
  }

  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-pop]');
    if (!b) return;
    const row = b.closest('.lr');
    const kind = b.dataset.pop;
    if (kind === 'gpu' || kind === 'ram') openPop(b, kind, () => memoryPop(kind));
    else if (kind === 'mem') openPop(b, kind, async () => memoryFitPop(row));
    else if (kind === 'score') openPop(b, kind, () => scorePop(row));
  });

  // ---- scores: filled in once Hugging Face has been asked ----
  function paintScore(row, s) {
    const cell = row.querySelector('.lr-score');
    const b = cell.querySelector('.lr-scorebtn');
    if (!b) return;
    if (s && s.rated) {
      row.dataset.score = s.score;
      b.className = 'lr-num lr-scorebtn';
      b.textContent = Math.round(s.score);
      b.title = 'Intelligence score. Click to compare with similar models.';
    } else {
      row.dataset.score = '';
      b.className = 'lr-scorebtn lr-none';
      b.textContent = 'not rated';
      b.title = 'No score. Click to see why.';
    }
  }
  api('GET', '/api/library/scores').then((d) => {
    for (const r of rows()) paintScore(r, d.scores[r.dataset.model]);
    if (sort === 'score') order(true);
  }).catch(() => {
    for (const r of rows()) if (!r.dataset.score) paintScore(r, null);
  });

  // ---- measuring ----
  let pollTimer = null;
  const rowOf = (name) => rows().find((r) => r.dataset.model === name);
  function setSpeed(row, item) {
    const cell = row.querySelector('.lr-speed');
    if (item.state === 'done' && item.genTps) {
      row.dataset.speed = item.genTps;
      const a = el('a', 'lr-num', String(Math.round(item.genTps)));
      a.href = '/compare?add=' + encodeURIComponent(row.dataset.model);
      a.title = 'Measured on this computer just now. Click for the full speed test.';
      cell.replaceChildren(a, document.createTextNode(' '), el('span', 'lr-unit', 'tok/s'));
      const menuItem = row.querySelector('.lr-menu [data-measure]');
      if (menuItem) menuItem.textContent = 'Measure speed again';
    } else if (item.state === 'running' || item.state === 'pending') {
      if (+row.dataset.speed > 0 && item.state === 'pending') return;
      cell.replaceChildren(el('span', 'lr-busy', item.state === 'running' ? 'Measuring…' : 'Queued'));
    } else if (!(+row.dataset.speed > 0)) {
      if (cell.querySelector('.lr-measure')) return;
      const b = el('button', 'lr-measure', 'Measure');
      b.type = 'button';
      b.dataset.measure = row.dataset.model;
      b.title = item.reason || 'Run the speed test on this model';
      cell.replaceChildren(b);
    }
  }
  function markBest() {
    const speeds = rows().map((r) => +r.dataset.speed || 0);
    const best = Math.max(0, ...speeds);
    for (const r of rows()) r.querySelector('.lr-speed .lr-num')?.classList.toggle('best', best > 0 && +r.dataset.speed === best);
  }
  function untested() {
    return rows().filter((r) => r.dataset.embed !== 'true' && !(+r.dataset.speed > 0)).length;
  }
  function paintMeasureButton(running) {
    const n = untested();
    document.getElementById('measure-n').textContent = n;
    measureAll.hidden = running || n === 0;
  }

  function paintProgress(st) {
    for (const it of st.items) {
      const r = rowOf(it.name);
      if (r) setSpeed(r, it);
    }
    markBest();
    paintMeasureButton(st.running);
    if (sort === 'speed' && st.items.some((i) => i.state === 'done')) order(true);
    status.replaceChildren();
    if (!st.items.length) { status.hidden = true; return; }
    status.hidden = false;
    status.classList.remove('bad');
    const line = el('div', 'lib-progress-line');
    const skipped = st.items.filter((i) => i.state === 'skipped' || i.state === 'failed');
    const made = st.items.filter((i) => i.state === 'done').length;
    if (st.running) {
      const at = Math.min(st.total, st.done + 1);
      line.append(el('span', 'lib-spin'));
      line.append(el('span', null, st.waiting ? `${st.waiting}… (${st.done} of ${st.total} done)` : `Measuring ${at} of ${st.total}${st.current ? ' · ' + st.current : ''}`));
      const c = el('button', 'lib-linkbtn', 'Cancel');
      c.type = 'button';
      c.addEventListener('click', async () => { c.disabled = true; c.textContent = 'Stopping…'; try { paintProgress(await api('DELETE', '/api/library/measure')); } catch {} });
      line.append(c);
    } else {
      line.append(el('span', null, `${st.cancelled ? 'Stopped. ' : ''}Measured ${made} of ${st.total}${skipped.length ? `, ${skipped.length} skipped` : ''}.`));
      const x = el('button', 'lib-linkbtn', 'Dismiss');
      x.type = 'button';
      x.addEventListener('click', () => { status.hidden = true; });
      line.append(x);
    }
    status.append(line);
    if (st.keptPinned && st.keptPinned.length && st.running) {
      status.append(el('div', 'lib-progress-note', `Keeping ${st.keptPinned.map(titleOfLoaded).join(', ')} loaded because it’s pinned, so models are measured next to it.`));
    }
    if (skipped.length && !st.running) {
      const d = el('details', 'lib-skipped');
      d.append(el('summary', null, 'See what was skipped'));
      for (const it of skipped) {
        const p = el('div', 'lib-skip-row');
        p.append(el('b', null, it.title), document.createTextNode(' · ' + (it.reason || 'failed')));
        d.append(p);
      }
      const pinnedSkips = skipped.filter((i) => /pinned/.test(i.reason || ''));
      if (st.keptPinned && st.keptPinned.length && pinnedSkips.length) {
        const b = el('button', 'btn', 'Eject the pinned model and measure these');
        b.type = 'button';
        b.addEventListener('click', () => startMeasure(pinnedSkips.map((i) => i.name), true));
        d.append(b);
      }
      status.append(d);
    }
  }

  async function poll() {
    clearTimeout(pollTimer);
    try {
      const st = await api('GET', '/api/library/measure');
      if (st.running || !status.hidden) paintProgress(st);
      if (st.running) pollTimer = setTimeout(poll, 1500);
    } catch { pollTimer = setTimeout(poll, 4000); }
  }

  async function startMeasure(names, evictPinned) {
    try {
      const st = await api('POST', '/api/library/measure', { names, evictPinned: !!evictPinned });
      paintProgress(st);
      pollTimer = setTimeout(poll, 1500);
    } catch (err) {
      status.hidden = false;
      status.classList.add('bad');
      status.replaceChildren(el('div', 'lib-progress-line', err.message));
    }
  }
  measureAll.addEventListener('click', () => startMeasure([]));
  document.addEventListener('click', async (e) => {
    const b = e.target.closest('[data-measure]');
    if (!b) return;
    const name = b.dataset.measure;
    let evict = false;
    try {
      const loaded = await api('GET', '/api/loaded');
      const row = rowOf(name);
      const names = row ? row.dataset.names.split(' ') : [name];
      const pinned = loaded.find((m) => m.pinned && names.includes(m.name));
      if (pinned) {
        if (!confirm(`${row ? row.dataset.title : name} is loaded and pinned. Measuring it unloads it first, then loads it fresh. Go ahead?`)) return;
        evict = true;
      }
    } catch {}
    startMeasure([name], evict);
  });
  poll();

  paintSort();
  order(false);
  applyFilter();
  paintMeasureButton(false);
})();
