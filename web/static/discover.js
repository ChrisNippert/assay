// Discover: fetch every match for a search once, then sort, filter and page
// through it locally. Sorting only reorders; it never changes what matches.
(() => {
  const root = document.getElementById('discover');
  if (!root) return;
  const { el, errorText, fmtBytes, pollDownloads } = window.assay;
  const $ = (id) => document.getElementById(id);

  const q = $('q');
  const author = $('author');
  const sort = $('sort');
  const meta = $('result-meta');
  const list = $('results');
  const more = $('show-more');
  const minIn = $('f-min');
  const maxIn = $('f-max');
  const archBox = $('f-arch');
  const checks = { long: 'f-longctx', rated: 'f-rated', fits: 'f-fits', hideInstalled: 'f-hide-installed', hideGated: 'f-hide-gated', hideDrafts: 'f-hide-drafts' };

  const PAGE = 20;
  const memory = JSON.parse(root.dataset.memory || '{}');
  $('memory-line').textContent = memory.vram
    ? `${memory.gpu}: ${fmtBytes(memory.vram)} VRAM · ${fmtBytes(memory.ram)} RAM`
    : `No GPU detected · ${fmtBytes(memory.ram)} RAM`;

  let results = [];
  let truncated = false;
  let max = 1000;
  let type = '';
  const takes = new Set(); // modalities the model must take; empty means any
  const makes = new Set(); // ...and must make
  const MODS = ['text', 'image', 'video', 'audio'];
  const archs = new Set(); // selected architectures; empty means any
  let limit = PAGE;
  let controller = null;
  let openId = null; // repo shown in the detail panel
  let detailController = null;
  const panel = $('detail-panel');
  const panelBody = $('detail-body');

  // ---- styled drop-downs over the real <select>s (which stay as the state) ----
  // A button with a muted label and the current choice opens a small menu.
  // Changing the select from script (author tags, the URL) fires 'sync'.
  let openDD = null;
  function closeDD(focus) {
    if (!openDD) return;
    openDD.menu.hidden = true;
    openDD.btn.setAttribute('aria-expanded', 'false');
    if (focus) openDD.btn.focus();
    openDD = null;
  }
  function dropdown(select) {
    const wrap = el('div', 'dd');
    const btn = el('button', 'dd-btn');
    btn.type = 'button';
    btn.setAttribute('aria-haspopup', 'listbox');
    btn.setAttribute('aria-expanded', 'false');
    btn.setAttribute('aria-label', select.getAttribute('aria-label'));
    const lab = el('span', 'dd-label', select.dataset.ddLabel);
    const cur = el('span', 'dd-value');
    btn.append(lab, cur, chevron());
    const menu = el('div', 'dd-menu');
    menu.setAttribute('role', 'listbox');
    menu.hidden = true;
    wrap.append(btn, menu);
    select.hidden = true;
    select.after(wrap);

    const paint = () => {
      const o = select.selectedOptions[0];
      cur.textContent = o ? o.textContent : '';
    };
    const items = () => [...menu.children];
    function fill() {
      menu.replaceChildren(...[...select.options].map((o) => {
        const it = el('button', 'dd-item', o.textContent);
        it.type = 'button';
        it.setAttribute('role', 'option');
        it.dataset.value = o.value;
        const on = o.value === select.value;
        it.setAttribute('aria-selected', String(on));
        it.classList.toggle('on', on);
        it.addEventListener('click', () => {
          const changedTo = select.value !== o.value;
          select.value = o.value;
          paint();
          closeDD(true);
          if (changedTo) select.dispatchEvent(new Event('change'));
        });
        return it;
      }));
    }
    function open() {
      closeDD();
      fill();
      menu.hidden = false;
      btn.setAttribute('aria-expanded', 'true');
      openDD = { menu, btn };
      (menu.querySelector('.on') || menu.firstChild)?.focus();
    }
    btn.addEventListener('click', () => { if (openDD?.btn === btn) closeDD(); else open(); });
    btn.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); open(); }
    });
    menu.addEventListener('keydown', (e) => {
      const list = items();
      const i = list.indexOf(document.activeElement);
      if (e.key === 'ArrowDown') { e.preventDefault(); list[Math.min(i + 1, list.length - 1)]?.focus(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); list[Math.max(i - 1, 0)]?.focus(); }
      else if (e.key === 'Home') { e.preventDefault(); list[0]?.focus(); }
      else if (e.key === 'End') { e.preventDefault(); list[list.length - 1]?.focus(); }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeDD(true); }
      else if (e.key === 'Tab') closeDD();
    });
    select.addEventListener('sync', paint);
    select.addEventListener('change', paint);
    paint();
  }
  function chevron() {
    const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    s.setAttribute('viewBox', '0 0 24 24');
    s.setAttribute('width', '12');
    s.setAttribute('height', '12');
    s.setAttribute('aria-hidden', 'true');
    s.setAttribute('class', 'dd-chev');
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', 'M6 9l6 6 6-6');
    s.append(p);
    return s;
  }
  document.addEventListener('pointerdown', (e) => { if (openDD && !openDD.menu.parentNode.contains(e.target)) closeDD(); });
  for (const s of document.querySelectorAll('select.dd-src')) dropdown(s);
  const syncSelects = () => { for (const s of document.querySelectorAll('select.dd-src')) s.dispatchEvent(new Event('sync')); };

  const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
  const billions = (n) => (n / 1e9).toFixed(n >= 10e9 ? 0 : 1) + 'B';
  function ago(iso) {
    const days = (Date.now() - new Date(iso)) / 864e5;
    if (days < 1) return 'today';
    if (days < 60) return `${Math.floor(days)}d ago`;
    if (days < 730) return `${Math.floor(days / 30)}mo ago`;
    return `${Math.floor(days / 365)}y ago`;
  }

  // ---- URL state: the search and every filter, so views can be linked ----
  function currentURL() {
    const p = new URLSearchParams();
    if (q.value.trim()) p.set('q', q.value.trim());
    p.set('author', author.value);
    if (sort.value !== 'downloads') p.set('sort', sort.value);
    if (minIn.value) p.set('min', minIn.value);
    if (maxIn.value) p.set('max', maxIn.value);
    if (type) p.set('type', type);
    if (archs.size) p.set('arch', [...archs].join(','));
    if (takes.size) p.set('takes', [...takes].join(','));
    if (makes.size) p.set('makes', [...makes].join(','));
    for (const [key, id] of Object.entries(checks)) if ($(id).checked) p.set(key, '1');
    if (openId) p.set('open', openId);
    return '?' + p;
  }
  function saveURL() {
    history.replaceState(history.state, '', currentURL());
  }

  function loadURL() {
    const p = new URLSearchParams(location.search);
    if (p.has('q')) q.value = p.get('q');
    if (p.has('author')) author.value = p.get('author');
    if (p.has('sort') && [...sort.options].some((o) => o.value === p.get('sort'))) sort.value = p.get('sort');
    minIn.value = p.get('min') || '';
    maxIn.value = p.get('max') || '';
    setType(p.get('type') || '');
    for (const a of (p.get('arch') || '').split(',')) if (a) archs.add(a);
    for (const m of (p.get('takes') || '').split(',')) if (MODS.includes(m)) takes.add(m);
    for (const m of (p.get('makes') || '').split(',')) if (MODS.includes(m)) makes.add(m);
    if (p.get('vision') === '1') takes.add('image'); // the old Vision checkbox
    syncIO();
    for (const [key, id] of Object.entries(checks)) $(id).checked = p.get(key) === '1';
    syncSelects();
  }

  // ---- search (server) ----
  async function search() {
    controller?.abort();
    controller = new AbortController();
    const params = new URLSearchParams({ q: q.value.trim(), author: author.value });
    if (takes.size) params.set('takes', [...takes].join(','));
    if (makes.size) params.set('makes', [...makes].join(','));
    fetched = ioKey();
    saveURL();
    startLoading();
    try {
      const res = await fetch('/api/hf/search?' + params, { signal: controller.signal });
      if (!res.ok) throw new Error(await errorText(res));
      const body = await res.json();
      results = body.results;
      truncated = body.truncated;
      max = body.max || max;
      limit = PAGE;
      stopLoading();
      buildArchFilter();
      render();
    } catch (err) {
      if (err.name === 'AbortError') return;
      stopLoading();
      results = [];
      list.replaceChildren();
      more.hidden = true;
      meta.textContent = `Search failed: ${err.message}`;
    }
  }

  // Slow searches show placeholder rows shaped like real ones, but only after
  // 200 ms: a fast answer never flashes them. Rows already on screen stay,
  // dimmed, until the new ones arrive.
  let loadTimer = 0;
  function startLoading() {
    clearTimeout(loadTimer);
    list.setAttribute('aria-busy', 'true');
    $('score-note').textContent = '';
    loadTimer = setTimeout(() => {
      meta.textContent = '\u00a0';
      if (list.querySelector('.result-row:not(.ghost)')) { list.classList.add('stale'); return; }
      list.replaceChildren(...Array.from({ length: 8 }, (_, i) => ghost(i)));
      more.hidden = true;
    }, 200);
  }
  function stopLoading() {
    clearTimeout(loadTimer);
    list.removeAttribute('aria-busy');
    list.classList.remove('stale');
  }
  function ghost(i) {
    const item = el('div', 'result-row ghost');
    item.setAttribute('aria-hidden', 'true');
    const main = el('div', 'r-main');
    const w = (n) => { const b = el('span', 'sk'); b.style.width = n + '%'; return b; };
    main.append(w(34 + ((i * 17) % 28)), w(46 + ((i * 11) % 22)));
    item.append(main, el('span', 'sk num'), el('span', 'sk num'), el('span', 'sk fit'), el('span', 'sk num'));
    return item;
  }

  // ---- filters ----
  // The server only needs to hear about Takes/Makes when they point at models
  // an ordinary search buries (media makers, audio or video input).
  let fetched = '';
  const ioKey = () => {
    const media = [...makes].some((m) => m !== 'text') || takes.has('audio') || takes.has('video');
    return media ? [...takes].join() + '>' + [...makes].join() : '';
  };

  function syncIO() {
    for (const [set, id] of [[takes, 'f-takes'], [makes, 'f-makes']]) {
      for (const b of $(id).querySelectorAll('.chip')) b.setAttribute('aria-pressed', String(set.has(b.dataset.mod)));
    }
    $('f-io-note').hidden = ![...makes].some((m) => m !== 'text');
  }

  function toggleIO(set, mod) {
    if (!set.delete(mod)) set.add(mod);
    ioChanged();
  }

  // Set Takes and Makes at once, e.g. from a tag on a result.
  function setIO(inn, out) {
    takes.clear(); makes.clear();
    for (const m of inn) takes.add(m);
    for (const m of out) makes.add(m);
    ioChanged();
  }

  function ioChanged() {
    syncIO();
    if (ioKey() !== fetched) { limit = PAGE; search(); } else changed();
  }

  function buildArchFilter() {
    const counts = new Map();
    for (const r of results) {
      const a = r.arch || 'unknown';
      counts.set(a, (counts.get(a) || 0) + 1);
    }
    for (const a of archs) if (!counts.has(a)) counts.set(a, 0); // keep selections visible
    archBox.replaceChildren();
    if (!counts.size) {
      archBox.append(el('span', 'muted small', 'No results'));
      return;
    }
    for (const [arch, n] of [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))) {
      const label = el('label');
      const box = el('input');
      box.type = 'checkbox';
      box.className = 'check';
      box.checked = archs.has(arch);
      box.addEventListener('change', () => {
        if (box.checked) archs.add(arch); else archs.delete(arch);
        changed();
      });
      label.append(box, el('span', 'arch-name', arch), el('span', 'count', String(n)));
      archBox.append(label);
    }
  }

  function setType(value) {
    type = value;
    for (const x of document.querySelectorAll('#f-type button')) x.classList.toggle('on', x.dataset.value === value);
  }

  function passes(r) {
    const b = r.params / 1e9;
    const min = parseFloat(minIn.value);
    const maxB = parseFloat(maxIn.value);
    if (!Number.isNaN(min) && (!r.params || b < min)) return false;
    if (!Number.isNaN(maxB) && (!r.params || b > maxB)) return false;
    if (type === 'moe' && !r.moe) return false;
    if (type === 'dense' && r.moe) return false;
    if (archs.size && !archs.has(r.arch || 'unknown')) return false;
    for (const m of takes) if (!r.in.includes(m)) return false;
    for (const m of makes) if (!r.out.includes(m)) return false;
    // Generators show only when something other than text is asked for.
    if (r.generator && ![...makes].some((m) => m !== 'text')) return false;
    if ($('f-longctx').checked && r.context < 131072) return false;
    if ($('f-rated').checked && r.score == null) return false;
    if ($('f-fits').checked && r.fit.level !== 'gpu' && r.fit.level !== 'tight') return false;
    if ($('f-hide-installed').checked && r.installed) return false;
    if ($('f-hide-gated').checked && r.gated) return false;
    if ($('f-hide-drafts').checked && r.draft) return false;
    return true;
  }

  const sorters = {
    downloads: (a, b) => b.downloads - a.downloads,
    trendingScore: (a, b) => b.trending - a.trending || b.downloads - a.downloads,
    likes: (a, b) => b.likes - a.likes,
    lastModified: (a, b) => new Date(b.updated) - new Date(a.updated),
    'params-desc': (a, b) => b.params - a.params,
    'params-asc': (a, b) => (a.params || Infinity) - (b.params || Infinity),
    // Unrated models sort last, then by downloads.
    score: (a, b) => (b.score ?? -1) - (a.score ?? -1) || b.downloads - a.downloads,
  };

  // The first number column follows the sort when it ranks by something
  // other than downloads.
  const POP = {
    likes: ['Likes', (r) => compact.format(r.likes), ''],
    lastModified: ['Updated', (r) => ago(r.updated), ''],
    trendingScore: ['Trending', (r) => String(Math.round(r.trending)), ''],
  };
  const popCol = () => POP[sort.value] || ['Downloads', (r) => compact.format(r.downloads), ''];

  // Any filter or sort change starts the list from the top again.
  function changed() {
    limit = PAGE;
    saveURL();
    render();
  }

  function render() {
    const shown = results.filter(passes).sort(sorters[sort.value] || sorters.downloads);
    if (!results.length) {
      meta.textContent = 'No GGUF repos match that search.';
    } else {
      meta.textContent = shown.length === results.length
        ? (truncated ? `${shown.length} most downloaded models. Narrow the search to see others` : `${shown.length} models on Hugging Face`)
        : `${shown.length} of ${results.length} match your filters`;
    }

    const rated = shown.filter((r) => r.score != null).length;
    $('score-note').textContent = shown.length ? `, ${rated} of ${shown.length} shown are rated` : '';

    $('th-pop').textContent = popCol()[0];
    const by = sort.value;
    for (const b of document.querySelectorAll('#results-head [data-sortto]')) {
      const to = b.dataset.sortto;
      b.classList.toggle('on', to === 'size' ? by.startsWith('params') : to === 'score' ? by === 'score' : by === 'downloads' || !!POP[by]);
    }

    for (const chip of document.querySelectorAll('#f-size-chips .chip')) {
      chip.classList.toggle('on', chip.dataset.min === minIn.value && chip.dataset.max === maxIn.value);
    }

    list.replaceChildren(...shown.slice(0, limit).map(row));
    markSelected();
    const left = shown.length - limit;
    more.hidden = left <= 0;
    more.textContent = `Show ${Math.min(left, PAGE)} more · ${left} left`;
  }

  // Facts under a name are quiet buttons that narrow the list to things like it.
  function facet(text, title, apply) {
    const b = el('button', 'r-facet', text);
    b.type = 'button';
    b.title = title;
    b.addEventListener('click', apply);
    return b;
  }

  function numCell(cls) { return el('div', 'r-cell ' + cls); }

  function row(r) {
    const item = el('div', 'result-row');
    item.dataset.id = r.id;

    // name, status chips, then facts
    const main = el('div', 'r-main');
    const head = el('div', 'r-title');
    const name = el('a', 'r-name', r.title || r.name);
    name.href = '/discover/' + r.id;
    name.title = r.id;
    name.addEventListener('click', (e) => {
      if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return; // let the browser open it
      e.preventDefault();
      openPanel(r.id);
    });
    head.append(name);
    for (const t of r.tags || []) head.append(el('span', 'tag', t));
    if (r.installed) head.append(el('span', 'tag good', 'installed'));
    if (r.gated) head.append(el('span', 'tag warn', 'gated'));
    if (r.draft) {
      const t = el('span', 'tag draft', 'draft model');
      t.title = 'A small helper for speculative decoding, not a model to chat with';
      head.append(t);
    }
    if (r.cantRun) {
      const t = el('span', 'tag draft', 'not runnable here');
      t.title = r.cantRun;
      head.append(t);
    }
    main.append(head);

    const facts = el('div', 'r-facts');
    facts.append(facet(r.author, `Only show models from ${r.author}`, () => {
      if (author.value === r.author) return;
      if (![...author.options].some((o) => o.value === r.author)) author.append(new Option(r.author, r.author));
      author.value = r.author;
      syncSelects();
      search();
    }));
    if (r.params) {
      facts.append(facet(`${r.moe ? 'MoE' : 'dense'} ${billions(r.params)}`, `Only ${r.moe ? 'MoE' : 'dense'} models`, () => {
        setType(r.moe ? 'moe' : 'dense');
        changed();
      }));
    }
    if (r.arch) {
      facts.append(facet(r.arch, `Only ${r.arch} models`, () => {
        archs.clear();
        archs.add(r.arch);
        buildArchFilter();
        changed();
      }));
    }
    const io = `${r.in.join('+') || 'nothing'} → ${r.out.join('+') || 'embeddings'}`;
    const iof = facet(io, 'Only models that take and make this', () => setIO(r.in, r.out));
    if (r.generator) iof.classList.add('generator');
    facts.append(iof);
    if (r.context >= 131072) {
      facts.append(facet(`${Math.round(r.context / 1024)}k ctx`, 'Only models with 128k+ context', () => { $('f-longctx').checked = true; changed(); }));
    }
    main.append(facts);

    // downloads (or whatever the sort ranks by)
    const pop = numCell('pop');
    const [, popText] = popCol();
    const pb = el('button', 'r-num', popText(r));
    pb.type = 'button';
    pb.title = sort.value === 'downloads' || !POP[sort.value] ? `${r.downloads.toLocaleString()} downloads · ${r.likes.toLocaleString()} likes` : `${compact.format(r.downloads)} downloads`;
    pb.addEventListener('click', () => setSort(POP[sort.value] ? sort.value : 'downloads'));
    pop.append(pb);

    // size and fit
    const size = numCell('size');
    const fit = numCell('fit');
    if (r.cantRun) {
      size.append(el('span', 'muted', '–'));
      const f = el('span', 'r-fit fit-draft', 'Can’t run here');
      f.title = r.cantRun;
      fit.append(f, el('span', 'r-fit-sub', r.generator ? `makes ${r.out.join(' and ')}` : 'needs whisper.cpp'));
    } else if (r.draft) {
      size.append(el('span', 'muted', '–'));
      const f = el('span', 'r-fit fit-draft', 'Draft model');
      f.title = 'Speeds up another model through speculative decoding; it can’t chat on its own';
      fit.append(f, el('span', 'r-fit-sub', r.draftFor ? `for ${r.draftFor.split('/').pop()}` : 'not the actual model'));
    } else if (r.q4Size) {
      const sb = el('button', 'r-num');
      sb.type = 'button';
      sb.append('~' + (r.q4Size / 1e9).toFixed(r.q4Size >= 1e10 ? 0 : 1), el('span', 'r-unit', ' GB'));
      sb.title = `About ${fmtBytes(r.q4Size)} at Q4 with an 8k context. Click to sort by size.`;
      sb.addEventListener('click', () => setSort(sort.value === 'params-desc' ? 'params-asc' : 'params-desc'));
      size.append(sb);
      const f = el('span', `r-fit ${r.fit.level}`, r.fit.label);
      f.title = `~${fmtBytes(r.q4Size)} at Q4 with 8k context`;
      const track = el('span', 'lib-track thin');
      const fill = el('span', `lib-fill fit-${r.fit.level}`);
      fill.style.width = `${r.fit.percent}%`;
      track.append(fill);
      fit.append(f, track);
    } else {
      size.append(el('span', 'muted', '–'));
      fit.append(el('span', 'r-fit', 'Size unknown'));
    }

    // intelligence score, linking to where it comes from
    const score = numCell('score');
    if (r.score == null) {
      const none = el('span', 'r-none', 'not rated');
      none.title = 'No Intelligence Index score for this model';
      score.append(none);
    } else {
      const a = el('a', 'r-num', r.score.toFixed(0));
      a.href = 'https://artificialanalysis.ai';
      a.target = '_blank';
      a.rel = 'noopener';
      a.title = `Artificial Analysis Intelligence Index of ${r.scoreOf}. Click to see where it comes from.`;
      score.append(a);
    }

    item.append(main, pop, size, fit, score);
    return item;
  }

  function setSort(v) {
    if (sort.value === v) return;
    sort.value = v;
    sort.dispatchEvent(new Event('sync'));
    changed();
  }

  // ---- detail panel: the model page's fragment, shown beside the list ----
  function markSelected() {
    for (const x of list.children) x.classList.toggle('selected', x.dataset.id === openId);
  }

  async function loadDetail(id, ctx) {
    detailController?.abort();
    detailController = new AbortController();
    const url = `/discover/${id}?partial=1` + (ctx ? '&ctx=' + encodeURIComponent(ctx) : '');
    panelBody.setAttribute('aria-busy', 'true');
    if (!ctx) {
      panelBody.replaceChildren(el('p', 'muted panel-loading', 'Loading…'));
      panel.scrollTop = 0;
    }
    try {
      const res = await fetch(url, { signal: detailController.signal });
      const html = await res.text();
      if (!res.ok && !html.includes('banner')) throw new Error(`HTTP ${res.status}`);
      panelBody.innerHTML = html;
      pollDownloads?.(); // show downloads already in progress on these quants
    } catch (err) {
      if (err.name === 'AbortError') return;
      panelBody.replaceChildren(el('div', 'banner error', `Couldn't load this model: ${err.message}`));
    }
    panelBody.removeAttribute('aria-busy');
  }

  // Show the panel for id (or hide it for null) without touching history.
  function showPanel(id) {
    if (id === openId) return;
    openId = id;
    panel.hidden = !id;
    root.classList.toggle('panel-open', !!id);
    if (!id) {
      detailController?.abort();
      root.classList.remove('filters-open');
      setToggle();
    } else {
      loadDetail(id);
    }
    markSelected();
  }

  function openPanel(id) {
    if (id === openId) return;
    const first = !openId;
    showPanel(id);
    // The first open adds a history entry so Back closes the panel.
    if (first) history.pushState({ panel: true }, '', currentURL());
    else history.replaceState({ panel: true }, '', currentURL());
  }

  function closePanel() {
    if (!openId) return;
    if (history.state?.panel) { history.back(); return; } // popstate does the rest
    showPanel(null);
    saveURL();
  }

  const phone = matchMedia('(max-width: 760px)');
  phone.addEventListener('change', () => setToggle());
  function setToggle() {
    const t = $('filters-toggle');
    t.hidden = !openId && !phone.matches;
    t.setAttribute('aria-expanded', String(root.classList.contains('filters-open')));
  }
  new MutationObserver(setToggle).observe(root, { attributes: true, attributeFilter: ['class'] });
  $('filters-toggle').addEventListener('click', () => root.classList.toggle('filters-open'));
  setToggle();

  panel.addEventListener('click', (e) => { if (e.target.closest('[data-close-panel]')) closePanel(); });
  panelBody.addEventListener('change', (e) => {
    if (e.target.matches('select[name="ctx"]') && openId) loadDetail(openId, e.target.value);
  });
  panelBody.addEventListener('submit', (e) => e.preventDefault());
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && openId && !e.defaultPrevented) closePanel();
  });
  window.addEventListener('popstate', () => {
    const id = new URLSearchParams(location.search).get('open');
    showPanel(id && /^[\w.-]+\/[\w.-]+$/.test(id) ? id : null);
  });

  // ---- wiring ----
  let debounce;
  q.addEventListener('input', () => { clearTimeout(debounce); debounce = setTimeout(search, 350); });
  q.addEventListener('keydown', (e) => { if (e.key === 'Enter') { clearTimeout(debounce); search(); } });
  author.addEventListener('change', search);
  for (const b of document.querySelectorAll('#results-head [data-sortto]')) {
    b.addEventListener('click', () => {
      const to = b.dataset.sortto;
      if (to === 'size') setSort(sort.value === 'params-desc' ? 'params-asc' : 'params-desc');
      else if (to === 'downloads') setSort(POP[sort.value] ? sort.value : 'downloads');
      else setSort(to);
    });
  }
  sort.addEventListener('change', changed);
  more.addEventListener('click', () => { limit += PAGE; render(); });

  for (const input of [minIn, maxIn]) input.addEventListener('input', changed);
  for (const id of Object.values(checks)) $(id).addEventListener('change', changed);
  for (const chip of document.querySelectorAll('#f-size-chips .chip')) {
    chip.addEventListener('click', () => {
      const same = chip.classList.contains('on');
      minIn.value = same ? '' : chip.dataset.min;
      maxIn.value = same ? '' : chip.dataset.max;
      changed();
    });
  }
  for (const b of document.querySelectorAll('#f-type button')) {
    b.addEventListener('click', () => { setType(b.dataset.value); changed(); });
  }
  for (const [set, id] of [[takes, 'f-takes'], [makes, 'f-makes']]) {
    for (const b of $(id).querySelectorAll('.chip')) b.addEventListener('click', () => toggleIO(set, b.dataset.mod));
  }
  $('f-clear').addEventListener('click', () => {
    const refetch = ioKey() !== '';
    takes.clear(); makes.clear(); syncIO();
    minIn.value = maxIn.value = '';
    setType('');
    archs.clear();
    for (const id of Object.values(checks)) $(id).checked = false;
    buildArchFilter();
    if (refetch) search(); else changed();
  });

  loadURL();
  if (![...author.options].some((o) => o.value === author.value)) {
    const wanted = new URLSearchParams(location.search).get('author');
    if (wanted) { author.append(new Option(wanted, wanted)); author.value = wanted; }
  }
  const deep = new URLSearchParams(location.search).get('open');
  window.assayWhenVisible(() => {
    if (deep && /^[\w.-]+\/[\w.-]+$/.test(deep)) showPanel(deep);
    search();
  });
})();
