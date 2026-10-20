// Discover: fetch every match for a search once, then sort, filter and page
// through it locally. Sorting only reorders; it never changes what matches.
(() => {
  const root = document.getElementById('discover');
  if (!root) return;
  const { el, errorText, fmtBytes, pollDownloads } = window.ollamui;
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
  const checks = { vision: 'f-vision', long: 'f-longctx', fits: 'f-fits', hideInstalled: 'f-hide-installed', hideGated: 'f-hide-gated' };

  const PAGE = 20;
  const memory = JSON.parse(root.dataset.memory || '{}');
  $('memory-line').textContent = memory.vram
    ? `${memory.gpu}: ${fmtBytes(memory.vram)} VRAM · ${fmtBytes(memory.ram)} RAM`
    : `No GPU detected · ${fmtBytes(memory.ram)} RAM`;

  let results = [];
  let truncated = false;
  let max = 1000;
  let type = '';
  const archs = new Set(); // selected architectures; empty means any
  let limit = PAGE;
  let controller = null;
  let openId = null; // repo shown in the detail panel
  let detailController = null;
  const panel = $('detail-panel');
  const panelBody = $('detail-body');

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
    for (const [key, id] of Object.entries(checks)) $(id).checked = p.get(key) === '1';
  }

  // ---- search (server) ----
  async function search() {
    controller?.abort();
    controller = new AbortController();
    const params = new URLSearchParams({ q: q.value.trim(), author: author.value });
    saveURL();
    meta.textContent = 'Searching Hugging Face…';
    try {
      const res = await fetch('/api/hf/search?' + params, { signal: controller.signal });
      if (!res.ok) throw new Error(await errorText(res));
      const body = await res.json();
      results = body.results;
      truncated = body.truncated;
      max = body.max || max;
      limit = PAGE;
      buildArchFilter();
      render();
    } catch (err) {
      if (err.name === 'AbortError') return;
      results = [];
      list.replaceChildren();
      more.hidden = true;
      meta.textContent = `Search failed: ${err.message}`;
    }
  }

  // ---- filters ----
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
      box.checked = archs.has(arch);
      box.addEventListener('change', () => {
        if (box.checked) archs.add(arch); else archs.delete(arch);
        changed();
      });
      label.append(box, document.createTextNode(arch), el('span', 'grow'), el('span', 'count', String(n)));
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
    if ($('f-vision').checked && !r.vision) return false;
    if ($('f-longctx').checked && r.context < 131072) return false;
    if ($('f-fits').checked && r.fit.level !== 'gpu' && r.fit.level !== 'tight') return false;
    if ($('f-hide-installed').checked && r.installed) return false;
    if ($('f-hide-gated').checked && r.gated) return false;
    return true;
  }

  const sorters = {
    downloads: (a, b) => b.downloads - a.downloads,
    trendingScore: (a, b) => b.trending - a.trending || b.downloads - a.downloads,
    likes: (a, b) => b.likes - a.likes,
    lastModified: (a, b) => new Date(b.updated) - new Date(a.updated),
    'params-desc': (a, b) => b.params - a.params,
    'params-asc': (a, b) => (a.params || Infinity) - (b.params || Infinity),
  };

  // What the current sort is ranking by, shown on the right of each row.
  function sortFact(r) {
    switch (sort.value) {
      case 'likes': return `${compact.format(r.likes)} likes`;
      case 'lastModified': return `updated ${ago(r.updated)}`;
      case 'trendingScore': return `trending ${Math.round(r.trending)}`;
      case 'params-desc': case 'params-asc': return `${compact.format(r.downloads)} downloads`;
      default: return `${compact.format(r.downloads)} downloads`;
    }
  }

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
      const scope = truncated ? `the ${results.length} most downloaded of many matches. Narrow the search to see others` : `${results.length} on Hugging Face`;
      meta.textContent = shown.length === results.length
        ? `${shown.length} models (${scope})`
        : `${shown.length} of ${results.length} match your filters${truncated ? ` (searched ${scope})` : ''}`;
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

  // Tags are buttons that narrow the list to things like them.
  function filterTag(text, title, apply, cls) {
    const b = el('button', 'tag tag-btn' + (cls ? ' ' + cls : ''), text);
    b.type = 'button';
    b.title = title;
    b.addEventListener('click', apply);
    return b;
  }

  function row(r) {
    const item = el('div', 'result-row');

    const main = el('div', 'r-main');
    item.dataset.id = r.id;
    const name = el('a', 'r-name', r.name);
    name.href = '/discover/' + r.id;
    name.addEventListener('click', (e) => {
      if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return; // let the browser open it
      e.preventDefault();
      openPanel(r.id);
    });
    main.append(name);

    const tags = el('div', 'tags');
    tags.append(filterTag(r.author, `Only show models from ${r.author}`, () => {
      if (author.value === r.author) return;
      if (![...author.options].some((o) => o.value === r.author)) author.append(new Option(r.author, r.author));
      author.value = r.author;
      search();
    }, 'plain'));
    if (r.params) {
      tags.append(filterTag(`${r.moe ? 'MoE' : 'dense'} ${billions(r.params)}`, `Only ${r.moe ? 'MoE' : 'dense'} models`, () => {
        setType(r.moe ? 'moe' : 'dense');
        changed();
      }));
    }
    if (r.arch) {
      tags.append(filterTag(r.arch, `Only ${r.arch} models`, () => {
        archs.clear();
        archs.add(r.arch);
        buildArchFilter();
        changed();
      }));
    }
    if (r.vision) tags.append(filterTag('vision', 'Only models that take images', () => { $('f-vision').checked = true; changed(); }));
    if (r.context >= 131072) {
      tags.append(filterTag(`${Math.round(r.context / 1024)}k ctx`, 'Only models with 128k+ context', () => { $('f-longctx').checked = true; changed(); }));
    }
    if (r.installed) tags.append(el('span', 'tag good', 'installed'));
    if (r.gated) tags.append(el('span', 'tag warn', 'gated'));
    main.append(tags);

    const side = el('div', 'r-side small');
    if (r.q4Size) {
      const fit = el('div', `r-fit fit-${r.fit.level}`, r.fit.label);
      fit.title = `~${fmtBytes(r.q4Size)} at Q4 with 8k context`;
      side.append(fit, el('div', 'muted', `~${fmtBytes(r.q4Size)} · ${sortFact(r)}`));
    } else {
      side.append(el('div', 'muted', 'size unknown'), el('div', 'muted', sortFact(r)));
    }

    item.append(main, side);
    return item;
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

  function setToggle() {
    const t = $('filters-toggle');
    t.hidden = !openId;
    t.setAttribute('aria-expanded', String(root.classList.contains('filters-open')));
  }
  new MutationObserver(setToggle).observe(root, { attributes: true, attributeFilter: ['class'] });
  $('filters-toggle').addEventListener('click', () => root.classList.toggle('filters-open'));

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
  $('f-clear').addEventListener('click', () => {
    minIn.value = maxIn.value = '';
    setType('');
    archs.clear();
    for (const id of Object.values(checks)) $(id).checked = false;
    buildArchFilter();
    changed();
  });

  loadURL();
  if (![...author.options].some((o) => o.value === author.value)) {
    const wanted = new URLSearchParams(location.search).get('author');
    if (wanted) { author.append(new Option(wanted, wanted)); author.value = wanted; }
  }
  const deep = new URLSearchParams(location.search).get('open');
  if (deep && /^[\w.-]+\/[\w.-]+$/.test(deep)) showPanel(deep);
  search();
})();
