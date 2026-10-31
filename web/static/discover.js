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
    const side = el('div', 'r-side');
    side.append(el('span', 'sk num'), el('span', 'sk num'), el('span', 'sk num'), el('span', 'sk pill'));
    item.append(main, side);
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

  // Takes/Makes live in two places (switches under "Can do" and chips under
  // "More"); both read the same sets.
  function syncIO() {
    for (const b of document.querySelectorAll('[data-io]')) {
      const [kind, mod] = b.dataset.io.split(':');
      const on = (kind === 'takes' ? takes : makes).has(mod);
      if (b.type === 'checkbox') b.checked = on; else b.setAttribute('aria-pressed', String(on));
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
      const nm = el('span', 'arch-name', prettyArch(arch));
      nm.title = arch;
      label.append(box, nm, el('span', 'count muted small', String(n)));
      archBox.append(label);
    }
  }

  // "qwen3moe" -> "Qwen 3 MoE", "qwen35" -> "Qwen 3.5", "glm4" -> "GLM 4".
  const ACRO = { glm: 'GLM', gpt: 'GPT', llm: 'LLM', oss: 'OSS', moe: 'MoE', vl: 'VL', ocr: 'OCR', bert: 'BERT', t5: 'T5' };
  function prettyArch(a) {
    const m = /^([a-z]+?)(\d{1,2})?([a-z]*)$/.exec(a);
    if (!m) return a;
    const word = (w) => ACRO[w] || w.charAt(0).toUpperCase() + w.slice(1);
    const num = m[2] && m[2].length === 2 ? `${m[2][0]}.${m[2][1]}` : m[2];
    const rest = [];
    let t = m[3];
    for (const k of ['vl', 'moe']) if (t.includes(k)) { rest.push(ACRO[k]); t = t.replace(k, ''); }
    if (t) rest.unshift(t);
    return [word(m[1]), num, ...rest].filter(Boolean).join(' ');
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

    for (const b of document.querySelectorAll('#f-size button')) {
      b.classList.toggle('on', b.dataset.min === minIn.value && b.dataset.max === maxIn.value);
    }
    renderActive();

    list.replaceChildren(...shown.slice(0, limit).map(row));
    markSelected();
    const left = shown.length - limit;
    more.hidden = left <= 0;
    more.textContent = `Show ${Math.min(left, PAGE)} more · ${left} left`;
  }

  // ---- active filters as removable chips ----
  const SIZE_NAMES = { '|9': 'Small', '9|35': 'Mid', '35|': 'Large' };
  function renderActive() {
    const box = $('active-chips');
    const items = [];
    const add = (label, undo) => items.push({ label, undo });
    for (const m of takes) add(`Takes ${m}`, () => { takes.delete(m); ioChanged(); });
    for (const m of makes) add(`Makes ${m}`, () => { makes.delete(m); ioChanged(); });
    if (minIn.value || maxIn.value) {
      add(SIZE_NAMES[`${minIn.value}|${maxIn.value}`] || `${minIn.value || 0}–${maxIn.value || 'any'}B`, () => { minIn.value = maxIn.value = ''; changed(); });
    }
    if (type) add(type === 'moe' ? 'MoE only' : 'Dense only', () => { setType(''); changed(); });
    for (const a of archs) add(prettyArch(a), () => { archs.delete(a); buildArchFilter(); changed(); });
    const flags = { long: 'Long memory', rated: 'Has score', fits: 'Fits my computer', hideInstalled: 'Hiding installed', hideGated: 'Hiding gated', hideDrafts: 'Hiding drafts' };
    for (const [key, label] of Object.entries(flags)) {
      if ($(checks[key]).checked) add(label, () => { $(checks[key]).checked = false; changed(); });
    }
    if (author.value !== 'unsloth') add(author.value ? `From ${author.value}` : 'From anyone', () => { author.value = 'unsloth'; search(); });
    box.hidden = !items.length;
    box.replaceChildren(...items.map((it) => {
      const b = el('button', 'achip');
      b.type = 'button';
      b.title = 'Remove this filter';
      b.append(it.label, el('span', 'x', '×'));
      b.addEventListener('click', it.undo);
      return b;
    }));
    if (items.length > 1) {
      const c = el('button', 'link-btn', 'Clear all');
      c.type = 'button';
      c.addEventListener('click', clearAll);
      box.append(c);
    }
  }

  // Facts under a name are quiet buttons that narrow the list to things like it.
  function facet(text, title, apply) {
    const b = el('button', 'r-facet', text);
    b.type = 'button';
    b.title = title;
    b.addEventListener('click', apply);
    return b;
  }

  // Plain words for what a model can do.
  function abilities(r) {
    const w = [];
    if (r.in.includes('image')) w.push(['reads images', () => { takes.add('image'); ioChanged(); }]);
    if (r.in.includes('video')) w.push(['watches video', () => { takes.add('video'); ioChanged(); }]);
    if (r.in.includes('audio')) w.push(['hears audio', () => { takes.add('audio'); ioChanged(); }]);
    for (const m of ['image', 'video', 'audio']) if (r.out.includes(m)) w.push([`makes ${m === 'audio' ? 'audio' : m + 's'}`, () => { makes.add(m); ioChanged(); }]);
    if (!r.out.length) w.push(['embeddings', () => setIO(r.in, [])]);
    return w;
  }

  const FIT_TEXT = { gpu: 'Fits', tight: 'Fits, tight', partial: 'Uses RAM · slower', cpu: 'CPU only', none: 'Too big' };

  function stat(cls, value, label) {
    const d = el('div', 'r-stat ' + cls);
    d.append(value, el('span', 'r-lab', label));
    return d;
  }

  function row(r) {
    const item = el('div', 'result-row');
    item.dataset.id = r.id;

    // name, status chips, then facts
    const main = el('div', 'r-main');
    const head = el('div', 'r-title');
    const name = el('a', 'r-name', r.title || r.name);
    name.href = '/discover/' + r.id;
    name.title = r.id + (r.arch ? ` · ${r.arch}` : '');
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

    // uploader, size, what it can do: each one narrows the list
    const facts = el('div', 'r-facts');
    facts.append(facet(r.author, `Only show models from ${r.author}`, () => {
      if (author.value === r.author) return;
      if (![...author.options].some((o) => o.value === r.author)) author.append(new Option(r.author, r.author));
      author.value = r.author;
      search();
    }));
    if (r.params) {
      facts.append(facet(`${r.moe ? 'MoE' : 'dense'} ${billions(r.params)}`, `Only ${r.moe ? 'MoE' : 'dense'} models`, () => {
        setType(r.moe ? 'moe' : 'dense');
        changed();
      }));
    }
    for (const [text, apply] of abilities(r)) {
      const f = facet(text, 'Only models that can do this', apply);
      if (text.startsWith('makes') && r.generator) f.classList.add('generator');
      facts.append(f);
    }
    if (r.context >= 131072) {
      facts.append(facet('long memory', 'Only models with 128k+ context', () => { $('f-longctx').checked = true; changed(); }));
    }
    main.append(facts);

    const side = el('div', 'r-side');

    // size in GB
    let sizeVal;
    if (r.q4Size && !r.cantRun && !r.draft) {
      sizeVal = el('button', 'r-num');
      sizeVal.type = 'button';
      sizeVal.append('~' + (r.q4Size / 1e9).toFixed(r.q4Size >= 1e10 ? 0 : 1), el('span', 'r-unit', ' GB'));
      sizeVal.title = `About ${fmtBytes(r.q4Size)} at Q4 with an 8k context. Click to sort by size.`;
      sizeVal.addEventListener('click', () => setSort(sort.value === 'params-desc' ? 'params-asc' : 'params-desc'));
    } else {
      sizeVal = el('span', 'r-none', '–');
    }

    // downloads (or whatever the sort ranks by)
    const [popLabel, popText] = popCol();
    const pb = el('button', 'r-num', popText(r));
    pb.type = 'button';
    pb.title = !POP[sort.value] ? `${r.downloads.toLocaleString()} downloads · ${r.likes.toLocaleString()} likes. Click to sort by this.` : `${compact.format(r.downloads)} downloads`;
    pb.addEventListener('click', () => setSort(POP[sort.value] ? sort.value : 'downloads'));

    // intelligence score, linking to where it comes from
    let scoreVal;
    if (r.score == null) {
      scoreVal = el('span', 'r-none', 'not rated');
      scoreVal.title = 'No Intelligence Index score for this model';
    } else {
      scoreVal = el('a', 'r-num', r.score.toFixed(0));
      scoreVal.href = 'https://artificialanalysis.ai';
      scoreVal.target = '_blank';
      scoreVal.rel = 'noopener';
      scoreVal.title = `Artificial Analysis Intelligence Index of ${r.scoreOf}. Click to see where it comes from.`;
    }

    // fit as a small labelled pill
    let pill;
    if (r.cantRun) {
      pill = el('span', 'fit-pill draft', 'Can’t run here');
      pill.title = r.cantRun;
    } else if (r.draft) {
      pill = el('span', 'fit-pill draft', 'Draft model');
      pill.title = 'Speeds up another model through speculative decoding; it can’t chat on its own';
    } else if (r.q4Size) {
      pill = el('span', `fit-pill ${r.fit.level}`, FIT_TEXT[r.fit.level] || r.fit.label);
      pill.title = `${r.fit.label}: ~${fmtBytes(r.q4Size)} at Q4 with 8k context`;
    } else {
      pill = el('span', 'fit-pill unknown', 'Size unknown');
    }

    side.append(stat('pop', pb, popLabel.toLowerCase()), stat('size', sizeVal, 'size'), stat('score', scoreVal, 'score'), pill);
    item.append(main, side);
    return item;
  }

  function setSort(v) {
    if (sort.value === v) return;
    sort.value = v;
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
  // Stat tiles in the panel narrow the list instead of leaving the page.
  panelBody.addEventListener('click', (e) => {
    const a = e.target.closest('[data-facet]');
    if (!a || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey) return;
    e.preventDefault();
    const [kind, val] = a.dataset.facet.split(':');
    if (kind === 'type') setType(val);
    else if (kind === 'long') $('f-longctx').checked = true;
    changed();
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
  for (const b of document.querySelectorAll('#f-size button')) {
    b.addEventListener('click', () => {
      minIn.value = b.dataset.min;
      maxIn.value = b.dataset.max;
      changed();
    });
  }
  for (const b of document.querySelectorAll('#f-type button')) {
    b.addEventListener('click', () => { setType(b.dataset.value); changed(); });
  }
  for (const b of document.querySelectorAll('[data-io]')) {
    const [kind, mod] = b.dataset.io.split(':');
    b.addEventListener(b.type === 'checkbox' ? 'change' : 'click', () => toggleIO(kind === 'takes' ? takes : makes, mod));
  }
  function clearAll() {
    const refetch = ioKey() !== '';
    takes.clear(); makes.clear(); syncIO();
    minIn.value = maxIn.value = '';
    setType('');
    archs.clear();
    for (const id of Object.values(checks)) $(id).checked = false;
    buildArchFilter();
    if (refetch) search(); else changed();
  }
  $('f-clear').addEventListener('click', clearAll);

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
