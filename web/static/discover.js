// Discover: search Hugging Face, then filter and sort the results locally.
(() => {
  const root = document.getElementById('discover');
  if (!root) return;
  const { el, errorText, fmtBytes } = window.ollamui;
  const $ = (id) => document.getElementById(id);

  const q = $('q');
  const author = $('author');
  const sort = $('sort');
  const meta = $('result-meta');
  const list = $('results');
  const minIn = $('f-min');
  const maxIn = $('f-max');
  const archBox = $('f-arch');

  const memory = JSON.parse(root.dataset.memory || '{}');
  $('memory-line').textContent = memory.vram
    ? `${memory.gpu}: ${fmtBytes(memory.vram)} VRAM · ${fmtBytes(memory.ram)} RAM`
    : `No GPU detected · ${fmtBytes(memory.ram)} RAM`;

  let results = [];
  let type = '';
  const excludedArch = new Set();
  let controller = null;

  const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
  const billions = (n) => (n / 1e9).toFixed(n >= 10e9 ? 0 : 1) + 'B';
  function ago(iso) {
    const days = (Date.now() - new Date(iso)) / 864e5;
    if (days < 1) return 'today';
    if (days < 60) return `${Math.floor(days)}d ago`;
    if (days < 730) return `${Math.floor(days / 30)}mo ago`;
    return `${Math.floor(days / 365)}y ago`;
  }

  // ---- search (server) ----
  async function search() {
    controller?.abort();
    controller = new AbortController();
    const params = new URLSearchParams({ q: q.value.trim(), author: author.value });
    if (!sort.value.startsWith('params')) params.set('sort', sort.value);
    history.replaceState(null, '', '?' + new URLSearchParams({ q: q.value.trim(), author: author.value, sort: sort.value }));
    meta.textContent = 'Searching Hugging Face…';
    try {
      const res = await fetch('/api/hf/search?' + params, { signal: controller.signal });
      if (!res.ok) throw new Error(await errorText(res));
      results = (await res.json()).results;
      buildArchFilter();
      render();
    } catch (err) {
      if (err.name === 'AbortError') return;
      results = [];
      list.replaceChildren();
      meta.textContent = `Search failed: ${err.message}`;
    }
  }

  // ---- filters (local) ----
  function buildArchFilter() {
    const counts = new Map();
    for (const r of results) {
      const a = r.arch || 'unknown';
      counts.set(a, (counts.get(a) || 0) + 1);
    }
    archBox.replaceChildren();
    if (!counts.size) {
      archBox.append(el('span', 'muted small', 'No results'));
      return;
    }
    for (const [arch, n] of [...counts].sort((a, b) => b[1] - a[1])) {
      const label = el('label');
      const box = el('input');
      box.type = 'checkbox';
      box.checked = !excludedArch.has(arch);
      box.addEventListener('change', () => {
        if (box.checked) excludedArch.delete(arch); else excludedArch.add(arch);
        render();
      });
      label.append(box, document.createTextNode(arch), el('span', 'grow'), el('span', 'count', String(n)));
      archBox.append(label);
    }
  }

  function passes(r) {
    const b = r.params / 1e9;
    const min = parseFloat(minIn.value);
    const max = parseFloat(maxIn.value);
    if (!Number.isNaN(min) && (!r.params || b < min)) return false;
    if (!Number.isNaN(max) && (!r.params || b > max)) return false;
    if (type === 'moe' && !r.moe) return false;
    if (type === 'dense' && r.moe) return false;
    if (excludedArch.has(r.arch || 'unknown')) return false;
    if ($('f-vision').checked && !r.vision) return false;
    if ($('f-longctx').checked && r.context < 131072) return false;
    if ($('f-fits').checked && r.fit.level !== 'gpu' && r.fit.level !== 'tight') return false;
    if ($('f-hide-installed').checked && r.installed) return false;
    if ($('f-hide-gated').checked && r.gated) return false;
    return true;
  }

  function render() {
    let shown = results.filter(passes);
    if (sort.value === 'params-desc') shown = [...shown].sort((a, b) => b.params - a.params);
    if (sort.value === 'params-asc') shown = [...shown].sort((a, b) => (a.params || Infinity) - (b.params || Infinity));
    meta.textContent = results.length
      ? `${shown.length} of ${results.length} shown (top ${results.length} from Hugging Face; filters apply to these)`
      : 'No GGUF repos match that search.';

    for (const chip of document.querySelectorAll('#f-size-chips .chip')) {
      chip.classList.toggle('on', chip.dataset.min === minIn.value && chip.dataset.max === maxIn.value);
    }

    list.replaceChildren(...shown.map(row));
  }

  function tag(text, cls) { return el('span', 'tag' + (cls ? ' ' + cls : ''), text); }

  function row(r) {
    const a = el('a', 'result-row');
    a.href = '/discover/' + r.id;

    const main = el('div', 'r-main');
    main.append(el('div', 'r-name', r.name));
    const tags = el('div', 'tags');
    tags.append(el('span', 'muted', r.author));
    if (r.params) tags.append(tag(`${r.moe ? 'MoE' : 'dense'} ${billions(r.params)}`));
    if (r.arch) tags.append(tag(r.arch));
    if (r.vision) tags.append(tag('vision'));
    if (r.context >= 131072) tags.append(tag(`${Math.round(r.context / 1024)}k ctx`));
    if (r.installed) tags.append(tag('installed', 'good'));
    if (r.gated) tags.append(tag('gated', 'warn'));
    main.append(tags);

    const size = el('div', 'r-size');
    if (r.q4Size) {
      size.append(el('div', null, `~${fmtBytes(r.q4Size)} at Q4`));
      size.append(el('div', `small fit-${r.fit.level}`, r.fit.label));
    } else {
      size.append(el('div', 'muted small', 'size unknown'));
    }

    const pop = el('div', 'r-pop small muted');
    pop.append(el('div', null, `${compact.format(r.downloads)} downloads · ${compact.format(r.likes)} likes`));
    pop.append(el('div', null, `updated ${ago(r.updated)}`));

    a.append(main, size, pop);
    return a;
  }

  // ---- wiring ----
  let debounce;
  q.addEventListener('input', () => { clearTimeout(debounce); debounce = setTimeout(search, 350); });
  q.addEventListener('keydown', (e) => { if (e.key === 'Enter') { clearTimeout(debounce); search(); } });
  author.addEventListener('change', search);
  sort.addEventListener('change', () => (sort.value.startsWith('params') && results.length ? render() : search()));

  for (const input of [minIn, maxIn]) input.addEventListener('input', render);
  for (const id of ['f-vision', 'f-longctx', 'f-fits', 'f-hide-installed', 'f-hide-gated']) $(id).addEventListener('change', render);
  for (const chip of document.querySelectorAll('#f-size-chips .chip')) {
    chip.addEventListener('click', () => {
      const same = chip.classList.contains('on');
      minIn.value = same ? '' : chip.dataset.min;
      maxIn.value = same ? '' : chip.dataset.max;
      render();
    });
  }
  for (const b of document.querySelectorAll('#f-type button')) {
    b.addEventListener('click', () => {
      type = b.dataset.value;
      for (const x of document.querySelectorAll('#f-type button')) x.classList.toggle('on', x === b);
      render();
    });
  }
  $('f-clear').addEventListener('click', () => {
    minIn.value = maxIn.value = '';
    type = '';
    for (const x of document.querySelectorAll('#f-type button')) x.classList.toggle('on', x.dataset.value === '');
    excludedArch.clear();
    for (const id of ['f-vision', 'f-longctx', 'f-fits', 'f-hide-installed', 'f-hide-gated']) $(id).checked = false;
    buildArchFilter();
    render();
  });

  const initial = new URLSearchParams(location.search);
  if (initial.has('q')) q.value = initial.get('q');
  if (initial.has('author')) author.value = initial.get('author');
  if (initial.has('sort')) sort.value = initial.get('sort');
  search();
})();
