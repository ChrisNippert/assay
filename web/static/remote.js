// Benchmarks › API models: endpoints (OpenRouter or any OpenAI-compatible
// API) and the ways of running their models. After a change it fires
// "assay:remote" so the leaderboard reloads.
(() => {
  const box = document.getElementById('bm-api');
  if (!box) return;
  const $ = (id) => document.getElementById(id);
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  async function api(method, url, body) {
    const res = await fetch(url, {
      method,
      headers: body == null ? undefined : { 'Content-Type': 'application/json' },
      body: body == null ? undefined : JSON.stringify(body),
    });
    let out = null;
    try { out = await res.json(); } catch {}
    if (!res.ok) throw new Error((out && out.error) || res.statusText);
    return out;
  }
  const REASON = { off: 'reasoning off', low: 'low effort', medium: 'medium effort', high: 'high effort' };
  let st = { endpoints: [], models: [], defaultUrl: 'https://openrouter.ai/api/v1' };
  const lists = {}; // endpoint id → its model list (a promise)
  const changed = () => document.dispatchEvent(new CustomEvent('assay:remote'));

  async function load() {
    try { st = await api('GET', '/api/remote'); } catch { return; }
    render();
  }

  function render() {
    const list = $('api-list');
    list.innerHTML = '';
    for (const m of st.models) {
      const ep = st.endpoints.find((e) => e.id === m.endpoint);
      const row = el('button', 'row bm-api-row');
      row.type = 'button';
      const t = el('div', 'row-text');
      t.append(el('div', 'row-title', m.title));
      const bits = [m.model, ep ? ep.name : 'no endpoint'];
      if (m.reasoning) bits.push(REASON[m.reasoning]);
      t.append(el('div', 'row-sub', bits.join(' · ')));
      if (m.system) t.append(el('div', 'row-sub bm-api-prompt', `Prompt: ${m.system}`));
      row.append(t, el('span', 'muted small', 'Edit'));
      row.onclick = () => openModel(m);
      list.append(row);
    }
    const eps = $('api-eps');
    eps.innerHTML = '';
    if (st.endpoints.length) eps.append(el('span', 'muted small', 'Endpoints:'));
    for (const e of st.endpoints) {
      const b = el('button', 'bm-linkbtn', `${e.name}${e.hasKey ? ` (key ${e.keyHint || 'set'})` : ' (no key)'}`);
      b.type = 'button';
      b.title = `${e.url}. Click to edit.`;
      b.onclick = () => openEndpoint(e);
      eps.append(b);
    }
    const add = el('button', 'bm-linkbtn', st.endpoints.length ? 'Add endpoint' : 'Add an endpoint (OpenRouter or another API)');
    add.type = 'button';
    add.onclick = () => openEndpoint(null);
    eps.append(add);
  }

  // Endpoints.
  let editingEp = null;
  let afterEp = null; // opens the model dialog after a first endpoint is saved
  function openEndpoint(e, then) {
    editingEp = e;
    afterEp = then || null;
    $('api-ep-title').textContent = e ? 'Edit endpoint' : 'Add endpoint';
    $('api-ep-name').value = e ? e.name : st.endpoints.length ? '' : 'OpenRouter';
    $('api-ep-url').value = e ? e.url : st.endpoints.length ? '' : st.defaultUrl;
    $('api-ep-key').value = '';
    $('api-ep-key').placeholder = e && e.hasKey ? `Saved (${e.keyHint || 'set'}); leave empty to keep it` : 'sk-…';
    $('api-ep-keynote').textContent = !e && !st.endpoints.length ? '(get one at openrouter.ai/keys)' : '';
    $('api-ep-delete').hidden = !e;
    $('api-ep-err').hidden = true;
    $('api-ep-dlg').showModal();
  }
  $('api-ep-form').onsubmit = async (ev) => {
    ev.preventDefault();
    const body = { name: $('api-ep-name').value, url: $('api-ep-url').value, key: $('api-ep-key').value };
    try {
      const r = await api('PUT', `/api/remote/endpoints/${editingEp ? editingEp.id : 'new'}`, body);
      delete lists[r.id];
      $('api-ep-dlg').close();
      await load();
      changed();
      if (afterEp) afterEp(r.id);
    } catch (e) {
      $('api-ep-err').textContent = e.message;
      $('api-ep-err').hidden = false;
    }
  };
  $('api-ep-delete').onclick = async () => {
    const n = st.models.filter((m) => m.endpoint === editingEp.id).length;
    if (!confirm(`Remove ${editingEp.name}${n ? ` and its ${n} model${n === 1 ? '' : 's'}` : ''}? Results already run stay saved.`)) return;
    await api('DELETE', `/api/remote/endpoints/${editingEp.id}`);
    $('api-ep-dlg').close();
    await load();
    changed();
  };

  // Models.
  let editingM = null;
  function listFor(id) {
    if (!lists[id]) lists[id] = api('GET', `/api/remote/endpoints/${id}/models`).catch(() => []);
    return lists[id];
  }
  async function fillList() {
    const id = $('api-m-ep').value;
    const dl = $('api-m-list');
    dl.innerHTML = '';
    const items = id ? await listFor(id) : [];
    for (const m of items) {
      const o = document.createElement('option');
      o.value = m.id;
      if (m.name || m.out) o.label = [m.name, m.out ? `$${fmt(m.in)} / $${fmt(m.out)} per M` : ''].filter(Boolean).join(' · ');
      dl.append(o);
    }
    price();
  }
  const fmt = (v) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(2).replace(/\.?0+$/, '') : v.toFixed(3).replace(/\.?0+$/, ''));
  async function price() {
    const id = $('api-m-ep').value;
    const m = id ? (await listFor(id)).find((x) => x.id === $('api-m-model').value.trim()) : null;
    $('api-m-price').textContent = m && (m.in || m.out) ? `$${fmt(m.in)} in, $${fmt(m.out)} out per million tokens` : '';
    if (m && m.name && !$('api-m-titleinp').dataset.touched) $('api-m-titleinp').value = m.name.replace(/^[^:]+:\s*/, '');
  }
  function openModel(m, copy) {
    if (!st.endpoints.length) return openEndpoint(null, (id) => openModel({ endpoint: id }));
    editingM = m && m.id && !copy ? m : null;
    $('api-m-title').textContent = copy ? 'Copy with another prompt' : editingM ? 'Edit API model' : 'Add API model';
    const sel = $('api-m-ep');
    sel.innerHTML = '';
    for (const e of st.endpoints) sel.append(new Option(e.name, e.id));
    sel.value = (m && m.endpoint) || st.endpoints[0].id;
    $('api-m-model').value = (m && m.model) || '';
    $('api-m-titleinp').value = m && m.title ? (copy ? `${m.title} (prompt 2)` : m.title) : '';
    $('api-m-titleinp').dataset.touched = m && m.title ? '1' : '';
    $('api-m-system').value = (m && m.system) || '';
    $('api-m-reason').value = (m && m.reasoning) || '';
    $('api-m-delete').hidden = $('api-m-copy').hidden = !editingM;
    $('api-m-err').hidden = true;
    $('api-m-dlg').showModal();
    fillList();
    if (copy) $('api-m-system').focus();
  }
  $('api-m-ep').onchange = fillList;
  $('api-m-model').oninput = price;
  $('api-m-titleinp').oninput = () => { $('api-m-titleinp').dataset.touched = '1'; };
  $('api-m-form').onsubmit = async (ev) => {
    ev.preventDefault();
    const body = { endpoint: $('api-m-ep').value, model: $('api-m-model').value, title: $('api-m-titleinp').value, system: $('api-m-system').value, reasoning: $('api-m-reason').value };
    try {
      await api('PUT', `/api/remote/models/${editingM ? editingM.id : 'new'}`, body);
      $('api-m-dlg').close();
      await load();
      changed();
    } catch (e) {
      $('api-m-err').textContent = e.message;
      $('api-m-err').hidden = false;
    }
  };
  $('api-m-copy').onclick = () => { const m = editingM; $('api-m-dlg').close(); openModel(m, true); };
  $('api-m-delete').onclick = async () => {
    if (!confirm(`Remove ${editingM.title}? Its results stay saved and come back if you add it again the same way.`)) return;
    await api('DELETE', `/api/remote/models/${editingM.id}`);
    $('api-m-dlg').close();
    await load();
    changed();
  };
  $('api-add-model').onclick = () => openModel(null);
  for (const d of [$('api-ep-dlg'), $('api-m-dlg')]) d.querySelectorAll('[data-close]').forEach((b) => { b.onclick = () => d.close(); });
  load();
})();
