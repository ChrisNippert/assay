// API models (Library) and teams (Benchmarks): endpoints (OpenRouter or
// any OpenAI-compatible API), the ways of running their models, and teams
// of a lead and workers. After a change it fires "assay:remote" so the
// leaderboard reloads.
(() => {
  const onAPI = !!document.getElementById('api-list');
  const onTeams = !!document.getElementById('team-dlg');
  if (!onAPI && !onTeams) return;
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
  const money = (v) => (v >= 1 ? `$${v.toFixed(2)}` : v >= 0.01 ? `$${v.toFixed(3)}` : v > 0 ? `$${v.toFixed(4)}` : '$0');
  const REASON = { off: 'reasoning off', low: 'low effort', medium: 'medium effort', high: 'high effort' };
  let st = { endpoints: [], models: [], defaultUrl: 'https://openrouter.ai/api/v1' };
  const lists = {}; // endpoint id → its model list (a promise)
  const changed = () => { document.dispatchEvent(new CustomEvent('assay:remote')); document.dispatchEvent(new CustomEvent('assay:remote-done')); };

  if (onAPI) {
  async function load() {
    try { st = await api('GET', '/api/remote'); } catch { return; }
    render();
  }

  function render() {
    const list = $('api-list');
    list.innerHTML = '';
    $('api-count').textContent = st.models.length ? `${st.models.length} set up` : '';
    // Spending: every API call, from chats, benchmarks, teams and speed tests.
    const sp = st.spend || { total: {}, models: {} };
    const box = $('api-spend');
    box.innerHTML = '';
    box.hidden = !(sp.total.all && sp.total.all.calls);
    if (!box.hidden) {
      for (const [k, label] of [['today', 'today'], ['week', 'last 7 days'], ['month', 'last 30 days'], ['all', 'in all']]) {
        const t = sp.total[k];
        const tile = el('div', 'api-spend-tile');
        tile.title = `${t.calls.toLocaleString()} call${t.calls === 1 ? '' : 's'}, ${t.tokens.toLocaleString()} tokens written`;
        tile.append(el('span', 'api-spend-n', money(t.cost)), el('span', 'muted small', label));
        box.append(tile);
      }
      if (sp.since) box.append(el('span', 'muted small api-spend-since', `Spent on API models since ${new Date(sp.since + 'T00:00').toLocaleDateString()}, as each API reported it.`));
    }
    for (const m of st.models) {
      const ep = st.endpoints.find((e) => e.id === m.endpoint);
      const row = el('div', 'row bm-api-row');
      const t = el('div', 'row-text');
      t.append(el('div', 'row-title', m.title));
      const bits = [m.model, ep ? ep.name : 'no endpoint'];
      if (m.reasoning) bits.push(REASON[m.reasoning]);
      const ms = (st.spend && st.spend.models[m.name]) || null;
      if (ms && ms.all.calls) bits.push(`${money(ms.month.cost)} last 30 days`);
      t.append(el('div', 'row-sub', bits.join(' · ')));
      if (m.system) t.append(el('div', 'row-sub bm-api-prompt', `Prompt: ${m.system}`));
      const chat = el('a', 'btn subtle', 'Chat');
      chat.href = `/chat?model=${encodeURIComponent(m.name)}`;
      const edit = el('button', 'btn subtle', 'Edit');
      edit.type = 'button';
      edit.onclick = () => openModel(m);
      row.append(t, chat, edit);
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
  if (location.hash === '#api-models') setTimeout(() => $('api-models').scrollIntoView(), 50);
  load();
  }

  // Teams: the dialog works wherever it's on the page (Library, chat,
  // Benchmarks); the list only where there's one. A saved team is announced
  // with "assay:team-saved", so chat can switch to it.
  if (!onTeams) return;
  let teams = [];
  let editingT = null;
  async function loadTeams() {
    const list = $('team-list');
    if (!list) return;
    try { teams = await api('GET', '/api/teams'); } catch { return; }
    list.innerHTML = '';
    if ($('team-count')) $('team-count').textContent = teams.length ? `${teams.length} set up` : '';
    for (const t of teams) {
      const row = el('div', 'row bm-api-row');
      const txt = el('div', 'row-text');
      txt.append(el('div', 'row-title', t.title), el('div', 'row-sub', t.error ? `Can't run: ${t.error}` : t.quant));
      if (t.notes) txt.append(el('div', 'row-sub bm-api-prompt', `Lead: ${t.notes}`));
      row.append(txt);
      if (!t.error) {
        const chat = el('a', 'btn subtle', 'Chat');
        chat.href = `/chat?model=${encodeURIComponent(t.name)}`;
        row.append(chat);
      }
      const edit = el('button', 'btn subtle', 'Edit');
      edit.type = 'button';
      edit.onclick = () => openTeam(t);
      row.append(edit);
      list.append(row);
    }
  }
  async function openTeam(t) {
    editingT = t;
    let ms = [];
    try { ms = await api('GET', '/api/teams/members'); } catch {}
    for (const id of ['team-lead', 'team-worker']) {
      const sel = $(id);
      sel.innerHTML = '';
      const local = el('optgroup');
      local.label = 'On this computer';
      const remote = el('optgroup');
      remote.label = 'API';
      for (const r of ms) (r.api ? remote : local).append(new Option(`${r.title}${r.meta ? ' · ' + r.meta : ''}`, r.name));
      if (local.children.length) sel.append(local);
      if (remote.children.length) sel.append(remote);
    }
    $('team-title').textContent = t ? 'Edit team' : 'New team';
    $('team-name').dataset.touched = t ? '1' : '';
    $('team-name').value = t ? t.title : '';
    $('team-strategy').value = t ? t.strategy : 'pipeline';
    if (t) { $('team-lead').value = t.lead; $('team-worker').value = t.worker; }
    $('team-size').value = t ? t.size : 3;
    $('team-rounds').value = t && t.rounds ? t.rounds : 2;
    $('team-notes').value = t ? t.notes || '' : '';
    $('team-delete').hidden = !t;
    $('team-err').hidden = true;
    teamForm();
    $('team-dlg').showModal();
  }
  function teamForm() {
    const loop = $('team-strategy').value === 'loop';
    $('team-rounds-row').hidden = !loop;
    $('team-how').textContent = loop
      ? 'The lead hands out tasks when it wants, reads the results, may hand out more for a few rounds, then answers. Workers see only the task the lead writes.'
      : 'Always three steps: the lead splits the question into parts, the workers each do one at the same time, and the lead answers from their results. Workers see the whole question and their part.';
    const l = $('team-lead').value, w = $('team-worker').value;
    $('team-gpu-note').hidden = !l || !w || l === w || l.startsWith('api:') || w.startsWith('api:');
    // A name to start from, until one is typed.
    if (!$('team-name').dataset.touched && l && w) {
      const short = (id) => ($(id).selectedOptions[0]?.textContent || '').split(' · ')[0];
      $('team-name').value = l === w ? `${short('team-lead')} ×${$('team-size').value}` : `${short('team-lead')} leads ${short('team-worker')}`;
    }
  }
  for (const id of ['team-strategy', 'team-lead', 'team-worker', 'team-size']) $(id).addEventListener('change', teamForm);
  $('team-name').addEventListener('input', () => { $('team-name').dataset.touched = '1'; });
  $('team-form').onsubmit = async (ev) => {
    ev.preventDefault();
    const loop = $('team-strategy').value === 'loop';
    const body = { title: $('team-name').value, strategy: $('team-strategy').value, lead: $('team-lead').value, worker: $('team-worker').value, size: Number($('team-size').value), rounds: loop ? Number($('team-rounds').value) : 0, notes: $('team-notes').value };
    try {
      const r = await api('PUT', `/api/teams/${editingT ? editingT.id : 'new'}`, body);
      $('team-dlg').close();
      await loadTeams();
      changed();
      const saved = (await api('GET', '/api/teams').catch(() => [])).find((x) => x.name === r.name);
      document.dispatchEvent(new CustomEvent('assay:team-saved', { detail: { name: r.name, title: body.title, meta: saved ? saved.quant : '' } }));
    } catch (e) {
      $('team-err').textContent = e.message;
      $('team-err').hidden = false;
    }
  };
  $('team-delete').onclick = async () => {
    if (!confirm(`Remove ${editingT.title}? Its results stay saved.`)) return;
    await api('DELETE', `/api/teams/${editingT.id}`);
    $('team-dlg').close();
    await loadTeams();
    changed();
  };
  window.assayNewTeam = () => openTeam(null);
  if ($('team-add')) $('team-add').onclick = window.assayNewTeam;
  $('team-dlg').querySelectorAll('[data-close]').forEach((b) => { b.onclick = () => $('team-dlg').close(); });
  document.addEventListener('assay:remote-done', loadTeams);
  loadTeams();
})();
