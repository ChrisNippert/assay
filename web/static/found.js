// Library: "Found on this computer" — models other apps downloaded, which
// can be linked into the library without copying.
(() => {
  const root = document.getElementById('found');
  if (!root) return;
  const body = document.getElementById('found-body');
  const say = document.getElementById('found-msg');
  let busy = false;

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const bytes = (n) => {
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (n >= 1000 && i < u.length - 1) { n /= 1000; i++; }
    return (i > 1 ? n.toFixed(1) : Math.round(n)) + ' ' + u[i];
  };
  const msg = (t, bad) => { say.textContent = t || ''; say.classList.toggle('bad', !!bad); say.hidden = !t; };

  async function call(method, path, data) {
    const res = await fetch(path, {
      method, headers: { 'Content-Type': 'application/json' },
      body: data ? JSON.stringify(data) : undefined,
    });
    if (!res.ok) {
      let t = res.statusText;
      try { t = (await res.json()).error || t; } catch {}
      throw new Error(t);
    }
    return res.status === 204 ? null : res.json();
  }

  async function run(label, fn) {
    if (busy) return;
    busy = true;
    root.classList.add('busy');
    msg(label);
    try { await fn(); } catch (e) { msg(e.message, true); }
    busy = false;
    root.classList.remove('busy');
  }

  async function doImport(source, names, count) {
    await run('Linking ' + (count === 1 ? names[0] : count + ' models') + '…', async () => {
      const r = await call('POST', '/api/models/sources/import', { source: source.id, dir: source.dir, names });
      msg('');
      if (r.imported > 0) {
        // The library list is server-rendered: reload to show the new rows.
        location.reload();
      } else {
        await load(true);
      }
    });
  }

  function sourceBlock(s) {
    const box = el('div', 'boxed-list found-src');
    const head = el('div', 'row found-head');
    const text = el('div', 'row-text');
    text.append(el('span', 'row-title', s.title));
    const how = s.link === 'hard'
      ? 'hard link, not a copy'
      : 'used where it is (another disk), not a copy';
    const sub = el('span', 'row-sub');
    sub.title = s.link === 'hard'
      ? 'The file is shared, so it stays if that app removes it.'
      : 'It is on another disk, so ollamui uses the file where it is. If that app removes it, the model shows as missing.';
    sub.append(el('span', 'mono', s.dir), document.createTextNode(' · ' + how));
    text.append(sub);
    head.append(text);
    const ctrl = el('div', 'row-ctrl');
    if (s.models.length > 1) {
      const all = el('button', 'btn', 'Import all');
      all.type = 'button';
      all.addEventListener('click', () => doImport(s, [], s.models.length));
      ctrl.append(all);
    }
    if (s.id === 'folder') {
      const rm = el('button', 'btn subtle', 'Remove');
      rm.type = 'button';
      rm.title = 'Stop looking here. The folder itself is left alone.';
      rm.addEventListener('click', () => run('', async () => {
        await call('DELETE', '/api/models/sources/folder', { path: s.dir });
        await load(true);
      }));
      ctrl.append(rm);
    }
    head.append(ctrl);
    box.append(head);
    if (!s.models.length) {
      const r = el('div', 'row');
      r.append(el('span', 'muted small', 'No .gguf files here that aren’t in your library.'));
      box.append(r);
    }
    for (const m of s.models) {
      const row = el('div', 'row found-row');
      const t = el('div', 'row-text');
      t.append(el('span', 'row-title found-name', m.name));
      t.append(el('span', 'row-sub', [m.quant, bytes(m.size)].filter(Boolean).join(' · ')));
      row.append(t);
      const c = el('div', 'row-ctrl');
      const b = el('button', 'btn primary', 'Import');
      b.type = 'button';
      c.append(b);
      row.append(c);
      row.addEventListener('click', () => doImport(s, [m.name], 1));
      box.append(row);
    }
    return box;
  }

  function addFolderRow() {
    const box = el('div', 'boxed-list found-add');
    const row = el('div', 'row found-addrow');
    const t = el('div', 'row-text');
    t.append(el('span', 'row-title', 'Add a folder…'));
    t.append(el('span', 'row-sub', 'Any folder with .gguf files in it, like /mnt/games/models. It is only read, never changed.'));
    row.append(t);
    const c = el('div', 'row-ctrl');
    const input = document.createElement('input');
    input.type = 'text'; input.size = 28; input.placeholder = '/path/to/models';
    input.setAttribute('aria-label', 'Folder to look in');
    input.autocomplete = 'off'; input.spellcheck = false;
    const go = el('button', 'btn', 'Add');
    go.type = 'button';
    c.append(input, go);
    row.append(c);
    const add = () => {
      if (!input.value.trim()) return input.focus();
      run('', async () => {
        await call('POST', '/api/models/sources/folder', { path: input.value.trim() });
        input.value = '';
        await load(true);
      });
    };
    go.addEventListener('click', add);
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') add(); });
    row.addEventListener('click', (e) => { if (e.target === row || t.contains(e.target)) input.focus(); });
    box.append(row);
    return box;
  }

  async function load(refresh) {
    let data;
    try { data = await call('GET', '/api/models/sources' + (refresh ? '?refresh=1' : '')); } catch { return; }
    body.replaceChildren();
    for (const s of data.sources) body.append(sourceBlock(s));
    body.append(addFolderRow());
    const n = data.sources.reduce((a, s) => a + s.models.length, 0);
    document.getElementById('found-count').textContent = n ? n + ' to import' : '';
    root.hidden = false;
  }

  document.getElementById('found-refresh')?.addEventListener('click', () => run('', () => load(true)));
  load(false);
})();
