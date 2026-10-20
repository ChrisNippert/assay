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

  // ---- loaded models ----
  // The app bar's "in memory" pills, one per loaded model.
  function renderLoaded(models) {
    const box = document.getElementById('loaded-list');
    if (!box) return;
    box.replaceChildren();
    if (!models.length) {
      box.append(el('span', 'mem-empty', 'No model loaded'));
      return;
    }
    for (const m of models) {
      const pct = m.size ? Math.floor((m.sizeVram * 100) / m.size) : 100;
      const pill = el('div', 'mem-pill' + (pct < 100 ? ' partial' : ''));
      pill.title = m.pinned ? 'Stays loaded until ejected' : 'Unloads after a few idle minutes';
      const dot = el('span', 'mem-dot');
      dot.setAttribute('aria-hidden', 'true');
      const link = el('a', 'row-link mem-name', m.name);
      link.href = '/chat?model=' + encodeURIComponent(m.name);
      link.title = 'Chat with ' + m.name;
      let detail = fmtBytes(m.sizeVram);
      if (m.context) detail += ` · ${Math.round(m.context / 1024)}k`;
      if (pct < 100) detail += ` · ${pct}% GPU`;
      const btn = el('button', 'icon-btn mem-eject');
      btn.type = 'button';
      btn.dataset.eject = m.name;
      btn.title = 'Eject ' + m.name;
      btn.setAttribute('aria-label', 'Eject ' + m.name);
      btn.innerHTML = ejectIcon;
      pill.append(dot, link, el('span', 'mem-detail', detail), btn);
      box.append(pill);
    }
  }

  async function refreshLoaded() {
    try {
      const res = await fetch('/api/loaded');
      if (!res.ok) return null;
      const models = await res.json();
      renderLoaded(models);
      document.dispatchEvent(new CustomEvent('loaded-changed', { detail: models }));
      return models;
    } catch {
      return null;
    }
  }

  // Makes a text button act as Load or Eject for a model.
  function setLoadButton(btn, name, loaded) {
    if (!btn) return;
    delete btn.dataset.load;
    delete btn.dataset.eject;
    if (loaded) btn.dataset.eject = name; else btn.dataset.load = name;
    btn.textContent = loaded ? 'Eject' : 'Load';
    btn.disabled = false;
  }

  // ---- click handling for any [data-load], [data-eject], [data-delete] ----
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-load], [data-eject], [data-delete]');
    if (!btn || btn.disabled) return;

    if (btn.dataset.delete) {
      const name = btn.dataset.delete;
      const all = (btn.closest('[data-names]')?.dataset.names || name).split(' ');
      const label = all.length > 1 ? `${name}\n(also ${all.slice(1).join(', ')})` : name;
      if (!confirm(`Delete ${label} from disk?\n\nOther apps using Ollama will lose it too.`)) return;
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
    try {
      await post(loading ? '/api/models/load' : '/api/models/eject', { name });
    } catch (err) {
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
  document.addEventListener('change', async (e) => {
    const box = e.target.closest('[data-publish]');
    if (!box) return;
    box.disabled = true;
    try {
      const res = await fetch('/api/presets');
      const preset = (await res.json()).find((p) => p.name === box.dataset.publish);
      const put = await fetch('/api/presets/' + encodeURIComponent(preset.name), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...preset, published: box.checked }),
      });
      if (!put.ok) throw new Error(await errorText(put));
    } catch (err) {
      box.checked = !box.checked;
      alert(err.message);
    }
    box.disabled = false;
  });
  document.addEventListener('click', async (e) => {
    const del = e.target.closest('[data-delete-preset]');
    if (!del) return;
    const name = del.dataset.deletePreset;
    if (!confirm(`Delete the preset “${name}”? If it's available to other apps, ollamui/${name} is removed from Ollama too.`)) return;
    const res = await fetch('/api/presets/' + encodeURIComponent(name), { method: 'DELETE' });
    if (res.ok) del.closest('tr')?.remove();
    else alert(`Couldn't delete: ${await errorText(res)}`);
  });

  // ---- sidebar chats ----
  const trashIcon = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>';
  const chatListEl = document.getElementById('chat-list');
  const chatFilter = document.getElementById('chat-filter');

  function applyChatFilter() {
    if (!chatListEl) return;
    const q = (chatFilter?.value || '').trim().toLowerCase();
    for (const row of chatListEl.querySelectorAll('.chat-row')) {
      row.hidden = !!q && !row.querySelector('.chat-link').textContent.toLowerCase().includes(q);
    }
    for (const g of chatListEl.querySelectorAll('.chat-group')) {
      g.hidden = !g.querySelector('.chat-row:not([hidden])');
    }
  }
  chatFilter?.addEventListener('input', applyChatFilter);

  // Adds a newly created chat to the top of the Today group (creating it if missing).
  function addSidebarChat(id, title) {
    if (!chatListEl) return;
    for (const a of chatListEl.querySelectorAll('.active')) a.classList.remove('active');
    let group = chatListEl.querySelector('.chat-group');
    if (group?.querySelector('.chat-group-label')?.textContent !== 'Today') {
      group = el('div', 'chat-group');
      group.append(el('div', 'chat-group-label', 'Today'));
      chatListEl.prepend(group);
    }
    const row = el('div', 'chat-row');
    const a = el('a', 'chat-link active', title.split('\n')[0].slice(0, 60));
    a.href = `/chat/${id}`;
    a.title = a.textContent;
    const del = el('button', 'icon-btn chat-del');
    del.type = 'button';
    del.dataset.chatDel = id;
    del.title = 'Delete chat';
    del.setAttribute('aria-label', 'Delete chat');
    del.innerHTML = trashIcon;
    row.append(a, del);
    group.querySelector('.chat-group-label').after(row);
    group.hidden = false;
    if (chatFilter && chatListEl.querySelectorAll('.chat-row').length > 8) chatFilter.hidden = false;
  }

  chatListEl?.addEventListener('click', async (e) => {
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
      row.remove();
      if (group && !group.querySelector('.chat-row')) group.remove();
      applyChatFilter();
    }, 200);
  });

  window.ollamui = { addSidebarChat, post, errorText, refreshLoaded, setLoadButton, el, fmtBytes, pollDownloads };
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
  const overlayOpen = () => root.classList.contains('nav-open') || !!tune?.classList.contains('open');
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
    if (tune?.classList.contains('open')) document.getElementById('tune-close')?.click();
    setTimeout(() => { popping = false; });
  });

  // Swipes: right opens the menu (or closes Tune), left closes the menu (or
  // opens Tune on the chat page). Ignored on fields and scrollable code.
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
    const tuneOpen = tune?.classList.contains('open');
    if (dx > 0) {
      if (tuneOpen) document.getElementById('tune-close')?.click();
      else if (!navOpen && phone.matches) set(true);
    } else if (navOpen) set(false);
    else if (tune && !tuneOpen) document.getElementById('tune-btn')?.click();
  }, { passive: true });
})();
