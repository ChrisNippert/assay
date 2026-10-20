// Chat page: streaming replies, markdown, model load state.
(() => {
  const chatEl = document.getElementById('chat');
  if (!chatEl) return;
  const { errorText, refreshLoaded, setLoadButton, el } = window.ollamui;
  const tune = window.ollamuiTune;

  const select = document.getElementById('model');
  const meta = document.getElementById('model-meta');
  const loadBtn = document.getElementById('load-btn');
  const list = document.getElementById('message-list');
  const scroller = document.getElementById('messages');
  const form = document.getElementById('composer');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  let chatId = chatEl.dataset.chatId || '';
  let controller = null;

  // ---- markdown (small and safe: everything is escaped first) ----
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function inline(text) {
    return text
      .split('`')
      .map((part, i) => (i % 2 ? `<code>${esc(part)}</code>` : esc(part).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')))
      .join('');
  }

  function blocks(text) {
    let html = '';
    let para = [];
    let list = null;
    const flushPara = () => {
      if (para.length) html += `<p>${para.map(inline).join('<br>')}</p>`;
      para = [];
    };
    const flushList = () => {
      if (list) html += `<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join('')}</${list.tag}>`;
      list = null;
    };
    const addItem = (tag, item) => {
      flushPara();
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push(item);
    };
    for (const line of text.split('\n')) {
      let m;
      if (!line.trim()) { flushPara(); flushList(); }
      else if ((m = line.match(/^(#{1,4})\s+(.*)$/))) {
        flushPara(); flushList();
        const level = m[1].length + 2;
        html += `<h${level}>${inline(m[2])}</h${level}>`;
      }
      else if ((m = line.match(/^\s*[-*]\s+(.*)$/))) addItem('ul', m[1]);
      else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) addItem('ol', m[1]);
      else { flushList(); para.push(line); }
    }
    flushPara();
    flushList();
    return html;
  }

  function codeBlock(code, lang) {
    return `<div class="code"><div class="code-head"><span>${esc(lang)}</span>` +
      '<button type="button" class="copy-btn" title="Copy code">Copy</button></div>' +
      `<pre><code>${esc(code)}</code></pre></div>`;
  }

  // Splits out ``` fences; the rest is prose. An unclosed fence (mid-stream)
  // still renders as code.
  function renderMarkdown(src) {
    let html = '';
    let prose = [];
    let code = null;
    let lang = '';
    const flushProse = () => {
      if (prose.length) html += blocks(prose.join('\n'));
      prose = [];
    };
    for (const line of src.split('\n')) {
      const fence = line.match(/^\s*```\s*([\w+#.-]*)/);
      if (fence && code === null) {
        flushProse();
        code = [];
        lang = fence[1];
      } else if (fence) {
        html += codeBlock(code.join('\n'), lang);
        code = null;
      } else if (code !== null) {
        code.push(line);
      } else {
        prose.push(line);
      }
    }
    if (code !== null) html += codeBlock(code.join('\n'), lang);
    flushProse();
    return html;
  }

  // The clipboard API needs a secure context (localhost or HTTPS); over plain
  // HTTP, e.g. a Tailscale address, fall back to a hidden textarea.
  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    const area = el('textarea');
    area.value = text;
    area.style.cssText = 'position:fixed;opacity:0';
    document.body.append(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    if (!ok) throw new Error('copy failed');
  }

  list.addEventListener('click', async (e) => {
    const btn = e.target.closest('.copy-btn');
    if (!btn) return;
    try {
      await copyText(btn.closest('.code').querySelector('code').textContent);
      btn.textContent = 'Copied';
    } catch {
      btn.textContent = 'Couldn’t copy';
    }
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  });

  // ---- message actions: copy, edit, regenerate ----
  const ICONS = {
    copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>',
    edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/>',
    redo: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
    check: '<path d="M5 12l5 5 9-10"/>',
  };
  function actionBtn(action, label) {
    const b = el('button', 'msg-action');
    b.type = 'button';
    b.dataset.action = action;
    b.title = label;
    b.setAttribute('aria-label', label);
    b.innerHTML = `<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[action]}</svg>`;
    return b;
  }

  // Markdown is rendered in place, so the raw text is kept for copying.
  function setMarkdown(md, src) {
    md.dataset.src = src;
    md.innerHTML = renderMarkdown(src);
  }

  function addActions(msg) {
    const bar = el('div', 'msg-actions');
    if (msg.classList.contains('user')) bar.append(actionBtn('copy', 'Copy'), actionBtn('edit', 'Edit and resend'));
    else bar.append(actionBtn('copy', 'Copy'), actionBtn('redo', 'Regenerate'));
    const stats = msg.querySelector(':scope > .stats');
    if (stats) {
      // Stats and actions share one quiet line under a reply.
      const foot = el('div', 'msg-foot');
      stats.replaceWith(foot);
      foot.append(stats, bar);
    } else {
      msg.append(bar);
    }
  }

  for (const md of list.querySelectorAll('.md')) setMarkdown(md, md.textContent);
  for (const msg of list.querySelectorAll(':scope > .msg')) addActions(msg);

  const userMsgs = () => [...list.querySelectorAll(':scope > .msg.user')];

  // Removes msg and everything after it from the page.
  function dropFrom(msg) {
    while (msg.nextElementSibling) msg.nextElementSibling.remove();
    msg.remove();
  }

  function startEdit(msg) {
    const bubble = msg.querySelector('.bubble');
    const box = el('div', 'msg-edit');
    const area = el('textarea');
    area.value = bubble.textContent;
    area.rows = Math.min(12, area.value.split('\n').length + 1);
    const btns = el('div', 'msg-edit-btns');
    const cancel = el('button', 'btn subtle', 'Cancel');
    const save = el('button', 'btn primary', 'Send');
    cancel.type = save.type = 'button';
    btns.append(cancel, save);
    box.append(area, btns);
    msg.classList.add('editing');
    msg.append(box);
    area.focus();
    area.setSelectionRange(area.value.length, area.value.length);
    const close = () => { box.remove(); msg.classList.remove('editing'); };
    const submit = () => {
      const text = area.value.trim();
      if (!text || controller) return;
      const n = userMsgs().indexOf(msg);
      dropFrom(msg);
      send(text, n);
    };
    cancel.addEventListener('click', close);
    save.addEventListener('click', submit);
    area.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.stopPropagation(); close(); }
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
    });
  }

  function regenerate(reply) {
    let user = reply.previousElementSibling;
    while (user && !user.classList.contains('user')) user = user.previousElementSibling;
    if (!user) return;
    const n = userMsgs().indexOf(user);
    const text = user.querySelector('.bubble').textContent;
    dropFrom(user);
    send(text, n);
  }

  list.addEventListener('click', async (e) => {
    const btn = e.target.closest('.msg-action');
    if (!btn) return;
    const msg = btn.closest('.msg');
    const action = btn.dataset.action;
    if (action === 'copy') {
      const src = msg.classList.contains('user')
        ? msg.querySelector('.bubble').textContent
        : msg.querySelector('.md')?.dataset.src ?? '';
      try {
        await copyText(src);
        btn.innerHTML = btn.innerHTML.replace(/<svg([^>]*)>.*<\/svg>/s, `<svg$1>${ICONS.check}</svg>`);
        setTimeout(() => { btn.replaceWith(actionBtn('copy', 'Copy')); }, 1200);
      } catch { btn.title = 'Couldn’t copy'; }
      return;
    }
    if (controller || !chatId) return; // busy streaming, or nothing saved yet
    if (action === 'edit') startEdit(msg);
    if (action === 'redo') regenerate(msg);
  });

  // ---- model picker + load state ----
  function selectedOption() { return select.selectedOptions[0]; }

  // Presets run on their base model, so loading and memory follow the base.
  const baseOf = (opt) => opt?.dataset.base || '';
  let loadedNames = new Set(JSON.parse(chatEl.dataset.loaded || '[]'));

  // The app bar shows what's in memory (with eject). Next to the picker we
  // only offer "Load" when the picked model isn't loaded yet.
  function showLoadState() {
    const base = baseOf(selectedOption());
    const names = (selectedOption()?.dataset.names || base).split(' ');
    const loaded = !base || names.some((n) => loadedNames.has(n));
    loadBtn.hidden = loaded;
    if (!loaded) {
      setLoadButton(loadBtn, base, false);
      loadBtn.title = `Load ${base} into memory now, so the first reply starts faster`;
    }
  }

  const isLoaded = (opt) => (opt?.dataset.names || baseOf(opt)).split(' ').some((n) => loadedNames.has(n));
  const metaOf = (opt) => opt?.value.startsWith('preset:')
    ? 'preset on ' + baseOf(opt)
    : opt?.dataset.meta?.replace(/^ · | · $/g, '') || '';

  // The empty chat says what will happen when you send.
  function showWelcomeHint() {
    const hint = document.getElementById('welcome-hint');
    const opt = selectedOption();
    if (!hint || !opt?.value) return;
    hint.textContent = isLoaded(opt)
      ? `${opt.textContent} is loaded and ready.`
      : `${opt.textContent} will load when you send your first message.`;
  }

  function syncPicker() {
    const opt = selectedOption();
    switcherName.textContent = opt?.textContent || 'Choose a model';
    meta.textContent = metaOf(opt);
    showLoadState();
    showWelcomeHint();
  }

  // ---- model switcher card (the hidden <select> stays the source of truth) ----
  const switcherBtn = document.getElementById('switcher-btn');
  const switcherName = document.getElementById('switcher-name');
  const card = document.getElementById('switcher-card');
  const filter = document.getElementById('switcher-filter');
  const options = document.getElementById('switcher-list');

  function renderSwitcher() {
    options.replaceChildren();
    const q = filter.value.trim().toLowerCase();
    for (const group of select.querySelectorAll('optgroup')) {
      const opts = [...group.querySelectorAll('option')].filter((o) => o.value && (!q || o.textContent.toLowerCase().includes(q)));
      if (!opts.length) continue;
      options.append(el('div', 'switcher-group', group.label));
      for (const o of opts) {
        const row = el('button', 'switcher-row');
        row.type = 'button';
        row.setAttribute('role', 'option');
        row.setAttribute('aria-selected', String(o.selected));
        row.dataset.value = o.value;
        const dot = el('span', 'switcher-dot' + (isLoaded(o) ? ' on' : ''));
        dot.title = isLoaded(o) ? 'Loaded' : '';
        const text = el('span', 'row-text');
        text.append(el('span', 'row-title', o.textContent), el('span', 'row-sub', metaOf(o)));
        row.append(dot, text);
        if (o.dataset.size) row.append(el('span', 'switcher-size', o.dataset.size));
        if (o.selected) row.insertAdjacentHTML('beforeend', `<svg class="switcher-check" aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">${ICONS.check}</svg>`);
        options.append(row);
      }
    }
    if (!options.children.length) options.append(el('div', 'switcher-none', q ? 'Nothing matches.' : 'No models installed.'));
  }

  function openSwitcher(open) {
    card.hidden = !open;
    switcherBtn.setAttribute('aria-expanded', String(open));
    if (open) {
      filter.value = '';
      renderSwitcher();
      filter.focus();
    }
  }

  function pick(value) {
    select.value = value;
    select.dispatchEvent(new Event('change'));
    openSwitcher(false);
    switcherBtn.focus();
  }

  switcherBtn.addEventListener('click', () => openSwitcher(card.hidden));
  filter.addEventListener('input', renderSwitcher);
  options.addEventListener('click', (e) => {
    const row = e.target.closest('.switcher-row');
    if (row) pick(row.dataset.value);
  });
  card.addEventListener('keydown', (e) => {
    const rows = [...options.querySelectorAll('.switcher-row')];
    const i = rows.indexOf(document.activeElement);
    if (e.key === 'Escape') { e.stopPropagation(); openSwitcher(false); switcherBtn.focus(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); rows[Math.min(i + 1, rows.length - 1)]?.focus(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); (i <= 0 ? filter : rows[i - 1]).focus(); }
    else if (e.key === 'Enter' && document.activeElement === filter && rows.length) { e.preventDefault(); pick(rows[0].dataset.value); }
  });
  document.addEventListener('click', (e) => {
    if (!card.hidden && !e.target.closest('.switcher')) openSwitcher(false);
  });

  select.addEventListener('change', syncPicker);
  document.addEventListener('loaded-changed', (e) => {
    loadedNames = new Set(e.detail.map((m) => m.name));
    showLoadState();
    showWelcomeHint();
    if (!card.hidden) renderSwitcher();
  });
  syncPicker();

  // ---- empty chat suggestions ----
  document.getElementById('empty')?.addEventListener('click', (e) => {
    const s = e.target.closest('.suggestion');
    if (!s) return;
    input.value = s.dataset.prompt.replace(/\\n/g, '\n');
    input.dispatchEvent(new Event('input'));
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  });

  // ---- delete chat ----
  document.getElementById('delete-chat')?.addEventListener('click', async () => {
    if (!confirm('Delete this chat?')) return;
    const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}`, { method: 'DELETE' });
    if (res.ok) location.href = '/chat';
    else alert(`Couldn't delete chat: ${await errorText(res)}`);
  });

  // ---- sending ----
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  const nearBottom = () => scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
  const scrollDown = () => { scroller.scrollTop = scroller.scrollHeight; };

  async function readNDJSON(body, onEvent) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) onEvent(JSON.parse(line));
      }
    }
  }

  function statsText(st) {
    let text = `${st.tokensPerSec.toFixed(1)} tok/s · ${st.outputTokens} tokens`;
    if (st.context) text += ` · ${Math.round(st.context / 1024)}k context${st.autoContext ? ' (auto)' : ''}`;
    return text;
  }

  function lengthNote(onlyThinking, stats) {
    const n = (v) => v.toLocaleString();
    const used = stats ? ` (${n(stats.promptTokens)} prompt + ${n(stats.outputTokens)} reply tokens)` : '';
    return onlyThinking
      ? `The model used up the context${used} on thinking and never answered. Raise “Context length” in Tune, or turn Thinking off.`
      : `Cut off: the conversation filled the context${used}. Raise “Context length” in Tune for longer replies.`;
  }

  function addSidebarChat(id, title) {
    window.ollamui?.addSidebarChat(id, title);
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (controller) { controller.abort(); return; } // the button is "Stop" while streaming
    const text = input.value.trim();
    if (!text || !select.value) return;
    input.value = '';
    send(text);
  });

  // send streams a reply to text. fromUser (edit/regenerate) replaces that
  // user message, counted from 0, and everything after it.
  async function send(text, fromUser) {
    document.getElementById('empty')?.remove();
    const userMsg = el('div', 'msg user');
    userMsg.append(el('div', 'bubble', text));
    addActions(userMsg);
    list.append(userMsg);
    const reply = el('div', 'msg assistant');
    const thinkBox = el('details', 'thinking');
    thinkBox.hidden = true;
    thinkBox.append(el('summary', null, 'Thinking…'));
    const thinkText = el('div', 'thinking-text');
    thinkBox.append(thinkText);
    const md = el('div', 'md');
    const stats = el('div', 'stats', 'Waiting for model…');
    stats.setAttribute('role', 'button');
    stats.tabIndex = 0;
    stats.title = 'Open Tune';
    reply.append(thinkBox, md, stats);
    list.append(reply);
    scrollDown();

    sendBtn.textContent = 'Stop';
    controller = new AbortController();
    let content = '';
    let thinking = '';

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chatId, content: text, fromUser, ...tune.selection(), ...tune.settings() }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(await errorText(res));
      await readNDJSON(res.body, (ev) => {
        const stick = nearBottom();
        switch (ev.type) {
          case 'chat':
            if (!chatId) {
              chatId = ev.id;
              chatEl.dataset.chatId = chatId;
              history.replaceState(null, '', `/chat/${chatId}`);
              addSidebarChat(chatId, text);
            }
            break;
          case 'thinking':
            if (!thinking) { thinkBox.hidden = false; thinkBox.open = true; stats.textContent = ''; }
            thinking += ev.text;
            thinkText.textContent = thinking;
            break;
          case 'content':
            if (!content) {
              stats.textContent = '';
              if (thinking) { thinkBox.open = false; thinkBox.querySelector('summary').textContent = 'Thinking'; }
            }
            content += ev.text;
            setMarkdown(md, content);
            break;
          case 'done':
            if (ev.stats) stats.textContent = statsText(ev.stats);
            if (ev.stats?.doneReason === 'length') reply.append(el('div', 'msg-warn', lengthNote(!content && !!thinking, ev.stats)));
            break;
          case 'error':
            throw new Error(ev.message);
        }
        if (stick) scrollDown();
      });
    } catch (err) {
      if (err.name === 'AbortError') {
        stats.textContent = 'Stopped';
      } else {
        stats.textContent = '';
        reply.append(el('div', 'msg-error', err.message));
      }
    } finally {
      if (thinking) thinkBox.querySelector('summary').textContent = 'Thinking';
      controller = null;
      sendBtn.textContent = 'Send';
      addActions(reply);
      input.focus();
      refreshLoaded(); // chatting loads the model, so the memory list may have changed
    }
  }

  // Stats lines (server-rendered and streamed) open the Tune panel.
  list.addEventListener('click', (e) => { if (e.target.closest('.stats')) tune?.open?.(); });
  list.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.closest?.('.stats')) { e.preventDefault(); tune?.open?.(); }
  });
})();
