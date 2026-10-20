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

  for (const md of list.querySelectorAll('.md')) md.innerHTML = renderMarkdown(md.textContent);

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

  function syncPicker() {
    const opt = selectedOption();
    if (opt?.value.startsWith('preset:')) {
      meta.textContent = 'on ' + baseOf(opt);
    } else {
      meta.textContent = opt?.dataset.meta?.replace(/^ · | · $/g, '') || '';
    }
    showLoadState();
  }

  select.addEventListener('change', syncPicker);
  document.addEventListener('loaded-changed', (e) => {
    loadedNames = new Set(e.detail.map((m) => m.name));
    showLoadState();
  });
  syncPicker();

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
    const sidebar = document.getElementById('chat-list');
    if (!sidebar) return;
    for (const a of sidebar.querySelectorAll('.active')) a.classList.remove('active');
    const a = el('a', 'chat-link active', title.split('\n')[0].slice(0, 60));
    a.href = `/chat/${id}`;
    sidebar.prepend(a);
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (controller) { controller.abort(); return; } // the button is "Stop" while streaming
    const text = input.value.trim();
    if (!text || !select.value) return;

    document.getElementById('empty')?.remove();
    list.append(el('div', 'msg user', text));
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

    input.value = '';
    sendBtn.textContent = 'Stop';
    controller = new AbortController();
    let content = '';
    let thinking = '';

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chatId, content: text, ...tune.selection(), ...tune.settings() }),
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
            md.innerHTML = renderMarkdown(content);
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
      input.focus();
      refreshLoaded(); // chatting loads the model, so the memory list may have changed
    }
  });

  // Stats lines (server-rendered and streamed) open the Tune panel.
  list.addEventListener('click', (e) => { if (e.target.closest('.stats')) tune?.open?.(); });
  list.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.closest?.('.stats')) { e.preventDefault(); tune?.open?.(); }
  });
})();
