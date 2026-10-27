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
  // The long hint doesn't fit one line on a phone and its second line is clipped.
  if (input && window.matchMedia('(max-width: 760px)').matches) input.placeholder = 'Message';
  const sendBtn = document.getElementById('send');
  let chatId = chatEl.dataset.chatId || '';

  // Back from a saved chat goes to an empty one; back from there leaves.
  // An empty-chat entry is slipped in under the chat to make that work.
  if (chatId && !history.state?.chat) {
    const here = location.pathname + location.search;
    history.replaceState({ emptyChat: true }, '', '/chat');
    history.pushState({ chat: true }, '', here);
  }
  addEventListener('popstate', (e) => { if (e.state?.emptyChat) location.replace(location.href); });
  let controller = null;

  // ---- markdown (small and safe: everything is escaped first) ----
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // Inline marks. Code spans first (their insides stay as typed), then
  // links, then emphasis; everything is escaped before any tag is added.
  // Links only go to http(s) and mailto addresses.
  function inline(text) {
    return text
      .split('`')
      .map((part, i) => (i % 2 ? `<code>${esc(part)}</code>` : marks(part)))
      .join('');
  }

  const SAFE_URL = /^(https?:\/\/|mailto:)/i;
  function marks(raw) {
    const links = [];
    const keep = (html) => { links.push(html); return `\u0001${links.length - 1}\u0001`; };
    let t = raw
      .replace(/\[([^\]\n]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (m, label, url) =>
        SAFE_URL.test(url) ? keep(`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${emphasis(esc(label))}</a>`) : m)
      .replace(/<(https?:\/\/[^\s>]+)>/g, (m, url) => keep(link(url)))
      .replace(/\bhttps?:\/\/[^\s<>()]+[^\s<>().,;:!?'"*_]/g, (url) => keep(link(url)));
    t = emphasis(esc(t));
    return t.replace(/\u0001(\d+)\u0001/g, (_, i) => links[i]);
  }
  const link = (url) => `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(url)}</a>`;

  function emphasis(t) {
    return t
      .replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?!\w)/g, '$1<strong>$2</strong>')
      .replace(/~~(?=\S)([\s\S]*?\S)~~/g, '<del>$1</del>')
      .replace(/(^|[^*\w])\*(?=[^\s*])([^*\n]*?[^\s*])\*(?!\*)/g, '$1<em>$2</em>')
      .replace(/(^|[^\w])_(?=[^\s_])([^_\n]*?[^\s_])_(?!\w)/g, '$1<em>$2</em>');
  }

  // ---- tables (GitHub style): a header row, a |---|:--:| row, then rows ----
  const TABLE_SEP = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;
  // Splits a row on | outside code spans and \| escapes.
  function cells(line) {
    let s = line.trim();
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
    const out = [];
    let cur = '';
    let code = false;
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (c === '\\' && s[i + 1] === '|') { cur += '|'; i++; continue; }
      if (c === '`') code = !code;
      if (c === '|' && !code) { out.push(cur.trim()); cur = ''; continue; }
      cur += c;
    }
    out.push(cur.trim());
    return out;
  }
  function table(head, sep, rows) {
    const align = cells(sep).map((c) => (c.startsWith(':') && c.endsWith(':') ? 'center' : c.endsWith(':') ? 'right' : ''));
    const cols = Math.max(head.length, align.length);
    const cell = (tag, text, i) => `<${tag}${align[i] ? ` style="text-align:${align[i]}"` : ''}>${inline(text ?? '')}</${tag}>`;
    const row = (tag, cs) => `<tr>${Array.from({ length: cols }, (_, i) => cell(tag, cs[i], i)).join('')}</tr>`;
    return `<div class="table-wrap"><table><thead>${row('th', head)}</thead>` +
      `<tbody>${rows.map((r) => row('td', cells(r))).join('')}</tbody></table></div>`;
  }

  // Block structure: headings, paragraphs, nested lists (task lists too),
  // quotes, rules and tables.
  function blocks(text) {
    const lines = text.split('\n');
    let html = '';
    let para = [];
    let stack = []; // open lists: { tag, indent, items: [html], open: bool }
    const flushPara = () => {
      if (para.length) html += `<p>${para.map(inline).join('<br>')}</p>`;
      para = [];
    };
    const closeList = (to = 0) => {
      while (stack.length > to) {
        const l = stack.pop();
        const out = `<${l.tag}>${l.items.map((i) => `<li${i.task ? ' class="task"' : ''}>${i.html}</li>`).join('')}</${l.tag}>`;
        if (stack.length) stack[stack.length - 1].items.at(-1).html += out;
        else html += out;
      }
    };
    const item = (indent, tag, body) => {
      flushPara();
      while (stack.length && indent < stack.at(-1).indent) closeList(stack.length - 1);
      const top = stack.at(-1);
      if (!top || indent > top.indent) stack.push({ tag, indent, items: [] });
      else if (top.tag !== tag) { closeList(stack.length - 1); stack.push({ tag, indent, items: [] }); }
      const task = body.match(/^\[([ xX])\]\s+(.*)$/);
      stack.at(-1).items.push(task
        ? { task: true, html: `<input type="checkbox" disabled${task[1] !== ' ' ? ' checked' : ''}> ${inline(task[2])}` }
        : { html: inline(body) });
    };
    for (let n = 0; n < lines.length; n++) {
      const line = lines[n];
      let m;
      if (!line.trim()) { flushPara(); closeList(); continue; }
      if ((m = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/)) && !/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
        item(m[1].replace(/\t/g, '    ').length, /\d/.test(m[2]) ? 'ol' : 'ul', m[3]);
        continue;
      }
      if (stack.length && /^\s{2,}\S/.test(line)) { // a list item's next line
        stack.at(-1).items.at(-1).html += '<br>' + inline(line.trim());
        continue;
      }
      closeList();
      if ((m = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/))) {
        flushPara();
        const level = Math.min(m[1].length + 2, 6);
        html += `<h${level}>${inline(m[2])}</h${level}>`;
      } else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
        flushPara();
        html += '<hr>';
      } else if (/^\s*>/.test(line)) {
        flushPara();
        const quote = [];
        while (n < lines.length && /^\s*>/.test(lines[n])) quote.push(lines[n++].replace(/^\s*>\s?/, ''));
        n--;
        html += `<blockquote>${blocks(quote.join('\n'))}</blockquote>`;
      } else if (line.includes('|') && n + 1 < lines.length && TABLE_SEP.test(lines[n + 1]) && lines[n + 1].includes('-')) {
        flushPara();
        const head = cells(line);
        const sep = lines[n + 1];
        const rows = [];
        n += 2;
        while (n < lines.length && lines[n].includes('|') && lines[n].trim()) rows.push(lines[n++]);
        n--;
        html += table(head, sep, rows);
      } else {
        para.push(line);
      }
    }
    flushPara();
    closeList();
    return html;
  }

  // Extensions for Download, by fence label (lowercased).
  const EXT = { python: 'py', py: 'py', javascript: 'js', js: 'js', typescript: 'ts', ts: 'ts', go: 'go', rust: 'rs', rs: 'rs', c: 'c', cpp: 'cpp', 'c++': 'cpp', java: 'java', kotlin: 'kt', ruby: 'rb', rb: 'rb', php: 'php', bash: 'sh', sh: 'sh', shell: 'sh', zsh: 'sh', json: 'json', yaml: 'yaml', yml: 'yaml', toml: 'toml', html: 'html', css: 'css', sql: 'sql', markdown: 'md', md: 'md', xml: 'xml', lua: 'lua', swift: 'swift', csharp: 'cs', cs: 'cs', latex: 'tex', tex: 'tex' };
  const WHOLE_NAME = { dockerfile: 'Dockerfile', makefile: 'Makefile' };

  // A file named in a comment on the first line (// main.go), else snippet.<ext>.
  function fileNameFor(code, lang) {
    const first = code.split('\n', 1)[0];
    const m = first.match(/^\s*(?:\/\/|#|--|\/\*|<!--)\s*([\w./-]+\.\w+)\s*(?:\*\/|-->)?\s*$/);
    if (m) return m[1].split('/').pop();
    const l = lang.toLowerCase();
    return WHOLE_NAME[l] || `snippet.${EXT[l] || 'txt'}`;
  }

  function codeBlock(code, lang) {
    return `<div class="code"><div class="code-head"><span>${esc(lang)}</span>` +
      '<span class="code-actions"><button type="button" class="copy-btn" title="Copy code">Copy</button>' +
      '<button type="button" class="copy-btn dl-btn" title="Save as a file">Download</button></span></div>' +
      `<pre>${codeInner(code, lang)}</pre></div>`;
  }

  // Highlights only when the fence names a language hljs knows; no
  // auto-detect (slow, and wrong on short or half-streamed snippets).
  function codeInner(code, lang) {
    const l = (lang || '').trim().toLowerCase().split(/\s+/)[0];
    if (window.hljs && l && /^[\w+#.-]+$/.test(l) && hljs.getLanguage(l)) {
      try {
        const v = hljs.highlight(code, { language: l, ignoreIllegals: true }).value;
        return `<code class="hljs language-${l}">${v}</code>`;
      } catch { /* fall through to plain */ }
    }
    return `<code>${esc(code)}</code>`;
  }

  // Math: $$…$$ and \[…\] as blocks, $…$ and \(…\) inline. Inline code
  // spans are matched first so `$x` in code stays as typed. A single $ needs
  // no space inside it and no digit after it, so "$5 and $10" stays text.
  const MATH = /(`[^`\n]*`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$(?!\s)([^$\n`]+?)(?<!\s)\$(?!\d)/g;

  // Swaps math for placeholders the markdown pass leaves alone, renders the
  // prose, then puts KaTeX output in their place.
  function proseWithMath(text) {
    if (!window.katex) return blocks(text);
    const parts = [];
    const masked = text.replace(MATH, (m, code, d1, d2, i1, i2) => {
      if (code) return m;
      const display = d1 != null || d2 != null;
      const tex = d1 ?? d2 ?? i1 ?? i2;
      let html;
      try { html = katex.renderToString(tex, { displayMode: display, throwOnError: false }); }
      catch { return m; }
      parts.push(html);
      return `\u0000${parts.length - 1}\u0000`;
    });
    return blocks(masked).replace(/\u0000(\d+)\u0000/g, (_, i) => parts[i]);
  }

  // Splits out ``` fences; the rest is prose. An unclosed fence (mid-stream)
  // still renders as code.
  function renderMarkdown(src) {
    let html = '';
    let prose = [];
    let code = null;
    let lang = '';
    const flushProse = () => {
      if (prose.length) html += proseWithMath(prose.join('\n'));
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
    const btn = e.target.closest('.copy-btn:not(.dl-btn)');
    if (!btn) return;
    try {
      await copyText(btn.closest('.code').querySelector('code').textContent);
      btn.textContent = 'Copied';
    } catch {
      btn.textContent = 'Couldn’t copy';
    }
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  });

  // Download saves the block as a file through a temporary link.
  list.addEventListener('click', (e) => {
    const btn = e.target.closest('.dl-btn');
    if (!btn) return;
    const code = btn.closest('.code');
    const text = code.querySelector('code').textContent;
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    const a = el('a');
    a.href = url;
    a.download = fileNameFor(text, code.querySelector('.code-head > span').textContent);
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    btn.textContent = 'Saved';
    setTimeout(() => { btn.textContent = 'Download'; }, 1500);
  });

  // ---- message actions: copy, edit, regenerate ----
  const ICONS = {
    copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>',
    edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/>',
    redo: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
    check: '<path d="M5 12l5 5 9-10"/>',
    speak: '<path d="M4 9v6h4l5 4V5L8 9z"/><path d="M16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12"/>',
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
  // With a JSON reply format the text shows as a highlighted JSON block. The
  // fence is added for display only; the saved reply stays as written.
  function setMarkdown(md, src) {
    md.dataset.src = src;
    const mode = tune?.replyFormat?.();
    // Only text that is JSON (or the start of it, while streaming): older
    // prose replies in the chat stay prose, and a ``` would break the fence.
    const asJSON = (mode === 'json' || mode === 'schema') && /^\s*[{[]/.test(src) && !src.includes('```');
    md.innerHTML = renderMarkdown(asJSON ? '```json\n' + src + '\n```' : src);
  }

  // The reply's own text: text the model wrote between tool calls (.said)
  // is part of the transcript, not of the answer.
  function finalText(msg) {
    const mds = [...msg.querySelectorAll(':scope > .md:not(.said)')];
    return mds.at(-1)?.dataset.src ?? '';
  }

  function addActions(msg) {
    const bar = el('div', 'msg-actions');
    if (msg.classList.contains('user')) bar.append(actionBtn('copy', 'Copy'), actionBtn('edit', 'Edit and resend'));
    else bar.append(actionBtn('copy', 'Copy'), actionBtn('speak', 'Read aloud'), actionBtn('redo', 'Regenerate'));
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
      if ((!text && !msg.dataset.att) || controller) return;
      const n = userMsgs().indexOf(msg);
      const kept = JSON.parse(msg.dataset.att || '[]');
      dropFrom(msg);
      send(text, n, [], kept);
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
    const kept = JSON.parse(user.dataset.att || '[]');
    dropFrom(user);
    send(text, n, [], kept);
  }

  list.addEventListener('click', async (e) => {
    const btn = e.target.closest('.msg-action');
    if (!btn) return;
    const msg = btn.closest('.msg');
    const action = btn.dataset.action;
    if (action === 'copy') {
      const src = msg.classList.contains('user')
        ? msg.querySelector('.bubble').textContent
        : finalText(msg);
      try {
        await copyText(src);
        btn.innerHTML = btn.innerHTML.replace(/<svg([^>]*)>.*<\/svg>/s, `<svg$1>${ICONS.check}</svg>`);
        setTimeout(() => { btn.replaceWith(actionBtn('copy', 'Copy')); }, 1200);
      } catch { btn.title = 'Couldn’t copy'; }
      return;
    }
    if (action === 'speak') { window.ollamuiVoice?.read(finalText(msg)); return; }
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
  // The model option a name (any of its aliases) belongs to.
  const modelOption = (name) => [...select.querySelectorAll('option[data-names]')]
    .find((o) => o.dataset.names.split(' ').includes(name));
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
        const title = el('span', 'row-title', o.textContent);
        if (o.value.startsWith('preset:')) title.prepend(el('span', 'switcher-star', '★ '));
        text.append(title, el('span', 'row-sub', metaOf(o)));
        row.append(dot, text);
        if (!o.value.startsWith('preset:')) {
          // A span, since the row is a button itself; app.js handles the click.
          const star = el('span', 'icon-btn star-btn switcher-fav');
          star.setAttribute('role', 'button');
          star.dataset.star = o.value;
          star.setAttribute('aria-pressed', String(o.hasAttribute('data-starred')));
          star.title = o.hasAttribute('data-starred') ? 'Default model for new chats. Click to unstar' : 'Star: use as the default model when none is loaded';
          star.innerHTML = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></svg>';
          row.append(star);
        }
        // How it fits and how fast it ran; a preset shows its base model's.
        const m = o.value.startsWith('preset:') ? modelOption(baseOf(o)) : o;
        const side = el('span', 'switcher-side');
        if (m?.dataset.fitLabel) {
          const fit = el('span', `switcher-fit fit-${m.dataset.fit}`, m.dataset.speed || m.dataset.fitLabel);
          fit.title = m.dataset.fitLabel + (m.dataset.speed ? ', measured' : ', estimated');
          side.append(fit);
        }
        if (o.dataset.size) side.append(el('span', 'switcher-size', o.dataset.size));
        if (side.children.length) row.append(side);
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
  document.addEventListener('star-changed', (e) => {
    for (const o of select.querySelectorAll('option')) o.toggleAttribute('data-starred', !!e.detail && o.value === e.detail);
    if (!card.hidden) renderSwitcher();
  });
  document.addEventListener('loaded-changed', (e) => {
    loadedNames = new Set(e.detail.map((m) => m.name));
    showLoadState();
    showWelcomeHint();
    if (!card.hidden) renderSwitcher();
  });
  syncPicker();

  // ---- empty chat suggestions ----
  document.getElementById('empty')?.addEventListener('click', (e) => {
    const p = e.target.closest('.preset-start');
    if (p) {
      select.value = 'preset:' + p.dataset.preset;
      select.dispatchEvent(new Event('change'));
      for (const b of document.querySelectorAll('.preset-start')) b.classList.toggle('on', b === p);
      input.focus();
      return;
    }
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
  // Streaming follows the reply only while the user hasn't scrolled away:
  // any upward scroll by hand lets go, getting back to the bottom picks up.
  let follow = true;
  const letGo = () => { if (controller) follow = false; };
  scroller.addEventListener('wheel', (e) => { if (e.deltaY < 0) letGo(); }, { passive: true });
  scroller.addEventListener('touchmove', letGo, { passive: true });
  scroller.addEventListener('keydown', (e) => { if (['ArrowUp', 'PageUp', 'Home'].includes(e.key)) letGo(); });
  scroller.addEventListener('scroll', () => {
    if (scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 4) follow = true;
  }, { passive: true });

  // ---- context meter ----
  // How full the model's context is: the last reply's prompt plus its
  // output, counted up while a reply streams.
  const meter = document.getElementById('ctx-meter');
  const kfmt = (n) => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n);
  let ctxUsed = 0, ctxMax = 0;
  function showContext(used, max) {
    if (used != null) ctxUsed = used;
    if (max) ctxMax = max;
    if (!ctxMax) ctxMax = window.ollamuiTune?.contextSize?.() || 0; // before the first reply says
    if (!meter) return;
    meter.hidden = !ctxUsed;
    const pct = ctxMax ? Math.min(100, (ctxUsed * 100) / ctxMax) : 0;
    document.getElementById('ctx-fill').style.width = pct + '%';
    meter.classList.toggle('warn', pct >= 80);
    meter.classList.toggle('nomax', !ctxMax);
    document.getElementById('ctx-text').textContent = ctxMax
      ? `${kfmt(ctxUsed)} / ${kfmt(ctxMax)} context · ${Math.round(pct)}%`
      : `${kfmt(ctxUsed)} tokens of context used`;
  }
  {
    const last = [...list.querySelectorAll('.stats[data-used]')].pop();
    if (last) showContext(last.dataset.used.split('+').reduce((a, b) => a + +b, 0), +last.dataset.ctx || 0);
  }
  meter?.addEventListener('click', () => document.getElementById('tune-btn')?.click());
  meter?.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); meter.click(); } });

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

  function addSidebarChat(id, title, projectId) {
    window.ollamui?.addSidebarChat(id, title, projectId);
  }

  // ---- projects ----
  // The chat's project: a new chat joins the one picked; an existing chat
  // moves when the picker changes.
  const projectSel = document.getElementById('chat-project');
  const project = () => projectSel?.value ?? chatEl.dataset.project ?? '';
  projectSel?.addEventListener('change', async () => {
    projectSel.closest('.proj-chip').classList.toggle('on', !!projectSel.value);
    if (!chatId) return; // sent with the first message
    try {
      const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ project: projectSel.value }),
      });
      if (!res.ok) throw new Error(await errorText(res));
      location.reload(); // the sidebar regroups
    } catch (err) {
      alert(`Couldn’t move the chat: ${err.message}`);
    }
  });

  // A tool call as a foldable row, like thinking: a one-line summary that
  // opens to the arguments and the result. Rendered the same way by the
  // server for saved chats (templates/chat.html).
  const PENDING = {
    web_search: 'Searching the web…', fetch_url: 'Reading a page…', calculator: 'Calculating…',
    search_chats: 'Searching your chats…', search_project: 'Searching the project’s files…', current_time: 'Checking the time…',
  };
  function prettyJSON(s) {
    try { return JSON.stringify(JSON.parse(s), null, 2); } catch { return s; }
  }
  function toolRow(call) {
    const box = el('details', 'tool pending');
    box.append(el('summary', null, call.label || PENDING[call.name] || `Running ${call.name}…`));
    const body = el('div', 'tool-body');
    body.append(el('div', 'tool-k', call.name), el('pre', 'tool-pre', prettyJSON(call.args)));
    box.append(body);
    return box;
  }
  function fillTool(box, call) {
    box.classList.remove('pending');
    box.classList.toggle('err', !!call.error);
    box.querySelector('summary').textContent = call.label;
    box.querySelector('.tool-body').append(el('div', 'tool-k', 'Result'), el('pre', 'tool-pre', call.result));
  }

  // The excerpts a reply was given, folded under it like the server renders them.
  function sourcesBox(sources) {
    const box = el('details', 'sources');
    const n = sources.length;
    box.append(el('summary', null, `${n} passage${n === 1 ? '' : 's'} from the project’s files`));
    const ol = el('ol');
    for (const x of sources) {
      const li = el('li');
      li.value = x.n;
      const where = [x.file, x.page ? `page ${x.page}` : '', x.heading || ''].filter(Boolean).join(' · ');
      li.append(el('span', 'src-file', where), el('span', 'src-text', x.text));
      ol.append(li);
    }
    box.append(ol);
    return box;
  }

  // ---- queue ----
  // While a reply streams, what's typed waits here and goes out, in order,
  // as each reply finishes. Stopping or an error pauses it.
  const queueBox = document.getElementById('queue');
  let queue = [];
  const hasDraft = () => !!input.value.trim() || window.ollamuiAttach.count() > 0;
  // Stopping takes a second press (click or Esc) within a few seconds.
  let stopAsked = null;
  // The reply runs on the server, so stopping asks the server; the stream
  // then ends with a "stopped" event. Voice replies aren't kept in the
  // background, so for those the request is just dropped.
  async function stopRun() {
    const c = controller;
    try {
      const res = chatId ? await fetch(`/api/chats/${chatId}/stop`, { method: 'POST' }) : null;
      if (!res?.ok) c?.abort();
    } catch { c?.abort(); }
  }
  function askStop() {
    if (!controller) return;
    if (stopAsked) { clearTimeout(stopAsked); stopAsked = null; stopRun(); return; }
    sendBtn.textContent = 'Stop?';
    sendBtn.classList.add('confirm');
    sendBtn.title = 'Press again to stop the reply';
    stopAsked = setTimeout(() => { stopAsked = null; syncSendBtn(); }, 3000);
  }
  function syncSendBtn() {
    if (stopAsked && controller && !hasDraft()) return;
    clearTimeout(stopAsked);
    stopAsked = null;
    sendBtn.classList.remove('confirm');
    sendBtn.textContent = !controller ? 'Send' : hasDraft() ? 'Queue' : 'Stop';
    sendBtn.title = controller && hasDraft() ? 'Send this after the reply finishes (Esc stops the reply)' : '';
  }
  input.addEventListener('input', syncSendBtn);
  // Attaching a file also makes a draft.
  new MutationObserver(syncSendBtn).observe(document.getElementById('att-row'), { childList: true, subtree: true });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && controller) { e.preventDefault(); askStop(); }
  });
  function paintQueue(paused) {
    queueBox.hidden = !queue.length;
    queueBox.replaceChildren();
    if (!queue.length) return;
    const head = el('div', 'queue-head');
    head.append(el('span', null, paused ? `Queued · paused` : `Queued · sends after this reply`));
    if (paused && !controller) {
      const go = el('button', 'btn', 'Send next');
      go.type = 'button';
      go.addEventListener('click', nextQueued);
      head.append(go);
    }
    queueBox.append(head);
    queue.forEach((q, i) => {
      const row = el('div', 'queue-row');
      let label = q.text || '';
      if (q.files.length) label += (label ? ' · ' : '') + q.files.map((f) => f.name).join(', ');
      row.append(el('span', 'queue-text', label));
      const rm = el('button', 'icon-btn', '×');
      rm.type = 'button';
      rm.title = 'Remove from the queue';
      rm.setAttribute('aria-label', 'Remove from the queue');
      rm.addEventListener('click', () => { queue.splice(i, 1); paintQueue(paused); });
      row.append(rm);
      queueBox.append(row);
    });
  }
  function nextQueued() {
    if (controller || !queue.length) return;
    const q = queue.shift();
    paintQueue();
    send(q.text, undefined, q.files);
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const att = window.ollamuiAttach;
    if (controller && !hasDraft()) { askStop(); return; } // the button is "Stop" while streaming
    const text = input.value.trim();
    if ((!text && !att.count()) || !select.value || att.busy()) return;
    if (await att.blockedImage()) return; // the chip says the model can't see it
    input.value = '';
    if (controller) {
      queue.push({ text, files: att.take() });
      paintQueue();
      syncSendBtn();
      return;
    }
    send(text, undefined, att.take());
  });

  // send streams a reply to text. fromUser (edit/regenerate) replaces that
  // user message, counted from 0, and everything after it. voice.js may ask
  // for a reply in spoken style and read it aloud as it streams.
  async function send(text, fromUser, files = [], kept = [], resume = null) {
    const voice = window.ollamuiVoice?.turn() ?? {};
    document.getElementById('empty')?.remove();
    // resume: attaching to a reply already running (its user message is on the page).
    const userMsg = el('div', 'msg user');
    const atts = [...kept, ...files];
    if (!resume) {
      if (atts.length) {
        userMsg.dataset.att = JSON.stringify(atts.map(({ url, state, error, ...a }) => a));
        userMsg.append(window.ollamuiAttach.render(atts, chatId));
      }
      userMsg.append(el('div', 'bubble', text));
      addActions(userMsg);
      list.append(userMsg);
    }
    const reply = el('div', 'msg assistant');
    // The reply is a run of segments (thinking, text), with a row for each
    // tool call between them; these are the current segment's elements.
    let thinkBox, thinkText, md;
    const newSegment = () => {
      thinkBox = el('details', 'thinking');
      thinkBox.hidden = true;
      thinkBox.append(el('summary', null, 'Thinking…'));
      thinkText = el('div', 'thinking-text');
      thinkBox.append(thinkText);
      md = el('div', 'md');
    };
    newSegment();
    const spoken = el('div', 'spoken');
    spoken.hidden = true;
    const stats = el('div', 'stats', 'Waiting for model…');
    stats.setAttribute('role', 'button');
    stats.tabIndex = 0;
    stats.title = 'Open Tune';
    reply.append(thinkBox, spoken, md, stats);
    list.append(reply);
    follow = true;
    scrollDown();

    controller = new AbortController();
    syncSendBtn();
    let streamed = 0;
    const ctxStart = ctxUsed + Math.round(text.length / 3.8);
    let outcome = 'done';
    // If the reply has to load the model first, the memory list shows it.
    const loadCheck = isLoaded(selectedOption()) ? null : setTimeout(() => refreshLoaded(true), 500);
    let content = '';
    let thinking = '';
    const running = new Map(); // tool call id -> its row, until the result is in

    try {
      const res = resume || await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chatId, project: chatId ? undefined : project(), content: text, fromUser, attachments: files.map((f) => f.file), spoken: !!voice.spoken, ...tune.selection(), ...tune.settings() }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(await errorText(res));
      await readNDJSON(res.body, (ev) => {
        voice.onEvent?.(ev);
        // A chunk can hold several tokens: estimate from the text (~3.8
        // characters a token) until the reply's own count arrives.
        if (ev.type === 'content' || ev.type === 'thinking') {
          streamed += ev.text.length;
          showContext(ctxStart + Math.round(streamed / 3.8));
        }
        switch (ev.type) {
          case 'chat':
            if (!chatId) {
              chatId = ev.id;
              chatEl.dataset.chatId = chatId;
              history.replaceState({ emptyChat: true }, '');
              history.pushState({ chat: true }, '', `/chat/${chatId}`);
              addSidebarChat(chatId, text, project());
            }
            window.ollamui.trackRunning(chatId);
            break;
          case 'attachments': {
            const cut = ev.attachments.filter((a) => a.cut).map((a) => a.name).join(', ');
            if (cut) userMsg.append(el('div', 'msg-warn', `Part of ${cut} was left out because it didn’t fit the context. Raise “Context length” in Tune to include more.`));
            break;
          }
          case 'sources':
            reply.insertBefore(sourcesBox(ev.sources), stats);
            break;
          case 'status':
            stats.textContent = ev.text || '';
            break;
          case 'compressed':
            placeCompression(ev.compression);
            break;
          case 'uncompressed':
            placeCompression(null);
            break;
          case 'tool': {
            // End this segment: keep what was written as part of the transcript.
            if (thinking) thinkBox.querySelector('summary').textContent = `Thinking · ~${kfmt(Math.round(thinking.length / 3.8))} tokens`;
            if (thinking && !content) thinkBox.open = false;
            if (content) { md.classList.add('said'); thinkBox.open = false; } else md.remove();
            if (!thinking) thinkBox.remove();
            const row = toolRow(ev.call);
            running.set(ev.call.id, row);
            stats.before(row);
            content = thinking = '';
            newSegment();
            stats.before(thinkBox, md);
            stats.textContent = '';
            break;
          }
          case 'toolresult': {
            const row = running.get(ev.call.id);
            if (row) { fillTool(row, ev.call); running.delete(ev.call.id); }
            stats.textContent = 'Waiting for model…';
            break;
          }
          case 'thinking':
            if (!thinking) { thinkBox.hidden = false; thinkBox.open = true; stats.textContent = ''; }
            thinking += ev.text;
            thinkText.textContent = thinking;
            // A long think shows it's still going, even folded up.
            thinkBox.querySelector('summary').textContent = `Thinking… ~${kfmt(Math.round(thinking.length / 3.8))} tokens`;
            break;
          case 'say':
            spoken.hidden = false;
            spoken.textContent += (spoken.textContent ? ' ' : '') + ev.text;
            if (!thinking) stats.textContent = '';
            break;
          case 'content':
            if (!content) {
              stats.textContent = '';
              if (thinking) { thinkBox.open = false; thinkBox.querySelector('summary').textContent = `Thinking · ~${kfmt(Math.round(thinking.length / 3.8))} tokens`; }
            }
            content += ev.text;
            setMarkdown(md, content);
            break;
          case 'done':
            if (ev.stats) {
              stats.textContent = statsText(ev.stats);
              showContext(ev.stats.promptTokens + ev.stats.outputTokens, ev.stats.context);
            }
            if (ev.stats?.doneReason === 'length') reply.append(el('div', 'msg-warn', lengthNote(!content && !!thinking, ev.stats)));
            break;
          case 'stopped':
            outcome = 'stopped';
            stats.textContent = 'Stopped';
            break;
          case 'error':
            throw new Error(ev.message);
        }
        if (follow) scrollDown();
      });
    } catch (err) {
      outcome = err.name === 'AbortError' ? 'stopped' : 'error';
      if (err.name === 'AbortError') {
        stats.textContent = 'Stopped';
      } else {
        stats.textContent = '';
        reply.append(el('div', 'msg-error', err.message));
      }
    } finally {
      if (thinking) thinkBox.querySelector('summary').textContent = `Thinking · ~${kfmt(Math.round(thinking.length / 3.8))} tokens`;
      for (const row of running.values()) { // stopped mid-call
        row.classList.remove('pending');
        row.querySelector('summary').textContent = 'Stopped before the call finished';
      }
      if (!content && !thinking && reply.querySelector('.tool')) md.remove(); // nothing after the last call
      clearTimeout(loadCheck);
      controller = null;
      syncSendBtn();
      if (queue.length) {
        if (outcome === 'done') setTimeout(nextQueued, 300);
        else paintQueue(true);
      }
      voice.end?.();
      if (chatId) window.ollamui.trackRunning(chatId, false);
      addActions(reply);
      input.focus();
      refreshLoaded(); // chatting loads the model, so the memory list may have changed
      window.ollamuiTune?.setChat({ messages: list.querySelectorAll(':scope > .msg').length });
    }
  }

  // Opening a chat whose reply is still being written (the page was left
  // and come back to) shows it as it goes: the server replays what it has.
  if (chatEl.dataset.running && chatId) {
    fetch(`/api/chats/${chatId}/stream`).then((res) => {
      if (res.status === 200) send('', undefined, [], [], res);
    }).catch(() => {});
  }

  // The divider for a chat's summary sits before the message it stands
  // above: message number upTo. Only one is shown; null removes it.
  function placeCompression(c) {
    for (const d of list.querySelectorAll('.compress-divider')) d.remove();
    if (c) {
      const div = compressDivider(c);
      const msgs = list.querySelectorAll(':scope > .msg');
      if (msgs[c.upTo]) msgs[c.upTo].before(div);
      else list.append(div);
    }
    window.ollamuiTune?.setChat({ compression: c || null });
  }

  function compressDivider(c) {
    const box = el('details', 'compress-divider');
    const n = c.saved >= 1000 ? (c.saved / 1000).toFixed(1) + 'k' : String(c.saved || 0);
    const saved = c.saved ? `, saved ~${n} tokens` : '';
    box.append(el('summary', null, `Context compacted · ${c.upTo} earlier messages summarized${saved}`));
    const body = el('div', 'compress-text md');
    setMarkdown(body, c.text);
    box.append(body);
    return box;
  }

  // voice.js works with the chat through this.
  window.ollamuiChat = {
    busy: () => !!controller,
    ready: () => !!select.value,
    model: () => baseOf(selectedOption()),
    input,
  };

  // Stats lines (server-rendered and streamed) open the Tune panel.
  list.addEventListener('click', (e) => { if (e.target.closest('.stats')) tune?.open?.(); });
  list.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.closest?.('.stats')) { e.preventDefault(); tune?.open?.(); }
  });
})();
