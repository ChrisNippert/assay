// Chat page: streaming replies, markdown, model load state.
(() => {
  const chatEl = document.getElementById('chat');
  if (!chatEl) return;
  const { errorText, refreshLoaded, setLoadButton, el } = window.assay;
  const tune = window.assayTune;

  const select = document.getElementById('model');
  const meta = document.getElementById('model-meta');
  const loadBtn = document.getElementById('load-btn');
  const list = document.getElementById('message-list');
  const scroller = document.getElementById('messages');
  const form = document.getElementById('composer');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const sendLabel = sendBtn.querySelector('.send-label');
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

  const { render: renderMarkdown, fileNameFor } = window.assayMarkdown;

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
  function actionBtn(action, label, text) {
    const b = el('button', text ? 'msg-action txt' : 'msg-action');
    b.type = 'button';
    b.dataset.action = action;
    b.title = label;
    b.setAttribute('aria-label', label);
    b.innerHTML = `<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[action]}</svg>${text ? `<span class="msg-action-text">${text}</span>` : ''}`;
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
    else bar.append(actionBtn('copy', 'Copy', 'Copy'), actionBtn('redo', 'Regenerate', 'Retry'), actionBtn('speak', 'Read aloud'));
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
        const label = btn.querySelector('.msg-action-text');
        if (label) label.textContent = 'Copied';
        setTimeout(() => { btn.replaceWith(actionBtn('copy', 'Copy', label ? 'Copy' : undefined)); }, 1200);
      } catch { btn.title = 'Couldn’t copy'; }
      return;
    }
    if (action === 'speak') { window.assayVoice?.read(finalText(msg)); return; }
    if (controller || !chatId) return; // busy streaming, or nothing saved yet
    if (action === 'edit') startEdit(msg);
    if (action === 'redo') regenerate(msg);
  });

  // ---- model picker + load state ----
  function selectedOption() { return select.selectedOptions[0]; }

  // Presets run on their base model, so loading and memory follow the base.
  const baseOf = (opt) => opt?.dataset.base || '';
  let loadedNames = new Set(JSON.parse(chatEl.dataset.loaded || '[]'));

  // The chip carries the state (dot); the menu says it in words and offers
  // Eject, and a small Load button sits beside the chip while it isn't loaded.
  const switcherDot = document.getElementById('switcher-dot');
  const stateRow = document.getElementById('switcher-state');
  const ejectBtn = document.getElementById('eject-btn');
  function showLoadState() {
    const opt = selectedOption();
    const base = baseOf(opt);
    const as = (opt?.dataset.names || base).split(' ').find((n) => loadedNames.has(n)) || '';
    const api = base.startsWith('api:'); // nothing to load: it runs on the API
    const loaded = !!as || api;
    switcherDot.classList.toggle('on', loaded);
    const shown = prettyOf(opt);
    switcherBtn.title = !base ? 'Choose a model or preset' : api ? `${shown} runs on ${modelOption(base)?.dataset.quant || 'its API'}. Choose another model or preset` : `${shown} is ${loaded ? 'loaded' : 'not loaded yet'}. Choose another model or preset`;
    loadBtn.hidden = loaded || !base;
    if (!loadBtn.hidden) {
      setLoadButton(loadBtn, base, false);
      loadBtn.title = `Load ${prettyOf(modelOption(base) || opt)} into memory now, so the first reply starts faster`;
    }
    stateRow.hidden = !base;
    stateRow.querySelector('.switcher-dot').classList.toggle('on', loaded);
    document.getElementById('switcher-state-text').textContent = api ? `Runs on ${modelOption(base)?.dataset.quant || 'its API'}, not this computer` : loaded ? 'Loaded in memory' : 'Not loaded. It loads on your first message';
    ejectBtn.hidden = !loaded || api;
    if (loaded) { ejectBtn.dataset.eject = as; ejectBtn.disabled = false; ejectBtn.textContent = 'Eject'; }
  }

  const isLoaded = (opt) => baseOf(opt).startsWith('api:') || (opt?.dataset.names || baseOf(opt)).split(' ').some((n) => loadedNames.has(n));
  // The model option a name (any of its aliases) belongs to.
  const modelOption = (name) => [...select.querySelectorAll('option[data-names]')]
    .find((o) => o.dataset.names.split(' ').includes(name));
  // The readable name ("Gemma 4 12B") for a model option; the raw name stays the identifier.
  const prettyOf = (opt) => {
    if (!opt) return '';
    if (opt.value.startsWith('preset:')) return opt.textContent;
    return opt.dataset.pretty || opt.textContent;
  };
  const quantOfOpt = (opt) => (opt?.value.startsWith('preset:') ? modelOption(baseOf(opt)) : opt)?.dataset.quant || '';
  // The chip's muted part and a list row's subtitle.
  const metaOf = (opt) => opt?.value.startsWith('preset:')
    ? 'preset on ' + (prettyOf(modelOption(baseOf(opt))) || baseOf(opt))
    : [opt?.dataset.quant, opt?.dataset.meta].filter(Boolean).join(' · ');

  // The empty chat says what will happen when you send.
  function showWelcomeHint() {
    const hint = document.getElementById('welcome-hint');
    const opt = selectedOption();
    if (!hint || !opt?.value) return;
    hint.textContent = baseOf(opt).startsWith('api:')
      ? `${prettyOf(opt)} answers through ${modelOption(baseOf(opt))?.dataset.quant || 'its API'}.`
      : isLoaded(opt)
      ? `${prettyOf(opt)} is loaded and ready.`
      : `${prettyOf(opt)} will load when you send your first message.`;
  }

  function syncPicker() {
    const opt = selectedOption();
    switcherName.textContent = prettyOf(opt) || 'Choose a model';
    meta.textContent = opt?.value.startsWith('preset:') ? metaOf(opt) : quantOfOpt(opt);
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
      const opts = [...group.querySelectorAll('option')].filter((o) => o.value && (!q || (o.textContent + ' ' + prettyOf(o)).toLowerCase().includes(q)));
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
        const title = el('span', 'row-title', prettyOf(o));
        row.title = o.textContent; // the raw name, for the curious
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

  // On a phone the card is a bottom sheet: it slides, so [hidden] waits for
  // the exit. `open` (not hidden) is the state.
  const sheets = window.assaySheets;
  const isPhone = () => sheets.phone.matches;
  const switcherOpen = () => card.classList.contains('open');
  let switcherT = null;
  function openSwitcher(open) {
    clearTimeout(switcherT);
    switcherBtn.setAttribute('aria-expanded', String(open));
    sheets.track(card, open);
    if (open) {
      card.hidden = false;
      void card.offsetWidth;
      card.classList.add('open');
      filter.value = '';
      renderSwitcher();
      if (!isPhone()) filter.focus(); // the keyboard would cover the sheet
    } else {
      card.classList.remove('open');
      if (isPhone()) switcherT = setTimeout(() => { if (!switcherOpen()) card.hidden = true; }, 280);
      else card.hidden = true;
    }
  }
  sheets.register(card, () => openSwitcher(false));

  function pick(value) {
    select.value = value;
    select.dispatchEvent(new Event('change'));
    openSwitcher(false);
    switcherBtn.focus();
  }

  switcherBtn.addEventListener('click', () => openSwitcher(!switcherOpen()));
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
    if (switcherOpen() && !e.target.closest('.switcher')) openSwitcher(false);
  });

  // A new chat starts on the model picked last (in this browser), unless
  // the link names one: picking shouldn't be undone by visiting another page.
  const PICK_KEY = 'chat:lastModel';
  if (!chatId && !new URLSearchParams(location.search).has('model')) {
    let last = null;
    try { last = localStorage.getItem(PICK_KEY); } catch {}
    if (last && last !== select.value && [...select.options].some((o) => o.value === last)) {
      select.value = last;
      // After every script (Tune too) is listening.
      setTimeout(() => select.dispatchEvent(new Event('change')));
    }
  }
  select.addEventListener('change', () => {
    try { localStorage.setItem(PICK_KEY, select.value); } catch {}
  });
  select.addEventListener('change', syncPicker);
  document.addEventListener('star-changed', (e) => {
    for (const o of select.querySelectorAll('option')) o.toggleAttribute('data-starred', !!e.detail && o.value === e.detail);
    if (switcherOpen()) renderSwitcher();
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
  async function deleteChat() {
    if (!confirm('Delete this chat?')) return;
    const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}`, { method: 'DELETE' });
    if (res.ok) location.href = '/chat';
    else alert(`Couldn't delete chat: ${await errorText(res)}`);
  }
  document.getElementById('delete-chat')?.addEventListener('click', deleteChat);

  // ---- phone: the "⋯" sheet (Tune, load/eject, memory, delete) ----
  const moreSheet = document.getElementById('more-sheet');
  const moreBtn = document.getElementById('more-btn');
  if (moreSheet) {
    const $m = (id) => document.getElementById(id);
    const loadedAs = () => {
      const opt = selectedOption();
      return (opt?.dataset.names || baseOf(opt)).split(' ').find((n) => loadedNames.has(n)) || '';
    };
    const setMore = (open) => {
      if (open) {
        const base = baseOf(selectedOption());
        const as = loadedAs();
        $m('more-load').hidden = !base;
        $m('more-load-title').textContent = as ? 'Eject model' : 'Load model';
        $m('more-load-sub').textContent = as || base;
        $m('more-tune-sub').textContent = $m('tune-summary')?.textContent.trim() || 'Reply style, sampling, context';
        $m('more-mem-sub').textContent = document.querySelector('#mem-btn .mem-label')?.textContent.trim() || '';
        sheets.open(moreSheet);
      } else {
        sheets.close(moreSheet);
      }
      moreBtn.setAttribute('aria-expanded', String(open));
    };
    sheets.register(moreSheet, () => setMore(false));
    moreBtn.addEventListener('click', () => setMore(!moreSheet.classList.contains('open')));
    const after = (fn) => () => { setMore(false); setTimeout(fn, 60); };
    $m('more-tune').addEventListener('click', after(() => document.getElementById('tune-btn').click()));
    $m('more-mem').addEventListener('click', after(() => document.getElementById('mem-btn').click()));
    $m('more-load').addEventListener('click', after(() => {
      const as = loadedAs();
      if (as) document.querySelector(`.mem-eject[data-eject="${CSS.escape(as)}"]`)?.click();
      else if (!loadBtn.hidden) loadBtn.click();
    }));
    $m('more-delete')?.addEventListener('click', after(deleteChat));
  }

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

  // ---- header numbers: speed, context, GPU memory ----
  // Always shown. Context: the last reply's prompt plus its output (counted
  // up while a reply streams), of the length the model runs with, else the
  // length Tune shows. Speed: counted live while a reply streams, then the
  // server's figure. All painting happens at most once a frame.
  const kfmt = (n) => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n);
  const $n = (id) => document.getElementById(id);
  const numEls = { tps: $n('num-tps'), used: $n('ctx-used'), max: $n('ctx-max'), fill: $n('ctx-fill'), wrap: $n('ctx-meter'), compress: $n('ctx-compress'), gpuUsed: $n('gpu-used'), gpuTotal: $n('gpu-total'), gpu: $n('num-gpu') };
  let ctxUsed = 0, ctxMax = 0; // ctxMax: what the last reply or the loaded model says
  let liveTps = 0, lastTps = 0, gpuNums = null, numFrame = 0, numLive = false;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  // Sets a number's text; a changed value outside live counting eases in.
  function setNum(node, text) {
    if (!node || node.textContent === text) return;
    node.textContent = text;
    if (!numLive && !reduceMotion.matches) {
      node.classList.remove('bump');
      void node.offsetWidth;
      node.classList.add('bump');
    }
  }
  function paintNumbers() {
    numFrame = 0;
    if (!numEls.wrap) return;
    const max = ctxMax || window.assayTune?.contextSize?.() || 0;
    const pct = max ? Math.min(100, (ctxUsed * 100) / max) : 0;
    numEls.fill.style.width = pct + '%';
    numEls.wrap.classList.toggle('warn', pct >= 80);
    setNum(numEls.used, kfmt(ctxUsed));
    setNum(numEls.max, !max ? 'auto' : max % 1024 === 0 ? max / 1024 + 'k' : kfmt(max));
    numEls.wrap.classList.toggle('no-used', !max && !ctxUsed); // "Context auto", not "0 of auto"
    // Compress is offered when the chat nears the limit and has something to fold up.
    const msgs = list.querySelectorAll(':scope > .msg').length, folded = window.assayTune?.compression?.()?.upTo || 0;
    numEls.compress.hidden = !(pct >= 80 && chatId && msgs - 4 > folded && !controller);
    const tps = numLive ? liveTps : lastTps;
    setNum(numEls.tps, tps ? String(Math.round(tps)) : '–');
    numEls.tps.classList.toggle('live', numLive);
    const opt = selectedOption();
    const off = !!opt?.value && !isLoaded(opt);
    numEls.gpu.classList.toggle('off', off);
    numEls.gpu.title = off ? 'Not loaded yet. Click to load it now' : 'GPU memory used by the loaded model. Click to see what is using it';
    if (gpuNums) {
      setNum(numEls.gpuUsed, gpuNums.used);
      setNum(numEls.gpuTotal, gpuNums.total);
      numEls.gpu.classList.toggle('cpu', !!gpuNums.cpu);
    }
  }
  const paintSoon = () => { if (!numFrame) numFrame = requestAnimationFrame(paintNumbers); };
  function showContext(used, max) {
    if (used != null) ctxUsed = used;
    if (max) ctxMax = max;
    paintSoon();
  }
  // The loaded model's share of GPU memory, of the card's total.
  const gb = (b) => { const v = b / 2 ** 30; return v >= 10 ? String(Math.round(v)) : v.toFixed(1); };
  async function refreshGpu() {
    try {
      const [g, l] = await Promise.all([fetch('/api/gpu').then((r) => r.json()), fetch('/api/loaded').then((r) => r.json())]);
      const model = baseOf(selectedOption());
      const mine = (l || []).find((m) => !m.loading && (m.name === model || m.name === selectedOption()?.value)) || (l || []).find((m) => !m.loading);
      const used = mine ? (mine.sizeVram || 0) : 0;
      if (g?.vramTotal > 0) gpuNums = { used: gb(used), total: gb(g.vramTotal), cpu: !!mine && mine.size > 0 && !used };
      if (mine?.context) ctxMax = mine.context; // what the loaded model really runs with
    } catch {}
    paintSoon();
  }
  {
    const lastStats = [...list.querySelectorAll('.stats[data-used]')].pop();
    if (lastStats) {
      let used = lastStats.dataset.used.split('+').reduce((a, b) => a + +b, 0);
      // A summary made after the last reply (Compress now) shrank the prompt.
      const c = window.assayTune?.compression?.();
      if (c?.saved && c.created && Date.parse(c.created) / 1000 > +lastStats.dataset.at) used = Math.max(0, used - c.saved);
      ctxUsed = used;
      ctxMax = +lastStats.dataset.ctx || 0;
      lastTps = +lastStats.dataset.tps || 0;
    } else {
      // No reply yet (a new chat, or one from before stats were kept): guess from the text.
      let chars = 0;
      for (const m of list.querySelectorAll(':scope > .msg')) chars += m.textContent.length;
      ctxUsed = Math.round(chars / 3.8);
    }
    paintNumbers();
  }
  $n('ctx-open')?.addEventListener('click', () => window.assayTune?.open?.('num_ctx'));
  numEls.compress?.addEventListener('click', async () => {
    numEls.compress.disabled = true;
    numEls.compress.textContent = 'Summarizing…';
    try {
      const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}/compress`, { method: 'POST' });
      if (!res.ok) throw new Error(await errorText(res));
      location.reload();
    } catch (err) {
      numEls.compress.disabled = false;
      numEls.compress.textContent = 'Compress';
      numEls.compress.title = err.message;
    }
  });
  numEls.gpu?.addEventListener('click', () => {
    if (numEls.gpu.classList.contains('off') && !loadBtn.hidden) { loadBtn.click(); return; }
    const chip = document.getElementById('mem-btn');
    if (chip) chip.click(); else location.href = '/library';
  });
  // Tune loads its numbers after the page, and changes with the model.
  document.addEventListener('tune-changed', paintSoon);
  document.addEventListener('loaded-changed', refreshGpu);
  refreshGpu();

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

  // The stats line under a reply; the server renders the same markup.
  function statsHTML(st) {
    const part = (v, label) => `<span class="st"><b>${v}</b> ${label}</span>`;
    return part(Math.round(st.tokensPerSec), 'tok/s')
      + (st.promptPerSec ? part(Math.round(st.promptPerSec), 'tok/s reading') : '')
      + part(st.outputTokens, 'tokens')
      + (st.totalMs ? part((st.totalMs / 1000).toFixed(1), 's') : '');
  }

  function lengthNote(onlyThinking, stats) {
    const n = (v) => v.toLocaleString();
    const used = stats ? ` (${n(stats.promptTokens)} prompt + ${n(stats.outputTokens)} reply tokens)` : '';
    return onlyThinking
      ? `The model used up the context${used} on thinking and never answered. Raise “Context length” in Tune, or turn Thinking off.`
      : `Cut off: the conversation filled the context${used}. Raise “Context length” in Tune for longer replies.`;
  }

  function addSidebarChat(id, title, projectId) {
    window.assay?.addSidebarChat(id, title, projectId);
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
    edit_code: 'Editing a code block…', forget: 'Trimming earlier messages…',
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
  // The line under a forget call says what it left out; undo puts it back.
  function trimLine(call) {
    const line = el('div', 'trim-line');
    const undo = el('button', 'trim-undo', 'undo');
    undo.type = 'button';
    undo.dataset.undoTrim = call.trim;
    line.append(`${call.label} · `, undo);
    return line;
  }
  document.addEventListener('click', async (e) => {
    const b = e.target.closest?.('.trim-undo');
    if (!b || !chatId) return;
    b.disabled = true;
    try {
      const r = await fetch(`/api/chats/${encodeURIComponent(chatId)}/trims/${encodeURIComponent(b.dataset.undoTrim)}`, { method: 'DELETE' });
      if (!r.ok) throw new Error(r.statusText);
      b.parentElement.textContent = 'Trim undone: those messages are back in later prompts';
    } catch {
      b.disabled = false;
      b.textContent = 'undo failed, try again';
    }
  });
  // Of a reply's edits only the newest stays open: each holds the whole
  // block, so several open ones repeat the file.
  function lastEditOpen(root) {
    const boxes = root ? [...root.querySelectorAll('details[data-edit]')] : [];
    boxes.forEach((d, i) => { d.open = i === boxes.length - 1; });
  }
  document.querySelectorAll('.msg.assistant').forEach(lastEditOpen);
  function fillTool(box, call) {
    box.classList.remove('pending');
    box.classList.toggle('err', !!call.error);
    box.querySelector('summary').textContent = call.label;
    if (call.name === 'edit_code' && !call.error) { // the updated block is the point
      box.dataset.edit = '';
      lastEditOpen(box.closest('.msg') || box.parentElement);
    }
    if (call.trim) box.after(trimLine(call));
    box.querySelector('.tool-body').append(el('div', 'tool-k', 'Result'), el('pre', 'tool-pre', call.block || call.result));
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
  const hasDraft = () => !!input.value.trim() || window.assayAttach.count() > 0;
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
    sendLabel.textContent = 'Stop?';
    sendBtn.dataset.state = 'stop';
    sendBtn.setAttribute('aria-label', 'Stop? Press again to stop the reply');
    sendBtn.classList.add('confirm');
    sendBtn.title = 'Press again to stop the reply';
    stopAsked = setTimeout(() => { stopAsked = null; syncSendBtn(); }, 3000);
  }
  function syncSendBtn() {
    if (stopAsked && controller && !hasDraft()) return;
    clearTimeout(stopAsked);
    stopAsked = null;
    sendBtn.classList.remove('confirm');
    const state = !controller ? 'send' : hasDraft() ? 'queue' : 'stop';
    sendLabel.textContent = { send: 'Send', queue: 'Queue', stop: 'Stop' }[state];
    sendBtn.dataset.state = state;
    sendBtn.classList.toggle('idle', state === 'send' && !hasDraft());
    sendBtn.setAttribute('aria-label', sendLabel.textContent);
    fitInput();
    sendBtn.title = controller && hasDraft() ? 'Send this after the reply finishes (Esc stops the reply)' : '';
  }
  input.addEventListener('input', syncSendBtn);
  // On a phone the box grows with the text (up to about six lines).
  function fitInput() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight + 2, isPhone() ? 152 : 240) + 'px';
  }
  sheets.phone.addEventListener('change', fitInput);
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
    const att = window.assayAttach;
    if (controller && !hasDraft()) { askStop(); return; } // the button is "Stop" while streaming
    const text = input.value.trim();
    if ((!text && !att.count()) || !select.value || att.busy()) return;
    if (await att.blockedImage()) return; // the chip says the model can't see it
    input.value = '';
    fitInput();
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
    const voice = window.assayVoice?.turn() ?? {};
    document.getElementById('empty')?.remove();
    // resume: attaching to a reply already running (its user message is on the page).
    const userMsg = el('div', 'msg user');
    const atts = [...kept, ...files];
    if (!resume) {
      if (atts.length) {
        userMsg.dataset.att = JSON.stringify(atts.map(({ url, state, error, ...a }) => a));
        userMsg.append(window.assayAttach.render(atts, chatId));
      }
      userMsg.append(el('div', 'bubble', text));
      addActions(userMsg);
      list.append(userMsg);
    }
    const reply = el('div', 'msg assistant');
    // The reply is a run of segments (thinking, text), with a row for each
    // tool call between them; these are the current segment's elements.
    let thinkBox, thinkText, md, thinkStart = 0;
    const thoughtFor = () => {
      const n = Math.max(1, Math.round((performance.now() - thinkStart) / 1000));
      return `Thought for ${n} second${n === 1 ? '' : 's'}`;
    };
    const newSegment = () => {
      thinkBox = el('details', 'thinking');
      thinkBox.hidden = true;
      thinkStart = 0;
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
    // Live speed: characters streamed over time (about 3.8 a token), smoothed.
    // Counted from the first token, so reading the prompt doesn't drag it down.
    let tpsAt = 0, tpsChars = 0, tpsSeen = 0;
    const tickSpeed = () => {
      if (resume || !tpsAt) return;
      const now = performance.now();
      if (now - tpsAt < 250) return;
      const rate = ((streamed - tpsChars) / 3.8) / ((now - tpsAt) / 1000);
      liveTps = liveTps && tpsSeen > 1 ? liveTps * 0.7 + rate * 0.3 : rate;
      tpsSeen++;
      tpsAt = now;
      tpsChars = streamed;
      numLive = true;
    };
    const ctxStart = ctxUsed + Math.round(text.length / 3.8);
    let outcome = 'done';
    // If the reply has to load the model first, the memory list shows it.
    refreshGpu();
    const loadCheck = isLoaded(selectedOption()) ? null : setTimeout(() => refreshLoaded(true), 500);
    let content = '';
    let thinking = '';
    const running = new Map(); // tool call id -> its row, until the result is in
    // Text is drawn at most once a frame. A resumed reply arrives as one
    // burst of thousands of tokens; redrawing the markdown for each one
    // froze the page for seconds.
    let frame = 0, drawnThinking = 0, drawnContent = 0;
    const draw = () => {
      cancelAnimationFrame(frame);
      frame = 0;
      if (thinking.length !== drawnThinking) {
        drawnThinking = thinking.length;
        thinkText.textContent = thinking;
        // A long think shows it's still going, even folded up.
        if (!content) thinkBox.querySelector('summary').textContent = `Thinking… ${Math.max(1, Math.round((performance.now() - thinkStart) / 1000))}s`;
      }
      if (content.length !== drawnContent) {
        drawnContent = content.length;
        setMarkdown(md, content);
      }
      if (follow) scrollDown();
    };
    const drawSoon = () => { if (!frame) frame = requestAnimationFrame(draw); };

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
          if (!tpsAt) { tpsAt = performance.now(); tpsChars = streamed; liveTps = 0; }
          streamed += ev.text.length;
          showContext(ctxStart + Math.round(streamed / 3.8));
          tickSpeed();
        } else if (frame) draw(); // anything else lands after the text so far
        switch (ev.type) {
          case 'chat':
            if (!chatId) {
              chatId = ev.id;
              chatEl.dataset.chatId = chatId;
              history.replaceState({ emptyChat: true }, '');
              history.pushState({ chat: true }, '', `/chat/${chatId}`);
              addSidebarChat(chatId, text, project());
            }
            window.assay.trackRunning(chatId);
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
            if (thinking) thinkBox.querySelector('summary').textContent = thoughtFor();
            if (thinking && !content) thinkBox.open = false;
            if (content) { md.classList.add('said'); thinkBox.open = false; } else md.remove();
            if (!thinking) thinkBox.remove();
            const row = toolRow(ev.call);
            running.set(ev.call.id, row);
            stats.before(row);
            content = thinking = '';
            drawnContent = drawnThinking = 0;
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
            if (!thinking) { thinkBox.hidden = false; thinkBox.open = true; stats.textContent = ''; thinkStart = performance.now(); }
            thinking += ev.text;
            drawSoon();
            break;
          case 'say':
            spoken.hidden = false;
            spoken.textContent += (spoken.textContent ? ' ' : '') + ev.text;
            if (!thinking) stats.textContent = '';
            break;
          case 'content':
            if (!content) {
              stats.textContent = '';
              if (thinking) { thinkBox.open = false; thinkBox.querySelector('summary').textContent = thoughtFor(); }
            }
            content += ev.text;
            drawSoon();
            break;
          case 'done':
            if (ev.stats) {
              stats.innerHTML = statsHTML(ev.stats);
              lastTps = ev.stats.tokensPerSec || lastTps;
              numLive = false;
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
        if (follow && !frame) scrollDown(); // text scrolls when drawn
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
      draw();
      if (thinking) thinkBox.querySelector('summary').textContent = thoughtFor();
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
      if (chatId) window.assay.trackRunning(chatId, false);
      addActions(reply);
      input.focus();
      numLive = false;
      refreshGpu();
      refreshLoaded(); // chatting loads the model, so the memory list may have changed
      window.assayTune?.setChat({ messages: list.querySelectorAll(':scope > .msg').length });
    }
  }

  // Opening a chat whose reply is still being written (the page was left
  // and come back to) shows it as it goes: the server replays what it has.
  if (chatEl.dataset.running && chatId) window.assayWhenVisible(() => {
    fetch(`/api/chats/${chatId}/stream`).then((res) => {
      if (res.status === 200) send('', undefined, [], [], res);
    }).catch(() => {});
  });

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
    window.assayTune?.setChat({ compression: c || null });
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
  window.assayChat = {
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
