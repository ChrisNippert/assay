// Benchmarks: results per model and set, the fit to the Intelligence Index,
// running sets in the background, reading answers, and editing sets.
(() => {
  const root = document.getElementById('bm');
  if (!root) return;
  const init = window.assayBench || {};
  let data = init.results || { sets: [], rows: [], suite: [], fit: {} };
  let run = init.run || { running: false, items: [] };
  const params = new URLSearchParams(location.search);
  let think = params.get('think') || '';
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch {} },
  };
  let suite = (() => { try { return JSON.parse(store.get('bm-suite') || 'null'); } catch { return null; } })();

  const $ = (id) => document.getElementById(id);
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  async function api(method, url, body, raw) {
    const res = await fetch(url, {
      method,
      headers: body == null ? undefined : { 'Content-Type': raw ? 'application/toml' : 'application/json' },
      body: body == null ? undefined : raw ? body : JSON.stringify(body),
    });
    let out = null;
    try { out = await res.json(); } catch {}
    if (!res.ok) throw new Error((out && out.error) || res.statusText);
    return out;
  }
  const pct = (x) => Math.round(x * 100);
  const setName = (id) => (data.sets.find((s) => s.id === id) || { name: id }).name;
  const shortName = (s) => s.short || s.name;

  // ---- loading ----
  let loadTimer = 0;
  async function reload() {
    const q = new URLSearchParams();
    if (think) q.set('think', think);
    (suite || []).forEach((s) => q.append('suite', s));
    try {
      data = await api('GET', '/api/evals/results?' + q);
      render();
    } catch (e) {
      status(e.message, true);
    }
  }
  const reloadSoon = () => { clearTimeout(loadTimer); loadTimer = setTimeout(reload, 400); };

  // ---- rendering ----
  function render() {
    renderStats();
    renderTable();
    renderFit();
    renderSets();
    root.querySelectorAll('[data-think]').forEach((b) => {
      const on = b.dataset.think === think;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', on ? 'true' : 'false');
    });
  }

  function renderStats() {
    const q = data.sets.reduce((n, s) => n + s.questions, 0);
    $('st-sets').innerHTML = '';
    $('st-sets').append(String(data.sets.length), ' ', el('span', 'lib-unit', `${q} questions`));
    $('st-sets-note').textContent = `${data.sets.filter((s) => !s.builtIn).length} of your own`;
    const tested = data.rows.filter((r) => r.overall != null).length;
    $('st-tested').innerHTML = '';
    $('st-tested').append(String(tested), ' ', el('span', 'lib-unit', `of ${data.rows.length}`));
    $('st-tested-bar').style.width = (data.rows.length ? (tested / data.rows.length) * 100 : 0) + '%';
    const f = data.fit || {};
    $('st-fit').innerHTML = '';
    if (f.ok) {
      $('st-fit').append(`R² ${f.r2.toFixed(2)}`, ' ', el('span', 'lib-unit', `±${f.se.toFixed(1)} points`));
      $('st-fit-note').textContent = `${f.n} anchor models · ${f.comment}`;
    } else {
      $('st-fit').append(el('span', 'lib-unit', `${f.n || 0} of ${f.need || 3} anchors`));
      $('st-fit-note').textContent = f.comment || '';
    }
  }

  function cellFor(row, set) {
    const c = row.cells[set.id];
    const item = run.running && run.think === think && run.items.find((it) => it.model === row.name && it.set === set.id && (it.state === 'running' || it.state === 'pending'));
    const box = el('div', 'bm-cell');
    if (item && item.state === 'running') {
      box.classList.add('busy');
      box.append(el('span', 'lib-spin'), el('span', 'bm-sub', `${item.answered}/${item.total}`));
      return box;
    }
    if (item) {
      box.append(el('span', 'bm-sub', 'queued'));
      return box;
    }
    if (!c) {
      const b = el('button', 'bm-run1', 'Run');
      b.type = 'button';
      b.title = `Run ${set.name} on this model`;
      b.onclick = () => openRun([row.name], [set.id]);
      box.append(b);
      return box;
    }
    const b = el('button', 'bm-score');
    b.type = 'button';
    b.title = `${c.correct} of ${c.answered} right${c.done ? '' : ` (${c.total - c.answered} not asked yet)`}${c.stale ? '. The set has changed since.' : ''}${c.cut ? `. ${c.cut} hit the token limit.` : ''}${c.errors ? `. ${c.errors} failed.` : ''} Click to read the answers.`;
    const n = el('span', 'bm-num' + (c.done && !c.stale ? '' : ' partial'), c.done ? `${pct(c.score)}%` : `${c.answered}/${c.total}`);
    if (c.stale) n.append(el('span', 'bm-mark', '*'));
    const track = el('span', 'lib-track thin');
    const fill = el('span', 'lib-fill' + (c.done && !c.stale ? ' accent' : ''));
    fill.style.width = pct(c.score) + '%';
    track.append(fill);
    b.append(n, track);
    b.onclick = () => openAnswers(row, set.id);
    box.append(b);
    return box;
  }

  function renderTable() {
    const table = $('bm-table');
    table.innerHTML = '';
    const sets = data.sets;
    table.style.setProperty('--sets', sets.length);
    const inSuite = (id) => data.suite.includes(id);
    const head = el('div', 'bm-row bm-head');
    head.append(el('span', 'bm-th left', 'Model'));
    sets.forEach((s) => {
      const b = el('button', 'bm-th' + (inSuite(s.id) ? ' on' : ''), shortName(s));
      b.type = 'button';
      b.title = `${s.name}: ${s.questions} questions. ${inSuite(s.id) ? 'Counts toward' : 'Left out of'} the overall score; click to change.`;
      b.setAttribute('aria-pressed', inSuite(s.id) ? 'true' : 'false');
      b.onclick = () => {
        let next = inSuite(s.id) ? data.suite.filter((x) => x !== s.id) : [...data.suite, s.id];
        if (!next.length) return;
        suite = next;
        store.set('bm-suite', JSON.stringify(suite));
        reload();
      };
      head.append(b);
    });
    for (const [t, tip] of [['Overall', 'Average over the highlighted sets'], ['Index', 'Published Intelligence Index of the base model'], ['Estimate', 'Placed on the index from your overall score']]) {
      const h = el('span', 'bm-th', t);
      h.title = tip;
      head.append(h);
    }
    table.append(head);
    const f = data.fit || {};
    for (const r of data.rows) {
      const row = el('div', 'bm-row');
      const name = el('div', 'bm-model');
      const a = el('a', 'lr-name', r.title);
      a.href = '/chat?model=' + encodeURIComponent(r.name);
      a.title = r.name;
      name.append(a);
      const sub = el('div', 'lr-sub');
      if (r.quant) sub.append(el('span', 'lr-quant', r.quant));
      const more = el('button', 'bm-linkbtn', 'Run…');
      more.type = 'button';
      more.title = 'Run sets on this model';
      more.onclick = () => openRun([r.name], data.suite);
      sub.append(r.quant ? ' · ' : '', more);
      name.append(sub);
      row.append(name);
      sets.forEach((s) => row.append(cellFor(r, s)));
      const ov = el('div', 'bm-cell bm-big');
      ov.append(r.overall != null ? `${pct(r.overall)}%` : el('span', 'muted', '–'));
      if (r.overall == null) ov.title = 'Finish every highlighted set for an overall score';
      row.append(ov);
      const idx = el('div', 'bm-cell');
      if (r.index != null) {
        const l = el('a', 'bm-plain', Math.round(r.index).toString());
        l.href = data.scoreUrl;
        l.target = '_blank';
        l.rel = 'noopener';
        l.title = `Artificial Analysis Intelligence Index of ${r.of}`;
        idx.append(l);
      } else idx.append(el('span', 'muted', '–'));
      row.append(idx);
      const est = el('div', 'bm-cell');
      if (r.estimate != null) {
        const e = el('span', 'bm-est' + (r.anchor ? ' anchor' : '') + (r.outside ? ' outside' : ''), `${r.outside ? '≈' : ''}${Math.round(r.estimate)}`);
        e.title = (r.anchor ? 'Anchor: this model helped fit the line. ' : '') + `Estimate ±${(f.se || 0).toFixed(1)}${r.outside ? '; outside the range the line was fitted on, so less certain' : ''}.`;
        est.append(e);
      } else est.append(el('span', 'muted', '–'));
      row.append(est);
      table.append(row);
    }
    const none = $('bm-none');
    none.hidden = data.rows.length > 0;
    if (!data.rows.length) none.innerHTML = 'No models installed yet. Find some on <a href="/discover">Discover</a>.';
  }

  function renderFit() {
    const svg = $('bm-plot');
    const f = data.fit || {};
    const W = 360, H = 220, L = 34, R = 10, T = 10, B = 30;
    const pts = data.rows.filter((r) => r.overall != null && (r.index != null || r.estimate != null));
    const ymax = Math.max(20, Math.ceil((Math.max(0, ...pts.map((r) => (r.index != null ? r.index : r.estimate || 0))) * 1.1) / 20) * 20);
    const x = (v) => L + v * (W - L - R);
    const y = (v) => H - B - (v / ymax) * (H - T - B);
    const parts = [];
    for (let g = 0; g <= 4; g++) {
      const gy = y((ymax / 4) * g);
      parts.push(`<line x1="${L}" x2="${W - R}" y1="${gy}" y2="${gy}" class="bm-grid"/><text x="${L - 6}" y="${gy + 4}" class="bm-axis" text-anchor="end">${Math.round((ymax / 4) * g)}</text>`);
      parts.push(`<text x="${x(g / 4)}" y="${H - B + 16}" class="bm-axis" text-anchor="middle">${g * 25}%</text>`);
    }
    parts.push(`<text x="${(L + W - R) / 2}" y="${H - 2}" class="bm-axis" text-anchor="middle">your overall score</text>`);
    if (f.ok) parts.push(`<line x1="${x(0)}" y1="${y(Math.max(0, f.a))}" x2="${x(1)}" y2="${y(Math.max(0, f.a + f.b))}" class="bm-line"/>`);
    const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
    for (const r of pts) {
      if (r.index != null) parts.push(`<circle cx="${x(r.overall)}" cy="${y(r.index)}" r="4.5" class="bm-dot"><title>${esc(r.title)}: ${pct(r.overall)}%, index ${Math.round(r.index)}</title></circle>`);
      else parts.push(`<circle cx="${x(r.overall)}" cy="${y(r.estimate)}" r="4" class="bm-dot est"><title>${esc(r.title)}: ${pct(r.overall)}%, estimated ${Math.round(r.estimate)}</title></circle>`);
    }
    svg.innerHTML = parts.join('');
    const text = $('bm-fit-text');
    text.innerHTML = '';
    if (f.ok) {
      text.append(el('p', null, `The line goes through ${f.n} models that have both your overall score and a published index (filled dots). It ${f.comment}: R² ${f.r2.toFixed(2)}, and a typical miss of ${f.se.toFixed(1)} index points.`));
      text.append(el('p', 'muted small', 'Hollow dots are estimates. With few anchors the line moves a lot when one is added; more anchors across sizes make it steadier. Published scores belong to the full-precision base model, so a small quant may sit a little below the line.'));
    } else {
      text.append(el('p', null, `An estimate needs at least ${f.need || 3} models that have finished every highlighted set and have a published index; there ${f.n === 1 ? 'is' : 'are'} ${f.n || 0} now.`));
      const withIdx = data.rows.filter((r) => r.overall == null && r.cells && Object.keys(r.cells).length === 0);
      if (withIdx.length) text.append(el('p', 'muted small', 'Run the highlighted sets on a few models of different sizes from families Artificial Analysis rates (Qwen, Gemma, Llama, Mistral…).'));
    }
  }

  function renderSets() {
    const list = $('bm-setlist');
    list.innerHTML = '';
    for (const s of data.sets) {
      const row = el('div', 'row');
      const t = el('div', 'row-text');
      const title = el('div', 'row-title', s.name);
      if (s.builtIn) title.append(' ', el('span', 'tag', 'built in'));
      t.append(title, el('div', 'row-sub', `${s.questions} questions${s.check ? ' · ' + s.check : ''}${s.description ? ' · ' + s.description : ''}`));
      const ctrl = el('div', 'row-ctrl');
      const open = el('button', 'btn', s.builtIn ? 'View' : 'Edit');
      open.type = 'button';
      open.onclick = () => openEditor(s.id, false);
      const copy = el('button', 'btn ghost', 'Copy');
      copy.type = 'button';
      copy.title = 'Start a new set from this one';
      copy.onclick = () => openEditor(s.id, true);
      ctrl.append(copy, open);
      row.append(t, ctrl);
      list.append(row);
    }
    fetch('/api/evals/sets').then((r) => r.json()).then((j) => {
      const errs = Object.entries(j.errors || {});
      const p = $('bm-set-errors');
      p.hidden = !errs.length;
      p.textContent = errs.map(([id, e]) => `${id}.toml couldn't be read: ${e}`).join(' · ');
    }).catch(() => {});
  }

  // ---- run progress ----
  function status(msg, bad) {
    const s = $('bm-status');
    s.hidden = !msg;
    s.classList.toggle('bad', !!bad);
    s.textContent = msg || '';
  }
  let pollTimer = 0;
  let lastDone = -1;
  function renderRun() {
    const s = $('bm-status');
    if (!run.running) {
      const failed = run.items.filter((it) => it.state === 'failed');
      if (run.cancelled) status('Stopped. Starting again picks up where it left off.');
      else if (failed.length) status(failed.map((it) => `${it.title} · ${it.setName}: ${it.reason}`).join(' · '), true);
      else s.hidden = true;
      $('bm-run-open').disabled = false;
      return;
    }
    $('bm-run-open').disabled = true;
    const cur = run.items.find((it) => it.state === 'running');
    s.hidden = false;
    s.classList.remove('bad');
    s.innerHTML = '';
    const line = el('div', 'lib-progress-line');
    line.append(el('span', 'lib-spin'));
    const txt = el('span');
    txt.textContent = (cur ? `${cur.title} · ${cur.setName}: ${cur.answered} of ${cur.total} asked, ${cur.correct} right` : 'Starting…') + (run.think === 'off' ? ' · thinking off' : '');
    const stop = el('button', 'lib-linkbtn', 'Stop');
    stop.type = 'button';
    stop.onclick = async () => { stop.disabled = true; run = await api('DELETE', '/api/evals/run'); };
    line.append(txt, el('span', 'muted', `${run.done} / ${run.total}`), stop);
    const bar = el('span', 'lib-track');
    const fill = el('span', 'lib-fill accent');
    fill.style.width = (run.total ? (run.done / run.total) * 100 : 0) + '%';
    bar.append(fill);
    s.append(line, bar);
    if (run.waiting) s.append(el('div', 'lib-progress-note', run.waiting + '.'));
  }
  async function poll() {
    clearTimeout(pollTimer);
    try { run = await api('GET', '/api/evals/run'); } catch { pollTimer = setTimeout(poll, 4000); return; }
    renderRun();
    if (run.done !== lastDone) { lastDone = run.done; reloadSoon(); }
    if (run.running) pollTimer = setTimeout(poll, document.hidden ? 5000 : 1500);
  }

  // ---- run dialog ----
  const runDlg = $('bm-run');
  function check(container, value, label, sub, checked) {
    const l = el('label', 'bm-check');
    const i = el('input');
    i.type = 'checkbox';
    i.value = value;
    i.checked = checked;
    l.append(i, el('span', null, label));
    if (sub) l.append(el('span', 'muted small', sub));
    container.append(l);
    return i;
  }
  function updateRunNote() {
    const ms = [...$('bm-run-models').querySelectorAll('input:checked')].map((i) => i.value);
    const ss = [...$('bm-run-sets').querySelectorAll('input:checked')].map((i) => i.value);
    const q = ss.reduce((n, id) => n + (data.sets.find((s) => s.id === id) || { questions: 0 }).questions, 0);
    $('bm-run-go').disabled = !ms.length || !ss.length;
    $('bm-run-note').textContent = ms.length && ss.length
      ? `Up to ${ms.length * q} questions, ${think === 'off' ? 'thinking off' : 'thinking as each model does by default'}. Sets already finished are skipped unless you tick the box above. It runs in the background; you can leave this page.`
      : 'Pick at least one model and one set.';
  }
  function openRun(models, sets) {
    const mBox = $('bm-run-models'), sBox = $('bm-run-sets');
    mBox.innerHTML = sBox.innerHTML = '';
    for (const r of data.rows) {
      const n = Object.values(r.cells).filter((c) => c.done && !c.stale).length;
      check(mBox, r.name, r.title, [r.quant, n ? `${n} sets done` : ''].filter(Boolean).join(' · '), (models || []).includes(r.name));
    }
    for (const s of data.sets) check(sBox, s.id, s.name, `${s.questions}`, (sets || data.suite).includes(s.id));
    $('bm-run-redo').checked = false;
    updateRunNote();
    runDlg.showModal();
  }
  runDlg.addEventListener('change', updateRunNote);
  $('bm-run-open').onclick = () => openRun([], data.suite);
  $('bm-run-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const body = {
      models: [...$('bm-run-models').querySelectorAll('input:checked')].map((i) => i.value),
      sets: [...$('bm-run-sets').querySelectorAll('input:checked')].map((i) => i.value),
      think,
      redo: $('bm-run-redo').checked,
    };
    const go = $('bm-run-go');
    go.disabled = true;
    try {
      run = await api('POST', '/api/evals/run', body);
      runDlg.close();
      renderRun();
      render();
      poll();
    } catch (err) {
      $('bm-run-note').textContent = err.message;
    } finally {
      go.disabled = false;
    }
  });

  // ---- answers ----
  const ansDlg = $('bm-answers');
  let ansFor = null;
  async function openAnswers(row, setId) {
    ansFor = { row, setId };
    $('bm-ans-title').textContent = `${row.title} · ${setName(setId)}`;
    $('bm-ans-sub').textContent = 'Loading…';
    $('bm-ans-list').innerHTML = '';
    ansDlg.showModal();
    const q = new URLSearchParams({ digest: row.digest, set: setId, think });
    let j;
    try { j = await api('GET', '/api/evals/result?' + q); } catch (e) { $('bm-ans-sub').textContent = e.message; return; }
    const res = j.result, set = j.set || { questions: [] };
    const byId = Object.fromEntries((set.questions || []).map((qq) => [qq.id, qq]));
    const secs = res.answers.reduce((n, a) => n + a.ms, 0) / 1000;
    const toks = res.answers.reduce((n, a) => n + a.tokens, 0);
    $('bm-ans-sub').textContent = `${res.correct} of ${res.answers.length} right${res.answers.length < res.total ? ` · ${res.total - res.answers.length} not asked yet` : ''} · ${think === 'off' ? 'thinking off' : "model's default thinking"} · up to ${res.maxTokens} tokens each · ${Math.round(secs)} s, ${toks.toLocaleString()} tokens · ${new Date(res.at).toLocaleString()}${set.hash && set.hash !== res.setHash ? ' · the set has changed since this run' : ''}`;
    const list = $('bm-ans-list');
    for (const a of res.answers) {
      const qq = byId[a.id] || {};
      const item = el('div', 'bm-ans' + (a.correct ? ' right' : ' wrong'));
      item.dataset.right = a.correct ? '1' : '0';
      const head = el('div', 'bm-ans-head');
      head.append(el('span', 'bm-ans-mark', a.correct ? '✓' : '✗'), el('span', 'bm-ans-id', a.id));
      const facts = el('span', 'bm-ans-facts muted small');
      const parts = [];
      if (qq.answer != null && (qq.check || set.check) !== 'regex') parts.push(`expected ${qq.answer}`);
      if (a.got) parts.push(`got ${a.got}`);
      parts.push(`${a.tokens} tokens`, `${(a.ms / 1000).toFixed(1)} s`);
      if (a.cut) parts.push('hit the token limit');
      if (a.error) parts.push('error: ' + a.error);
      facts.textContent = parts.join(' · ');
      head.append(facts);
      item.append(head);
      if (qq.prompt) {
        const d = el('details', 'bm-ans-q');
        d.open = !a.correct;
        d.append(el('summary', null, 'Question'), el('pre', 'bm-pre', qq.prompt + (qq.choices ? '\n\n' + qq.choices.map((c, i) => String.fromCharCode(65 + i) + '. ' + c).join('\n') : '')));
        if ((qq.check || set.check) === 'regex') d.append(el('div', 'muted small', 'Must match: ' + qq.answer));
        item.append(d);
      }
      if (a.thinking) {
        const d = el('details', 'bm-ans-q');
        d.append(el('summary', null, `Thinking`), el('pre', 'bm-pre muted', a.thinking));
        item.append(d);
      }
      item.append(el('pre', 'bm-pre bm-reply', a.reply || '(empty reply)'));
      list.append(item);
    }
    applyShow();
  }
  let show = 'all';
  function applyShow() {
    ansDlg.querySelectorAll('[data-show]').forEach((b) => b.classList.toggle('on', b.dataset.show === show));
    $('bm-ans-list').querySelectorAll('.bm-ans').forEach((it) => {
      it.hidden = (show === 'wrong' && it.dataset.right === '1') || (show === 'right' && it.dataset.right === '0');
    });
  }
  ansDlg.querySelectorAll('[data-show]').forEach((b) => { b.onclick = () => { show = b.dataset.show; applyShow(); }; });
  $('bm-ans-delete').onclick = async () => {
    if (!ansFor || !confirm('Delete this result? The answers are gone for good.')) return;
    const q = new URLSearchParams({ digest: ansFor.row.digest, set: ansFor.setId, think });
    await api('DELETE', '/api/evals/result?' + q);
    ansDlg.close();
    reload();
  };
  $('bm-ans-rerun').onclick = () => {
    if (!ansFor) return;
    ansDlg.close();
    openRun([ansFor.row.name], [ansFor.setId]);
    $('bm-run-redo').checked = true;
    updateRunNote();
  };

  // ---- set editor ----
  const editDlg = $('bm-edit');
  let editing = null; // id being edited, or null for a new set
  const blank = `name = "My set"
description = "What it tests."
check = "number"

[[questions]]
prompt = "What is 17 × 23?"
answer = "391"

[[questions]]
prompt = "A recipe needs 3 eggs per 4 people. How many eggs for 10 people?"
answer = "7.5"
`;
  async function openEditor(id, asCopy) {
    const err = $('bm-edit-err');
    err.hidden = true;
    let toml = blank, builtIn = false;
    if (id) {
      try {
        const j = await api('GET', '/api/evals/sets/' + encodeURIComponent(id));
        toml = j.toml;
        builtIn = !!(j.set && j.set.builtIn) || data.sets.some((s) => s.id === id && s.builtIn);
      } catch (e) { alert(e.message); return; }
    }
    const readOnly = builtIn && !asCopy;
    editing = id && !asCopy ? id : null;
    $('bm-edit-title').textContent = readOnly ? setName(id) + ' (built in)' : asCopy ? 'New set from ' + setName(id) : id ? 'Edit ' + setName(id) : 'New set';
    $('bm-edit-id').value = asCopy ? id + '-mine' : id || '';
    $('bm-edit-id').readOnly = !!editing || readOnly;
    $('bm-edit-toml').value = toml;
    $('bm-edit-toml').readOnly = readOnly;
    $('bm-edit-save').hidden = readOnly;
    $('bm-edit-delete').hidden = !editing || builtIn;
    editDlg.showModal();
  }
  $('bm-new').onclick = () => openEditor(null, false);
  $('bm-edit-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const id = $('bm-edit-id').value.trim();
    const err = $('bm-edit-err');
    if (!editing && data.sets.some((s) => s.id === id) && !confirm(`A set called ${id} exists. Replace it?`)) return;
    try {
      await api('PUT', '/api/evals/sets/' + encodeURIComponent(id), $('bm-edit-toml').value, true);
      editDlg.close();
      reload();
    } catch (x) {
      err.textContent = x.message;
      err.hidden = false;
    }
  });
  $('bm-edit-delete').onclick = async () => {
    if (!editing || !confirm(`Delete the set ${setName(editing)}? Its results stay but no longer show.`)) return;
    await api('DELETE', '/api/evals/sets/' + encodeURIComponent(editing));
    editDlg.close();
    reload();
  };

  // ---- shared ----
  document.querySelectorAll('.bm-dialog [data-close]').forEach((b) => { b.onclick = () => b.closest('dialog').close(); });
  document.querySelectorAll('.bm-dialog').forEach((d) => d.addEventListener('click', (e) => { if (e.target === d) d.close(); }));
  root.querySelectorAll('[data-think]').forEach((b) => {
    b.onclick = () => {
      think = b.dataset.think;
      const u = new URL(location.href);
      if (think) u.searchParams.set('think', think); else u.searchParams.delete('think');
      history.replaceState(null, '', u);
      reload();
    };
  });
  if (suite) reload(); else render();
  renderRun();
  if (run.running) poll();
  document.addEventListener('visibilitychange', () => { if (!document.hidden && run.running) poll(); });
})();
