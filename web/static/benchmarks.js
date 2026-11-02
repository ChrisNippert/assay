// Benchmarks: results per model and set, the fit to the Intelligence Index,
// running sets in the background, reading answers, and editing sets.
(() => {
  const root = document.getElementById('bm');
  if (!root) return;
  const init = window.assayBench || {};
  let data = init.results || { sets: [], rows: [], suite: [], fit: {} };
  window.assayBenchRows = data.rows; // remote.js offers them as team members
  let run = init.run || { running: false, items: [] };
  let leanSt = init.lean || {};
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

  // What the table shows per cell: the score or one of the speed stats.
  const num = (v) => (v >= 100 ? Math.round(v).toLocaleString() : v >= 10 ? v.toFixed(1) : v.toFixed(2));
  const METRICS = {
    score: { name: 'Score' },
    tps: { name: 'Tokens/s', val: (c) => (c.ms ? c.tokens / (c.ms / 1000) : null) },
    tpt: { name: 'Tokens/answer', val: (c) => (c.answered ? c.tokens / c.answered : null) },
    tpm: { name: 'Answers/min', val: (c) => (c.ms ? c.answered / (c.ms / 60000) : null) },
    rpm: { name: 'Right/min', val: (c) => (c.ms ? c.correct / (c.ms / 60000) : null) },
    // The per-right costs charge each wrong answer a multiple of the board's
    // typical answer on top, the same for every model, so a model can't win
    // them by being brief and wrong.
    tpr: { name: 'Tokens/right', val: (c) => (c.correct ? (c.tokens + (c.answered - c.correct) * penalty * typical.tokens) / c.correct : null) },
    spr: { name: 'Seconds/right', val: (c) => (c.correct && c.ms ? (c.ms / 1000 + (c.answered - c.correct) * penalty * typical.secs) / c.correct : null) },
  };
  let penalty = (() => { const v = Number(store.get('bm-penalty-x')); return store.get('bm-penalty-x') != null && v >= 0 ? v : 1.5; })();
  // typical is the median answer on the board being ranked; standings sets it.
  let typical = { tokens: 0, secs: 0 };
  const median = (xs) => {
    const a = xs.filter((x) => x != null && isFinite(x)).sort((p, q) => p - q);
    return a.length ? (a.length % 2 ? a[a.length >> 1] : (a[a.length / 2 - 1] + a[a.length / 2]) / 2) : 0;
  };
  let metric = METRICS[store.get('bm-metric')] ? store.get('bm-metric') : 'score';
  // sumCells adds cells up, so speed stats can be had over several sets.
  const sumCells = (cs) => cs.reduce((a, c) => ({ tokens: a.tokens + (c.tokens || 0), ms: a.ms + (c.ms || 0), answered: a.answered + c.answered, correct: a.correct + c.correct, cost: a.cost + (c.cost || 0) }), { tokens: 0, ms: 0, answered: 0, correct: 0, cost: 0 });
  const dur = (ms) => {
    const s = Math.round(ms / 1000);
    if (s < 90) return `${s} s`;
    const m = Math.round(s / 60);
    return m < 90 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
  };
  const money = (v) => (v >= 1 ? `$${v.toFixed(2)}` : v >= 0.01 ? `$${v.toFixed(3)}` : `$${v.toPrecision(2)}`);
  const cellStats = (c) => {
    const parts = [];
    if (c.ms) parts.push(`${num(METRICS.tps.val(c))} tokens/s`, `${num(METRICS.tpm.val(c))} answers/min`, `${num(METRICS.rpm.val(c))} right/min`, `${dur(c.ms)} in all`);
    if (c.answered) parts.push(`${Math.round(METRICS.tpt.val(c)).toLocaleString()} tokens/answer`);
    if (c.correct) parts.push(`${Math.round(c.tokens / c.correct).toLocaleString()} tokens per right answer`);
    if (c.cost) parts.push(`${money(c.cost)} in all${c.correct ? `, ${money(c.cost / c.correct)} per right answer` : ''}`);
    return parts.join(' · ');
  };
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
      window.assayBenchRows = data.rows;
      render();
    } catch (e) {
      status(e.message, true);
    }
  }
  const reloadSoon = () => { clearTimeout(loadTimer); loadTimer = setTimeout(reload, 400); };

  // ---- rendering ----
  function render() {
    renderStats();
    renderModel();
    renderBoard();
    renderTable();
    renderFit();
    renderSets();
    const shown = view === 'trade' && metric === 'score' ? 'tpm' : metric;
    root.querySelectorAll('[data-metric]').forEach((b) => {
      const on = b.dataset.metric === shown;
      if (b.dataset.metric === 'score') b.hidden = view === 'trade';
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', on ? 'true' : 'false');
    });
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
    const tested = data.rows.filter((r) => Object.values(r.cells).some(finished)).length;
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
      let text = item.judging ? `judging ${item.judging}` : `${item.answered}/${item.total}`;
      const done = item.answered - (item.from || 0);
      if (!item.judging && item.since && done > 0) {
        const ms = Date.now() - new Date(item.since).getTime();
        const per = ms / done;
        text += ` · ${dur(per * (item.total - item.answered))} left`;
        box.title = `${num(60000 / per)} answers/min this run, ${dur(per)} per answer`;
      }
      box.append(el('span', 'lib-spin'), el('span', 'bm-sub', text));
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
    b.title = `${c.correct} of ${c.answered} right${c.pending ? ` (${c.pending} waiting for a judge)` : ''}${c.answered < c.total ? ` (${c.total - c.answered} not asked yet)` : ''}${c.stale ? '. Out of date: the set or the way runs sample has changed since.' : ''}${c.cut ? `. ${c.cut} hit the token limit or looped.` : ''}${c.errors ? `. ${c.errors} failed.` : ''} Click to read the answers.`;
    const stats = cellStats(c);
    if (stats) b.title += `\n${stats}`;
    const m = METRICS[metric];
    const v = m.val ? m.val(c) : null;
    const label = m.val ? (v == null ? '–' : num(v)) : c.done ? `${pct(c.score)}%` : c.pending && c.answered >= c.total ? 'judge' : `${c.answered}/${c.total}`;
    const n = el('span', 'bm-num' + (c.done && !c.stale ? '' : ' partial'), label);
    if (c.stale) n.append(el('span', 'bm-mark', '*'));
    const track = el('span', 'lib-track thin');
    const fill = el('span', 'lib-fill' + (c.done && !c.stale ? ' accent' : ''));
    fill.style.width = (m.val ? (v == null ? 0 : (v / (colMax[set.id] || v)) * 100) : pct(c.score)) + '%';
    track.append(fill);
    b.append(n, track);
    b.onclick = () => openAnswers(row, set.id);
    box.append(b);
    return box;
  }

  let colMax = {}; // per set, the largest value of the shown speed stat, for the bars
  function renderTable() {
    const table = $('bm-table');
    table.innerHTML = '';
    const sets = data.sets;
    const m = METRICS[metric];
    colMax = {};
    if (m.val) for (const s of sets) colMax[s.id] = Math.max(0, ...data.rows.map((r) => (r.cells[s.id] && m.val(r.cells[s.id])) || 0));
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
    for (const [t, tip] of [[m.val ? m.name : 'Overall', m.val ? `${m.name} over the highlighted sets` : 'Average over the highlighted sets'], ['Index', 'Published Intelligence Index of the base model'], ['Estimate', 'Placed on the index from your overall score']]) {
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
      if (m.val) {
        const all = sumCells(data.suite.map((id) => r.cells[id]).filter(Boolean));
        const v = m.val(all);
        ov.append(v != null ? num(v) : el('span', 'muted', '–'));
        if (all.answered) ov.title = `Over ${all.answered} answers: ${cellStats(all)}`;
      } else {
        ov.append(r.overall != null ? `${pct(r.overall)}%` : el('span', 'muted', '–'));
        if (r.overall == null) ov.title = 'Finish every highlighted set for an overall score';
      }
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
      if (s.code) title.append(' ', el('span', 'tag', 'runs code'));
      if (s.lean) title.append(' ', el('span', 'tag', 'checked by Lean'));
      if (s.judge) title.append(' ', el('span', 'tag', 'judged'));
      t.append(title, el('div', 'row-sub', `${s.questions} questions${s.check ? ' · ' + s.check : ''}${s.description ? ' · ' + s.description : ''}`));
      const ctrl = el('div', 'row-ctrl');
      const open = el('button', 'btn', 'Questions');
      open.type = 'button';
      open.title = s.builtIn ? 'Browse the questions' : 'Browse the questions; edit them from there';
      open.onclick = () => openQuestions(s.id);
      const copy = el('button', 'btn ghost', 'Copy');
      copy.type = 'button';
      copy.title = 'Start a new set from this one';
      copy.onclick = () => openEditor(s.id, true);
      ctrl.append(copy, open);
      row.append(t, ctrl);
      if (s.lean && !leanReady()) t.append(leanBox());
      list.append(row);
    }
    fetch('/api/evals/sets').then((r) => r.json()).then((j) => {
      const errs = Object.entries(j.errors || {});
      const p = $('bm-set-errors');
      p.hidden = !errs.length;
      p.textContent = errs.map(([id, e]) => `${id}.toml couldn't be read: ${e}`).join(' · ');
    }).catch(() => {});
  }

  // ---- one model: pick it, see its sets, run some ----
  let picked = params.get('model') || store.get('bm-model') || '';
  let ticked = null; // set ids ticked to run; null until the model's rows are first drawn
  const pickedRow = () => {
    let r = data.rows.find((x) => x.name === picked);
    if (!r && data.rows.length) {
      // Default: the model with the most finished sets.
      const done = (x) => Object.values(x.cells).filter((c) => c.done && !c.stale).length;
      r = [...data.rows].sort((a, b) => done(b) - done(a))[0];
      picked = r.name;
    }
    return r;
  };
  function pick(name) {
    if (name !== picked) ticked = null;
    picked = name;
    store.set('bm-model', name);
    renderModel();
    renderBoard();
    const bar = root.querySelector('.bm-vbar.sel');
    if (bar) bar.scrollIntoView({ block: 'nearest', inline: 'center' });
  }
  const itemFor = (row, setId) => run.running && run.think === think && run.items.find((it) => it.model === row.name && it.set === setId && (it.state === 'running' || it.state === 'pending'));
  const finished = (c) => c && c.done && !c.stale;
  function progressText(item) {
    let text = item.judging ? `judging, ${item.judging} left` : `${item.answered} of ${item.total}`;
    const done = item.answered - (item.from || 0);
    if (!item.judging && item.since && done > 0) {
      const per = (Date.now() - new Date(item.since).getTime()) / done;
      text += ` · ${dur(per * (item.total - item.answered))} left`;
    }
    return text;
  }

  function renderModel() {
    const r = pickedRow();
    const input = $('bm-combo-input');
    if (document.activeElement !== input) input.value = r ? r.title + (r.quant ? ' · ' + r.quant : '') : '';
    const sum = $('bm-model-sum');
    sum.innerHTML = '';
    const rows = $('bm-setrows');
    rows.innerHTML = '';
    $('bm-runbar').hidden = !r;
    $('bm-model-h').textContent = r ? `Sets for ${r.title}` : 'Sets';
    if (!r) return;
    const bits = [];
    if (r.overall != null) bits.push(['Overall', `${pct(r.overall)}%`]);
    if (r.index != null) bits.push(['Index', String(Math.round(r.index))]);
    else if (r.estimate != null) bits.push(['Estimate', `${r.outside ? '≈' : ''}${Math.round(r.estimate)}`]);
    const all = sumCells(Object.values(r.cells));
    if (all.ms) bits.push(['Tokens/s', num(METRICS.tps.val(all))]);
    for (const [k, v] of bits) {
      const t = el('div', 'bm-sum');
      t.append(el('span', 'bm-sum-k', k), el('span', 'bm-sum-v', v));
      sum.append(t);
    }
    if (!ticked) ticked = new Set(data.sets.filter((s) => !finished(r.cells[s.id]) && data.suite.includes(s.id)).map((s) => s.id));
    for (const s of data.sets) {
      const c = r.cells[s.id];
      const item = itemFor(r, s.id);
      const row = el('div', 'row bm-setrow');
      const box = el('label', 'bm-tick');
      const cb = el('input');
      cb.type = 'checkbox';
      cb.checked = ticked.has(s.id);
      cb.setAttribute('aria-label', `Run ${s.name}`);
      cb.onchange = () => { if (cb.checked) ticked.add(s.id); else ticked.delete(s.id); runLabel(); };
      box.append(cb);
      const t = el('div', 'row-text');
      const title = el('div', 'row-title', s.name);
      if (s.code) title.append(' ', el('span', 'tag', 'runs code'));
      if (s.lean) title.append(' ', el('span', 'tag', 'Lean'));
      if (s.judge) title.append(' ', el('span', 'tag', 'judged'));
      const subText = c && c.answered ? cellStats(c) : `${s.questions} questions${s.description ? ' · ' + s.description : ''}`;
      t.append(title, el('div', 'row-sub', subText));
      const right = el('div', 'bm-setscore');
      if (item) {
        right.classList.add('busy');
        if (item.state === 'running') right.append(el('span', 'lib-spin'));
        right.append(el('span', 'muted small', item.state === 'running' ? progressText(item) : 'queued'));
      } else if (c) {
        const b = el('button', 'bm-score');
        b.type = 'button';
        b.title = `${c.correct} of ${c.answered} right${c.stale ? '. Out of date: the set or the way runs sample has changed since.' : ''}${c.cut ? `. ${c.cut} ran out of tokens or looped.` : ''} Click to read the answers.`;
        const n = el('span', 'bm-num' + (finished(c) ? '' : ' partial'), c.done ? `${pct(c.score)}%` : `${c.answered}/${c.total}`);
        if (c.stale) n.append(el('span', 'bm-mark', '*'));
        const track = el('span', 'lib-track thin');
        const fill = el('span', 'lib-fill' + (finished(c) ? ' accent' : ''));
        fill.style.width = pct(c.score) + '%';
        track.append(fill);
        b.append(n, track);
        b.onclick = () => openAnswers(r, s.id);
        right.append(b);
      } else right.append(el('span', 'muted small', 'not run'));
      row.append(box, t, right);
      rows.append(row);
    }
    runLabel();
  }
  function runLabel() {
    const b = $('bm-one-run');
    const n = ticked ? ticked.size : 0;
    b.textContent = n ? `Run ${n} set${n === 1 ? '' : 's'}` : 'Run';
    b.disabled = !n || run.running;
    b.title = run.running ? 'A run is going; wait for it to finish or stop it' : '';
  }
  root.querySelectorAll('[data-tick]').forEach((b) => {
    b.onclick = () => {
      const r = pickedRow();
      if (!r) return;
      const k = b.dataset.tick;
      ticked = new Set(k === 'all' ? data.sets.map((s) => s.id) : k === 'new' ? data.sets.filter((s) => !r.cells[s.id]).map((s) => s.id) : []);
      renderModel();
    };
  });
  $('bm-one-run').onclick = async () => {
    const r = pickedRow();
    const sets = data.sets.filter((s) => ticked.has(s.id));
    if (!r || !sets.length) return;
    if (sets.some((s) => s.judge || (s.lean && !leanReady()))) { openRun([r.name], sets.map((s) => s.id)); return; } // needs the dialog's judge or Lean install
    const b = $('bm-one-run');
    b.disabled = true;
    try {
      run = await api('POST', '/api/evals/run', { models: [r.name], sets: sets.map((s) => s.id), think, redo: $('bm-one-redo').checked });
      ticked = new Set();
      renderRun();
      render();
      poll();
    } catch (e) {
      status(e.message, true);
      b.disabled = false;
    }
  };

  // The searchable model picker.
  const comboIn = $('bm-combo-input'), comboList = $('bm-combo-list');
  let comboAt = 0;
  function comboMatches() {
    const q = comboIn.value.trim().toLowerCase();
    return data.rows.filter((r) => !q || `${r.title} ${r.name} ${r.quant || ''}`.toLowerCase().includes(q));
  }
  function showCombo() {
    const ms = comboMatches();
    comboAt = Math.min(comboAt, Math.max(0, ms.length - 1));
    comboList.innerHTML = '';
    ms.forEach((r, i) => {
      const o = el('div', 'bm-opt' + (i === comboAt ? ' at' : '') + (r.name === picked ? ' sel' : ''));
      o.setAttribute('role', 'option');
      o.setAttribute('aria-selected', r.name === picked ? 'true' : 'false');
      const done = Object.values(r.cells).filter(finished).length;
      o.append(el('span', 'bm-opt-name', r.title), el('span', 'bm-opt-sub', [r.quant, r.params ? `${r.params < 10 ? r.params.toFixed(1) : Math.round(r.params)}B` : '', done ? `${done} set${done === 1 ? '' : 's'} done` : 'not tested', r.overall != null ? `${pct(r.overall)}% overall` : ''].filter(Boolean).join(' · ')));
      o.onmousedown = (e) => { e.preventDefault(); choose(r); };
      comboList.append(o);
    });
    if (!ms.length) comboList.append(el('div', 'bm-opt muted', 'No model matches'));
    comboList.hidden = false;
    comboIn.setAttribute('aria-expanded', 'true');
    comboList.querySelector('.at')?.scrollIntoView({ block: 'nearest' });
  }
  function hideCombo() {
    comboList.hidden = true;
    comboIn.setAttribute('aria-expanded', 'false');
  }
  function choose(r) {
    hideCombo();
    comboIn.blur();
    pick(r.name);
  }
  comboIn.addEventListener('focus', () => { comboIn.select(); comboAt = 0; comboIn.value = ''; showCombo(); });
  comboIn.addEventListener('input', () => { comboAt = 0; showCombo(); });
  comboIn.addEventListener('blur', () => { hideCombo(); renderModel(); });
  comboIn.addEventListener('keydown', (e) => {
    const ms = comboMatches();
    if (e.key === 'ArrowDown') { comboAt = Math.min(comboAt + 1, ms.length - 1); showCombo(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { comboAt = Math.max(comboAt - 1, 0); showCombo(); e.preventDefault(); }
    else if (e.key === 'Enter') { if (ms[comboAt]) choose(ms[comboAt]); e.preventDefault(); }
    else if (e.key === 'Escape') { comboIn.blur(); }
  });

  // ---- leaderboard: rank the models on a category of sets ----
  // Categories group the built-in sets; Custom ranks on any sets you tick.
  const CATS = [
    ['Math', ['math', 'math-medium', 'math-hard', 'calculus']],
    ['Science', ['physics']],
    ['Code', ['code', 'coding', 'formulas']],
    ['Projects', ['projects-games', 'projects-sims']],
    ['Reasoning', ['logic']],
    ['Knowledge', ['knowledge']],
    ['Instructions', ['instructions']],
    ['Proofs', ['lean']],
  ];
  const METRIC_NOTE = {
    score: 'Share of questions right, averaged over the sets',
    tpm: 'Questions finished a minute, thinking included',
    rpm: 'Right answers a minute: answers a minute times the share right',
    tps: 'Tokens written per second on this computer',
    tpt: 'Tokens written per question, thinking included',
    tpr: 'Tokens written for each right answer, each wrong one counting extra',
    spr: 'Seconds taken for each right answer, each wrong one counting extra',
  };
  const LOWER_BETTER = { tpt: true, tpr: true, spr: true };
  const BOARD_COLS = ['score', 'rpm', 'tpm', 'tps', 'tpt', 'tpr', 'spr'];
  let cat = store.get('bm-cat') || 'overall';
  let view = store.get('bm-view') === 'trade' ? 'trade' : 'bars';
  let subset = (() => { try { return JSON.parse(store.get('bm-subset') || 'null'); } catch { return null; } })();
  const catSets = (name) => {
    const c = CATS.find(([n]) => n === name);
    return c ? c[1].filter((id) => data.sets.some((s) => s.id === id)) : [];
  };
  // boardIds is the sets the board ranks on, and its name.
  function boardIds() {
    if (cat === 'custom') return [subset && subset.length ? subset : data.suite, 'your sets'];
    const ids = catSets(cat);
    if (cat === 'overall' || !ids.length) return [data.suite, 'Overall'];
    return [ids, cat];
  }
  const metricVal = (key, cs) => (METRICS[key].val ? METRICS[key].val(sumCells(cs)) : cs.reduce((n, c) => n + c.score, 0) / cs.length);
  const fmtMetric = (key, v) => (v == null ? '–' : key === 'score' ? `${pct(v)}` : key === 'tpt' || key === 'tpr' ? Math.round(v).toLocaleString() : num(v));
  // standings ranks the models that finished every set in ids, by key;
  // the rest that finished some are listed as unranked.
  function standings(ids, key) {
    const ranked = [], unranked = [];
    const done = data.rows.map((r) => ids.map((id) => r.cells[id])).filter((cs) => cs.every(finished)).map(sumCells);
    typical = {
      tokens: median(done.map((c) => (c.answered ? c.tokens / c.answered : null))),
      secs: median(done.map((c) => (c.answered && c.ms ? c.ms / 1000 / c.answered : null))),
    };
    for (const r of data.rows) {
      const cs = ids.map((id) => r.cells[id]);
      const left = ids.filter((id, i) => !finished(cs[i]));
      if (!left.length) {
        const vals = {};
        for (const k of Object.keys(METRICS)) vals[k] = metricVal(k, cs);
        ranked.push({ r, vals });
      } else if (left.length < ids.length) unranked.push({ r, left });
    }
    const v = (x) => x.vals[key] ?? (LOWER_BETTER[key] ? Infinity : -Infinity);
    ranked.sort((a, b) => (LOWER_BETTER[key] ? v(a) - v(b) : v(b) - v(a)));
    return { ranked, unranked };
  }
  function setMetric(k) { metric = k; store.set('bm-metric', k); render(); }
  // runRest picks a model and ticks the sets it's missing.
  function runRest(r, left) {
    pick(r.name);
    ticked = new Set(left);
    renderModel();
    $('bm-model').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  function renderBoard() {
    if (subset) subset = subset.filter((id) => data.sets.some((s) => s.id === id));
    const [all, label] = boardIds();
    // Rank on the sets at least half the models that tried this category
    // have finished, so one new set doesn't empty the board; Custom ranks on
    // exactly what was picked.
    let ids = all, dropped = [];
    if (cat !== 'custom') {
      const tried = data.rows.filter((r) => all.some((id) => finished(r.cells[id])));
      const need = Math.max(1, Math.ceil(tried.length / 2));
      const kept = all.filter((id) => tried.filter((r) => finished(r.cells[id])).length >= need);
      if (kept.length) { ids = kept; dropped = all.filter((id) => !kept.includes(id)); }
    }
    // Categories.
    const cats = $('bm-cats');
    cats.innerHTML = '';
    const catBtn = (key, text, title) => {
      const b = el('button', 'lib-sortchip' + (cat === key ? ' on' : ''), text);
      b.type = 'button';
      b.setAttribute('aria-pressed', cat === key ? 'true' : 'false');
      if (title) b.title = title;
      b.onclick = () => { cat = key; store.set('bm-cat', key); renderBoard(); };
      cats.append(b);
    };
    catBtn('overall', 'Overall', `The sets that make up the overall score: ${data.suite.map(setName).join(', ')}`);
    for (const [n] of CATS) if (catSets(n).length) catBtn(n, n, catSets(n).map(setName).join(', '));
    catBtn('custom', 'Custom…', 'Rank on any sets you pick');
    // Custom: pick the sets.
    const chips = $('bm-subset');
    chips.hidden = cat !== 'custom';
    chips.innerHTML = '';
    if (cat === 'custom') {
      for (const s of data.sets) {
        const on = !!subset && subset.includes(s.id);
        const b = el('button', 'lib-sortchip' + (on ? ' on' : ''), shortName(s));
        b.type = 'button';
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
        b.onclick = () => {
          subset = on ? (subset || []).filter((x) => x !== s.id) : [...(subset || []), s.id];
          store.set('bm-subset', JSON.stringify(subset));
          renderBoard();
        };
        chips.append(b);
      }
      if (subset && subset.length && (subset.length !== data.suite.length || subset.some((id) => !data.suite.includes(id)))) {
        const use = el('button', 'bm-linkbtn', 'Make these the overall');
        use.type = 'button';
        use.title = 'The overall score and the index estimate will use these sets';
        use.onclick = () => { suite = [...subset]; store.set('bm-suite', JSON.stringify(suite)); cat = 'overall'; store.set('bm-cat', cat); reload(); };
        chips.append(use);
      }
    }
    const m = METRICS[metric];
    const { ranked, unranked } = standings(ids, metric);
    const setsText = `${ids.length === 1 ? setName(ids[0]) : `${ids.length} sets: ${ids.map(setName).join(', ')}`}.` +
      (dropped.length ? ` Leaving out ${dropped.map(setName).join(', ')} until more models have run ${dropped.length === 1 ? 'it' : 'them'}.` : '');
    root.querySelectorAll('[data-view]').forEach((b) => {
      const on = b.dataset.view === view;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', on ? 'true' : 'false');
    });
    $('bm-trade').hidden = view !== 'trade';
    $('bm-penalty').hidden = metric !== 'tpr' && metric !== 'spr';
    $('bm-pen-typ').textContent = `${Math.round(typical.tokens).toLocaleString()} tokens, ${Math.round(typical.secs)} s`;
    $('bm-bars').parentElement.hidden = view === 'trade';
    if (view === 'trade') {
      const xk = metric === 'score' ? 'tpm' : metric;
      $('bm-chart-title').textContent = `Score against ${METRICS[xk].name} · ${label}`;
      $('bm-chart-note').textContent = `Up and to the right is better${LOWER_BETTER[xk] ? ' (lower to the right)' : ''}. The line joins the best tradeoffs: no other model is both better and ${LOWER_BETTER[xk] ? 'cheaper' : 'faster'}. ${setsText}`;
      renderTradeoff(ranked, xk, ids);
    } else {
      $('bm-chart-title').textContent = `${metric === 'score' ? 'Score' : m.name} · ${label}`;
      $('bm-chart-note').textContent = `${METRIC_NOTE[metric]}. ${LOWER_BETTER[metric] ? 'Lower' : 'Higher'} is better. ${setsText}`;
    }
    // Bars.
    const bars = $('bm-bars');
    bars.innerHTML = '';
    const vals = ranked.map((x) => x.vals[metric]).filter((v) => v != null && isFinite(v));
    const max = Math.max(...vals, metric === 'score' ? 1 : 0);
    if (!ranked.length) bars.append(el('p', 'muted small bm-vbars-empty', `No model has finished ${ids.length === 1 ? setName(ids[0]) : 'all of these sets'} yet.`));
    for (const { r, vals: v } of ranked) {
      const b = el('button', 'bm-vbar' + (r.name === picked ? ' sel' : ''));
      b.type = 'button';
      b.title = `${r.title}${r.quant ? ' · ' + r.quant : ''}: ${fmtMetric(metric, v[metric])}${metric === 'score' ? '%' : ' ' + m.name.toLowerCase()}. Click to pick it.`;
      const col = el('span', 'bm-vbar-col');
      const fill = el('span', 'bm-vbar-fill');
      fill.style.height = (max && v[metric] != null ? Math.max(1, (v[metric] / max) * 100) : 0) + '%';
      col.append(el('span', 'bm-vbar-val', fmtMetric(metric, v[metric])), fill);
      b.append(col, el('span', 'bm-vbar-name', r.title));
      b.onclick = () => pick(r.name);
      bars.append(b);
    }
    // Table: every metric, sorted like the bars; a header sorts by its metric.
    const t = $('bm-rank');
    t.innerHTML = '';
    const head = el('tr');
    head.append(el('th', 'bm-rank-n', '#'), el('th', 'bm-rank-model', 'Model'));
    for (const k of BOARD_COLS) {
      const th = el('th', 'bm-rank-num' + (k === metric ? ' on' : '') + (k === 'score' ? ' k-score' : ''));
      const hb = el('button', 'bm-rank-sort' + (k === metric ? ' on' : ''), k === 'score' ? 'Score' : METRICS[k].name);
      hb.type = 'button';
      hb.title = `Rank by ${METRIC_NOTE[k].toLowerCase()}`;
      hb.onclick = () => setMetric(k);
      th.setAttribute('aria-sort', k === metric ? (LOWER_BETTER[k] ? 'ascending' : 'descending') : 'none');
      th.append(hb);
      head.append(th);
    }
    const thead = el('thead');
    thead.append(head);
    const tbody = el('tbody');
    ranked.forEach(({ r, vals: v }, i) => {
      const tr = el('tr', r.name === picked ? 'sel' : '');
      const nameCell = el('td', 'bm-rank-model');
      const nb = el('button', 'bm-rank-pick');
      nb.type = 'button';
      nb.append(el('span', 'bm-rank-title', r.title));
      if (r.quant) nb.append(el('span', 'bm-rank-quant', r.quant));
      nb.onclick = () => pick(r.name);
      nameCell.append(nb);
      tr.append(el('td', 'bm-rank-n', String(i + 1)), nameCell);
      for (const k of BOARD_COLS) tr.append(el('td', 'bm-rank-num' + (k === metric ? ' on' : '') + (k === 'score' ? ' k-score' : ''), fmtMetric(k, v[k])));
      tbody.append(tr);
    });
    t.append(thead, tbody);
    t.parentElement.hidden = !ranked.length;
    // Models that haven't finished every set here.
    const un = $('bm-unranked');
    un.innerHTML = '';
    if (unranked.length) {
      un.append(el('span', 'muted small', 'Not ranked yet:'));
      for (const { r, left } of unranked) {
        const b = el('button', 'bm-linkbtn', `${r.title} (${ids.length - left.length} of ${ids.length})`);
        b.type = 'button';
        b.title = `Hasn't finished ${left.map(setName).join(', ')}. Click to run them.`;
        b.onclick = () => runRest(r, left);
        un.append(b);
      }
    }
    renderRanks(ids, label, ranked, unranked);
  }
  // renderTradeoff plots score against a speed or token metric (log scale,
  // better to the right) and joins the frontier: the models no other model
  // beats on both.
  function renderTradeoff(ranked, xk, ids) {
    const box = $('bm-trade');
    box.innerHTML = '';
    const pts = ranked.filter((x) => x.vals[xk] > 0 && isFinite(x.vals[xk]));
    if (!pts.length) { box.append(el('p', 'muted small bm-vbars-empty', `No model has finished ${ids.length === 1 ? setName(ids[0]) : 'all of these sets'} yet.`)); return; }
    const W = 760, H = 380, L = 48, R = 24, T = 16, B = 44;
    const lo = LOWER_BETTER[xk];
    // good: bigger is better along x.
    const good = (v) => (lo ? -Math.log10(v) : Math.log10(v));
    const gs = pts.map((p) => good(p.vals[xk]));
    let g0 = Math.min(...gs), g1 = Math.max(...gs);
    const pad = Math.max(0.15, (g1 - g0) * 0.08);
    g0 -= pad; g1 += pad;
    const scores = pts.map((p) => p.vals.score);
    const y0 = Math.max(0, Math.floor(Math.min(...scores) * 10 - 0.5) / 10), y1 = 1;
    const X = (v) => L + ((good(v) - g0) / (g1 - g0)) * (W - L - R);
    const Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
    const NS = 'http://www.w3.org/2000/svg';
    const mk = (tag, attrs, text) => {
      const n = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      if (text != null) n.textContent = text;
      return n;
    };
    const svg = mk('svg', { viewBox: `0 0 ${W} ${H}`, class: 'bm-trade-svg', role: 'img', 'aria-label': `Score against ${METRICS[xk].name.toLowerCase()}` });
    // Grid: score every 10 points, x at 1-2-5 steps.
    for (let v = Math.ceil(y0 * 10) / 10; v <= 1.0001; v += 0.1) {
      svg.append(mk('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: 'bm-trade-grid' }));
      svg.append(mk('text', { x: L - 8, y: Y(v) + 4, class: 'bm-trade-tick', 'text-anchor': 'end' }, String(Math.round(v * 100))));
    }
    const raw = (g) => Math.pow(10, lo ? -g : g);
    const vmin = Math.min(raw(g0), raw(g1)), vmax = Math.max(raw(g0), raw(g1));
    for (let e = Math.floor(Math.log10(vmin)); e <= Math.ceil(Math.log10(vmax)); e++) {
      for (const f of [1, 2, 5]) {
        const v = f * Math.pow(10, e);
        if (v < vmin || v > vmax) continue;
        svg.append(mk('line', { x1: X(v), x2: X(v), y1: T, y2: H - B, class: 'bm-trade-grid' }));
        svg.append(mk('text', { x: X(v), y: H - B + 18, class: 'bm-trade-tick', 'text-anchor': 'middle' }, v >= 1000 ? `${v / 1000}k` : String(v)));
      }
    }
    svg.append(mk('text', { x: W - R, y: H - 6, class: 'bm-trade-axis', 'text-anchor': 'end' }, `${METRICS[xk].name}${lo ? ' (fewer →)' : ' →'}`));
    svg.append(mk('text', { x: 12, y: T + 4, class: 'bm-trade-axis', transform: `rotate(-90 12 ${T + 4})`, 'text-anchor': 'end' }, 'Score'));
    // Frontier: from the best on x leftwards, keep each model that scores
    // higher than every one to its right.
    const byX = [...pts].sort((a, b) => good(b.vals[xk]) - good(a.vals[xk]));
    const front = [];
    let best = -1;
    for (const p of byX) if (p.vals.score > best) { front.push(p); best = p.vals.score; }
    const onFront = new Set(front.map((p) => p.r.name));
    if (front.length > 1) {
      // A step line: a model on the frontier holds its score leftwards until
      // the next better one.
      const f = [...front].reverse();
      let d = `M${X(f[0].vals[xk])},${Y(f[0].vals.score)}`;
      for (let i = 1; i < f.length; i++) d += ` H${X(f[i].vals[xk])} V${Y(f[i].vals.score)}`;
      svg.append(mk('path', { d, class: 'bm-trade-front' }));
    }
    // Labels: beside each dot, nudged up or down (or to the other side)
    // until they don't cover one another.
    const placed = [];
    const labelAt = new Map();
    const hits = (a) => placed.some((b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1);
    for (const p of [...pts].sort((a, b) => (b.r.name === picked) - (a.r.name === picked) || b.vals.score - a.vals.score)) {
      const x = X(p.vals[xk]), y = Y(p.vals.score), w = p.r.title.length * 7.4 + 4;
      const pref = x > W - R - 150 ? ['end', 'start'] : ['start', 'end'];
      let spot = null;
      for (const dy of [0, -15, 15, -30, 30, -45, 45]) {
        for (const side of pref) {
          const lx = side === 'start' ? x + 11 : x - 11;
          const box = { x0: side === 'start' ? lx : lx - w, x1: side === 'start' ? lx + w : lx, y0: y + dy - 9, y1: y + dy + 7 };
          if (!hits(box)) { spot = { lx, ly: y + dy + 4, side, box }; break; }
        }
        if (spot) break;
      }
      if (!spot) spot = { lx: x + 11, ly: y + 4, side: 'start', box: { x0: 0, x1: 0, y0: 0, y1: 0 } };
      placed.push(spot.box);
      labelAt.set(p.r.name, spot);
    }
    // Dots, the picked one last so it sits on top.
    const order = [...pts].sort((a, b) => (a.r.name === picked) - (b.r.name === picked));
    for (const p of order) {
      const x = X(p.vals[xk]), y = Y(p.vals.score);
      const sel = p.r.name === picked;
      const g = mk('g', { class: 'bm-trade-pt' + (onFront.has(p.r.name) ? ' front' : '') + (sel ? ' sel' : ''), tabindex: '0', role: 'button',
        'aria-label': `${p.r.title}: score ${pct(p.vals.score)}, ${fmtMetric(xk, p.vals[xk])} ${METRICS[xk].name.toLowerCase()}` });
      g.append(mk('title', {}, `${p.r.title}${p.r.quant ? ' · ' + p.r.quant : ''}\nScore ${pct(p.vals.score)}% · ${fmtMetric(xk, p.vals[xk])} ${METRICS[xk].name.toLowerCase()}`));
      g.append(mk('circle', { cx: x, cy: y, r: 16, class: 'bm-trade-hit' }));
      g.append(mk('circle', { cx: x, cy: y, r: sel ? 7 : 5.5, class: 'bm-trade-dot' }));
      const at = labelAt.get(p.r.name);
      g.append(mk('text', { x: at.lx, y: at.ly, 'text-anchor': at.side, class: 'bm-trade-label' }, p.r.title));
      g.addEventListener('click', () => pick(p.r.name));
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(p.r.name); } });
      svg.append(g);
    }
    box.append(svg);
  }

  // renderRanks says where the picked model stands on each metric.
  function renderRanks(ids, label, ranked, unranked) {
    const box = $('bm-ranks');
    box.innerHTML = '';
    const r = data.rows.find((x) => x.name === picked);
    if (!r) return;
    const un = unranked.find((x) => x.r === r);
    if (!ranked.some((x) => x.r === r)) {
      if (!un) { box.append(el('span', 'muted small', `${r.title} hasn't run any ${label === 'Overall' ? 'overall' : label} sets yet.`)); return; }
      box.append(el('span', 'muted small', `Not ranked on ${label}: hasn't finished ${un.left.map(setName).join(', ')}.`));
      const b = el('button', 'bm-linkbtn', 'Run them');
      b.type = 'button';
      b.onclick = () => runRest(r, un.left);
      box.append(b);
      return;
    }
    for (const k of BOARD_COLS) {
      const order = standings(ids, k).ranked;
      const i = order.findIndex((x) => x.r === r);
      const b = el('button', 'bm-rankchip' + (k === metric ? ' on' : '') + (i === 0 ? ' top' : ''));
      b.type = 'button';
      b.title = `${r.title} is #${i + 1} of ${order.length} on ${METRIC_NOTE[k].toLowerCase()} (${label}). Click to show it.`;
      b.append(el('span', 'bm-rankchip-n', `#${i + 1}`), el('span', 'bm-rankchip-k', `${k === 'score' ? 'Score' : METRICS[k].name} · ${fmtMetric(k, order[i].vals[k])}${k === 'score' ? '%' : ''}`));
      b.onclick = () => setMetric(k);
      box.append(b);
    }
    box.append(el('span', 'muted small', `of ${ranked.length} on ${label}`));
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
    renderModel();
    const s = $('bm-status');
    if (!run.running) {
      const failed = run.items.filter((it) => it.state === 'failed');
      const waiting = run.items.reduce((n, it) => n + (it.waiting || 0), 0);
      if (waiting && !failed.length && !run.cancelled) status(`${waiting} answer${waiting === 1 ? '' : 's'} wait for a judge. Run the set again with a judge picked to grade them.`);
      else if (run.cancelled) status('Stopped. Starting again picks up where it left off.');
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
    txt.textContent = (cur ? (cur.judging ? `${cur.title} · ${cur.setName}: the judge has ${cur.judging} left` : `${cur.title} · ${cur.setName}: ${cur.answered} of ${cur.total} asked, ${cur.correct} right`) : 'Starting…') + (run.think === 'off' ? ' · thinking off' : '');
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
    const picked = data.sets.filter((s) => ss.includes(s.id));
    $('bm-judge-row').hidden = !picked.some((s) => s.judge);
    $('bm-code-note').hidden = !picked.some((s) => s.code);
    const needLean = picked.some((s) => s.lean) && !leanReady();
    const ln = $('bm-lean-need');
    ln.hidden = !needLean;
    ln.innerHTML = '';
    if (needLean) ln.append(leanBox());
    const q = ss.reduce((n, id) => n + (data.sets.find((s) => s.id === id) || { questions: 0 }).questions, 0);
    $('bm-run-go').disabled = !ms.length || !ss.length || needLean;
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
    const judge = $('bm-run-judge');
    judge.innerHTML = '';
    const none = el('option', null, 'No judge: grade later');
    none.value = '';
    judge.append(none);
    const byParams = [...data.rows].sort((a, b) => (b.params || 0) - (a.params || 0));
    for (const r of byParams) {
      const o = el('option', null, `${r.title}${r.quant ? ' · ' + r.quant : ''}${r.params ? ` · ${r.params < 10 ? r.params.toFixed(1) : Math.round(r.params)}B` : ''}`);
      o.value = r.name;
      judge.append(o);
    }
    const saved = store.get('bm-judge');
    judge.value = saved && byParams.some((r) => r.name === saved) ? saved : byParams.length ? byParams[0].name : '';
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
      judge: $('bm-judge-row').hidden ? '' : $('bm-run-judge').value,
    };
    if (body.judge) store.set('bm-judge', body.judge);
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

  // The fun numbers over one run's answers.
  function renderAnswerStats(answers) {
    const box = $('bm-ans-stats');
    box.innerHTML = '';
    const as = answers.filter((a) => !a.error);
    if (!as.length) return;
    const sum = (xs, f) => xs.reduce((n, a) => n + f(a), 0);
    const median = (xs) => {
      const s = [...xs].sort((a, b) => a - b);
      return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
    };
    const toks = sum(as, (a) => a.tokens), ms = sum(as, (a) => a.ms);
    const right = as.filter((a) => a.correct), wrong = as.filter((a) => !a.correct && !a.pending);
    const think = sum(as, (a) => (a.thinking || '').length), said = sum(as, (a) => (a.reply || '').length);
    const longest = as.reduce((x, a) => (a.tokens > x.tokens ? a : x));
    const quickRight = right.length ? right.reduce((x, a) => (a.ms < x.ms ? a : x)) : null;
    const stuck = as.filter((a) => a.cut || a.looped).length;
    const tiles = [
      ['Tokens/s', ms ? num(toks / (ms / 1000)) : '–', 'tokens written per second, reading the question included'],
      ['Tasks/min', ms ? num(as.length / (ms / 60000)) : '–', `${dur(ms)} for ${as.length} answers`],
      ['Tokens/task', Math.round(toks / as.length).toLocaleString(), `median ${Math.round(median(as.map((a) => a.tokens))).toLocaleString()}`],
      ['Time/task', dur(median(as.map((a) => a.ms))), 'median'],
      ['Tokens/right', right.length ? Math.round(toks / right.length).toLocaleString() : '–', 'everything written, divided by the right answers'],
    ];
    if (right.length && wrong.length) {
      const r = sum(right, (a) => a.tokens) / right.length, w = sum(wrong, (a) => a.tokens) / wrong.length;
      tiles.push(['Right vs wrong', `${Math.round(r).toLocaleString()} / ${Math.round(w).toLocaleString()}`, w > r * 1.15 ? `wrong answers take ${num(w / r)}× the tokens` : r > w * 1.15 ? `right answers take ${num(r / w)}× the tokens` : 'about the same tokens either way']);
    }
    if (think + said) tiles.push(['Thinking', `${pct(think / (think + said))}%`, 'of the text written was thinking']);
    tiles.push(['Longest', `${longest.tokens.toLocaleString()}`, `tokens, on ${longest.id} (${dur(longest.ms)})`]);
    if (quickRight) tiles.push(['Quickest right', dur(quickRight.ms), `${quickRight.id}, ${quickRight.tokens.toLocaleString()} tokens`]);
    if (stuck) tiles.push(['Ran out', String(stuck), 'hit the token limit or looped']);
    for (const [k, v, note] of tiles) {
      const t = el('div', 'bm-ans-stat');
      t.title = note;
      t.append(el('span', 'bm-ans-stat-k', k), el('span', 'bm-ans-stat-v', v), el('span', 'bm-ans-stat-n', note));
      box.append(t);
    }
  }

  // ---- answers: a task browser (list on the left, one task on the right) ----
  const ansDlg = $('bm-answers');
  let ansFor = null; // {row, setId} for a model's run; null when only browsing a set's questions
  let ans = null; // {set, byId, items, res}
  let ansSel = '';
  let show = 'all';
  const md = (text) => {
    const d = el('div', 'md bm-md');
    if (window.assayMarkdown) d.innerHTML = window.assayMarkdown.render(text || '');
    else d.textContent = text || '';
    return d;
  };
  function openBrowser(title, sub) {
    $('bm-ans-title').textContent = title;
    $('bm-ans-sub').textContent = sub;
    $('bm-ans-list').innerHTML = '';
    $('bm-ans-detail').innerHTML = '';
    $('bm-ans-stats').innerHTML = '';
    $('bm-ans-search').value = '';
    show = 'all';
    if (!ansDlg.open) ansDlg.showModal();
  }
  async function openAnswers(row, setId) {
    ansFor = { row, setId };
    openBrowser(`${row.title} · ${setName(setId)}`, 'Loading…');
    const q = new URLSearchParams({ digest: row.digest, set: setId, think });
    let j;
    try { j = await api('GET', '/api/evals/result?' + q); } catch (e) { $('bm-ans-sub').textContent = e.message; return; }
    const res = j.result, set = j.set || { questions: [] };
    const toks = res.answers.reduce((n, a) => n + a.tokens, 0);
    const ms = res.answers.reduce((n, a) => n + a.ms, 0);
    $('bm-ans-sub').textContent = `${res.correct} of ${res.answers.length} right${res.answers.length < res.total ? ` · ${res.total - res.answers.length} not asked yet` : ''} · ${think === 'off' ? 'thinking off' : "model's default thinking"} · up to ${res.maxTokens.toLocaleString()} tokens each · ${dur(ms)}, ${toks.toLocaleString()} tokens · ${new Date(res.at).toLocaleString()}${j.stale ? ' · ' + j.stale : ''}`;
    renderAnswerStats(res.answers);
    ans = { set, byId: Object.fromEntries((set.questions || []).map((qq) => [qq.id, qq])), items: res.answers, res };
    browserMode();
    const firstWrong = res.answers.find((a) => !a.correct);
    selectTask((firstWrong || res.answers[0] || {}).id);
  }
  async function openQuestions(setId) {
    ansFor = null;
    openBrowser(setName(setId), 'Loading…');
    let j;
    try { j = await api('GET', '/api/evals/sets/' + encodeURIComponent(setId)); } catch (e) { $('bm-ans-sub').textContent = e.message; return; }
    const set = j.set || { questions: [] };
    $('bm-ans-sub').textContent = `${set.questions.length} questions · checked by ${set.check || 'exact match'}${set.generated ? ' · written by code from a fixed seed' : ''}${set.description ? ' · ' + set.description : ''}`;
    ans = { set, byId: Object.fromEntries(set.questions.map((qq) => [qq.id, qq])), items: set.questions.map((qq) => ({ id: qq.id, only: true })), setId };
    browserMode();
    selectTask(set.questions[0]?.id);
  }
  function browserMode() {
    const run = !!ansFor;
    $('bm-ans-statbox').hidden = !run;
    ansDlg.querySelector('.bm-ans-filter').hidden = !run;
    $('bm-ans-delete').hidden = $('bm-ans-rerun').hidden = !run;
    $('bm-ans-toml').hidden = run;
    $('bm-ans-toml').textContent = ans.set.builtIn ? 'View as TOML' : 'Edit as TOML';
  }
  function visibleTasks() {
    const q = $('bm-ans-search').value.trim().toLowerCase();
    return ans.items.filter((a) => {
      if (show === 'wrong' && a.correct) return false;
      if (show === 'right' && !a.correct) return false;
      if (!q) return true;
      const qq = ans.byId[a.id] || {};
      return `${a.id} ${qq.prompt || ''} ${a.got || ''}`.toLowerCase().includes(q);
    });
  }
  function verdictOf(a) {
    if (a.only) return { mark: '•', cls: 'only', word: '' };
    if (a.error) return { mark: '!', cls: 'wrong', word: 'Failed' };
    if (a.pending) return { mark: '…', cls: 'pending', word: 'Waiting for a judge' };
    return a.correct ? { mark: '✓', cls: 'right', word: 'Right' } : { mark: '✗', cls: 'wrong', word: 'Wrong' };
  }
  function renderTaskList() {
    const list = $('bm-ans-list');
    list.innerHTML = '';
    ansDlg.querySelectorAll('[data-show]').forEach((b) => b.classList.toggle('on', b.dataset.show === show));
    const vis = visibleTasks();
    for (const a of vis) {
      const qq = ans.byId[a.id] || {};
      const v = verdictOf(a);
      const b = el('button', `bm-tk ${v.cls}${a.id === ansSel ? ' sel' : ''}`);
      b.type = 'button';
      b.setAttribute('role', 'option');
      b.setAttribute('aria-selected', a.id === ansSel ? 'true' : 'false');
      b.dataset.id = a.id;
      const top = el('span', 'bm-tk-top');
      top.append(el('span', 'bm-tk-mark', v.mark), el('span', 'bm-tk-id', a.id));
      const sub = a.only
        ? (qq.prompt || '').replace(/\s+/g, ' ').slice(0, 90)
        : [a.got ? `got ${a.got}` : a.reply ? '' : 'no answer', `${a.tokens.toLocaleString()} tok`, dur(a.ms), a.looped ? 'looped' : a.cut ? 'ran out' : ''].filter(Boolean).join(' · ');
      b.append(top, el('span', 'bm-tk-sub', sub));
      b.onclick = () => selectTask(a.id);
      list.append(b);
    }
    if (!vis.length) list.append(el('div', 'muted small bm-tk-none', 'Nothing matches.'));
  }
  function selectTask(id) {
    if (!ans) return;
    ansSel = id || '';
    renderTaskList();
    $('bm-ans-list').querySelector('.bm-tk.sel')?.scrollIntoView({ block: 'nearest' });
    renderTask();
  }
  function moveTask(d) {
    const vis = visibleTasks();
    const i = vis.findIndex((a) => a.id === ansSel);
    const next = vis[Math.max(0, Math.min(vis.length - 1, i + d))];
    if (next) selectTask(next.id);
  }
  function section(title, ...kids) {
    const s = el('section', 'bm-tsec');
    s.append(el('h3', 'bm-tsec-h', title), ...kids);
    return s;
  }
  function renderTask() {
    const box = $('bm-ans-detail');
    box.innerHTML = '';
    box.scrollTop = 0;
    const a = ans.items.find((x) => x.id === ansSel);
    if (!a) return;
    const qq = ans.byId[a.id] || {};
    const chk = qq.check || ans.set.check;
    const v = verdictOf(a);
    const head = el('div', 'bm-task-head');
    head.append(el('h3', 'bm-task-id', a.id));
    if (v.word) head.append(el('span', `bm-pill ${v.cls}`, v.word));
    box.append(head);
    const facts = [];
    if (qq.answer && !['regex', 'judge', 'code', 'lean'].includes(chk)) {
      let want = qq.answer;
      if (chk === 'number' && /^-?[\d.]+(e-?\d+)?$/i.test(want) && want.replace(/[^\d]/g, '').length > 6) want = String(Number(Number(want).toPrecision(6)));
      if (chk === 'number' && qq.tolerance && qq.tolerance !== 0.5) want += ` ± ${Number(qq.tolerance.toPrecision(2))}`;
      facts.push(['Expected', want]);
    }
    if (!a.only) {
      if (a.got) facts.push(['Got', a.got]);
      facts.push(['Tokens', a.tokens.toLocaleString()], ['Time', dur(a.ms)]);
      if (a.ms && a.tokens) facts.push(['Speed', `${num(a.tokens / (a.ms / 1000))} tok/s`]);
    }
    if (facts.length) {
      const f = el('div', 'bm-facts');
      for (const [k, val] of facts) {
        const t = el('div', 'bm-fact');
        t.append(el('span', 'bm-fact-k', k), el('span', 'bm-fact-v', val));
        f.append(t);
      }
      box.append(f);
    }
    if (a.looped) box.append(el('div', 'msg-warn', 'Stopped early: it kept writing the same thing over and over.'));
    else if (a.cut) box.append(el('div', 'msg-warn', `Ran out of room: it hit the ${ans.res ? ans.res.maxTokens.toLocaleString() + '-token' : ''} limit before finishing.`));
    if (a.error) box.append(el('div', 'msg-warn', 'Error: ' + a.error));
    const q = section('Question', md(qq.prompt || '(question not found)'));
    if (qq.choices) {
      const ol = el('ol', 'bm-choices');
      qq.choices.forEach((c, i) => {
        const li = el('li', String.fromCharCode(65 + i) === (qq.answer || '').toUpperCase() ? 'key' : '');
        li.append(el('span', 'bm-choice-l', String.fromCharCode(65 + i)), el('span', null, c));
        ol.append(li);
      });
      q.append(ol);
    }
    if (chk === 'regex') q.append(el('div', 'muted small', `${qq.not ? 'Must not match' : 'Must match'}: ${qq.answer}`));
    box.append(q);
    if (chk === 'lean') box.append(section('Must prove', md('```lean\ntheorem goal : ' + qq.statement + (qq.preamble ? '\n\n-- given:\n' + qq.preamble : '') + '\n```')));
    if (chk === 'code' && qq.tests) {
      const d = el('details', 'bm-tdet');
      d.append(el('summary', null, 'Hidden tests'), md('```python\n' + qq.tests + '\n```'));
      box.append(d);
    }
    if (chk === 'judge' && qq.answer) {
      const d = el('details', 'bm-tdet');
      d.append(el('summary', null, 'Reference solution'), md(qq.answer));
      box.append(d);
    }
    if (a.only) return;
    if (a.thinking) {
      const d = el('details', 'bm-tdet');
      const words = a.thinking.split(/\s+/).length;
      const team = ansFor && ansFor.row && ansFor.row.team;
      d.append(el('summary', null, `${team ? "The team's work: plan and every worker's reply" : 'Thinking'} · about ${words.toLocaleString()} words`));
      d.addEventListener('toggle', () => { if (d.open && d.children.length === 1) d.append(md(a.thinking)); }, { once: false });
      if (team) d.open = true;
      box.append(d);
    }
    box.append(section('Answer', a.reply ? md(a.reply) : el('p', 'muted', '(no answer written)')));
    if (a.output) {
      const d = el('details', 'bm-tdet');
      d.open = !a.correct;
      d.append(el('summary', null, chk === 'lean' ? "Lean's verdict" : 'Test output'), el('pre', 'bm-pre bm-out', a.output));
      box.append(d);
    }
    if (a.judge) {
      const d = el('details', 'bm-tdet');
      d.open = !a.correct;
      d.append(el('summary', null, `Judged by ${a.judgeModel}${a.judgeModel === ans.res.model ? ' (itself)' : ''}`), md(a.judge));
      box.append(d);
    }
    const nav = el('div', 'bm-task-nav');
    const prev = el('button', 'btn ghost', '← Previous'), next = el('button', 'btn ghost', 'Next →');
    prev.type = next.type = 'button';
    prev.onclick = () => moveTask(-1);
    next.onclick = () => moveTask(1);
    nav.append(prev, next);
    box.append(nav);
  }
  ansDlg.querySelectorAll('[data-show]').forEach((b) => { b.onclick = () => { show = b.dataset.show; const vis = visibleTasks(); if (!vis.some((a) => a.id === ansSel)) selectTask(vis[0]?.id); else renderTaskList(); }; });
  $('bm-ans-search').addEventListener('input', () => { const vis = visibleTasks(); if (vis.length && !vis.some((a) => a.id === ansSel)) selectTask(vis[0].id); else renderTaskList(); });
  ansDlg.addEventListener('keydown', (e) => {
    if (e.target.closest('input, textarea, select') || e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === 'ArrowDown' || e.key === 'j') { moveTask(1); e.preventDefault(); }
    else if (e.key === 'ArrowUp' || e.key === 'k') { moveTask(-1); e.preventDefault(); }
  });
  $('bm-ans-toml').onclick = () => { const id = ans?.setId; if (!id) return; ansDlg.close(); openEditor(id, false); };
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

  // ---- Lean ----
  const leanReady = () => !!(leanSt.installed || leanSt.onPath);
  const leanBoxes = new Set();
  function leanBox() {
    const box = el('div', 'bm-lean');
    leanBoxes.add(box);
    paintLean(box);
    return box;
  }
  function paintLean(box) {
    box.innerHTML = '';
    if (leanReady()) {
      box.append(el('span', 'muted small', `Lean ${leanSt.installed || ''} is ready.`));
      return;
    }
    if (leanSt.busy) {
      const p = leanSt.progress || {};
      const mb = (n) => Math.round(n / 1e6);
      box.append(el('span', 'lib-spin'), el('span', 'small', p.phase === 'Downloading' && p.total ? `Downloading Lean ${p.version || ''}: ${mb(p.done)} of ${mb(p.total)} MB` : p.phase === 'Unpacking' && p.total ? `Unpacking: ${Math.round((p.done / p.total) * 100)}%` : (p.phase || 'Starting') + '…'));
      const c = el('button', 'lib-linkbtn small', 'Cancel');
      c.type = 'button';
      c.onclick = () => api('DELETE', '/api/lean/install').then((st) => { leanSt = st; paintAllLean(); });
      box.append(c);
      return;
    }
    const b = el('button', 'btn', 'Install Lean');
    b.type = 'button';
    b.onclick = async () => {
      b.disabled = true;
      try { leanSt = await api('POST', '/api/lean/install'); } catch (e) { leanSt.error = e.message; }
      paintAllLean();
      pollLean();
    };
    box.append(b, el('span', 'muted small', `These proofs are checked by the Lean prover, which isn't installed. About 0.9 GB to download, 3.2 GB on disk, in Assay's data folder.${leanSt.error ? ' Last try: ' + leanSt.error : ''}`));
  }
  function paintAllLean() {
    for (const b of leanBoxes) {
      if (!b.isConnected) leanBoxes.delete(b);
      else paintLean(b);
    }
    if (!runDlg.open) return;
    updateRunNote();
  }
  let leanTimer = 0;
  async function pollLean() {
    clearTimeout(leanTimer);
    try { leanSt = await api('GET', '/api/lean'); } catch {}
    paintAllLean();
    if (leanSt.busy) leanTimer = setTimeout(pollLean, 1000);
    else if (leanReady()) renderSets();
  }
  if (leanSt.busy) pollLean();

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
  $('bm-chart-note').onclick = () => $('bm-chart-note').classList.toggle('open');
  root.querySelectorAll('[data-view]').forEach((b) => {
    b.onclick = () => { view = b.dataset.view; store.set('bm-view', view); render(); };
  });
  $('bm-pen-x').value = penalty;
  $('bm-pen-x').onchange = () => {
    const v = Number($('bm-pen-x').value);
    if (!(v >= 0) || $('bm-pen-x').value === '') { $('bm-pen-x').value = penalty; return; }
    penalty = v;
    store.set('bm-penalty-x', String(v));
    render();
  };
  root.querySelectorAll('[data-metric]').forEach((b) => {
    b.onclick = () => {
      metric = b.dataset.metric;
      store.set('bm-metric', metric);
      render();
    };
  });
  document.addEventListener('assay:remote', reload);
  if (suite) reload(); else render();
  renderRun();
  if (run.running) poll();
  document.addEventListener('visibilitychange', () => { if (!document.hidden && run.running) poll(); });
})();
