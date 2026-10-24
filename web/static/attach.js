// File attachments in the composer: paperclip, drag and drop, paste.
// Files upload as soon as they are added; chat.js takes them when sending.
(() => {
  const form = document.getElementById('composer');
  const chatEl = document.getElementById('chat');
  if (!form || !chatEl) return;
  const { el } = window.ollamui;
  const row = document.getElementById('att-row');
  const btn = document.getElementById('attach-btn');
  const picker = document.getElementById('attach-input');
  const input = document.getElementById('input');
  const select = document.getElementById('model');
  const MAX = 25 << 20;
  const IMG = /^(png|jpe?g|webp|gif)$/i;

  const size = (n) => n >= 1 << 20 ? (n / (1 << 20)).toFixed(1) + ' MB' : n >= 1024 ? Math.round(n / 1024) + ' KB' : n + ' B';
  const kindOf = (name) => IMG.test(name.split('.').pop()) ? 'image' : 'doc';

  // Whether the picked model can see images, asked once per model.
  const visionCache = new Map();
  function modelName() {
    const o = select?.selectedOptions[0];
    return o?.dataset.base || select?.value || '';
  }
  async function hasVision() {
    const m = modelName();
    if (!m) return true;
    if (!visionCache.has(m)) {
      visionCache.set(m, fetch('/api/vision?model=' + encodeURIComponent(m)).then((r) => r.json()).then((j) => !!j.vision).catch(() => true));
    }
    return visionCache.get(m);
  }

  let items = []; // {name, kind, size, file, url, state: 'up'|'ok'|'err', error}

  const icon = (kind) => kind === 'image'
    ? '<path d="M4 5h16v14H4z"/><circle cx="9" cy="10" r="1.6"/><path d="M4 17l5-4 4 3 3-2 4 3"/>'
    : '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M10 13h6M10 17h6"/>';
  const svg = (inner, n = 18) => `<svg aria-hidden="true" width="${n}" height="${n}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${inner}</svg>`;

  async function paint() {
    row.hidden = items.length === 0;
    row.replaceChildren();
    const vision = await hasVision();
    for (const it of items) {
      const chip = el('div', 'att-chip pending');
      if (it.state === 'err') chip.classList.add('bad');
      else if (it.kind === 'image' && !vision) chip.classList.add('warn');
      if (it.kind === 'image' && it.url) {
        const img = el('img', 'att-thumb');
        img.src = it.url;
        img.alt = '';
        chip.append(img);
      } else {
        const ic = el('span', 'att-icon');
        ic.innerHTML = svg(icon(it.kind));
        chip.append(ic);
      }
      const txt = el('span', 'att-text');
      txt.append(el('span', 'att-name', it.name));
      let sub = size(it.size);
      if (it.state === 'up') sub = 'Uploading…';
      else if (it.state === 'err') sub = it.error;
      else if (it.kind === 'image' && !vision) sub = 'This model can’t see images';
      txt.append(el('span', 'att-sub', sub));
      chip.append(txt);
      const x = el('button', 'att-x');
      x.type = 'button';
      x.title = x.ariaLabel = 'Remove ' + it.name;
      x.innerHTML = svg('<path d="M6 6l12 12M18 6L6 18"/>', 14);
      x.addEventListener('click', () => { items = items.filter((i) => i !== it); paint(); });
      chip.append(x);
      row.append(chip);
    }
  }

  async function upload(it, file) {
    try {
      const fd = new FormData();
      fd.append('file', file, it.name);
      const res = await fetch('/api/attachments', { method: 'POST', body: fd });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || 'Upload failed');
      const out = await res.json();
      if (out.files?.length) Object.assign(it, out.files[0], { state: 'ok' });
      else Object.assign(it, { state: 'err', error: (out.problems?.[0] || 'Could not add it').replace(/^.*?: /, '') });
    } catch (err) {
      Object.assign(it, { state: 'err', error: err.message });
    }
    paint();
  }

  function add(files) {
    for (const f of files) {
      const it = { name: f.name || 'file', kind: kindOf(f.name || ''), size: f.size, state: 'up' };
      if (it.kind === 'image') it.url = URL.createObjectURL(f);
      if (f.size > MAX) { it.state = 'err'; it.error = 'Larger than 25 MB'; }
      items.push(it);
      if (it.state === 'up') upload(it, f);
    }
    paint();
  }

  btn.addEventListener('click', () => picker.click());
  picker.addEventListener('change', () => { add([...picker.files]); picker.value = ''; });
  select?.addEventListener('change', paint);

  // Paste: images from the clipboard (and files copied in a file manager).
  input.addEventListener('paste', (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (!files.length) return;
    e.preventDefault();
    add(files.map((f) => f.name && f.name !== 'image.png' ? f : new File([f], `pasted-${new Date().toISOString().replace(/\D/g, '').slice(0, 14)}.${(f.type.split('/')[1] || 'png').replace('jpeg', 'jpg')}`, { type: f.type })));
  });

  // Drag and drop anywhere on the chat.
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  chatEl.addEventListener('dragenter', (e) => { if (hasFiles(e)) { depth++; chatEl.classList.add('dropping'); } });
  chatEl.addEventListener('dragleave', (e) => { if (hasFiles(e) && --depth <= 0) { depth = 0; chatEl.classList.remove('dropping'); } });
  chatEl.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  chatEl.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    chatEl.classList.remove('dropping');
    add([...e.dataTransfer.files]);
  });

  // The attachment elements of a sent message. Images are thumbnails that
  // open full size; documents are chips with name and size.
  function render(atts, chatId) {
    const box = el('div', 'msg-atts');
    for (const a of atts) {
      const href = a.url || `/api/chats/${chatId}/files/${a.file}`;
      if (a.kind === 'image') {
        const link = el('a', 'att-img');
        link.href = href;
        link.target = '_blank';
        link.rel = 'noopener';
        link.title = a.name;
        const img = el('img');
        img.src = href;
        img.alt = a.name;
        img.loading = 'lazy';
        link.append(img);
        box.append(link);
      } else {
        const link = el('a', 'att-chip');
        link.href = href;
        link.download = a.name;
        const ic = el('span', 'att-icon');
        ic.innerHTML = svg(icon('doc'));
        const txt = el('span', 'att-text');
        txt.append(el('span', 'att-name', a.name), el('span', 'att-sub', size(a.size) + (a.cut ? ' · shortened' : '')));
        link.append(ic, txt);
        box.append(link);
      }
    }
    return box;
  }

  window.ollamuiAttach = {
    // busy: a file is still uploading.
    busy: () => items.some((i) => i.state === 'up'),
    count: () => items.filter((i) => i.state === 'ok').length,
    // take hands over the uploaded files and empties the row.
    take() {
      const ok = items.filter((i) => i.state === 'ok');
      items = [];
      paint();
      return ok;
    },
    render,
    // blockedImage: the first attached image, if the model can't see images.
    async blockedImage() {
      const img = items.find((i) => i.state === 'ok' && i.kind === 'image');
      return img && !(await hasVision()) ? img : null;
    },
  };
})();
