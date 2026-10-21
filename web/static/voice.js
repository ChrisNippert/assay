// Dictation and reading aloud on the chat page.
//
// The mic button dictates into the message box. The mic is read through an
// AudioWorklet and a small energy-based detector cuts speech into
// utterances at pauses: each is transcribed live while you talk and kept
// once you pause. Nothing is sent until you press Send, which ends the
// dictation first. Two toggles shape replies: "Spoken style" asks for plain
// prose meant to be heard, and "Read aloud" speaks each reply a sentence at
// a time as it streams in. Any reply can also be read from its action bar.
(() => {
  const chat = window.ollamuiChat;
  const btn = document.getElementById('mic-btn');
  if (!chat || !btn) return;
  const bar = document.getElementById('voice-bar');
  const stateEl = document.getElementById('voice-state');
  const hush = document.getElementById('voice-hush');
  const dl = document.getElementById('voice-download');
  const notNow = document.getElementById('voice-not-now');
  const spokenBtn = document.getElementById('voice-spoken');
  const readBtn = document.getElementById('voice-read');
  const input = chat.input;
  const form = input.form;

  const FRAME_MS = 20;
  const PREROLL = 15; // frames kept from before speech is detected (300 ms)
  const END_SILENCE = 40; // quiet frames that end an utterance (800 ms)
  const MIN_SPEECH = 12; // loud frames for an utterance to count (240 ms)
  const MAX_FRAMES = 1500; // 30 s
  const PARTIAL_MS = 900;
  // From the Voice settings' speech detection: frames of the last WINDOW
  // that must be voice, and a multiplier on the loudness threshold.
  const WINDOW = 15;
  const SENSITIVITY = { low: [13, 1.6], normal: [11, 1], high: [8, 0.7] };
  let START_VOICED = 11;
  let LOUDNESS = 1;
  // Silence between spoken clips (ms), by how the clip ended.
  let SENTENCE_PAUSE = 450; // from the Voice settings
  const CLAUSE_PAUSE = 200;
  const WORD_PAUSE = 120;

  let status = {};
  let active = false; // dictating
  let audio = null; // { ctx, stream, node }
  let speaker = null;
  let listenState = ''; // listening, hearing, transcribing

  // ---- the toggles, remembered in this browser ----
  const pressed = (b) => b.getAttribute('aria-pressed') === 'true';
  for (const [b, key] of [[spokenBtn, 'spoken'], [readBtn, 'read']]) {
    try { b.setAttribute('aria-pressed', String(localStorage.getItem('voice.' + key) === '1')); } catch {}
    b.addEventListener('click', () => {
      const on = !pressed(b);
      b.setAttribute('aria-pressed', String(on));
      try { localStorage.setItem('voice.' + key, on ? '1' : '0'); } catch {}
      if (b === readBtn && !on) speaker?.stop();
    });
  }

  // ---- status and the bar under the composer ----
  const ready = () => document.body.classList.contains('voice-ok');

  async function checkStatus(start) {
    try { status = await (await fetch('/api/voice' + (start ? '?start=1' : ''))).json(); } catch { status = { available: false, error: 'Couldn’t reach ollamui.' }; }
    // Stopped means "starts when you use it", which is fine.
    const usable = status.available || status.stopped || status.starting;
    btn.classList.toggle('unavailable', !usable);
    btn.title = usable ? 'Dictate' : status.error;
    document.body.classList.toggle('voice-ok', !!usable && !status.needsDownload);
    const st = status.settings || {};
    if (st.pauseMs >= 0) SENTENCE_PAUSE = st.pauseMs;
    [START_VOICED, LOUDNESS] = SENSITIVITY[st.sensitivity] || SENSITIVITY.normal;
    showBar();
  }
  checkStatus();

  // The bar holds the toggles once voice is set up, and shows what's going on.
  function showBar() {
    bar.hidden = !(ready() || active || bar.classList.contains('setup'));
  }

  function render(text) {
    const s = active ? listenState : speaker?.busy() ? 'speaking' : '';
    bar.dataset.state = s;
    stateEl.classList.remove('error');
    stateEl.textContent = text ?? {
      listening: 'Listening · press the mic to stop',
      hearing: 'Hearing you…',
      transcribing: 'Writing it down…',
      speaking: 'Reading aloud',
    }[s] ?? '';
    hush.hidden = !speaker?.busy();
  }

  function fail(msg) {
    bar.hidden = false;
    bar.dataset.state = 'error';
    stateEl.classList.add('error');
    stateEl.textContent = msg;
  }

  // ensureService starts the speech service, first asking to download it
  // if need be. Voice is optional: its models download only after a yes.
  async function ensureService() {
    await checkStatus(true);
    if (status.needsDownload) {
      bar.classList.add('setup');
      showBar();
      render(`Voice needs a one-time download (${status.downloadSize}).`);
      stateEl.title = 'Speech recognition and a voice, kept in your cache folder. Remove them any time in Settings → Voice.';
      dl.hidden = notNow.hidden = false;
      const ok = await new Promise((resolve) => {
        dl.onclick = () => resolve(true);
        notNow.onclick = () => resolve(false);
      });
      dl.hidden = notNow.hidden = true;
      stateEl.title = '';
      if (!ok) { bar.classList.remove('setup'); return false; }
      render('Downloading…');
      try { status = await (await fetch('/api/voice?install=1&start=1')).json(); } catch {}
    }
    // The first start installs the speech packages, which takes a minute.
    for (let i = 0; status.starting && i < 480; i++) {
      render(i < 10 ? 'Starting speech…' : 'Downloading voice… this takes a minute or two the first time');
      await new Promise((r) => setTimeout(r, 500));
      await checkStatus(true);
    }
    bar.classList.remove('setup');
    if (!status.available) { fail(status.error || 'The speech service didn’t start.'); return false; }
    return true;
  }

  // ---- dictation on and off ----
  btn.addEventListener('click', () => (active ? stop() : start()));
  hush.addEventListener('click', () => { speaker?.stop(); render(); });

  async function start() {
    if (!navigator.mediaDevices?.getUserMedia) {
      fail('The microphone only works over HTTPS or on localhost. From another device, open ollamui through “tailscale serve” or another HTTPS proxy.');
      return;
    }
    active = true;
    btn.setAttribute('aria-pressed', 'true');
    listenState = '';
    showBar();
    render('Starting…');
    if (!(await ensureService())) {
      active = false;
      btn.setAttribute('aria-pressed', 'false');
      if (bar.dataset.state !== 'error') { render(); showBar(); }
      return;
    }
    if (!active) return; // pressed again while starting
    try {
      audio = await openMic();
    } catch (err) {
      stop();
      fail(`Couldn’t open the microphone: ${err.message}`);
      return;
    }
    if (!active) { closeMic(); return; }
    listenState = 'listening';
    render();
  }

  // stop ends dictation, keeping what was said before the press.
  async function stop() {
    active = false;
    btn.setAttribute('aria-pressed', 'false');
    const u = utt;
    utt = null;
    closeMic();
    if (u) endUtterance(u);
    if (writing) render('Writing it down…');
    await queue;
    render();
    showBar();
  }

  // ---- microphone ----
  const WORKLET = `
    class Tap extends AudioWorkletProcessor {
      constructor() { super(); this.buf = new Float32Array(Math.round(sampleRate * ${FRAME_MS} / 1000)); this.n = 0; }
      process(inputs) {
        const ch = inputs[0][0];
        if (ch) for (let i = 0; i < ch.length; i++) {
          this.buf[this.n++] = ch[i];
          if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice()); this.n = 0; }
        }
        return true;
      }
    }
    registerProcessor('tap', Tap);`;

  async function openMic() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    const ctx = new AudioContext();
    await ctx.resume();
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' })));
    const node = new AudioWorkletNode(ctx, 'tap');
    const mute = ctx.createGain();
    mute.gain.value = 0;
    ctx.createMediaStreamSource(stream).connect(node);
    node.connect(mute).connect(ctx.destination); // keeps the worklet running everywhere
    node.port.onmessage = (e) => onFrame(e.data);
    return { ctx, stream, node };
  }

  function closeMic() {
    if (!audio) return;
    audio.stream.getTracks().forEach((t) => t.stop());
    audio.ctx.close();
    audio = null;
  }

  // ---- speech detection ----
  // Loudness alone can't tell speech from typing, so a frame only counts as
  // voice when it's loud without being a click (a click is a sharp peak in
  // an otherwise quiet frame), and an utterance only starts when most of
  // the last 300 ms was voice. Typing is short bursts with gaps; speech is
  // sustained.
  const MAX_CREST = 7; // peak / RMS; voiced speech is ~3–5, clicks are far higher
  let floor = 0.002; // running estimate of background noise (RMS)
  let recent = []; // voice flags for the last WINDOW frames
  let preroll = [];
  let utt = null; // { frames, rate, voiced, quiet, partialAt, partialBusy, base }

  function onFrame(f) {
    if (!active) return;
    let sum = 0, peak = 0;
    for (let i = 0; i < f.length; i++) {
      sum += f[i] * f[i];
      const a = Math.abs(f[i]);
      if (a > peak) peak = a;
    }
    const rms = Math.sqrt(sum / f.length);
    // A reply being read aloud can leak into the mic, so it takes more then.
    const threshold = Math.max(floor * 3, 0.006) * LOUDNESS * (speaker?.busy() ? 2.5 : 1);
    const loud = rms > threshold;
    const voiced = loud && peak / rms < MAX_CREST;

    if (!utt) {
      if (!loud) floor = floor * 0.97 + rms * 0.03;
      preroll.push(f);
      if (preroll.length > PREROLL) preroll.shift();
      recent.push(voiced);
      if (recent.length > WINDOW) recent.shift();
      if (voiced && recent.filter(Boolean).length >= START_VOICED) beginUtterance();
      return;
    }
    utt.frames.push(f);
    if (voiced) utt.voiced++;
    utt.quiet = loud ? 0 : utt.quiet + 1;
    if (utt.quiet >= END_SILENCE || utt.frames.length >= MAX_FRAMES) {
      const u = utt;
      utt = null;
      endUtterance(u);
    } else if (performance.now() - utt.partialAt > PARTIAL_MS && !utt.partialBusy && utt.voiced >= MIN_SPEECH) partial(utt);
  }

  function beginUtterance() {
    utt = { frames: preroll, rate: audio.ctx.sampleRate, voiced: recent.filter(Boolean).length, quiet: 0, partialAt: performance.now(), partialBusy: false, base: null };
    preroll = [];
    recent = [];
    listenState = 'hearing';
    render();
  }

  // Utterances are written down in order: each one's text goes after
  // whatever is in the box once the one before it is done.
  let queue = Promise.resolve();
  let writing = 0;

  // Live text while you talk: what's been said so far, after the box's text.
  async function partial(u) {
    if (writing) return; // the last utterance isn't in the box yet
    u.partialBusy = true;
    u.partialAt = performance.now();
    const text = await transcribe(u.frames, u.rate);
    u.partialBusy = false;
    if (utt !== u || writing || !text) return;
    u.base ??= input.value;
    put(u.base, text);
  }

  function endUtterance(u) {
    if (u.voiced < MIN_SPEECH) { // a cough or a click
      if (u.base != null) put(u.base, '');
      listenState = 'listening';
      render();
      return;
    }
    writing++;
    if (active && !utt) { listenState = 'transcribing'; render(); }
    queue = queue.then(async () => {
      const text = await transcribe(u.frames, u.rate);
      u.base ??= input.value;
      put(u.base, isWords(text) ? text : '');
      writing--;
      if (active && !utt) { listenState = 'listening'; render(); }
    });
  }

  // Real words: not one of whisper's stock phrases for noise.
  function isWords(text) {
    return /[\p{L}\p{N}]/u.test(text) && !/^(thank you|thanks for watching|you|bye)[.!]?$/i.test(text.trim());
  }

  function put(base, text) {
    input.value = !text ? base : !base || /\s$/.test(base) ? base + text : base + ' ' + text;
    input.dispatchEvent(new Event('input'));
  }

  async function transcribe(frames, rate) {
    try {
      const res = await fetch('/api/voice/transcribe', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wav(frames, rate) });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
      return (await res.json()).text.trim();
    } catch (err) {
      fail(`Couldn’t transcribe: ${err.message}`);
      return '';
    }
  }

  // 16 kHz mono 16-bit WAV, which every speech service reads.
  function wav(frames, rate) {
    const n = frames.reduce((a, f) => a + f.length, 0);
    const all = new Float32Array(n);
    let o = 0;
    for (const f of frames) { all.set(f, o); o += f.length; }
    const ratio = rate / 16000;
    const len = Math.floor(n / ratio);
    const buf = new ArrayBuffer(44 + len * 2);
    const v = new DataView(buf);
    const str = (at, s) => { for (let i = 0; i < s.length; i++) v.setUint8(at + i, s.charCodeAt(i)); };
    str(0, 'RIFF'); v.setUint32(4, 36 + len * 2, true); str(8, 'WAVE');
    str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, 'data'); v.setUint32(40, len * 2, true);
    for (let i = 0; i < len; i++) {
      // Average the samples that fall in this output sample.
      const a = Math.floor(i * ratio), b = Math.max(a + 1, Math.floor((i + 1) * ratio));
      let s = 0;
      for (let j = a; j < b; j++) s += all[j];
      s = Math.max(-1, Math.min(1, s / (b - a)));
      v.setInt16(44 + i * 2, s * 32767, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
  }

  // Sending ends the dictation, once what was said is in the box.
  form.addEventListener('submit', async (e) => {
    if ((!active && !writing) || chat.busy()) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    await stop();
    form.requestSubmit();
  }, true);

  // ---- reading aloud ----
  // The speaker gets its own audio context, made while handling a click or
  // a key so the browser lets it play.
  function getSpeaker() {
    speaker ??= new Speaker(new AudioContext());
    speaker.ctx.resume();
    return speaker;
  }

  function read(markdown) {
    const sp = getSpeaker();
    sp.stop();
    const s = new Sentences((t) => sp.say(t));
    s.feed(markdown);
    s.flush();
  }

  // chat.js asks this as it sends a message.
  window.ollamuiVoice = {
    turn() {
      const shown = !bar.hidden && ready();
      speaker?.stop();
      const out = { spoken: shown && pressed(spokenBtn) };
      if (!shown || !pressed(readBtn)) return out;
      const sp = getSpeaker();
      const sentences = new Sentences((t) => sp.say(t));
      out.onEvent = (ev) => {
        if (ev.type === 'content') sentences.feed(ev.text);
        if (ev.type === 'done') sentences.flush();
      };
      out.end = () => sentences.flush();
      return out;
    },
    read,
  };

  // Sentences cuts streamed markdown into speakable sentences. Code blocks
  // are left for the screen.
  class Sentences {
    constructor(out) { this.out = out; this.full = ''; this.done = 0; }
    feed(t) { this.full += t; this.emit(false); }
    flush() { this.emit(true); }
    emit(final) {
      let text = this.full;
      // Hold back an unfinished code block; say a finished one in passing.
      const fences = text.split(/^\s*```.*$/m);
      if (fences.length % 2 === 0 && !final) fences.pop();
      text = fences.map((part, i) => (i % 2 ? ' (the code is on screen.) ' : part)).join('');
      let rest = text.slice(this.done);
      for (;;) {
        const m = rest.match(/^[\s\S]{12,}?(?:[.!?…]+["'”’)\]]*(?=\s)|\n)/);
        if (!m) break;
        this.say(m[0]);
        this.done += m[0].length;
        rest = rest.slice(m[0].length);
      }
      if (final && rest.trim()) { this.say(rest); this.done += rest.length; }
    }
    say(s) {
      s = speakable(s);
      if (s) this.out(s);
    }
  }

  function speakable(s) {
    return s
      .replace(/\$\$?([^$]+)\$\$?/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/https?:\/\/\S+/g, 'the link')
      .replace(/^\s*(#{1,6}|[-*+]|\d+[.)])\s+/gm, '')
      .replace(/[*_`#|>~]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  // Speaker plays lines in order. Each is synthesized as soon as it arrives,
  // so the next one is ready when the current one ends.
  class Speaker {
    constructor(ctx) {
      this.ctx = ctx; this.queue = []; this.source = null; this.gen = 0; this.running = false;
      this.resumeAt = 0; // when the pause after the last clip ends
    }
    say(text) {
      const gen = this.gen;
      const buf = fetch('/api/voice/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) })
        .then((r) => (r.ok ? r.arrayBuffer() : null))
        .then((b) => (b && gen === this.gen ? this.ctx.decodeAudioData(b) : null))
        .catch(() => null);
      // Each sentence is its own clip, and Kokoro trims the silence off its
      // ends, so the pause a full stop deserves is added between clips.
      const pause = /[.!?…]["'”’)\]]*$/.test(text) ? SENTENCE_PAUSE : /[,;:]$/.test(text) ? CLAUSE_PAUSE : WORD_PAUSE;
      this.queue.push({ buf, pause });
      if (!this.running) this.pump(gen);
    }
    async pump(gen) {
      this.running = true;
      render();
      while (this.queue.length && gen === this.gen) {
        const { buf: next, pause } = this.queue.shift();
        const buf = await next;
        if (!buf || gen !== this.gen) continue;
        // Leave the previous clip's pause, however soon this one was ready.
        const wait = this.resumeAt - performance.now();
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        if (gen !== this.gen) continue;
        await new Promise((done) => {
          const src = this.ctx.createBufferSource();
          src.buffer = buf;
          src.connect(this.ctx.destination);
          src.onended = done;
          this.source = src;
          src.start();
        });
        this.source = null;
        this.resumeAt = performance.now() + pause;
      }
      if (gen !== this.gen) return;
      this.running = false;
      render();
    }
    busy() { return this.running; }
    stop() {
      this.gen++;
      this.queue = [];
      try { this.source?.stop(); } catch {}
      this.source = null;
      this.running = false;
      render();
    }
  }

  addEventListener('pagehide', () => { active = false; utt = null; closeMic(); speaker?.stop(); });
})();
