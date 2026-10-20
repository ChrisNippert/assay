// Voice mode: hands-free conversation on the chat page.
//
// The mic is read through an AudioWorklet. A small energy-based detector
// decides when you start and stop talking; while you talk, the recording so
// far is transcribed every so often for live text, and once you stop the
// whole utterance is transcribed and sent as a voice turn. The reply is
// spoken a sentence at a time as it streams in, along with any "say" lines
// the server's talker produces while the model thinks. Talking over the
// reply interrupts it.
(() => {
  const chat = window.ollamuiChat;
  const btn = document.getElementById('mic-btn');
  if (!chat || !btn) return;
  const bar = document.getElementById('voice-bar');
  const stateEl = document.getElementById('voice-state');
  const hush = document.getElementById('voice-hush');
  const input = chat.input;

  const FRAME_MS = 20;
  const PREROLL = 15; // frames kept from before speech is detected (300 ms)
  const END_SILENCE = 40; // quiet frames that end an utterance (800 ms)
  const MIN_SPEECH = 12; // loud frames for an utterance to count (240 ms)
  const MAX_FRAMES = 1500; // 30 s
  const PARTIAL_MS = 900;
  // Silence between spoken clips (ms), by how the clip ended.
  let SENTENCE_PAUSE = 450; // from the Voice settings
  const CLAUSE_PAUSE = 200;
  const WORD_PAUSE = 120;

  let status = null;
  let active = false;
  let audio = null; // { ctx, stream, node }
  let speaker = null;
  let state = '';

  // With a model, this also has the server load the talker ahead of time
  // and start the speech service if it isn't running.
  async function checkStatus(model) {
    const q = model ? `?start=1&model=${encodeURIComponent(model)}` : '';
    try { status = await (await fetch('/api/voice' + q)).json(); } catch { status = { available: false, error: 'Couldn’t reach ollamui.' }; }
    // Stopped means "starts when you press it", which is fine.
    const usable = status.available || status.stopped || status.starting;
    btn.classList.toggle('unavailable', !usable);
    btn.title = usable ? 'Talk (voice mode)' : status.error;
  }
  checkStatus();

  function setState(s, text) {
    state = s;
    bar.dataset.state = s;
    stateEl.classList.toggle('error', s === 'error');
    stateEl.textContent = text ?? {
      listening: 'Listening…',
      hearing: 'Hearing you…',
      transcribing: 'Got it…',
      thinking: 'Thinking…',
      speaking: 'Speaking · talk to interrupt',
    }[s] ?? '';
    hush.hidden = s !== 'speaking';
  }

  // ---- turning voice mode on and off ----
  btn.addEventListener('click', () => (active ? stop() : start()));
  document.getElementById('voice-end').addEventListener('click', stop);
  hush.addEventListener('click', () => { speaker?.stop(); chat.stop(); });

  async function start() {
    bar.hidden = false;
    if (!navigator.mediaDevices?.getUserMedia) {
      setState('error', 'The microphone only works over HTTPS or on localhost. From another device, open ollamui through “tailscale serve” or another HTTPS proxy.');
      return;
    }
    await checkStatus(chat.model());
    // The first start installs the speech packages, which takes a minute.
    for (let i = 0; status.starting && i < 240 && !bar.hidden; i++) {
      setState('', i < 10 ? 'Starting speech…' : 'Starting speech… the first time downloads about 400 MB');
      await new Promise((r) => setTimeout(r, 500));
      await checkStatus(chat.model());
    }
    if (bar.hidden) return; // ended while starting
    if (!status.available) { setState('error', status.error || 'The speech service didn’t start.'); return; }
    const st = status.settings || {};
    if (st.pauseMs >= 0) SENTENCE_PAUSE = st.pauseMs;
    [START_VOICED, LOUDNESS] = SENSITIVITY[st.sensitivity] || SENSITIVITY.normal;
    if (!chat.ready()) { setState('error', 'Choose a model first.'); return; }
    active = true;
    btn.setAttribute('aria-pressed', 'true');
    setState('', 'Starting the microphone…');
    try {
      audio = await openMic();
    } catch (err) {
      stop();
      bar.hidden = false;
      setState('error', `Couldn’t open the microphone: ${err.message}`);
      return;
    }
    speaker = new Speaker(audio.ctx);
    listen();
  }

  function stop() {
    active = false;
    btn.setAttribute('aria-pressed', 'false');
    speaker?.stop();
    if (chat.busy()) chat.stop();
    if (audio) {
      audio.stream.getTracks().forEach((t) => t.stop());
      audio.ctx.close();
      audio = null;
    }
    utt = null;
    bar.hidden = true;
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

  // ---- speech detection ----
  // Loudness alone can't tell speech from typing, so a frame only counts as
  // voice when it's loud without being a click (a click is a sharp peak in
  // an otherwise quiet frame), and an utterance only starts when most of
  // the last 300 ms was voice. Typing is short bursts with gaps; speech is
  // sustained.
  const WINDOW = 15;
  // From the Voice settings' speech detection: frames of the last WINDOW
  // that must be voice, and a multiplier on the loudness threshold.
  const SENSITIVITY = { low: [13, 1.6], normal: [11, 1], high: [8, 0.7] };
  let START_VOICED = 11;
  let LOUDNESS = 1;
  const MAX_CREST = 7; // peak / RMS; voiced speech is ~3–5, clicks are far higher
  let floor = 0.002; // running estimate of background noise (RMS)
  let recent = []; // voice flags for the last WINDOW frames
  let preroll = [];
  let utt = null; // { frames, voiced, quiet, partialAt, partialBusy, interrupting }

  function onFrame(f) {
    if (!active) return;
    let sum = 0, peak = 0;
    for (let i = 0; i < f.length; i++) {
      sum += f[i] * f[i];
      const a = Math.abs(f[i]);
      if (a > peak) peak = a;
    }
    const rms = Math.sqrt(sum / f.length);
    const playing = speaker?.busy();
    const threshold = Math.max(floor * 3, 0.006) * LOUDNESS * (playing ? 2.5 : 1);
    const loud = rms > threshold;
    const voiced = loud && peak / rms < MAX_CREST;

    if (!utt) {
      if (!loud) floor = floor * 0.97 + rms * 0.03;
      preroll.push(f);
      if (preroll.length > PREROLL) preroll.shift();
      recent.push(voiced);
      if (recent.length > WINDOW) recent.shift();
      if (voiced && recent.filter(Boolean).length >= START_VOICED && state !== 'transcribing') beginUtterance();
      return;
    }
    utt.frames.push(f);
    if (voiced) utt.voiced++;
    utt.quiet = loud ? 0 : utt.quiet + 1;
    if (utt.quiet >= END_SILENCE || utt.frames.length >= MAX_FRAMES) endUtterance();
    else if (performance.now() - utt.partialAt > (utt.interrupting ? PARTIAL_MS / 2 : PARTIAL_MS) && !utt.partialBusy && utt.voiced >= MIN_SPEECH) partial();
  }

  // Talking over the reply only turns it down at first; it stops once the
  // transcript shows real words that aren't our own speech coming back.
  function beginUtterance() {
    const interrupting = speaker.busy() || chat.busy();
    if (interrupting) speaker.duck(true);
    utt = { frames: preroll, voiced: recent.filter(Boolean).length, quiet: 0, partialAt: performance.now(), partialBusy: false, interrupting };
    preroll = [];
    recent = [];
    setState('hearing');
  }

  function interrupt(u) {
    u.interrupting = false;
    turn++;
    speaker.stop();
    chat.stop();
  }

  // Real words: at least two, and not one of whisper's stock phrases for noise.
  function isWords(text) {
    const words = text.match(/[\p{L}\p{N}']+/gu) || [];
    return words.length >= 2 && !/^(thank you|thanks for watching|you|bye)[.!]?$/i.test(text.trim());
  }

  // Echo: mostly words the speaker said just now.
  function isEcho(text) {
    const words = (text.toLowerCase().match(/[\p{L}\p{N}']+/gu) || []);
    const said = new Set(speaker.recent.join(' ').toLowerCase().match(/[\p{L}\p{N}']+/gu) || []);
    return words.length > 0 && words.filter((w) => said.has(w)).length >= words.length * 0.6;
  }

  // Live text while you talk: transcribe what's been said so far.
  async function partial() {
    const u = utt;
    u.partialBusy = true;
    u.partialAt = performance.now();
    const text = await transcribe(u.frames);
    u.partialBusy = false;
    if (utt !== u || !text) return;
    if (u.interrupting) {
      if (!isWords(text) || isEcho(text)) return;
      interrupt(u);
    }
    input.value = text;
    input.dispatchEvent(new Event('input'));
  }

  async function endUtterance() {
    const u = utt;
    utt = null;
    if (u.voiced < MIN_SPEECH) { settle(u); return; } // a cough or a click
    setState('transcribing');
    const text = await transcribe(u.frames);
    if (!active) return;
    if (!text || !isWords(text) || (u.interrupting && isEcho(text))) {
      input.value = '';
      settle(u);
      return;
    }
    if (u.interrupting) interrupt(u);
    input.value = '';
    input.dispatchEvent(new Event('input'));
    reply(text);
  }

  // After a false alarm: back to whatever was going on.
  function settle(u) {
    if (u.interrupting) speaker.duck(false);
    if (!active || utt) return;
    setState(speaker.busy() ? 'speaking' : chat.busy() ? 'thinking' : 'listening');
  }

  function listen() {
    if (active && !utt) setState('listening');
  }

  async function transcribe(frames) {
    const rate = audio?.ctx.sampleRate;
    if (!rate) return '';
    try {
      const res = await fetch('/api/voice/transcribe', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wav(frames, rate) });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
      return (await res.json()).text.trim();
    } catch (err) {
      if (active) setState('error', `Couldn’t transcribe: ${err.message}`);
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

  // ---- the reply ----
  let turn = 0; // bumped by each reply and interruption, so stale ones bow out

  async function reply(text) {
    const mine = ++turn;
    setState('thinking');
    const sentences = new Sentences((s) => speaker.say(s));
    await chat.send(text, {
      onEvent(ev) {
        if (ev.type === 'say') { const s = new Sentences((t) => speaker.say(t)); s.feed(ev.text); s.flush(); }
        if (ev.type === 'content') sentences.feed(ev.text);
        if (ev.type === 'done') sentences.flush();
      },
    });
    sentences.flush();
    if (!active || mine !== turn) return;
    await speaker.idle();
    if (active && !utt && mine === turn) listen();
  }

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
      this.ctx = ctx; this.queue = []; this.source = null; this.gen = 0; this.running = false; this.waiters = [];
      this.recent = []; // the last few lines, to recognize our own voice in the mic
      this.resumeAt = 0; // when the pause after the last clip ends
      this.gain = ctx.createGain();
      this.gain.connect(ctx.destination);
    }
    duck(on) { this.gain.gain.setTargetAtTime(on ? 0.25 : 1, this.ctx.currentTime, 0.05); }
    say(text) {
      this.recent.push(text);
      if (this.recent.length > 4) this.recent.shift();
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
      if (state !== 'hearing') setState('speaking');
      while (this.queue.length && gen === this.gen) {
        const { buf: next, pause } = this.queue.shift();
        const buf = await next;
        if (!buf || gen !== this.gen) continue;
        // Leave the previous clip's pause, however soon this one was ready.
        const wait = this.resumeAt - performance.now();
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        if (gen !== this.gen) continue;
        if (state !== 'hearing') setState('speaking');
        await new Promise((done) => {
          const src = this.ctx.createBufferSource();
          src.buffer = buf;
          src.connect(this.gain);
          src.onended = done;
          this.source = src;
          src.start();
        });
        this.source = null;
        this.resumeAt = performance.now() + pause;
      }
      if (gen !== this.gen) return;
      this.running = false;
      if (state === 'speaking') setState(chat.busy() ? 'thinking' : 'listening');
      this.waiters.splice(0).forEach((w) => w());
    }
    busy() { return this.running; }
    idle() { return this.running ? new Promise((r) => this.waiters.push(r)) : Promise.resolve(); }
    stop() {
      this.duck(false);
      this.gen++;
      this.queue = [];
      try { this.source?.stop(); } catch {}
      this.source = null;
      this.running = false;
      this.waiters.splice(0).forEach((w) => w());
    }
  }

  addEventListener('pagehide', stop);
})();
