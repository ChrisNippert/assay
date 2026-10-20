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
  const START_FRAMES = 4; // loud frames in a row to start an utterance
  const START_FRAMES_PLAYING = 10; // more while we're talking, against echo
  const END_SILENCE = 40; // quiet frames that end an utterance (800 ms)
  const MIN_SPEECH = 12; // loud frames for an utterance to count (240 ms)
  const MAX_FRAMES = 1500; // 30 s
  const PARTIAL_MS = 900;

  let status = null;
  let active = false;
  let audio = null; // { ctx, stream, node }
  let speaker = null;
  let state = '';

  // With a model, this also has the server load the talker ahead of time.
  async function checkStatus(model) {
    try { status = await (await fetch('/api/voice' + (model ? '?model=' + encodeURIComponent(model) : ''))).json(); } catch { status = { available: false, error: 'Couldn’t reach ollamui.' }; }
    btn.classList.toggle('unavailable', !status.available);
    btn.title = status.available ? 'Talk (voice mode)' : status.error;
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
    if (!status.available) { setState('error', status.error); return; }
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
  let floor = 0.002; // running estimate of background noise (RMS)
  let loudRun = 0;
  let preroll = [];
  let utt = null; // { frames, loud, quiet, partialAt, partialBusy }

  function onFrame(f) {
    if (!active) return;
    let sum = 0;
    for (let i = 0; i < f.length; i++) sum += f[i] * f[i];
    const rms = Math.sqrt(sum / f.length);
    const playing = speaker?.busy();
    const threshold = Math.max(floor * 3, 0.006) * (playing ? 2.5 : 1);
    const loud = rms > threshold;

    if (!utt) {
      if (!loud) floor = floor * 0.97 + rms * 0.03;
      preroll.push(f);
      if (preroll.length > PREROLL) preroll.shift();
      loudRun = loud ? loudRun + 1 : 0;
      if (loudRun >= (playing ? START_FRAMES_PLAYING : START_FRAMES) && state !== 'transcribing') beginUtterance();
      return;
    }
    utt.frames.push(f);
    if (loud) { utt.loud++; utt.quiet = 0; } else utt.quiet++;
    if (utt.quiet >= END_SILENCE || utt.frames.length >= MAX_FRAMES) endUtterance();
    else if (performance.now() - utt.partialAt > PARTIAL_MS && !utt.partialBusy && utt.loud >= MIN_SPEECH) partial();
  }

  function beginUtterance() {
    // Talking over the reply interrupts it.
    if (speaker.busy() || chat.busy()) { speaker.stop(); chat.stop(); }
    utt = { frames: preroll, loud: loudRun, quiet: 0, partialAt: performance.now(), partialBusy: false };
    preroll = [];
    loudRun = 0;
    setState('hearing');
  }

  // Live text while you talk: transcribe what's been said so far.
  async function partial() {
    const u = utt;
    u.partialBusy = true;
    u.partialAt = performance.now();
    const text = await transcribe(u.frames);
    u.partialBusy = false;
    if (utt === u && text) {
      input.value = text;
      input.dispatchEvent(new Event('input'));
    }
  }

  async function endUtterance() {
    const u = utt;
    utt = null;
    if (u.loud < MIN_SPEECH) { listen(); return; } // a cough or a click
    setState('transcribing');
    const text = await transcribe(u.frames);
    if (!active) return;
    // Whisper hears words in noise now and then; very short ones are noise.
    if (!text || text.replace(/[^\p{L}\p{N}]/gu, '').length < 2 || /^(thank you|thanks for watching|you)[.!]?$/i.test(text)) {
      input.value = '';
      listen();
      return;
    }
    input.value = '';
    input.dispatchEvent(new Event('input'));
    reply(text);
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
  async function reply(text) {
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
    if (!active) return;
    await speaker.idle();
    if (active && !utt) listen();
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
    constructor(ctx) { this.ctx = ctx; this.queue = []; this.source = null; this.gen = 0; this.running = false; this.waiters = []; }
    say(text) {
      const gen = this.gen;
      const buf = fetch('/api/voice/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) })
        .then((r) => (r.ok ? r.arrayBuffer() : null))
        .then((b) => (b && gen === this.gen ? this.ctx.decodeAudioData(b) : null))
        .catch(() => null);
      this.queue.push(buf);
      if (!this.running) this.pump(gen);
    }
    async pump(gen) {
      this.running = true;
      if (state !== 'hearing') setState('speaking');
      while (this.queue.length && gen === this.gen) {
        const buf = await this.queue.shift();
        if (!buf || gen !== this.gen) continue;
        if (state !== 'hearing') setState('speaking');
        await new Promise((done) => {
          const src = this.ctx.createBufferSource();
          src.buffer = buf;
          src.connect(this.ctx.destination);
          src.onended = done;
          this.source = src;
          src.start();
        });
        this.source = null;
      }
      if (gen !== this.gen) return;
      this.running = false;
      if (state === 'speaking') setState(chat.busy() ? 'thinking' : 'listening');
      this.waiters.splice(0).forEach((w) => w());
    }
    busy() { return this.running; }
    idle() { return this.running ? new Promise((r) => this.waiters.push(r)) : Promise.resolve(); }
    stop() {
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
