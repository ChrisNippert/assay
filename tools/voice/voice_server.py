# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "faster-whisper>=1.1",
#   "kokoro-onnx>=0.4",
#   "fastapi>=0.110",
#   "uvicorn>=0.29",
#   "python-multipart>=0.0.9",
#   "numpy",
# ]
# ///
"""Speech for ollamui: speech to text (faster-whisper) and text to speech
(Kokoro), behind the same endpoints as OpenAI's audio API:

  POST /v1/audio/transcriptions   multipart "file", or a raw audio body
  POST /v1/audio/speech           {"input": "...", "voice": "af_heart"} -> WAV
  GET  /health

Both run on the CPU, so the GPU stays free for the language model.

    uv run tools/voice/voice_server.py              # listens on 127.0.0.1:8485

Models download on first use (~250 MB for whisper small.en, ~120 MB for
Kokoro int8) into ~/.cache.
"""

import argparse
import io
import os
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ollamui-voice"

args = None
app = FastAPI()
_stt = None
_stt_name = None
_tts = None
_stt_lock = threading.Lock()
_tts_lock = threading.Lock()


def stt(name=None):
    """The whisper model called name (default --whisper). Asking for another
    one swaps it in; the first use of a model downloads it."""
    global _stt, _stt_name
    name = name or args.whisper
    with _stt_lock:
        if _stt is None or name != _stt_name:
            from faster_whisper import WhisperModel

            t = time.time()
            _stt = None  # free the old one first
            _stt = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=args.threads)
            _stt_name = name
            print(f"whisper {name} ready in {time.time() - t:.1f}s", flush=True)
        return _stt


def whisper_name(model):
    """A faster-whisper model name from a request, or None for the default.
    OpenAI clients send "whisper-1", which means "whatever you have"."""
    from faster_whisper.utils import available_models

    return model if model in available_models() else None


def fetch(name):
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name}…", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(KOKORO_URL + name, tmp)
        tmp.rename(path)
    return str(path)


def tts():
    global _tts
    with _tts_lock:
        if _tts is None:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            t = time.time()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = args.tts_threads
            session = ort.InferenceSession(fetch("kokoro-v1.0.int8.onnx"), opts, providers=["CPUExecutionProvider"])
            _tts = Kokoro.from_session(session, fetch("voices-v1.0.bin"))
            print(f"kokoro ready in {time.time() - t:.1f}s", flush=True)
        return _tts


def decode(data: bytes):
    """16-bit PCM WAV goes straight to numpy; anything else through PyAV."""
    if data[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(data)) as w:
                if w.getsampwidth() == 2 and w.getframerate() == 16000:
                    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                    pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
                    return (pcm / 32768.0).astype(np.float32)
        except wave.Error:
            pass
    return io.BytesIO(data)


@app.get("/health")
def health():
    return {"ok": True, "whisper": _stt_name or args.whisper, "voice": args.voice}


@app.get("/v1/audio/voices")
async def voices():
    names = await run_in_threadpool(lambda: sorted(tts().get_voices()))
    return {"voices": names, "default": args.voice}


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    ctype = request.headers.get("content-type", "")
    language = request.query_params.get("language")
    prompt = request.query_params.get("prompt")
    model_name = request.query_params.get("model")
    if ctype.startswith("multipart/"):
        form = await request.form()
        f = form.get("file")
        if f is None:
            raise HTTPException(400, "missing file")
        data = await f.read()
        language = form.get("language") or language
        prompt = form.get("prompt") or prompt
        model_name = form.get("model") or model_name
    else:
        data = await request.body()
    if not data:
        raise HTTPException(400, "empty audio")
    t = time.time()

    name = whisper_name(model_name)

    def run():
        model = stt(name)
        with _stt_lock:  # one transcription at a time; they'd only fight for cores
            segments, _ = model.transcribe(
                decode(data), language=language or None, initial_prompt=prompt or None,
                beam_size=1, condition_on_previous_text=False, vad_filter=False,
            )
            return "".join(s.text for s in segments).strip()

    text = await run_in_threadpool(run)
    print(f"stt {len(data) // 1024} KB in {time.time() - t:.2f}s: {text[:60]!r}", flush=True)
    return JSONResponse({"text": text})


@app.post("/v1/audio/speech")
async def speech(request: Request):
    body = await request.json()
    text = (body.get("input") or "").strip()
    if not text:
        raise HTTPException(400, "empty input")
    voice = body.get("voice") or args.voice
    if voice in ("alloy", "echo", "fable", "onyx", "nova", "shimmer"):
        voice = args.voice  # OpenAI's names; Kokoro has its own
    speed = float(body.get("speed") or args.speed)
    t = time.time()

    def run():
        model = tts()
        with _tts_lock:
            return model.create(text, voice=voice, speed=speed, lang=args.lang)

    try:
        samples, rate = await run_in_threadpool(run)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, f"can't speak that: {e}")
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    print(f"tts {len(text)} chars -> {len(pcm) / rate:.1f}s audio in {time.time() - t:.2f}s", flush=True)
    return Response(buf.getvalue(), media_type="audio/wav")


def main():
    global args
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listen", default="127.0.0.1:8485")
    p.add_argument("--whisper", default="small.en", help="faster-whisper model: tiny.en, base.en, small.en, distil-small.en, …")
    p.add_argument("--voice", default="af_heart", help="Kokoro voice, e.g. af_heart, af_bella, am_michael, bf_emma")
    p.add_argument("--lang", default="en-us")
    p.add_argument("--speed", type=float, default=1.0)
    # Whisper, Kokoro and ollamui's talker model (4 threads) all run on the
    # CPU at once during a voice turn; on a shared pool each slows the others
    # several times over, so each gets its own share of the cores.
    cores = max(2, (os.cpu_count() or 4) // 2)
    p.add_argument("--threads", type=int, default=max(1, cores - 2), help="whisper threads")
    p.add_argument("--tts-threads", type=int, default=max(1, min(4, cores // 2)), help="Kokoro threads")
    p.add_argument("--lazy", action="store_true", help="load models on first request instead of at startup")
    args = p.parse_args()
    if not args.lazy:
        threading.Thread(target=lambda: (stt(), tts()), daemon=True).start()
    host, _, port = args.listen.rpartition(":")
    uvicorn.run(app, host=host or "127.0.0.1", port=int(port), log_level="warning")


if __name__ == "__main__":
    main()
