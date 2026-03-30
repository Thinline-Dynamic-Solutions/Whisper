#!/usr/bin/env python3
import os
import re
import sys
import time
import logging
import argparse
import tempfile
import subprocess
import gc
from collections import Counter
from typing import Optional, List

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import torch
import whisper

from fastapi import Request, UploadFile, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

import openedai

model = None
app = openedai.OpenAIStub()

SAMPLE_RATE = 16000

# Per-process transcription counter (each uvicorn worker has its own)
_transcription_count = 0


@app.on_event("startup")
async def load_whisper_model():
    """Load the Whisper model in each worker process on startup."""
    global model
    model_name = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
    device     = os.environ.get("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info(f"Loading Whisper model '{model_name}' on device '{device}' (pid {os.getpid()})...")
    model = whisper.load_model(model_name, device=device)
    app.register_model("whisper-1", model_name)
    logging.info(f"Model '{model_name}' ready (pid {os.getpid()})")

# ----------------------------
# Hallucination filter lists
# ----------------------------

# Segments containing any of these phrases are dropped entirely
BLOCK_PHRASES = [
    "castingwords", "eso", "rev.com", "amara.org", "amara",
    "esa, inc.", "u.s. department of", "transcripts translated by",
    "transcription outsourcing",
]

# Transcripts are truncated at the first match of any of these phrases
CUTOFF_PHRASES = [
    "we'll be right back", "we will be right back",
    "thank you for watching", "thanks for watching",
    "thank you for your patience", "stay tuned", "commercial break",
    "transcription by", "translation by", "captions by",
    "subtitle", "subtitles",
]

# Substrings that indicate an alert-tone hallucination (checked uppercase)
BEEP_PATTERNS = [
    "BEEE", "BEEEE", "EEEE", "BEEP", "BEEEP",
    "BEEEEEEEE", "EEEEEEEE",
    "AAAA", "AAAAA", "AAAAAA", "A A A",
    "AAAAAAAAA", "AAAAAAAAAA",
]

# ----------------------------
# ffmpeg helpers
# ----------------------------

def _run_ffmpeg(cmd: List[str]) -> None:
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def ffmpeg_to_wav_mono_16k(in_path: str, out_path: str) -> None:
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", in_path,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        out_path
    ])

# ----------------------------
# Audio pre-processing
# ----------------------------

def load_and_preprocess_audio(wav_path: str) -> np.ndarray:
    """
    Load a 16k mono WAV, apply a high-pass filter to remove low-frequency
    rumble/static, then apply percentile-based normalization so squelch pops
    don't crush quiet speech.  Returns a float32 numpy array ready for Whisper.
    """
    rate, data = wavfile.read(wav_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)

    # Flatten to mono just in case ffmpeg produced stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 5th-order Butterworth high-pass at 100 Hz — removes rumble / DC offset
    sos = signal.butter(5, 100 / (rate / 2), btype="high", output="sos")
    audio = signal.sosfilt(sos, audio).astype(np.float32)

    # Percentile-based normalization (95th percentile)
    percentile_val = np.percentile(np.abs(audio), 95)
    if percentile_val > 0:
        audio = audio / percentile_val
        audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32)

# ----------------------------
# Whisper segment filtering
# ----------------------------

def filter_segments(result):
    """Remove segments that fail Whisper's own confidence signals."""
    filtered_segments = []
    final_text = []

    for seg in result.get("segments", []):
        avg_logprob      = seg.get("avg_logprob", 0)
        no_speech_prob   = seg.get("no_speech_prob", 0)
        compression_ratio = seg.get("compression_ratio", 0)

        if no_speech_prob > 0.7:
            continue
        if avg_logprob < -1.2:
            continue
        if compression_ratio > 2.4:
            continue

        filtered_segments.append(seg)
        final_text.append(seg["text"].strip())

    result["segments"] = filtered_segments
    result["text"] = " ".join(final_text).strip()
    return result

# ----------------------------
# Text-level hallucination filtering
# ----------------------------

def filter_text(text: str, duration: float) -> Optional[str]:
    """
    Second-pass text filter that catches hallucinations which pass Whisper's
    confidence thresholds.  Returns the (possibly truncated/replaced) text,
    or None if the entire result should be discarded.
    """
    if not text:
        return text

    lower = text.lower()

    # Drop entire segment on known caption/credit hallucinations
    for phrase in BLOCK_PHRASES:
        if phrase in lower:
            return None

    # Truncate at known hallucinated endings
    for phrase in CUTOFF_PHRASES:
        idx = lower.find(phrase)
        if idx != -1:
            text = text[:idx].strip()
            lower = text.lower()
            break

    upper = text.upper()

    # Alert-tone hallucinations → [beeps]
    if duration < 10.0 and len(text) > 10:
        if any(pat in upper for pat in BEEP_PATTERNS):
            return "[beeps]"

    # Long repeated-character run → [noise]
    if re.search(r"([A-Z])\1{10,}", upper):
        return "[noise]"

    # Whisper repetition loop: one word makes up >50 % of output → collapse
    words = text.split()
    if len(words) > 3:
        common = Counter(words).most_common(1)
        if common and len(common[0][0]) <= 10 and common[0][1] > len(words) // 2:
            return common[0][0]

    return text

# ----------------------------
# Transcription
# ----------------------------

async def transcribe_audio(
    file: UploadFile,
    response_format: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    task: str = "transcribe",
    beam_size: int = 5,
    best_of: int = 5,
):
    global model

    audio_data = await file.read()

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(audio_data)
        temp_path = temp_file.name

    wav_path = temp_path + ".16k.wav"
    audio_array: Optional[np.ndarray] = None

    try:
        try:
            ffmpeg_to_wav_mono_16k(temp_path, wav_path)
            audio_array = load_and_preprocess_audio(wav_path)
            transcribe_input = audio_array
        except Exception:
            transcribe_input = temp_path

        duration = float(len(audio_array)) / SAMPLE_RATE if audio_array is not None else 0.0

        options = {
            "task": task,
            "temperature": temperature if temperature is not None else 0.0,
            "fp16": torch.cuda.is_available(),
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": 1.5,
            "suppress_blank": True,
        }

        if language:
            options["language"] = language
        if prompt:
            options["initial_prompt"] = prompt

        result = model.transcribe(transcribe_input, **options)

        result = filter_segments(result)

        filtered = filter_text(result["text"], duration)
        if filtered is None:
            result["text"] = ""
            result["segments"] = []
        else:
            result["text"] = filtered

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return result

    finally:
        for p in (temp_path, wav_path):
            try:
                os.unlink(p)
            except Exception:
                pass

# ----------------------------
# API endpoints
# ----------------------------

@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile,
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None),
    timestamp_granularities: List[str] = Form(["segment"]),
    beam_size: Optional[int] = Form(5),
    best_of: Optional[int] = Form(5),
):
    global _transcription_count

    system_label    = request.headers.get("X-TLR-System", "")
    talkgroup_label = request.headers.get("X-TLR-Talkgroup", "")
    call_id         = request.headers.get("X-TLR-Call-ID", "?")

    start = time.time()

    result = await transcribe_audio(
        file, response_format, language, prompt,
        temperature or 0.0, task="transcribe",
        beam_size=beam_size or 5, best_of=best_of or 5,
    )

    elapsed = time.time() - start
    _transcription_count += 1

    channel = f"{system_label} / {talkgroup_label}" if system_label or talkgroup_label else "unknown"
    logging.info(
        f"[transcription] call {call_id} | {channel} | "
        f"done in {elapsed:.2f}s | total #{_transcription_count} | pid {os.getpid()}"
    )

    filename_noext, ext = os.path.splitext(file.filename)

    if response_format == "text":
        return PlainTextResponse(result["text"].strip(),
                                 headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"})
    elif response_format == "json":
        return JSONResponse(content={"text": result["text"].strip()},
                            media_type="application/json",
                            headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})
    elif response_format == "verbose_json":
        segments = result.get("segments", [])
        response = {
            "task": "transcribe",
            "language": result.get("language", language or "en"),
            "duration": segments[-1]["end"] if segments else 0,
            "text": result["text"].strip(),
            "segments": [{
                "id": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            } for i, seg in enumerate(segments)]
        }
        return JSONResponse(content=response,
                            media_type="application/json",
                            headers={"Content-Disposition": f"attachment; filename={filename_noext}_verbose.json"})

    return JSONResponse(content={"text": result["text"].strip()})


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile,
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None),
    beam_size: Optional[int] = Form(5),
    best_of: Optional[int] = Form(5),
):
    result = await transcribe_audio(
        file, response_format, None, prompt,
        temperature or 0.0, task="translate",
        beam_size=beam_size or 5, best_of=best_of or 5,
    )
    filename_noext, ext = os.path.splitext(file.filename)
    return JSONResponse(content={"text": result["text"].strip()},
                        media_type="application/json",
                        headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",   default="large-v3-turbo")
    parser.add_argument("-d", "--device",  default="auto")
    parser.add_argument("-P", "--port",    default=8000, type=int)
    parser.add_argument("-H", "--host",    default="0.0.0.0")
    parser.add_argument("-w", "--workers", default=1, type=int,
                        help="Number of parallel uvicorn worker processes (each loads its own model)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pass config to worker processes via environment variables
    # (on Windows, workers are spawned fresh so globals don't carry over)
    os.environ["WHISPER_MODEL"]  = args.model
    os.environ["WHISPER_DEVICE"] = device

    print(f"Starting Whisper server: model={args.model} device={device} workers={args.workers}")

    uvicorn.run(
        "whisper_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
