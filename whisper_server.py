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

from faster_whisper import WhisperModel

from fastapi import Request, UploadFile, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

import openedai

model = None
app = openedai.OpenAIStub()

SAMPLE_RATE = 16000
_transcription_count = 0


@app.on_event("startup")
async def load_whisper_model():
    global model
    model_name   = os.environ.get("WHISPER_MODEL", "large-v3")
    device       = os.environ.get("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16" if device != "cpu" else "int8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info(f"Loading Whisper model '{model_name}' on device '{device}' (pid {os.getpid()})...")

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type
    )

    app.register_model("whisper-1", model_name)
    logging.info(f"Model '{model_name}' ready (pid {os.getpid()})")


# ----------------------------
# FILTER LISTS (UNCHANGED)
# ----------------------------

BLOCK_PHRASES = [
    "castingwords", "eso", "rev.com", "amara.org", "amara",
    "esa, inc.", "u.s. department of", "transcripts translated by",
    "transcription outsourcing",
]

CUTOFF_PHRASES = [
    "we'll be right back", "we will be right back",
    "thank you for watching", "thanks for watching",
    "thank you for your patience", "stay tuned", "commercial break",
    "transcription by", "translation by", "captions by",
    "subtitle", "subtitles",
]

BEEP_PATTERNS = [
    "BEEE", "BEEEE", "EEEE", "BEEP", "BEEEP",
    "BEEEEEEEE", "EEEEEEEE",
    "AAAA", "AAAAA", "AAAAAA", "A A A",
    "AAAAAAAAA", "AAAAAAAAAA",
]

# ----------------------------
# FFmpeg helpers (UNCHANGED)
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
# Audio pre-processing (UNCHANGED)
# ----------------------------

def load_and_preprocess_audio(wav_path: str) -> np.ndarray:
    rate, data = wavfile.read(wav_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    sos = signal.butter(5, 100 / (rate / 2), btype="high", output="sos")
    audio = signal.sosfilt(sos, audio).astype(np.float32)

    percentile_val = np.percentile(np.abs(audio), 95)
    if percentile_val > 0:
        audio = audio / percentile_val
        audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32)

# ----------------------------
# Text filtering (UNCHANGED)
# ----------------------------

def filter_text(text: str, duration: float) -> Optional[str]:
    if not text:
        return text

    lower = text.lower()

    for phrase in BLOCK_PHRASES:
        if phrase in lower:
            return None

    for phrase in CUTOFF_PHRASES:
        idx = lower.find(phrase)
        if idx != -1:
            text = text[:idx].strip()
            break

    upper = text.upper()

    if duration < 10.0 and len(text) > 10:
        if any(pat in upper for pat in BEEP_PATTERNS):
            return "[beeps]"

    if re.search(r"([A-Z])\1{10,}", upper):
        return "[noise]"

    words = text.split()
    if len(words) > 3:
        common = Counter(words).most_common(1)
        if common and common[0][1] > len(words) // 2:
            return common[0][0]

    return text

# ----------------------------
# TRANSCRIPTION (ADAPTED CLEANLY)
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

        segments, info = model.transcribe(
            transcribe_input,
            beam_size=beam_size,
            language=language,
            initial_prompt=prompt
        )

        text = " ".join([seg.text.strip() for seg in segments])

        filtered = filter_text(text, duration)
        if filtered is None:
            text = ""

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "text": text,
            "segments": list(segments),
            "language": info.language,
        }

    finally:
        for p in (temp_path, wav_path):
            try:
                os.unlink(p)
            except Exception:
                pass

# ----------------------------
# API (UNCHANGED)
# ----------------------------

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
        temperature or 0.0,
        beam_size=beam_size or 5,
        best_of=best_of or 5,
    )

    elapsed = time.time() - start
    _transcription_count += 1

    channel = f"{system_label} / {talkgroup_label}" if system_label or talkgroup_label else "unknown"

    logging.info(
        f"[transcription] call {call_id} | {channel} | "
        f"done in {elapsed:.2f}s | total #{_transcription_count} | pid {os.getpid()}"
    )

    return JSONResponse(content={"text": result["text"].strip()})

# ----------------------------
# CLI
# ----------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",   default="large-v3")
    parser.add_argument("-d", "--device",  default="auto")
    parser.add_argument("-P", "--port",    default=8000, type=int)
    parser.add_argument("-H", "--host",    default="0.0.0.0")
    parser.add_argument("-w", "--workers", default=1, type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.environ["WHISPER_MODEL"]  = args.model
    os.environ["WHISPER_DEVICE"] = device

    print(f"Starting Whisper server: model={args.model} device={device} workers={args.workers}")

    uvicorn.run(
        "whisper_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
