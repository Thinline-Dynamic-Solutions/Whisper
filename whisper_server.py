#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
import subprocess
import gc
from typing import Optional, List

import torch
import whisper

from fastapi import UploadFile, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

import openedai

model = None
app = openedai.OpenAIStub()

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
        "-ar", "16000",
        out_path
    ])

# ----------------------------
# Whisper Segment Filtering
# ----------------------------

def filter_segments(result):
    """
    Remove tone hallucination segments based on Whisper confidence signals.
    """

    filtered_segments = []
    final_text = []

    for seg in result.get("segments", []):
        avg_logprob = seg.get("avg_logprob", 0)
        no_speech_prob = seg.get("no_speech_prob", 0)
        compression_ratio = seg.get("compression_ratio", 0)

        # Tone / hallucination rejection rules
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
# Transcription
# ----------------------------

async def transcribe_audio(
    file: UploadFile,
    response_format: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    task: str = "transcribe"
):
    global model

    audio_data = await file.read()

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(audio_data)
        temp_path = temp_file.name

    wav_path = temp_path + ".16k.wav"

    try:
        try:
            ffmpeg_to_wav_mono_16k(temp_path, wav_path)
            transcribe_path = wav_path
        except Exception:
            transcribe_path = temp_path

        options = {
            "task": task,
            "temperature": temperature if temperature is not None else 0.0,
            "fp16": False,
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }

        if language:
            options["language"] = language

        result = model.transcribe(transcribe_path, **options)

        # 🔥 FILTER OUT TONE HALLUCINATIONS HERE
        result = filter_segments(result)

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
    file: UploadFile,
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None),
    timestamp_granularities: List[str] = Form(["segment"])
):
    result = await transcribe_audio(file, response_format, language, prompt, temperature or 0.0, task="transcribe")
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
):
    result = await transcribe_audio(file, response_format, None, prompt, temperature or 0.0, task="translate")
    filename_noext, ext = os.path.splitext(file.filename)
    return JSONResponse(content={"text": result["text"].strip()},
                        media_type="application/json",
                        headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="large-v3")
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("-P", "--port", default=8000, type=int)
    parser.add_argument("-H", "--host", default="0.0.0.0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Whisper model '{args.model}' on device '{device}'...")
    model = whisper.load_model(args.model, device=device)
    print("Model loaded successfully!")

    app.register_model("whisper-1", args.model)

    uvicorn.run(app, host=args.host, port=args.port)
