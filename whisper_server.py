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

# Silero VAD functions removed - no longer trimming tones before transcription
# Instead, Whisper is configured to handle tones through:
# - Higher no_speech_threshold to skip tone-only segments
# - Lower logprob_threshold to be more strict on low-confidence segments
# - Better initial_prompt to guide the model

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
        # Convert audio to consistent format (16kHz mono WAV) for Whisper
        try:
            ffmpeg_to_wav_mono_16k(temp_path, wav_path)
            transcribe_path = wav_path
        except Exception:
            # If conversion fails, use original file
            transcribe_path = temp_path

        # Whisper transcription options
        # NOTE: Removed Silero VAD tone-skipping logic - transcribing full audio instead
        # If Whisper hallucinates on tones, address through:
        # 1. Better initial_prompt to ignore non-speech sounds
        # 2. Higher no_speech_threshold to skip tone-only segments
        # 3. Adjust compression_ratio_threshold to detect repetitive hallucinations
        options = {
            "task": task,
            "temperature": temperature if temperature is not None else 0.0,
            "fp16": False,
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.4,  # Detect hallucinations (repetitive text)
            "logprob_threshold": -1.0,  # Lowered from -0.8 to be more strict (skip low-confidence segments)
            "no_speech_threshold": 0.6,  # Raised from 0.3 to better ignore tone-only/noise segments
        }

        if language:
            options["language"] = language
        
        # Add initial prompt to help Whisper understand the context and ignore tones
        if prompt:
            options["initial_prompt"] = prompt
        else:
            # Default prompt for radio dispatch audio
            options["initial_prompt"] = "This is radio dispatch audio. Ignore alert tones and beeps. Transcribe only spoken words."

        result = model.transcribe(transcribe_path, **options)

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
    elif response_format == "srt":
        def srt_time(t):
            return "{:02d}:{:02d}:{:06.3f}".format(int(t//3600), int(t//60)%60, t%60).replace(".", ",")
        segments = result.get("segments", [])
        srt_output = "\n".join([
            f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{seg['text'].strip()}\n"
            for i, seg in enumerate(segments, 1)
        ])
        return PlainTextResponse(srt_output, media_type="text/srt; charset=utf-8",
                                 headers={"Content-Disposition": f"attachment; filename={filename_noext}.srt"})
    elif response_format == "vtt":
        def vtt_time(t):
            return "{:02d}:{:06.3f}".format(int(t//60), t%60)
        segments = result.get("segments", [])
        vtt_output = "WEBVTT\n\n" + "\n".join([
            f"{vtt_time(seg['start'])} --> {vtt_time(seg['end'])}\n{seg['text'].strip()}\n"
            for seg in segments
        ])
        return PlainTextResponse(vtt_output, media_type="text/vtt; charset=utf-8",
                                 headers={"Content-Disposition": f"attachment; filename={filename_noext}.vtt"})

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
    if response_format == "text":
        return PlainTextResponse(result["text"].strip(),
                                 headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"})
    return JSONResponse(content={"text": result["text"].strip()},
                        media_type="application/json",
                        headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})

# ----------------------------
# CLI
# ----------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="whisper_server.py",
        description="OpenedAI Whisper API Server for radio dispatch audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--model", default="large-v3",
                        help="The model to use for transcription. tiny, base, small, medium, large, large-v2, large-v3")
    parser.add_argument("-d", "--device", default="auto",
                        help="Torch device. Ex. cuda:0 or cpu")
    parser.add_argument("-P", "--port", default=8000, type=int, help="Server tcp port")
    parser.add_argument("-H", "--host", default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--preload", action="store_true", help="Preload model and exit.")
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Whisper model '{args.model}' on device '{device}'...")
    model = whisper.load_model(args.model, device=device)
    print("Model loaded successfully!")

    if args.preload:
        sys.exit(0)

    app.register_model("whisper-1", args.model)

    uvicorn.run(app, host=args.host, port=args.port)
