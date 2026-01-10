#!/usr/bin/env python3
import os
import sys
import argparse
import io
import tempfile

import torch
import whisper
from typing import Optional, List
from fastapi import UploadFile, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn
import gc

import openedai

model = None
app = openedai.OpenAIStub()

async def transcribe_audio(file, response_format: str, language: Optional[str], prompt: Optional[str], temperature: float, task: str = "transcribe"):
    global model
    
    # Read audio data
    audio_data = await file.read()
    
    # Save to temporary file (whisper.load_audio needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(audio_data)
        temp_path = temp_file.name
    
    try:
        # Build transcribe options with proper hallucination prevention
        options = {
            "task": task,
            "temperature": temperature if temperature is not None else 0.0,
            "fp16": False,  # Use FP32 for better quality on CPU
            "condition_on_previous_text": False,  # Don't build on previous predictions - reduces hallucinations
            "compression_ratio_threshold": 2.4,  # Detect and reject repetitive patterns
            "logprob_threshold": -1.0,  # Reject low-confidence predictions
            "no_speech_threshold": 0.6,  # Detect silence/no speech and return empty
        }
        
        if language:
            options["language"] = language
        
        # Add prompt if provided
        if prompt:
            options["initial_prompt"] = prompt
        
        # Transcribe
        result = model.transcribe(temp_path, **options)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return result
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

@app.get("/health")
async def health():
    """Health check endpoint for monitoring service availability"""
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
        return PlainTextResponse(result["text"].strip(), headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"})
    
    elif response_format == "json":
        return JSONResponse(content={'text': result['text'].strip()}, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})
    
    elif response_format == "verbose_json":
        segments = result.get("segments", [])
        
        response = {
            "task": "transcribe",
            "language": result.get("language", language or "en"),
            "duration": segments[-1]['end'] if segments else 0,
            "text": result["text"].strip(),
            "segments": [{
                "id": i,
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip(),
            } for i, seg in enumerate(segments)]
        }
        
        return JSONResponse(content=response, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}_verbose.json"})
    
    elif response_format == "srt":
        def srt_time(t):
            return "{:02d}:{:02d}:{:06.3f}".format(int(t//3600), int(t//60)%60, t%60).replace(".", ",")
        
        segments = result.get("segments", [])
        srt_output = "\n".join([
            f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{seg['text'].strip()}\n"
            for i, seg in enumerate(segments, 1)
        ])
        
        return PlainTextResponse(srt_output, media_type="text/srt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.srt"})
    
    elif response_format == "vtt":
        def vtt_time(t):
            return "{:02d}:{:06.3f}".format(int(t//60), t%60)
        
        segments = result.get("segments", [])
        vtt_output = "WEBVTT\n\n" + "\n".join([
            f"{vtt_time(seg['start'])} --> {vtt_time(seg['end'])}\n{seg['text'].strip()}\n"
            for seg in segments
        ])
        
        return PlainTextResponse(vtt_output, media_type="text/vtt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.vtt"})


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
        return PlainTextResponse(result["text"].strip(), headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"})
    else:
        return JSONResponse(content={'text': result['text'].strip()}, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='whisper_server.py',
        description='OpenedAI Whisper API Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-m', '--model', action='store', default="large-v2", help="The model to use for transcription. Options: tiny, base, small, medium, large, large-v2, large-v3")
    parser.add_argument('-d', '--device', action='store', default="auto", help="Set the torch device for the model. Ex. cuda:0 or cpu")
    parser.add_argument('-P', '--port', action='store', default=8000, type=int, help="Server tcp port")
    parser.add_argument('-H', '--host', action='store', default='0.0.0.0', help="Host to listen on, Ex. 0.0.0.0")
    parser.add_argument('--preload', action='store_true', help="Preload model and exit.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Whisper model '{args.model}' on device '{device}'...")
    model = whisper.load_model(args.model, device=device)
    print(f"Model loaded successfully!")
    
    if args.preload:
        sys.exit(0)
    
    app.register_model('whisper-1', args.model)
    
    uvicorn.run(app, host=args.host, port=args.port)
