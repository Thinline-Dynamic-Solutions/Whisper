# Quick Start Guide

Get up and running with the Whisper transcription server in under 5 minutes!

## Prerequisites
- Python 3.8+ installed
- Internet connection (for downloading models)

## 1. Install FFmpeg

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```powershell
choco install ffmpeg
```

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- FastAPI/Uvicorn (API server)
- OpenAI Whisper (transcription)
- PyTorch (ML framework)
- Silero VAD (tone detection)

## 3. Start the Server

**For testing (small/fast model):**
```bash
python whisper_server.py --model small
```

**For production (best accuracy):**
```bash
python whisper_server.py --model large-v3
```

**First run note:** The Silero VAD model (~2MB) will be automatically downloaded on the first transcription. This is a one-time download.

## 4. Test It!

**Health check:**
```bash
curl http://localhost:8000/health
```

**Transcribe an audio file:**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model="whisper-1" \
  -F file="@your-audio.mp3"
```

## That's it! 🎉

### What happens automatically:
- ✅ Silero VAD detects and skips alert tones/silence
- ✅ Audio is converted to 16kHz mono for optimal processing
- ✅ GPU acceleration (if CUDA available)
- ✅ 150ms pre-speech buffer preserved for context

### Next Steps:
- Read [README.md](README.md) for full API documentation
- Check [INSTALL.md](INSTALL.md) for detailed installation options
- Review [CHANGES.md](CHANGES.md) to see what's new

## Common Options

```bash
# Use CPU instead of GPU
python whisper_server.py --model small --device cpu

# Change port
python whisper_server.py --model large-v3 --port 9000

# Listen on specific interface
python whisper_server.py --model large-v3 --host 127.0.0.1

# Preload model without starting server
python whisper_server.py --model large-v3 --preload
```

## Docker Quick Start

If you prefer Docker:

```bash
# Edit whisper.env to configure (optional)
nano whisper.env

# Build and run
docker compose up
```

Server will be available at `http://localhost:8000`

## Python API Example

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    api_key='not-needed',
    base_url='http://localhost:8000/v1'
)

# Transcribe
with open("audio.mp3", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio
    )
    print(result.text)
```

## Troubleshooting

**"Command not found: ffmpeg"**
→ Install FFmpeg (see step 1)

**"CUDA out of memory"**
→ Use smaller model or CPU: `--model small --device cpu`

**Port 8000 in use**
→ Use different port: `--port 8001`

**Import errors**
→ Reinstall dependencies: `pip install -r requirements.txt --upgrade`

## Model Comparison

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| tiny | ⚡⚡⚡⚡⚡ | ⭐⭐ | Testing |
| small | ⚡⚡⚡⚡ | ⭐⭐⭐ | Real-time, clear audio |
| medium | ⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| large-v3 | ⚡⚡ | ⭐⭐⭐⭐⭐ | Production, best quality |

---

**Need help?** Check the full documentation in [README.md](README.md) or [INSTALL.md](INSTALL.md)

