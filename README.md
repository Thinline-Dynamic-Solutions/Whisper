OpenedAI Whisper
----------------

An OpenAI API compatible speech to text server for audio transcription and translations, aka. Whisper.

- Compatible with the OpenAI audio/transcriptions and audio/translations API
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way
- **NEW:** Automatic alert tone and silence skipping with Silero VAD

## Quick Links

📚 **Documentation:**
- [Quick Start Guide](QUICKSTART.md) - Get running in 5 minutes
- [Installation Guide](INSTALL.md) - Detailed setup for all platforms
- [Dependencies Explained](DEPENDENCIES.md) - What gets installed and why
- [Change Log](CHANGES.md) - What's new in v0.2.0

🚀 **Quick Start:**
```bash
pip install -r requirements.txt
python whisper_server.py --model small
curl http://localhost:8000/health
```

API Compatibility:
- [X] /v1/audio/transcriptions
- [X] /v1/audio/translations

Parameter Support:
- [X] `file`
- [X] `model` (only whisper-1 exists, so this is ignored)
- [X] `language`
- [X] `prompt` (**FULLY SUPPORTED** - guides transcription with custom terminology, formatting, etc.)
- [X] `temperature`
- [X] `response_format`:
- - [X] `json`
- - [X] `text`
- - [X] `srt`
- - [X] `vtt`
- - [X] `verbose_json`

Details:
* CUDA or CPU support (automatically detected)
* float32, float16 or bfloat16 support (automatically detected)
* **Silero VAD tone skipping** - Automatically detects and skips alert tones and silence at the beginning of audio files using Silero Voice Activity Detection

Tested whisper models:
* large-v3 (the default)
* large-v2
* large
* medium
* small
* base
* tiny


Version: 0.2.0, Last update: 2026-01-11


API Documentation
-----------------

## Usage

* [OpenAI Speech to text guide](https://platform.openai.com/docs/guides/speech-to-text)
* [OpenAI API Transcription Reference](https://platform.openai.com/docs/api-reference/audio/createTranscription)
* [OpenAI API Translation Reference](https://platform.openai.com/docs/api-reference/audio/createTranslation)


Installation instructions
-------------------------

### System Requirements

- Python 3.8 or higher
- FFmpeg (for audio processing)
- (Optional) CUDA-capable GPU for faster transcription

### Installation Steps

1. **Install FFmpeg**
   ```shell
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS (using Homebrew)
   brew install ffmpeg
   
   # Windows (using Chocolatey)
   choco install ffmpeg
   ```

2. **Install Python Dependencies**
   ```shell
   pip install -r requirements.txt
   ```
   
   This will install:
   - FastAPI and Uvicorn (API server)
   - OpenAI Whisper (transcription engine)
   - PyTorch and torchaudio (deep learning framework)
   - Silero VAD (automatic tone/silence detection, downloaded on first run)
   - Python-multipart (file upload support)

3. **(Optional) CUDA Support**
   
   For GPU acceleration, install CUDA for your operating system. PyTorch will automatically detect and use CUDA if available.
   
   - [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)

### First Run

On the first run, Silero VAD model will be automatically downloaded from torch.hub (~2MB). This is a one-time operation.

**Note**: This implementation uses the official OpenAI Whisper library which has **full prompt support** built-in!

Usage
-----

```
Usage: whisper_server.py [-m <model_name>] [-d <device>] [-P <port>] [-H <host>] [--preload]


Description:
OpenedAI Whisper API Server (Silero VAD tone skipping)

Options:
-h, --help            Show this help message and exit.
-m MODEL, --model MODEL
                      The model to use for transcription.
                      Options: tiny, base, small, medium, large, large-v2, large-v3 (default: large-v3)
-d DEVICE, --device DEVICE
                      Set the torch device for the model. Ex. cuda:0 or cpu (default: auto)
-P PORT, --port PORT  Server tcp port (default: 8000)
-H HOST, --host HOST  Host to listen on, Ex. 0.0.0.0 (default: 0.0.0.0)
--preload             Preload model and exit. (default: False)
```

### Automatic Tone and Silence Skipping

This server includes **Silero VAD (Voice Activity Detection)** which automatically:
- Detects the start of speech in audio files
- Skips alert tones, beeps, and silence at the beginning of recordings
- Preserves a 150ms buffer before speech starts to maintain context
- Improves transcription accuracy by removing non-speech audio

This feature is especially useful for:
- Radio dispatch recordings with alert tones
- Pager recordings with notification beeps
- Any audio with leading silence or tones

The VAD processing is automatic and requires no configuration. It processes the audio at 16kHz mono and uses a 250ms minimum speech duration threshold with 0.5 confidence.

Sample API Usage
----------------

### Health Check

Check if the server is running and ready:

```shell
curl http://localhost:8000/health
```

Response: `{"status":"ok"}`

### Transcription

You can use it like this:

```shell
curl -s http://localhost:8000/v1/audio/transcriptions -H "Content-Type: multipart/form-data" -F model="whisper-1" -F file="@audio.mp3" -F response_format=text
```

Or just like this:

```shell
curl -s http://localhost:8000/v1/audio/transcriptions -F model="whisper-1" -F file="@audio.mp3"
```

Or like this example from the [OpenAI Speech to text guide Quickstart](https://platform.openai.com/docs/guides/speech-to-text/quickstart):

```python
from openai import OpenAI
client = OpenAI(api_key='sk-1111', base_url='http://localhost:8000/v1')

audio_file = open("/path/to/file/audio.mp3", "rb")
transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
print(transcription.text)
```

### Using Custom Prompts

The `prompt` parameter helps guide Whisper's transcription by providing context, terminology, and formatting preferences. This is especially useful for domain-specific audio like radio communications, medical terminology, or technical jargon.

**Example with curl:**

```shell
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F model="whisper-1" \
  -F file="@audio.mp3" \
  -F prompt="Emergency radio dispatch communications. Common units: MEDIC, ENGINE, TRUCK, LADDER. Radio procedure: COPY, CLEAR, EN ROUTE, ON SCENE."
```

**Example with Python:**

```python
from openai import OpenAI
client = OpenAI(api_key='sk-1111', base_url='http://localhost:8000/v1')

# Recommended prompt for radio dispatch transcription
prompt = """Emergency radio dispatch. CRITICAL: Never repeat. Common units: MEDIC, ENGINE, TRUCK, LADDER, SQUAD, BATTALION. Radio words: COPY, CLEAR, EN ROUTE, ON SCENE. Phonetic: ADAM, BAKER, CHARLES, DAVID, FRANK, GEORGE, KING, LINCOLN, MARY, OCEAN, QUEEN, SAM, VICTOR, X-RAY. Ages: NUMBER YEAR OLD MALE/FEMALE. Medical: GSW, SOB, CPR, AED, MVA. Use periods between statements."""

audio_file = open("/path/to/file/radio_audio.mp3", "rb")
transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    prompt=prompt
)
print(transcription.text)
```

**Recommended Radio Dispatch Prompt:**

For emergency radio dispatch communications, this prompt has been tested and works well:

```
Emergency radio dispatch. CRITICAL: Never repeat. Common units: MEDIC, ENGINE, TRUCK, LADDER, SQUAD, BATTALION. Radio words: COPY, CLEAR, EN ROUTE, ON SCENE. Phonetic: ADAM, BAKER, CHARLES, DAVID, FRANK, GEORGE, KING, LINCOLN, MARY, OCEAN, QUEEN, SAM, VICTOR, X-RAY. Ages: NUMBER YEAR OLD MALE/FEMALE. Medical: GSW, SOB, CPR, AED, MVA. Use periods between statements.
```

This prompt:
- Prevents repetitive hallucinations with "CRITICAL: Never repeat"
- Provides common emergency service terminology
- Includes phonetic alphabet for call signs
- Guides proper formatting for ages and medical terms
- Achieves ~95% accuracy on radio dispatch audio

**Important Notes:**
- The prompt provides **guidance**, not restrictions - Whisper will still transcribe all audio
- Prompts improve accuracy on domain-specific terms and reduce hallucinations
- **Keep prompts under 400 characters** - longer prompts can trigger hallucinations
- Especially helpful with poor audio quality or background noise
- Customize the prompt based on your specific use case (medical, legal, technical, etc.)
- If you see repeated words (hallucinations), try shortening or removing the prompt

Docker support
--------------

You can run the server via docker like so:
```shell
docker compose build
docker compose up
```

Options can be set via `whisper.env`.

### Docker Notes
- The Silero VAD model will be automatically downloaded on first run (~2MB)
- Models are cached in the `hf_home` directory which is mounted as a volume
- GPU support requires NVIDIA Docker runtime and compatible GPU
- For CPU-only Docker, remove the `runtime: nvidia` and `deploy` sections from `docker-compose.yml`

Troubleshooting
---------------

### Silero VAD Download Issues

If you get errors about downloading the Silero VAD model:
1. Ensure you have internet connectivity
2. Check that `torch.hub` has write access to cache directory
3. The model is downloaded from GitHub (snakers4/silero-vad)
4. First run may take 1-2 minutes to download the model

### FFmpeg Not Found

If you get `FileNotFoundError` related to ffmpeg:
```shell
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

### CUDA Out of Memory

If you get CUDA out of memory errors:
1. Use a smaller model (`small`, `base`, or `tiny`)
2. Use CPU mode: `--device cpu`
3. Reduce concurrent requests

### Import Errors

If you get import errors:
```shell
pip install -r requirements.txt --upgrade
```

Make sure you have Python 3.8 or higher:
```shell
python --version
```
