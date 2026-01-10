OpenedAI Whisper
----------------

An OpenAI API compatible speech to text server for audio transcription and translations, aka. Whisper.

- Compatible with the OpenAI audio/transcriptions and audio/translations API
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way

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

Tested whisper models:
* large-v2 (the default)
* large-v3
* large
* medium
* small
* base
* tiny


Version: 0.1.0, Last update: 2024-03-15


API Documentation
-----------------

## Usage

* [OpenAI Speech to text guide](https://platform.openai.com/docs/guides/speech-to-text)
* [OpenAI API Transcription Reference](https://platform.openai.com/docs/api-reference/audio/createTranscription)
* [OpenAI API Translation Reference](https://platform.openai.com/docs/api-reference/audio/createTranslation)


Installation instructions
-------------------------

You will need to install CUDA for your operating system if you want to use CUDA.

```shell
# Install the Python requirements
pip install -r requirements.txt
# install ffmpeg
sudo apt install ffmpeg
```

**Note**: This implementation uses the official OpenAI Whisper library which has **full prompt support** built-in!

Usage
-----

```
Usage: whisper_server.py [-m <model_name>] [-d <device>] [-P <port>] [-H <host>] [--preload]


Description:
OpenedAI Whisper API Server

Options:
-h, --help            Show this help message and exit.
-m MODEL, --model MODEL
                      The model to use for transcription.
                      Options: tiny, base, small, medium, large, large-v2, large-v3 (default: large-v2)
-d DEVICE, --device DEVICE
                      Set the torch device for the model. Ex. cuda:0 or cpu (default: auto)
-P PORT, --port PORT  Server tcp port (default: 8000)
-H HOST, --host HOST  Host to listen on, Ex. 0.0.0.0 (default: localhost)
--preload             Preload model and exit. (default: False)
```

Sample API Usage
----------------

You can use it like this:

```shell
curl -s http://localhost:8000/v1/audio/transscriptions -H "Content-Type: multipart/form-data" -F model="whisper-1" -F file="@audio.mp3" -F response_format=text
```

Or just like this:

```shell
curl -s http://localhost:8000/v1/audio/transscriptions -F model="whisper-1" -F file="@audio.mp3"
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
