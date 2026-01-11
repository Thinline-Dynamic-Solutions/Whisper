# Dependencies Explained

This document explains each dependency in `requirements.txt` and why it's needed.

## Core Dependencies

### fastapi>=0.104.0
**Purpose:** Modern web framework for building APIs  
**Why needed:** Provides the HTTP server and API endpoints for transcription requests  
**Size:** ~70 MB (with dependencies)  
**License:** MIT

### uvicorn[standard]>=0.24.0
**Purpose:** ASGI web server  
**Why needed:** Runs the FastAPI application and handles HTTP requests  
**Note:** `[standard]` includes recommended dependencies for production use  
**Size:** ~10 MB  
**License:** BSD-3-Clause

### pydantic>=2.0.0
**Purpose:** Data validation library  
**Why needed:** FastAPI uses it for request/response validation and serialization  
**Size:** ~15 MB  
**License:** MIT

### python-multipart>=0.0.6
**Purpose:** Multipart form data parser  
**Why needed:** Required for file uploads (audio files sent to API)  
**Size:** <1 MB  
**License:** Apache-2.0

## Whisper & AI Dependencies

### openai-whisper>=20231117
**Purpose:** OpenAI's Whisper speech recognition model  
**Why needed:** Core transcription engine that converts audio to text  
**Size:** ~5 MB (library only, models downloaded separately)  
**Models:**
- tiny: ~75 MB
- small: ~250 MB
- medium: ~775 MB
- large-v2/v3: ~1.5 GB each  
**License:** MIT

### torch>=2.0.0
**Purpose:** PyTorch deep learning framework  
**Why needed:** Required by Whisper and Silero VAD for neural network inference  
**Size:** ~800 MB (CPU), ~2 GB (with CUDA)  
**Features:**
- CPU inference (always available)
- CUDA GPU acceleration (if NVIDIA GPU present)
- Metal Performance Shaders (Apple Silicon)  
**License:** BSD-3-Clause

### torchaudio>=2.0.0
**Purpose:** Audio processing library for PyTorch  
**Why needed:** Required by Silero VAD for audio tensor operations  
**Size:** ~10 MB  
**License:** BSD-2-Clause

## Feature: Silero VAD (Automatic Download)

### Silero Voice Activity Detection
**Purpose:** Detects speech vs non-speech in audio  
**Why needed:** Automatically skips alert tones, beeps, and silence at start of recordings  
**How installed:** Downloaded automatically via `torch.hub.load()` on first transcription  
**Source:** https://github.com/snakers4/silero-vad  
**Size:** ~2 MB  
**Cache location:** `~/.cache/torch/hub/snakers4_silero-vad_*`  
**License:** MIT

**Benefits:**
- Improves transcription accuracy by removing non-speech audio
- Especially useful for radio dispatch and pager recordings
- Reduces hallucinations caused by alert tones
- No manual configuration required

## Optional System Dependencies

### FFmpeg (Required, but not in requirements.txt)
**Purpose:** Audio/video processing tool  
**Why needed:** Converts various audio formats to 16kHz mono WAV for processing  
**Installation:**
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: `choco install ffmpeg` or manual download  
**License:** LGPL/GPL (depending on build configuration)

## Total Disk Space Requirements

### Without models:
- Python packages: ~900 MB (CPU) or ~2.2 GB (GPU)
- FFmpeg: ~100 MB

### With models (one-time download):
- tiny model: +75 MB
- small model: +250 MB
- medium model: +775 MB
- large-v3 model: +1.5 GB
- Silero VAD: +2 MB

### Typical Installation Sizes:
- Minimal (CPU, tiny model): ~1 GB
- Recommended (CPU, large-v3): ~2.5 GB
- Full (GPU, large-v3): ~3.8 GB

## Memory Requirements

### Runtime Memory Usage:
- Base application: ~200 MB
- tiny model: +1 GB
- small model: +2 GB
- medium model: +5 GB
- large-v2/v3 model: +10 GB
- Silero VAD: +50 MB

### Recommended System RAM:
- tiny/small: 4 GB+
- medium: 8 GB+
- large-v2/v3: 16 GB+ (32 GB recommended)

### GPU VRAM Requirements (if using CUDA):
- tiny: 1 GB
- small: 2 GB
- medium: 5 GB
- large-v2/v3: 10 GB

## Version Constraints Explained

### Why minimum versions?
- `>=0.104.0` ensures critical bug fixes and features are available
- Older versions may lack security patches or compatibility
- Tested and verified to work with these versions

### Can I use newer versions?
- Yes! The `>=` constraint allows newer versions
- Whisper server is designed to work with future versions
- Breaking changes in major versions are rare

### Upgrading dependencies:
```bash
# Upgrade all to latest compatible versions
pip install -r requirements.txt --upgrade

# Upgrade specific package
pip install --upgrade fastapi
```

## Security Considerations

### Regular Updates
Keep dependencies updated for security patches:
```bash
pip list --outdated
pip install -r requirements.txt --upgrade
```

### Known Security Notes:
- All dependencies use well-maintained, popular packages
- FastAPI/Uvicorn have excellent security track record
- PyTorch is industry-standard ML framework
- OpenAI Whisper is official OpenAI release

### Network Access
Dependencies that require internet access:
- **Installation time:** All packages (from PyPI)
- **First run:** Silero VAD model (from GitHub via torch.hub)
- **Runtime:** None (fully offline after setup)

## Dependency Tree

```
whisper_server.py
├── fastapi (Web framework)
│   ├── pydantic (Validation)
│   └── python-multipart (File uploads)
├── uvicorn (Web server)
├── openai-whisper (Transcription)
│   └── torch (ML framework)
│       └── torchaudio (Audio processing)
└── Silero VAD (via torch.hub)
    └── torch (Already installed)
```

## Alternative Configurations

### Minimal Installation (CPU only, small model)
All required packages are in `requirements.txt` - no alternatives needed.

### GPU-Optimized Installation (NVIDIA CUDA)
```bash
# Install CUDA-enabled PyTorch explicitly
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements.txt
```

### Apple Silicon Optimized (M1/M2/M3)
Standard `requirements.txt` works perfectly - PyTorch automatically uses Metal Performance Shaders.

## Troubleshooting Dependencies

### Import Errors
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Clear cache and reinstall
pip cache purge
pip install -r requirements.txt
```

### Version Conflicts
```bash
# Check installed versions
pip list

# Check for conflicts
pip check
```

### Torch Hub Cache Issues
```bash
# Clear Silero VAD cache
rm -rf ~/.cache/torch/hub/snakers4_silero-vad_*

# Model will re-download on next transcription
```

## License Summary

All dependencies use permissive open-source licenses:
- MIT: fastapi, pydantic, openai-whisper, Silero VAD
- BSD: uvicorn, torch, torchaudio
- Apache-2.0: python-multipart

No GPL dependencies - safe for commercial use.

