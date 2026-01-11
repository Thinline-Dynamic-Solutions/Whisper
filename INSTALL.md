# Installation Guide - OpenedAI Whisper Server with Silero VAD

This guide provides detailed installation instructions for the Whisper transcription server with automatic tone/silence skipping.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation on Linux](#installation-on-linux)
- [Installation on macOS](#installation-on-macos)
- [Installation on Windows](#installation-on-windows)
- [Docker Installation](#docker-installation)
- [Verification](#verification)
- [Configuration](#configuration)

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4 GB RAM
- FFmpeg for audio processing
- Internet connection (for first-time model download)

### Recommended for GPU Acceleration
- NVIDIA GPU with CUDA support
- CUDA 11.8 or higher
- 8 GB+ GPU memory (for large models)

### Disk Space
- ~2 GB for small model
- ~3 GB for medium model
- ~6 GB for large-v2/large-v3 model
- Additional ~2 MB for Silero VAD model

## Installation on Linux

### Ubuntu/Debian

1. **Update system packages:**
```bash
sudo apt update
sudo apt upgrade
```

2. **Install Python and FFmpeg:**
```bash
sudo apt install python3 python3-pip python3-venv ffmpeg
```

3. **Create virtual environment (recommended):**
```bash
cd /path/to/whisper
python3 -m venv venv
source venv/bin/activate
```

4. **Install Python dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. **(Optional) Install CUDA for GPU support:**
```bash
# Follow NVIDIA's official guide:
# https://developer.nvidia.com/cuda-downloads
```

### CentOS/RHEL/Fedora

1. **Install Python and FFmpeg:**
```bash
sudo dnf install python3 python3-pip ffmpeg
```

2. **Follow steps 3-5 from Ubuntu section above**

## Installation on macOS

### Using Homebrew

1. **Install Homebrew (if not already installed):**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Install Python and FFmpeg:**
```bash
brew install python@3.11 ffmpeg
```

3. **Create virtual environment:**
```bash
cd /path/to/whisper
python3 -m venv venv
source venv/bin/activate
```

4. **Install Python dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Notes for Apple Silicon (M1/M2/M3)
- PyTorch will automatically use Metal Performance Shaders (MPS) for acceleration
- No CUDA support needed on Apple Silicon
- Performance is excellent on M-series chips

## Installation on Windows

### Using Command Prompt or PowerShell

1. **Install Python:**
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify: `python --version`

2. **Install FFmpeg:**
   
   **Option A - Using Chocolatey (recommended):**
   ```powershell
   # Install Chocolatey first (run as Administrator):
   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   
   # Install FFmpeg:
   choco install ffmpeg
   ```
   
   **Option B - Manual installation:**
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Extract to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to System PATH

3. **Create virtual environment:**
```cmd
cd C:\path\to\whisper
python -m venv venv
venv\Scripts\activate
```

4. **Install Python dependencies:**
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

5. **(Optional) Install CUDA for GPU support:**
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Install PyTorch with CUDA:
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Docker Installation

### Prerequisites
- Docker Engine 20.10+
- Docker Compose v2.0+
- (For GPU) NVIDIA Docker runtime

### Standard Installation

1. **Clone/navigate to the whisper directory:**
```bash
cd /path/to/whisper
```

2. **Configure settings (optional):**
```bash
# Edit whisper.env to change model or settings
nano whisper.env
```

3. **Build and run:**
```bash
docker compose build
docker compose up
```

### GPU Support (Linux with NVIDIA GPU)

1. **Install NVIDIA Docker runtime:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. **Verify GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

3. **Run with GPU:**
```bash
docker compose up
```

### CPU-Only Docker

If you don't have a GPU or want to use CPU only:

1. **Edit docker-compose.yml** and remove these lines:
```yaml
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

2. **Run:**
```bash
docker compose up
```

## Verification

### Test the Installation

1. **Start the server:**
```bash
# Without Docker:
python whisper_server.py --model base --host 0.0.0.0 --port 8000

# With Docker:
docker compose up
```

2. **Check health endpoint:**
```bash
curl http://localhost:8000/health
```
Expected response: `{"status":"ok"}`

3. **Test transcription with a sample audio file:**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model="whisper-1" \
  -F file="@/path/to/audio.mp3" \
  -F response_format="text"
```

### Verify Silero VAD

On first transcription, you should see Silero VAD downloading:
```
Downloading: "https://github.com/snakers4/silero-vad/..." to /home/user/.cache/torch/hub/...
```

This is normal and only happens once.

### Check GPU Usage

If using CUDA:
```bash
# In another terminal while server is running:
nvidia-smi

# You should see python process using GPU
```

## Configuration

### Model Selection

Choose model based on your needs:

| Model | Size | VRAM | Speed | Accuracy |
|-------|------|------|-------|----------|
| tiny | ~40 MB | ~1 GB | Fastest | Lowest |
| base | ~75 MB | ~1 GB | Very Fast | Low |
| small | ~245 MB | ~2 GB | Fast | Good |
| medium | ~775 MB | ~5 GB | Moderate | Better |
| large-v2 | ~1.5 GB | ~10 GB | Slow | Best |
| large-v3 | ~1.5 GB | ~10 GB | Slow | Best |

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Python environment
export HF_HOME=/path/to/cache  # Hugging Face cache directory
export TORCH_HOME=/path/to/cache  # PyTorch cache directory

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

### Server Settings

Start server with custom settings:

```bash
# CPU mode, small model
python whisper_server.py --model small --device cpu --host 0.0.0.0 --port 8000

# GPU mode, large model
python whisper_server.py --model large-v3 --device cuda:0 --host 0.0.0.0 --port 8000

# Preload model without starting server (useful for containers)
python whisper_server.py --model large-v3 --preload
```

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'torch'`
```bash
pip install -r requirements.txt --upgrade
```

**Issue:** `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
```bash
# Install FFmpeg (see platform-specific instructions above)
```

**Issue:** `CUDA out of memory`
```bash
# Use smaller model or CPU mode
python whisper_server.py --model small --device cpu
```

**Issue:** Silero VAD download fails
```bash
# Clear torch hub cache and try again
rm -rf ~/.cache/torch/hub/snakers4_silero-vad_*
```

**Issue:** Port 8000 already in use
```bash
# Use different port
python whisper_server.py --port 8001
```

### Getting Help

- Check logs for error messages
- Ensure all dependencies are installed
- Verify FFmpeg is in PATH: `ffmpeg -version`
- Verify Python version: `python --version` (should be 3.8+)
- Check CUDA installation: `nvidia-smi` (if using GPU)

## Next Steps

After successful installation:
1. Read the [README.md](README.md) for API usage examples
2. Test with your audio files
3. Configure custom prompts for better accuracy
4. Set up as a system service (optional)

## System Service Setup (Optional)

### Linux (systemd)

Create `/etc/systemd/system/whisper-server.service`:

```ini
[Unit]
Description=Whisper Transcription Server
After=network.target

[Service]
Type=simple
User=whisper
WorkingDirectory=/opt/whisper
ExecStart=/opt/whisper/venv/bin/python whisper_server.py --model large-v3 --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable whisper-server
sudo systemctl start whisper-server
```

### macOS (launchd)

Create `~/Library/LaunchAgents/com.whisper.server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.whisper.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/venv/bin/python</string>
        <string>/path/to/whisper_server.py</string>
        <string>--model</string>
        <string>large-v3</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load service:
```bash
launchctl load ~/Library/LaunchAgents/com.whisper.server.plist
```

### Windows (NSSM)

1. Download [NSSM](https://nssm.cc/download)
2. Install service:
```cmd
nssm install WhisperServer "C:\path\to\venv\Scripts\python.exe" "C:\path\to\whisper_server.py --model large-v3"
nssm start WhisperServer
```

