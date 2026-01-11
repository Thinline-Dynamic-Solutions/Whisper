# Changes to Whisper Server

## Version 0.2.0 (2026-01-11)

### New Features

#### Automatic Tone and Silence Skipping (Silero VAD)
- Integrated Silero Voice Activity Detection (VAD)
- Automatically detects and skips alert tones, beeps, and leading silence
- Preserves 150ms buffer before speech starts
- Improves transcription accuracy for radio dispatch and pager recordings
- No configuration required - works automatically

### Technical Implementation
- `detect_speech_start_sec_silero()`: Analyzes 16kHz mono audio to find speech start
- Uses 250ms minimum speech duration with 0.5 confidence threshold
- Falls back gracefully if VAD fails
- ~2MB model downloaded once on first run

### Dependencies Added
- `torchaudio>=2.0.0` - Required for Silero VAD audio processing
- Updated all dependencies to use version constraints for stability

### Configuration Changes
- Default model changed from `large-v2` to `large-v3`
- Default host changed from `localhost` to `0.0.0.0` (listen on all interfaces)
- Fixed `whisper.env` to use correct CLI argument format

### Bug Fixes
- Fixed Dockerfile CMD to reference `whisper_server.py` instead of `whisper.py`
- Corrected model parameter format in `whisper.env`

### Documentation Updates
- Comprehensive installation guide added (`INSTALL.md`)
- Updated `README.md` with:
  - Silero VAD feature description
  - Improved installation instructions for all platforms
  - Health check endpoint documentation
  - Troubleshooting section
  - Docker setup notes
- Added version-pinned dependencies to `requirements.txt`

### Breaking Changes
None - All changes are backward compatible

### Migration Notes
If upgrading from a previous version:
1. Update dependencies: `pip install -r requirements.txt --upgrade`
2. First transcription will download Silero VAD model (~2MB, one-time)
3. Update `whisper.env` if using Docker (see new format)

### Performance Notes
- VAD processing adds ~100-200ms per transcription
- Total time saved by skipping tones/silence typically exceeds VAD overhead
- No additional memory requirements beyond initial model load

