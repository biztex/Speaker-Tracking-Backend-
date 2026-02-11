# Backend config: audio format expected from frontend
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit
CHANNELS = 1

# Processing
# The buffer and interval values here control how "realtime" the backend feels.
# Previously these were tuned for stability (5s buffer, 2s interval), which
# caused several seconds of lag before the first updates. These values make
# the system much more responsive while keeping segments long enough for
# reliable speaker detection.
BUFFER_DURATION_SEC = 1.5          # total rolling buffer size in seconds (was 5.0)
PROCESS_INTERVAL_SEC = 0.5         # how often we run the pipeline (was 2.0)
MIN_SEGMENT_DURATION_SEC = 0.15    # ignore extremely short blips (was 0.25)
VAD_FRAME_MS = 30
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive filtering of non-speech

# Pyannote.audio configuration for advanced overlapped speaker diarization
# Set PYANNOTE_HF_TOKEN environment variable with your HuggingFace token
# Get token from: https://huggingface.co/settings/tokens
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTE_MIN_SPEAKERS = 1
PYANNOTE_MAX_SPEAKERS = 5
# Minimum audio duration (seconds) before running pyannote diarization
# pyannote works better on longer segments (5-30 seconds)
PYANNOTE_MIN_DURATION_SEC = 2.0
