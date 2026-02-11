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
