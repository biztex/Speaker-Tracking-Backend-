# Backend config: audio format expected from frontend
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit
CHANNELS = 1

# Processing
BUFFER_DURATION_SEC = 5.0
PROCESS_INTERVAL_SEC = 2.0
MIN_SEGMENT_DURATION_SEC = 0.25  # ignore very short segments (noise)
VAD_FRAME_MS = 30
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive filtering of non-speech
