# Speaker Tracker Backend

FastAPI server that performs real-time speaker tracking over a WebSocket. It receives raw PCM audio (16 kHz, 16-bit mono), runs VAD and speaker embedding + clustering, and streams back who is speaking and per-speaker timing. No audio is stored.

**Diarization stack:** Silero VAD (via `torch.hub`) for speech activity + SpeechBrain speaker embeddings with KMeans clustering. This keeps the backend lighter than a full pyannote.audio pipeline while still providing good accuracy for multi-speaker sessions.

## Setup

**Python 3.10+** recommended.

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Speaker Diarization (Silero + SpeechBrain)

The backend uses:

- **Silero VAD** (loaded via `torch.hub`) for speech activity detection on 16 kHz mono audio.
- **SpeechBrain** (`speechbrain/spkrec-ecapa-voxceleb`) to extract speaker embeddings for each detected speech segment.
- **KMeans clustering** to assign segments to up to N speakers (N is chosen by the client).

This stack:

- ✅ Avoids large pyannote.audio diarization models (~500MB downloads).
- ✅ Stays relatively lightweight for CPU-only deployments (e.g. Railway).
- ✅ Provides good accuracy for typical multi-speaker calls.

## Run

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
(Using `python -m uvicorn` ensures the venv’s uvicorn is used.)

- Health: [http://localhost:8000/health](http://localhost:8000/health)
- WebSocket: `ws://localhost:8000/ws/session`

## Protocol

1. **Connect** to `ws://localhost:8000/ws/session`.
2. **Start session:** send JSON `{"action": "start", "num_speakers": 2}` (default 2, max 5).
3. Server replies with `{"event": "started", "num_speakers": 2}`.
4. **Stream audio:** send binary PCM frames (16-bit little-endian, 16 kHz, mono). No other format supported.
5. Server sends periodic JSON updates: `{"event": "speaker", "currentSpeakerId": 0, "speakingTimeMs": {"0": 1200, "1": 500}}`.
6. **End session:** send JSON `{"action": "stop"}`. Server replies with `{"event": "report", "speakers": [{"id": 0, "totalTimeMs": 45000}, {"id": 1, "totalTimeMs": 12000}]}`.

Only the first N distinct speakers (by voice embedding) are tracked; extra voices/noise are ignored.

## Speaker Diarization Methods

Currently a single method is used:

1. **Silero VAD + SpeechBrain embeddings + KMeans**:
   - ✅ Good accuracy for typical multi-speaker conversations
   - ✅ Much lighter than pyannote.audio-based diarization
   - ❌ Does not provide full overlapped-speech segmentation like pyannote.audio
