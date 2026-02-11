# Speaker Tracker Backend

FastAPI server that performs real-time speaker tracking over a WebSocket. It receives raw PCM audio (16 kHz, 16-bit mono), runs VAD and speaker embedding + clustering, and streams back who is speaking and per-speaker timing. No audio is stored.

**Now includes advanced overlapped speaker diarization** using `pyannote.audio` - can detect when multiple speakers are talking simultaneously!

## Setup

**Python 3.10+** recommended.

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Advanced Speaker Diarization (pyannote.audio)

For **overlapped speaker detection** (detecting when two people speak at the same time), you need to set up `pyannote.audio`:

1. **Get a HuggingFace token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token (read access is sufficient)
   - Accept the terms for the `pyannote/speaker-diarization-3.1` model

2. **Set the token as an environment variable:**
   ```bash
   # Windows PowerShell
   $env:PYANNOTE_HF_TOKEN="your_token_here"
   
   # Windows CMD
   set PYANNOTE_HF_TOKEN=your_token_here
   
   # macOS/Linux
   export PYANNOTE_HF_TOKEN="your_token_here"
   ```

3. **The backend will automatically use pyannote** when the token is set. If not set, it falls back to Resemblyzer + KMeans clustering (which doesn't handle overlapped speech as well).

**Note:** PyTorch is included in requirements. If you have a GPU, it will be used automatically. On first run, pyannote will download the model (~500MB).

### Fallback Mode (Resemblyzer)

If pyannote is not available, the system uses Resemblyzer + KMeans clustering. Resemblyzer uses PyTorch. If you don't have a GPU, the CPU build is used automatically. On first run, Resemblyzer may download a small pretrained model.

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

The backend supports two methods (automatically selected):

1. **pyannote.audio** (when `PYANNOTE_HF_TOKEN` is set):
   - ✅ **Supports overlapped speech** - can detect when multiple speakers talk simultaneously
   - ✅ More accurate speaker separation
   - ✅ Better handling of speaker changes
   - ⚠️ Requires HuggingFace token and downloads ~500MB model on first run
   - ⚠️ Slightly higher CPU/GPU usage

2. **Resemblyzer + KMeans** (fallback):
   - ✅ No external token required
   - ✅ Lighter weight
   - ❌ Cannot detect overlapped speech (only one speaker at a time)
   - ❌ Less accurate for rapid speaker changes
