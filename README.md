# Speaker Tracker (Milestone 2)

Real-time speaker tracking: select how many speakers (2–5), press Start, speak; the backend identifies who is speaking and records time per speaker. Press Stop to get a summary. No audio is recorded or stored.

## Architecture

- **Frontend (React + Vite):** Captures microphone, resamples to 16 kHz, streams PCM to the backend over WebSocket, and displays current speaker and per-speaker times from backend events.
- **Backend (Python + FastAPI):** Receives PCM, runs VAD and speaker embeddings (Resemblyzer) + clustering (KMeans), tracks only the first N distinct speakers, and streams back speaker id and speaking times; on Stop, returns a final report.

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The app connects to `ws://localhost:8000/ws/session` by default. To use another URL, set `VITE_WS_URL` in `frontend/.env` (e.g. `VITE_WS_URL=ws://your-server:8000/ws/session`).

### 3. Use

1. Choose number of speakers (default 2).
2. Click **Start** and allow microphone access.
3. Speak; the UI shows who is speaking and live times.
4. Click **Stop** to end the session and see the final time-based report.

Only the first N voices that appear are counted as Speaker 1, 2, …; noise and other voices are ignored.
