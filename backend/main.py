"""
FastAPI backend: WebSocket for streaming audio, speaker tracking (fixed N speakers).
No audio is stored; process in memory and emit speaker/timing updates.
"""
import asyncio
import json
import logging
from collections import deque
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import (
    SAMPLE_RATE,
    BYTES_PER_SAMPLE,
    BUFFER_DURATION_SEC,
    PROCESS_INTERVAL_SEC,
)
from speaker_pipeline import SpeakerPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Speaker Tracker API",
    description="Real-time speaker tracking over WebSocket. Send PCM 16kHz 16-bit mono.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def buffer_duration_bytes() -> int:
    return int(SAMPLE_RATE * BUFFER_DURATION_SEC * BYTES_PER_SAMPLE)


@app.websocket("/ws/session")
async def session_websocket(websocket: WebSocket):
    await websocket.accept()
    buffer: deque[bytes] = deque()
    buffer_byte_count = 0
    pipeline: Optional[SpeakerPipeline] = None
    num_speakers = 2
    process_task: Optional[asyncio.Task] = None
    closed = False

    async def process_loop():
        nonlocal buffer_byte_count, buffer, pipeline, closed
        target_bytes = buffer_duration_bytes()
        while not closed and pipeline is not None:
            await asyncio.sleep(PROCESS_INTERVAL_SEC)
            if closed or pipeline is None:
                break
            if buffer_byte_count < target_bytes // 2:
                continue
            # Take up to target_bytes from buffer (FIFO)
            chunks: list[bytes] = []
            total = 0
            while buffer and total < target_bytes:
                chunk = buffer.popleft()
                buffer_byte_count -= len(chunk)
                chunks.append(chunk)
                total += len(chunk)
            if not chunks:
                continue
            pcm = b"".join(chunks)
            try:
                update = pipeline.process(pcm)
                await websocket.send_json(update)
            except Exception as e:
                logger.warning("Pipeline process error: %s", e)

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive(), timeout=300.0)
            except asyncio.TimeoutError:
                logger.info("WebSocket receive timeout")
                break
            if raw.get("type") == "websocket.disconnect":
                break
            if "text" in raw:
                msg = json.loads(raw["text"])
                action = msg.get("action")
                if action == "start":
                    num_speakers = max(1, min(5, int(msg.get("num_speakers", 2))))
                    pipeline = SpeakerPipeline(num_speakers=num_speakers)
                    buffer.clear()
                    buffer_byte_count = 0
                    if process_task is not None:
                        process_task.cancel()
                    process_task = asyncio.create_task(process_loop())
                    await websocket.send_json({
                        "event": "started",
                        "num_speakers": num_speakers,
                    })
                elif action == "stop":
                    closed = True
                    if process_task is not None:
                        process_task.cancel()
                        try:
                            await process_task
                        except asyncio.CancelledError:
                            pass
                    if pipeline is not None:
                        # Flush remaining buffer once
                        if buffer_byte_count > 0:
                            all_chunks = []
                            while buffer:
                                c = buffer.popleft()
                                buffer_byte_count -= len(c)
                                all_chunks.append(c)
                            if all_chunks:
                                try:
                                    pipeline.process(b"".join(all_chunks))
                                except Exception as e:
                                    logger.warning("Final process error: %s", e)
                        report = pipeline.get_report()
                        await websocket.send_json(report)
                    break
            elif "bytes" in raw:
                chunk = raw["bytes"]
                if pipeline is not None and chunk:
                    buffer.append(chunk)
                    buffer_byte_count += len(chunk)
                    # Cap buffer size
                    max_b = buffer_duration_bytes()
                    while buffer_byte_count > max_b and buffer:
                        old = buffer.popleft()
                        buffer_byte_count -= len(old)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        closed = True
        if process_task is not None and not process_task.done():
            process_task.cancel()
            try:
                await process_task
            except asyncio.CancelledError:
                pass


@app.get("/health")
async def health():
    return {"status": "ok"}
