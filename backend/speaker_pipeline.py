"""
Speaker tracking pipeline: VAD -> segment -> embed -> cluster (fixed N speakers).
Only the first N distinct speakers are tracked; noise/others are ignored.
"""
import struct
import numpy as np
from typing import List, Tuple, Optional

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

try:
    from resemblyzer import VoiceEncoder
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False

from config import (
    SAMPLE_RATE,
    MIN_SEGMENT_DURATION_SEC,
    VAD_FRAME_MS,
    VAD_AGGRESSIVENESS,
)


def pcm_bytes_to_float(pcm: bytes) -> np.ndarray:
    """Convert 16-bit little-endian PCM bytes to float32 in [-1, 1]."""
    n = len(pcm) // 2
    samples = struct.unpack(f"<{n}h", pcm)
    return np.array(samples, dtype=np.float32) / 32768.0


def resample_to_16k(audio_float: np.ndarray, from_sr: int) -> np.ndarray:
    """Resample to 16000 Hz using linear interpolation."""
    if from_sr == SAMPLE_RATE:
        return audio_float
    n = len(audio_float)
    out_len = int(n * SAMPLE_RATE / from_sr)
    if out_len <= 0:
        return np.array([], dtype=np.float32)
    x_old = np.linspace(0, n - 1, n)
    x_new = np.linspace(0, n - 1, out_len)
    return np.interp(x_new, x_old, audio_float).astype(np.float32)


def float_to_pcm_bytes(arr: np.ndarray) -> bytes:
    """Convert float32 [-1,1] to 16-bit little-endian PCM."""
    arr = np.clip(arr, -1.0, 1.0)
    samples = (arr * 32767).astype(np.int16)
    return samples.tobytes()


def _get_speech_segments_webrtc(
    pcm_bytes: bytes,
    sample_rate: int,
    frame_duration_ms: int,
    aggressiveness: int,
) -> List[Tuple[float, float]]:
    """VAD using webrtcvad (requires C build on Windows)."""
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = sample_rate * frame_duration_ms // 1000 * 2
    n = len(pcm_bytes) // frame_len
    segments: List[Tuple[float, float]] = []
    in_speech = False
    start_sec = 0.0
    for i in range(n):
        frame = pcm_bytes[i * frame_len : (i + 1) * frame_len]
        if len(frame) < frame_len:
            break
        is_speech = vad.is_speech(frame, sample_rate)
        t = (i + 1) * frame_duration_ms / 1000.0
        if is_speech and not in_speech:
            in_speech = True
            start_sec = i * frame_duration_ms / 1000.0
        elif not is_speech and in_speech:
            in_speech = False
            if t - start_sec >= MIN_SEGMENT_DURATION_SEC:
                segments.append((start_sec, t))
    if in_speech and (n * frame_duration_ms / 1000.0 - start_sec) >= MIN_SEGMENT_DURATION_SEC:
        segments.append((start_sec, n * frame_duration_ms / 1000.0))
    return segments


def _get_speech_segments_energy(
    audio_float: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int,
    energy_threshold: float = 0.01,
) -> List[Tuple[float, float]]:
    """Pure-Python energy-based VAD (no C compiler required)."""
    frame_len = sample_rate * frame_duration_ms // 1000
    if frame_len <= 0:
        return []
    n_frames = len(audio_float) // frame_len
    segments: List[Tuple[float, float]] = []
    in_speech = False
    start_sec = 0.0
    for i in range(n_frames):
        frame = audio_float[i * frame_len : (i + 1) * frame_len]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        is_speech = rms > energy_threshold
        t = (i + 1) * frame_duration_ms / 1000.0
        if is_speech and not in_speech:
            in_speech = True
            start_sec = i * frame_duration_ms / 1000.0
        elif not is_speech and in_speech:
            in_speech = False
            if t - start_sec >= MIN_SEGMENT_DURATION_SEC:
                segments.append((start_sec, t))
    if in_speech and (n_frames * frame_duration_ms / 1000.0 - start_sec) >= MIN_SEGMENT_DURATION_SEC:
        segments.append((start_sec, n_frames * frame_duration_ms / 1000.0))
    return segments


def get_speech_segments(
    pcm_bytes: bytes,
    sample_rate: int,
    frame_duration_ms: int = 30,
    aggressiveness: int = 2,
) -> List[Tuple[float, float]]:
    """
    Run VAD and return list of (start_sec, end_sec) for speech segments.
    Uses webrtcvad if available, else pure-Python energy VAD (works without C compiler).
    """
    if WEBRTCVAD_AVAILABLE and sample_rate in (8000, 16000, 32000):
        return _get_speech_segments_webrtc(
            pcm_bytes, sample_rate, frame_duration_ms, aggressiveness
        )
    audio_float = pcm_bytes_to_float(pcm_bytes)
    return _get_speech_segments_energy(
        audio_float, sample_rate, frame_duration_ms, energy_threshold=0.008
    )


class SpeakerPipeline:
    """
    Tracks up to num_speakers. First N distinct voices become Speaker 0..N-1.
    Extra voices (noise, others) are not assigned.
    """

    def __init__(self, num_speakers: int = 2):
        self.num_speakers = max(1, min(5, num_speakers))
        self.sample_rate = SAMPLE_RATE
        self._encoder: Optional[any] = None
        if RESEMBLYZER_AVAILABLE:
            self._encoder = VoiceEncoder()
        # Cumulative speaking time per speaker (ms)
        self.speaking_time_ms: dict[int, int] = {i: 0 for i in range(self.num_speakers)}
        # Cluster label -> speaker id (filled by first appearance)
        self._cluster_to_speaker: dict[int, int] = {}
        self._next_speaker_id = 0
        self._last_current_speaker: Optional[int] = None

    def _embed_segments(
        self,
        audio_float: np.ndarray,
        segments: List[Tuple[float, float]],
    ) -> List[Tuple[float, float, np.ndarray]]:
        """Return (start_sec, end_sec, embedding) for each segment."""
        if not RESEMBLYZER_AVAILABLE or self._encoder is None:
            return []
        out: List[Tuple[float, float, np.ndarray]] = []
        for start_sec, end_sec in segments:
            start_idx = int(start_sec * self.sample_rate)
            end_idx = int(end_sec * self.sample_rate)
            if end_idx <= start_idx or end_idx > len(audio_float):
                continue
            seg = audio_float[start_idx:end_idx]
            try:
                emb = self._encoder.embed_utterance(seg)
                out.append((start_sec, end_sec, emb))
            except Exception:
                continue
        return out

    def process(
        self,
        pcm_bytes: bytes,
        stream_start_offset_sec: float = 0.0,
    ) -> dict:
        """
        Process a chunk of PCM (16kHz 16-bit mono). Returns update dict:
        - currentSpeakerId: int | null
        - speakingTimeMs: { "0": ms, "1": ms, ... }
        - event: "speaker"
        """
        from sklearn.cluster import KMeans

        audio_float = pcm_bytes_to_float(pcm_bytes)
        segments = get_speech_segments(
            pcm_bytes,
            self.sample_rate,
            frame_duration_ms=VAD_FRAME_MS,
            aggressiveness=VAD_AGGRESSIVENESS,
        )
        if not segments:
            return {
                "event": "speaker",
                "currentSpeakerId": self._last_current_speaker,
                "speakingTimeMs": {str(i): self.speaking_time_ms[i] for i in range(self.num_speakers)},
            }

        embedded = self._embed_segments(audio_float, segments)
        if not embedded:
            # No embeddings (Resemblyzer not installed): count all speech as speaker 0
            for start_sec, end_sec in segments:
                dur_ms = int((end_sec - start_sec) * 1000)
                self.speaking_time_ms[0] = self.speaking_time_ms.get(0, 0) + dur_ms
            self._last_current_speaker = 0
            return self._current_state()

        # Cluster into num_speakers (or fewer if we have fewer segments)
        n_clusters = min(self.num_speakers, len(embedded))
        if n_clusters < 1:
            return self._current_state()

        X = np.stack([e[2] for e in embedded])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Map cluster id -> speaker id by first appearance (order of first segment)
        cluster_first_sec: dict[int, float] = {}
        for (start_sec, end_sec, _), label in zip(embedded, labels):
            if label not in cluster_first_sec:
                cluster_first_sec[label] = start_sec
        # Sort clusters by first appearance
        order = sorted(cluster_first_sec.keys(), key=lambda k: cluster_first_sec[k])
        for idx, c in enumerate(order):
            if idx >= self.num_speakers:
                break
            if c not in self._cluster_to_speaker:
                if self._next_speaker_id < self.num_speakers:
                    self._cluster_to_speaker[c] = self._next_speaker_id
                    self._next_speaker_id += 1
                else:
                    self._cluster_to_speaker[c] = 0  # fold extra into speaker 0 (noise)

        # Accumulate time and pick current speaker (last segment)
        current_speaker: Optional[int] = None
        for (start_sec, end_sec, _), label in zip(embedded, labels):
            speaker_id = self._cluster_to_speaker.get(label, 0)
            if speaker_id < self.num_speakers:
                dur_ms = int((end_sec - start_sec) * 1000)
                self.speaking_time_ms[speaker_id] = self.speaking_time_ms.get(speaker_id, 0) + dur_ms
                current_speaker = speaker_id
        self._last_current_speaker = current_speaker

        return self._current_state()

    def _current_state(self) -> dict:
        return {
            "event": "speaker",
            "currentSpeakerId": self._last_current_speaker,
            "speakingTimeMs": {str(i): self.speaking_time_ms.get(i, 0) for i in range(self.num_speakers)},
        }

    def get_report(self) -> dict:
        """Final report: list of { id, totalTimeMs } for each speaker."""
        return {
            "event": "report",
            "speakers": [
                {"id": i, "totalTimeMs": self.speaking_time_ms.get(i, 0)}
                for i in range(self.num_speakers)
            ],
        }

    def reset(self) -> None:
        self.speaking_time_ms = {i: 0 for i in range(self.num_speakers)}
        self._cluster_to_speaker = {}
        self._next_speaker_id = 0
        self._last_current_speaker = None
