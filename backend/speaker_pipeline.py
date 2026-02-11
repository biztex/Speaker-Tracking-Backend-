"""
Speaker tracking pipeline: VAD -> segment -> embed -> cluster (fixed N speakers).
Only the first N distinct speakers are tracked; noise/others are ignored.

Now includes pyannote.audio support for advanced overlapped speaker diarization.
"""
import struct
import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque

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

try:
    import torch
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

from config import (
    SAMPLE_RATE,
    MIN_SEGMENT_DURATION_SEC,
    VAD_FRAME_MS,
    VAD_AGGRESSIVENESS,
    PYANNOTE_MODEL,
    PYANNOTE_MIN_SPEAKERS,
    PYANNOTE_MAX_SPEAKERS,
    PYANNOTE_MIN_DURATION_SEC,
)

logger = logging.getLogger(__name__)


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


class PyannotePipeline:
    """
    Advanced speaker diarization using pyannote.audio.
    Supports overlapped speech detection - can identify when multiple speakers
    are active simultaneously.
    """
    
    def __init__(self, num_speakers: int = 2):
        self.num_speakers = max(1, min(5, num_speakers))
        self.sample_rate = SAMPLE_RATE
        self._pipeline: Optional[Pipeline] = None
        self._device = None
        
        # Accumulated audio buffer for processing
        self._audio_buffer: deque[bytes] = deque()
        self._buffer_duration_sec = 0.0
        
        # Speaker tracking state
        self.speaking_time_ms: Dict[int, int] = {i: 0 for i in range(self.num_speakers)}
        self._speaker_label_map: Dict[str, int] = {}  # pyannote label -> speaker ID
        self._last_current_speakers: List[int] = []  # Can have multiple active speakers
        
        if PYANNOTE_AVAILABLE:
            self._initialize_pipeline()
        else:
            logger.warning("pyannote.audio not available. Install with: pip install pyannote.audio")
    
    def _initialize_pipeline(self):
        """Initialize pyannote diarization pipeline."""
        try:
            hf_token = os.getenv("PYANNOTE_HF_TOKEN") or os.getenv("HF_TOKEN")
            if not hf_token:
                logger.warning(
                    "PYANNOTE_HF_TOKEN not set. pyannote models require HuggingFace token. "
                    "Get one at https://huggingface.co/settings/tokens"
                )
                return
            
            # Determine device
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Initializing pyannote pipeline on {self._device}")
            
            # Load pipeline
            # Support both old (use_auth_token) and new (token) API versions
            try:
                # Try new API first (pyannote >= 3.1)
                self._pipeline = Pipeline.from_pretrained(
                    PYANNOTE_MODEL,
                    token=hf_token
                ).to(self._device)
            except TypeError:
                # Fallback to old API (pyannote < 3.1)
                self._pipeline = Pipeline.from_pretrained(
                    PYANNOTE_MODEL,
                    use_auth_token=hf_token
                ).to(self._device)
            
            # Set min/max speakers
            self._pipeline.instantiate({
                "clustering": {
                    "min_cluster_size": PYANNOTE_MIN_SPEAKERS,
                    "threshold": None,  # Auto-determine based on num_speakers
                }
            })
            
            logger.info("Pyannote pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pyannote pipeline: {e}")
            self._pipeline = None
    
    def _map_speaker_label(self, pyannote_label: str) -> int:
        """
        Map pyannote speaker label (e.g., 'SPEAKER_00') to numeric speaker ID.
        Creates mapping on first appearance, respecting num_speakers limit.
        """
        if pyannote_label in self._speaker_label_map:
            return self._speaker_label_map[pyannote_label]
        
        # New speaker - assign next available ID
        if len(self._speaker_label_map) < self.num_speakers:
            speaker_id = len(self._speaker_label_map)
            self._speaker_label_map[pyannote_label] = speaker_id
            if speaker_id not in self.speaking_time_ms:
                self.speaking_time_ms[speaker_id] = 0
            return speaker_id
        
        # Already at max speakers - map to closest existing speaker
        # For simplicity, map to speaker 0 (could be improved with similarity matching)
        return 0
    
    def _process_diarization(self, diarization, chunk_start_offset: float = 0.0):
        """
        Process pyannote diarization output and update speaking times.
        Handles overlapping segments correctly - each speaker gets their own duration.
        
        Args:
            diarization: pyannote.core.Annotation object
            chunk_start_offset: Time offset for this chunk (for streaming)
        """
        if not self._pipeline:
            return
        
        current_speakers = set()
        
        # Process each segment in the diarization
        for segment, track, label in diarization.itertracks(yield_label=True):
            start_sec = segment.start + chunk_start_offset
            end_sec = segment.end + chunk_start_offset
            duration_ms = int((end_sec - start_sec) * 1000)
            
            if duration_ms <= 0:
                continue
            
            # Map pyannote label to speaker ID
            speaker_id = self._map_speaker_label(label)
            
            # Accumulate speaking time (overlapping segments are counted separately)
            if speaker_id < self.num_speakers:
                self.speaking_time_ms[speaker_id] = self.speaking_time_ms.get(speaker_id, 0) + duration_ms
                current_speakers.add(speaker_id)
        
        # Update current speakers (can be multiple due to overlap)
        self._last_current_speakers = sorted(list(current_speakers))
    
    def process(self, pcm_bytes: bytes, stream_start_offset_sec: float = 0.0) -> dict:
        """
        Process audio chunk with pyannote diarization.
        Accumulates audio until minimum duration is reached, then processes.
        
        Args:
            pcm_bytes: PCM audio bytes (16kHz, 16-bit, mono)
            stream_start_offset_sec: Offset for this chunk (for streaming)
        
        Returns:
            Update dict with currentSpeakerId(s) and speakingTimeMs
        """
        if not self._pipeline:
            # Fallback: return current state
            return self._current_state()
        
        # Add to buffer
        self._audio_buffer.append(pcm_bytes)
        chunk_duration = len(pcm_bytes) / (self.sample_rate * 2)  # 2 bytes per sample
        self._buffer_duration_sec += chunk_duration
        
        # Process if we have enough audio
        if self._buffer_duration_sec >= PYANNOTE_MIN_DURATION_SEC:
            # Concatenate buffer
            full_audio = b"".join(self._audio_buffer)
            audio_float = pcm_bytes_to_float(full_audio)
            
            # Convert to torch tensor (pyannote expects (1, T) shape)
            waveform = torch.tensor(audio_float, dtype=torch.float32).unsqueeze(0)
            
            try:
                # Run diarization
                diarization = self._pipeline({
                    "waveform": waveform.to(self._device),
                    "sample_rate": self.sample_rate
                })
                
                # Process results
                self._process_diarization(diarization, stream_start_offset_sec)
                
                # Clear buffer (keep last small chunk for continuity)
                keep_duration = PYANNOTE_MIN_DURATION_SEC * 0.3  # Keep 30% for overlap
                keep_bytes = int(keep_duration * self.sample_rate * 2)
                if len(full_audio) > keep_bytes:
                    # Keep last portion
                    self._audio_buffer.clear()
                    self._audio_buffer.append(full_audio[-keep_bytes:])
                    self._buffer_duration_sec = keep_bytes / (self.sample_rate * 2)
                else:
                    self._audio_buffer.clear()
                    self._buffer_duration_sec = 0.0
                    
            except Exception as e:
                logger.warning(f"Pyannote diarization error: {e}")
                # Clear buffer on error to prevent accumulation
                self._audio_buffer.clear()
                self._buffer_duration_sec = 0.0
        
        return self._current_state()
    
    def _current_state(self) -> dict:
        """Return current state dict compatible with existing API."""
        # For compatibility, return the first active speaker as currentSpeakerId
        # In the future, could extend API to support multiple active speakers
        current_speaker = self._last_current_speakers[0] if self._last_current_speakers else None
        
        return {
            "event": "speaker",
            "currentSpeakerId": current_speaker,
            "speakingTimeMs": {str(i): self.speaking_time_ms.get(i, 0) for i in range(self.num_speakers)},
            # Additional field for future use (multiple active speakers)
            "activeSpeakerIds": self._last_current_speakers,
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
        """Reset pipeline state."""
        self.speaking_time_ms = {i: 0 for i in range(self.num_speakers)}
        self._speaker_label_map = {}
        self._last_current_speakers = []
        self._audio_buffer.clear()
        self._buffer_duration_sec = 0.0


class SpeakerPipeline:
    """
    Tracks up to num_speakers. First N distinct voices become Speaker 0..N-1.
    Extra voices (noise, others) are not assigned.
    
    Uses pyannote.audio for advanced overlapped speaker diarization when available,
    otherwise falls back to Resemblyzer + KMeans clustering.
    """

    def __init__(self, num_speakers: int = 2):
        self.num_speakers = max(1, min(5, num_speakers))
        self.sample_rate = SAMPLE_RATE
        
        # Try to use pyannote first (best for overlapped speech)
        self._use_pyannote = False
        self._pyannote_pipeline: Optional[PyannotePipeline] = None
        
        if PYANNOTE_AVAILABLE:
            try:
                self._pyannote_pipeline = PyannotePipeline(num_speakers=self.num_speakers)
                if self._pyannote_pipeline._pipeline is not None:
                    self._use_pyannote = True
                    logger.info("Using pyannote.audio for speaker diarization (supports overlapped speech)")
                else:
                    logger.info("Pyannote pipeline not initialized, falling back to Resemblyzer")
            except Exception as e:
                logger.warning(f"Failed to initialize pyannote, falling back to Resemblyzer: {e}")
        
        # Fallback: Resemblyzer + KMeans
        self._encoder: Optional[any] = None
        if not self._use_pyannote and RESEMBLYZER_AVAILABLE:
            self._encoder = VoiceEncoder()
            logger.info("Using Resemblyzer + KMeans for speaker diarization")
        
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
        - activeSpeakerIds: List[int] (when using pyannote, can have multiple)
        """
        # Use pyannote if available (best for overlapped speech)
        if self._use_pyannote and self._pyannote_pipeline:
            result = self._pyannote_pipeline.process(pcm_bytes, stream_start_offset_sec)
            # Sync state
            self.speaking_time_ms = self._pyannote_pipeline.speaking_time_ms.copy()
            if result.get("currentSpeakerId") is not None:
                self._last_current_speaker = result["currentSpeakerId"]
            return result
        
        # Fallback: Resemblyzer + KMeans (original method)
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
        # Sync from pyannote if using it
        if self._use_pyannote and self._pyannote_pipeline:
            self.speaking_time_ms = self._pyannote_pipeline.speaking_time_ms.copy()
        
        return {
            "event": "report",
            "speakers": [
                {"id": i, "totalTimeMs": self.speaking_time_ms.get(i, 0)}
                for i in range(self.num_speakers)
            ],
        }

    def reset(self) -> None:
        if self._use_pyannote and self._pyannote_pipeline:
            self._pyannote_pipeline.reset()
            self.speaking_time_ms = self._pyannote_pipeline.speaking_time_ms.copy()
        else:
            self.speaking_time_ms = {i: 0 for i in range(self.num_speakers)}
        self._cluster_to_speaker = {}
        self._next_speaker_id = 0
        self._last_current_speaker = None
