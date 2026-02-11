import { useState, useRef, useCallback, useEffect } from 'react';
import type { AudioFeatures, Speaker, SessionStatus } from '../types';
import { DEFAULT_AUDIO_CONFIG, SPEAKER_COLORS } from '../types';
import { extractAudioFeatures } from '../utils/audioUtils';
import { resampleTo, float32ToPcm16Bytes } from '../utils/pcmUtils';
import { getBackendWsUrl, TARGET_SAMPLE_RATE } from '../config';

interface UseBackendSessionOptions {
  maxSpeakers?: number;
}

interface UseBackendSessionReturn {
  status: SessionStatus;
  speakers: Speaker[];
  currentSpeakerId: number | null;
  audioFeatures: AudioFeatures | null;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
  reset: () => void;
  isVoiceActive: boolean;
}

function buildSpeakersFromTimes(
  numSpeakers: number,
  speakingTimeMs: Record<string, number>,
  currentSpeakerId: number | null
): Speaker[] {
  const list: Speaker[] = [];
  for (let i = 0; i < numSpeakers; i++) {
    list.push({
      id: i,
      name: `Speaker ${i + 1}`,
      color: SPEAKER_COLORS[i] ?? '#64748b',
      totalTime: speakingTimeMs[String(i)] ?? 0,
      isActive: currentSpeakerId === i,
      lastActiveTime: currentSpeakerId === i ? Date.now() : null,
    });
  }
  return list;
}

export function useBackendSession(
  options: UseBackendSessionOptions = {}
): UseBackendSessionReturn {
  const { maxSpeakers = 2 } = options;

  const [status, setStatus] = useState<SessionStatus>('idle');
  const [speakers, setSpeakers] = useState<Speaker[]>(() =>
    buildSpeakersFromTimes(maxSpeakers, {}, null)
  );
  const [currentSpeakerId, setCurrentSpeakerId] = useState<number | null>(null);
  const [audioFeatures, setAudioFeatures] = useState<AudioFeatures | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const scriptNodeRef = useRef<ScriptProcessorNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const isRunningRef = useRef(false);

  const numSpeakersRef = useRef(maxSpeakers);

  const updateSpeakersFromBackend = useCallback(
    (speakingTimeMs: Record<string, number>, current: number | null) => {
      setSpeakers(
        buildSpeakersFromTimes(numSpeakersRef.current, speakingTimeMs, current)
      );
      setCurrentSpeakerId(current);
    },
    []
  );

  const processVisualizationFrame = useCallback(() => {
    if (!analyserRef.current || !audioContextRef.current || !isRunningRef.current)
      return;
    const analyser = analyserRef.current;
    const sampleRate = audioContextRef.current.sampleRate;
    const timeDomainData = new Float32Array(analyser.fftSize);
    const frequencyData = new Uint8Array(analyser.frequencyBinCount);
    const features = extractAudioFeatures(
      analyser,
      timeDomainData,
      frequencyData,
      sampleRate
    );
    setAudioFeatures(features);
    animationFrameRef.current = requestAnimationFrame(processVisualizationFrame);
  }, []);

  const start = useCallback(async () => {
    try {
      setError(null);
      numSpeakersRef.current = maxSpeakers;

      const wsUrl = getBackendWsUrl();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error('WebSocket connection failed'));
        setTimeout(() => reject(new Error('WebSocket timeout')), 5000);
      });

      ws.send(
        JSON.stringify({ action: 'start', num_speakers: maxSpeakers })
      );

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const audioContext = new AudioContext({
        sampleRate: DEFAULT_AUDIO_CONFIG.sampleRate,
      });
      audioContextRef.current = audioContext;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = DEFAULT_AUDIO_CONFIG.fftSize;
      analyser.smoothingTimeConstant = DEFAULT_AUDIO_CONFIG.smoothingTimeConstant;
      analyser.minDecibels = DEFAULT_AUDIO_CONFIG.minDecibels;
      analyser.maxDecibels = DEFAULT_AUDIO_CONFIG.maxDecibels;
      analyserRef.current = analyser;

      const gainNode = audioContext.createGain();
      gainNode.gain.value = 0;
      gainNodeRef.current = gainNode;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      analyser.connect(gainNode);
      gainNode.connect(audioContext.destination);
      sourceRef.current = source;

      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      const bufferLength = 4096;
      const scriptNode = audioContext.createScriptProcessor(bufferLength, 1, 1);
      scriptNode.onaudioprocess = (e) => {
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        const sr = audioContext.sampleRate;
        const resampled =
          sr !== TARGET_SAMPLE_RATE
            ? resampleTo(input, sr, TARGET_SAMPLE_RATE)
            : input;
        const bytes = float32ToPcm16Bytes(
          resampled instanceof Float32Array ? resampled : new Float32Array(resampled)
        );
        ws.send(bytes);
      };
      source.connect(scriptNode);
      scriptNode.connect(audioContext.destination);
      scriptNodeRef.current = scriptNode;

      setStatus('running');
      isRunningRef.current = true;
      setSpeakers(buildSpeakersFromTimes(maxSpeakers, {}, null));
      animationFrameRef.current =
        requestAnimationFrame(processVisualizationFrame);

      ws.onmessage = (event) => {
        if (typeof event.data !== 'string') return;
        try {
          const msg = JSON.parse(event.data) as {
            event?: string;
            currentSpeakerId?: number | null;
            speakingTimeMs?: Record<string, number>;
            speakers?: { id: number; totalTimeMs: number }[];
          };
          if (msg.event === 'started') {
            setSpeakers(
              buildSpeakersFromTimes(maxSpeakers, {}, null)
            );
          } else if (msg.event === 'speaker' && msg.speakingTimeMs) {
            updateSpeakersFromBackend(
              msg.speakingTimeMs,
              msg.currentSpeakerId ?? null
            );
          } else if (msg.event === 'report' && msg.speakers) {
            const times: Record<string, number> = {};
            for (const s of msg.speakers) times[String(s.id)] = s.totalTimeMs;
            updateSpeakersFromBackend(times, null);
            setStatus('stopped');
            isRunningRef.current = false;
            if (wsRef.current) {
              wsRef.current.close();
              wsRef.current = null;
            }
          }
        } catch (_) {
          // ignore parse errors
        }
      };

      ws.onclose = () => {
        if (isRunningRef.current && status !== 'stopped') {
          setError('Connection to backend closed');
        }
      };
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to start session';
      setError(message);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      console.error('Backend session start error:', err);
    }
  }, [maxSpeakers, updateSpeakersFromBackend, processVisualizationFrame]);

  const stop = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'stop' }));
    }
    isRunningRef.current = false;
    if (animationFrameRef.current != null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (scriptNodeRef.current) {
      try {
        scriptNodeRef.current.disconnect();
      } catch (_) {}
      scriptNodeRef.current = null;
    }
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch (_) {}
      sourceRef.current = null;
    }
    if (analyserRef.current) {
      try {
        analyserRef.current.disconnect();
      } catch (_) {}
      analyserRef.current = null;
    }
    if (gainNodeRef.current) {
      try {
        gainNodeRef.current.disconnect();
      } catch (_) {}
      gainNodeRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCurrentSpeakerId(null);
    setAudioFeatures(null);
  }, []);

  const reset = useCallback(() => {
    stop();
    setSpeakers(buildSpeakersFromTimes(maxSpeakers, {}, null));
    setCurrentSpeakerId(null);
    setError(null);
    setStatus('idle');
  }, [stop, maxSpeakers]);

  useEffect(() => {
    return () => {
      if (animationFrameRef.current != null)
        cancelAnimationFrame(animationFrameRef.current);
      if (streamRef.current)
        streamRef.current.getTracks().forEach((t) => t.stop());
      if (audioContextRef.current) audioContextRef.current.close();
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const isVoiceActive = status === 'running' && currentSpeakerId !== null;

  return {
    status,
    speakers,
    currentSpeakerId,
    audioFeatures,
    error,
    start,
    stop,
    reset,
    isVoiceActive,
  };
}

export default useBackendSession;
