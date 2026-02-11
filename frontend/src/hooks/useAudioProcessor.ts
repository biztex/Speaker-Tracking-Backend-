import { useState, useRef, useCallback, useEffect } from 'react';
import type { AudioFeatures, Speaker, SessionStatus } from '../types';
import { DEFAULT_AUDIO_CONFIG, DEFAULT_SPEAKER_CONFIG, SPEAKER_COLORS } from '../types';
import { extractAudioFeatures, detectVoiceActivity } from '../utils/audioUtils';
import {
  createSpeakerDetector,
  processSpeakerDetection,
  resetDetector,
  type SpeakerDetector,
} from '../utils/speakerDetection';

interface UseAudioProcessorOptions {
  maxSpeakers?: number;
  onSpeakerChange?: (speakerId: number | null) => void;
  onVoiceActivity?: (isActive: boolean) => void;
}

interface UseAudioProcessorReturn {
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

export function useAudioProcessor(
  options: UseAudioProcessorOptions = {}
): UseAudioProcessorReturn {
  const { maxSpeakers = 2, onSpeakerChange, onVoiceActivity } = options;

  const [status, setStatus] = useState<SessionStatus>('idle');
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [currentSpeakerId, setCurrentSpeakerId] = useState<number | null>(null);
  const [audioFeatures, setAudioFeatures] = useState<AudioFeatures | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isVoiceActive, setIsVoiceActive] = useState(false);

  // Refs for audio processing
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const speakerDetectorRef = useRef<SpeakerDetector>(createSpeakerDetector(maxSpeakers));
  
  // Ref for running state (avoids stale closure in requestAnimationFrame callback)
  const isRunningRef = useRef(false);

  // Refs for speaker timing
  const lastSpeakerIdRef = useRef<number | null>(null);
  const lastSpeakingTimeRef = useRef<number>(Date.now());
  const speakersRef = useRef<Speaker[]>([]);

  // Add a new speaker dynamically
  const addSpeaker = useCallback((speakerId: number) => {
    if (speakerId < 0 || speakerId >= maxSpeakers) return;
    
    // Check if speaker already exists
    if (speakersRef.current.find(s => s.id === speakerId)) return;
    
    const newSpeaker: Speaker = {
      id: speakerId,
      name: `Speaker ${speakerId + 1}`,
      color: SPEAKER_COLORS[speakerId],
      totalTime: 0,
      isActive: false,
      lastActiveTime: null,
    };
    
    speakersRef.current = [...speakersRef.current, newSpeaker].sort((a, b) => a.id - b.id);
    setSpeakers([...speakersRef.current]);
  }, [maxSpeakers]);

  // Process audio frame
  const processAudioFrame = useCallback(() => {
    if (!analyserRef.current || !audioContextRef.current || !isRunningRef.current) {
      return;
    }

    const analyser = analyserRef.current;
    const sampleRate = audioContextRef.current.sampleRate;
    
    const timeDomainData = new Float32Array(analyser.fftSize);
    const frequencyData = new Uint8Array(analyser.frequencyBinCount);

    const features = extractAudioFeatures(analyser, timeDomainData, frequencyData, sampleRate);
    setAudioFeatures(features);

    const voiceActive = detectVoiceActivity(features, DEFAULT_SPEAKER_CONFIG.silenceThreshold);
    setIsVoiceActive(voiceActive);
    onVoiceActivity?.(voiceActive);

    const now = Date.now();

    if (voiceActive) {
      // Process speaker detection
      const result = processSpeakerDetection(speakerDetectorRef.current, features);
      speakerDetectorRef.current = result.detector;

      const detectedSpeakerId = result.speakerId;

      // Ensure speaker exists in our list
      if (detectedSpeakerId >= 0 && detectedSpeakerId < maxSpeakers) {
        // Add new speaker if detected
        if (result.isNewSpeaker && !speakersRef.current.find(s => s.id === detectedSpeakerId)) {
          addSpeaker(detectedSpeakerId);
        }

        // Update speaking time for previous speaker
        if (lastSpeakerIdRef.current !== null && lastSpeakerIdRef.current !== detectedSpeakerId) {
          const elapsed = now - lastSpeakingTimeRef.current;
          speakersRef.current = speakersRef.current.map(s =>
            s.id === lastSpeakerIdRef.current
              ? { ...s, totalTime: s.totalTime + elapsed, isActive: false }
              : s
          );
        }

        // Update current speaker state
        if (lastSpeakerIdRef.current !== detectedSpeakerId) {
          lastSpeakingTimeRef.current = now;
          onSpeakerChange?.(detectedSpeakerId);
        }

        lastSpeakerIdRef.current = detectedSpeakerId;
        setCurrentSpeakerId(detectedSpeakerId);

        // Update speaker active state
        speakersRef.current = speakersRef.current.map(s =>
          s.id === detectedSpeakerId
            ? { ...s, isActive: true, lastActiveTime: now }
            : { ...s, isActive: false }
        );

        setSpeakers([...speakersRef.current]);
      }
    } else {
      // No voice activity - update last speaker's time
      if (lastSpeakerIdRef.current !== null) {
        const elapsed = now - lastSpeakingTimeRef.current;
        if (elapsed > DEFAULT_SPEAKER_CONFIG.minSpeechDuration) {
          speakersRef.current = speakersRef.current.map(s =>
            s.id === lastSpeakerIdRef.current
              ? { ...s, totalTime: s.totalTime + elapsed, isActive: false }
              : { ...s, isActive: false }
          );
          setSpeakers([...speakersRef.current]);
        }
        lastSpeakerIdRef.current = null;
        lastSpeakingTimeRef.current = now;
      }

      setCurrentSpeakerId(null);
    }

    // Continue processing
    animationFrameRef.current = requestAnimationFrame(processAudioFrame);
  }, [maxSpeakers, onSpeakerChange, onVoiceActivity, addSpeaker]);

  // Start audio capture
  const start = useCallback(async () => {
    try {
      setError(null);

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;

      // Create audio context
      const audioContext = new AudioContext({
        sampleRate: DEFAULT_AUDIO_CONFIG.sampleRate,
      });
      audioContextRef.current = audioContext;

      // Create analyzer node
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = DEFAULT_AUDIO_CONFIG.fftSize;
      analyser.smoothingTimeConstant = DEFAULT_AUDIO_CONFIG.smoothingTimeConstant;
      analyser.minDecibels = DEFAULT_AUDIO_CONFIG.minDecibels;
      analyser.maxDecibels = DEFAULT_AUDIO_CONFIG.maxDecibels;
      analyserRef.current = analyser;

      // Create a zero-gain node to connect to destination
      // This is CRITICAL: Web Audio nodes only process when connected to destination
      const gainNode = audioContext.createGain();
      gainNode.gain.value = 0; // Mute output (no echo/feedback)
      gainNodeRef.current = gainNode;

      // Connect the audio graph: source -> analyser -> gainNode -> destination
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      analyser.connect(gainNode);
      gainNode.connect(audioContext.destination);
      sourceRef.current = source;

      // CRITICAL: Resume AudioContext if suspended (required by browsers)
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      // Don't pre-initialize speakers - they will be added dynamically when detected
      // Clear any existing speakers
      speakersRef.current = [];
      setSpeakers([]);

      // Reset detector
      speakerDetectorRef.current = resetDetector(maxSpeakers);
      lastSpeakerIdRef.current = null;
      lastSpeakingTimeRef.current = Date.now();

      // Set running ref BEFORE scheduling frame so callback sees it immediately
      isRunningRef.current = true;
      setStatus('running');

      // Start processing loop
      animationFrameRef.current = requestAnimationFrame(processAudioFrame);
    } catch (err) {
      isRunningRef.current = false;
      const errorMessage =
        err instanceof Error ? err.message : 'Failed to access microphone';
      setError(errorMessage);
      console.error('Audio start error:', err);
    }
  }, [maxSpeakers, processAudioFrame]);

  // Stop audio capture
  const stop = useCallback(() => {
    isRunningRef.current = false;

    // Stop animation frame
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Finalize last speaker's time
    if (lastSpeakerIdRef.current !== null) {
      const now = Date.now();
      const elapsed = now - lastSpeakingTimeRef.current;
      speakersRef.current = speakersRef.current.map(s =>
        s.id === lastSpeakerIdRef.current
          ? { ...s, totalTime: s.totalTime + elapsed, isActive: false }
          : { ...s, isActive: false }
      );
      setSpeakers([...speakersRef.current]);
    }

    // Disconnect and clean up audio nodes
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }

    if (analyserRef.current) {
      analyserRef.current.disconnect();
      analyserRef.current = null;
    }

    if (gainNodeRef.current) {
      gainNodeRef.current.disconnect();
      gainNodeRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Stop media stream tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    setStatus('stopped');
    setCurrentSpeakerId(null);
    setIsVoiceActive(false);
  }, []);

  // Reset everything
  const reset = useCallback(() => {
    stop();
    setSpeakers([]);
    speakersRef.current = [];
    setAudioFeatures(null);
    setError(null);
    setStatus('idle');
    speakerDetectorRef.current = resetDetector(maxSpeakers);
    lastSpeakerIdRef.current = null;
  }, [stop, maxSpeakers]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

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

export default useAudioProcessor;
