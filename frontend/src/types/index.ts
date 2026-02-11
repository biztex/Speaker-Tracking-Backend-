// Speaker related types
export interface Speaker {
  id: number;
  name: string;
  color: string;
  totalTime: number; // in milliseconds
  isActive: boolean;
  lastActiveTime: number | null;
}

export interface SpeakerStats {
  speakerId: number;
  totalTime: number;
  percentage: number;
}

// Audio related types
export interface AudioFeatures {
  volume: number; // 0-1 normalized
  pitch: number; // Hz
  spectralCentroid: number;
  zeroCrossingRate: number;
  mfcc: number[];
  frequencyData: Uint8Array;
  waveformData: Uint8Array;
}

export interface VoiceProfile {
  speakerId: number;
  avgPitch: number;
  avgSpectralCentroid: number;
  pitchVariance: number;
  samples: number;
}

// Session related types
export type SessionStatus = 'idle' | 'running' | 'paused' | 'stopped';

export interface SessionState {
  status: SessionStatus;
  startTime: number | null;
  endTime: number | null;
  totalDuration: number;
  speakers: Speaker[];
  currentSpeakerId: number | null;
}

// Visualization related types
export interface VisualizationData {
  waveform: number[];
  frequencies: number[];
  currentSpeakerColor: string | null;
}

// Theme types
export type Theme = 'light' | 'dark';

// Event types for audio processing
export interface VoiceActivityEvent {
  isActive: boolean;
  timestamp: number;
  features: AudioFeatures | null;
}

export interface SpeakerChangeEvent {
  previousSpeakerId: number | null;
  currentSpeakerId: number | null;
  timestamp: number;
}

// Configuration types
export interface AudioConfig {
  sampleRate: number;
  fftSize: number;
  smoothingTimeConstant: number;
  minDecibels: number;
  maxDecibels: number;
}

export interface SpeakerDetectionConfig {
  minSpeechDuration: number; // minimum ms to consider speech
  silenceThreshold: number; // volume threshold for silence
  speakerChangeThreshold: number; // threshold for speaker change detection
  maxSpeakers: number;
}

export const DEFAULT_AUDIO_CONFIG: AudioConfig = {
  sampleRate: 44100,
  fftSize: 2048,
  smoothingTimeConstant: 0.8,
  minDecibels: -90,
  maxDecibels: -10,
};

export const DEFAULT_SPEAKER_CONFIG: SpeakerDetectionConfig = {
  minSpeechDuration: 50, // Very fast response
  silenceThreshold: 0.008, // Very sensitive threshold for VAD
  speakerChangeThreshold: 0.35, // Threshold for detecting different speaker
  maxSpeakers: 5,
};

export const SPEAKER_COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
];
