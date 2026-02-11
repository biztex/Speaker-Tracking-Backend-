import type { AudioFeatures, VoiceProfile } from '../types';

/**
 * Speaker detection using voice characteristics
 * This is a simplified approach using pitch and spectral features
 */

const PROFILE_SAMPLES_THRESHOLD = 5; // Minimum samples before profile is stable (lowered for faster learning)
const SPEAKER_CHANGE_THRESHOLD = 0.35; // Threshold for detecting speaker change

export interface SpeakerDetector {
  profiles: Map<number, VoiceProfile>;
  currentSpeakerId: number | null;
  lastFeatures: AudioFeatures | null;
  speakerCount: number;
  maxSpeakers: number;
}

export function createSpeakerDetector(maxSpeakers: number = 5): SpeakerDetector {
  return {
    profiles: new Map(),
    currentSpeakerId: null,
    lastFeatures: null,
    speakerCount: 0,
    maxSpeakers,
  };
}

/**
 * Calculate similarity between audio features and a voice profile
 * Returns a value between 0 (different) and 1 (same)
 */
export function calculateSimilarity(
  features: AudioFeatures,
  profile: VoiceProfile
): number {
  if (profile.samples < PROFILE_SAMPLES_THRESHOLD) {
    return 0.5; // Not enough data to make a determination
  }

  // Pitch similarity (weighted)
  const pitchDiff = Math.abs(features.pitch - profile.avgPitch);
  const pitchSimilarity = Math.max(0, 1 - pitchDiff / 150); // 150 Hz tolerance

  // Spectral centroid similarity
  const spectralDiff = Math.abs(features.spectralCentroid - profile.avgSpectralCentroid);
  const spectralSimilarity = Math.max(0, 1 - spectralDiff / 1000); // 1000 Hz tolerance

  // Combined similarity with weights
  return pitchSimilarity * 0.6 + spectralSimilarity * 0.4;
}

/**
 * Update a voice profile with new features
 */
export function updateProfile(
  profile: VoiceProfile,
  features: AudioFeatures
): VoiceProfile {
  const alpha = Math.min(0.1, 1 / (profile.samples + 1)); // Exponential moving average
  
  const newAvgPitch = profile.avgPitch * (1 - alpha) + features.pitch * alpha;
  const newAvgSpectral = profile.avgSpectralCentroid * (1 - alpha) + features.spectralCentroid * alpha;
  
  // Update variance estimate
  const pitchVariance = profile.pitchVariance * (1 - alpha) + 
    Math.pow(features.pitch - newAvgPitch, 2) * alpha;

  return {
    ...profile,
    avgPitch: newAvgPitch,
    avgSpectralCentroid: newAvgSpectral,
    pitchVariance,
    samples: profile.samples + 1,
  };
}

/**
 * Create a new voice profile from features
 */
export function createProfile(speakerId: number, features: AudioFeatures): VoiceProfile {
  return {
    speakerId,
    avgPitch: features.pitch,
    avgSpectralCentroid: features.spectralCentroid,
    pitchVariance: 0,
    samples: 1,
  };
}

/**
 * Detect which speaker is currently talking
 * Returns the speaker ID and whether it's a new speaker
 */
export function detectSpeaker(
  detector: SpeakerDetector,
  features: AudioFeatures
): { speakerId: number; isNew: boolean; confidence: number } {
  // If no profiles exist yet, create the first speaker
  if (detector.profiles.size === 0) {
    return {
      speakerId: 0,
      isNew: true,
      confidence: 1.0,
    };
  }

  // If pitch is invalid but we have a current speaker, keep it
  // This prevents constant speaker switching during unclear audio
  const hasPitch = features.pitch > 50 && features.pitch < 500;
  if (!hasPitch && detector.currentSpeakerId !== null) {
    return {
      speakerId: detector.currentSpeakerId,
      isNew: false,
      confidence: 0.5,
    };
  }

  let bestSpeakerId = -1;
  let bestSimilarity = 0;

  // Compare with existing profiles
  for (const [speakerId, profile] of detector.profiles) {
    const similarity = calculateSimilarity(features, profile);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestSpeakerId = speakerId;
    }
  }

  // Decide if this is a new speaker or matches an existing one
  let isNew = false;
  
  if (bestSimilarity < SPEAKER_CHANGE_THRESHOLD && detector.speakerCount < detector.maxSpeakers) {
    // New speaker detected - significantly different voice characteristics
    isNew = true;
    bestSpeakerId = detector.speakerCount;
  } else if (bestSpeakerId === -1) {
    // Fallback: if no match found at all, use current speaker or speaker 0
    bestSpeakerId = detector.currentSpeakerId ?? 0;
  }

  return {
    speakerId: bestSpeakerId,
    isNew,
    confidence: bestSimilarity,
  };
}

/**
 * Process audio features and update speaker detection state
 */
export function processSpeakerDetection(
  detector: SpeakerDetector,
  features: AudioFeatures
): {
  detector: SpeakerDetector;
  speakerId: number;
  isNewSpeaker: boolean;
  confidence: number;
} {
  const { speakerId, isNew, confidence } = detectSpeaker(detector, features);

  const newProfiles = new Map(detector.profiles);

  if (speakerId >= 0) {
    if (isNew) {
      // Create new profile
      newProfiles.set(speakerId, createProfile(speakerId, features));
    } else if (newProfiles.has(speakerId)) {
      // Update existing profile
      const existingProfile = newProfiles.get(speakerId)!;
      newProfiles.set(speakerId, updateProfile(existingProfile, features));
    }
  }

  return {
    detector: {
      profiles: newProfiles,
      currentSpeakerId: speakerId >= 0 ? speakerId : detector.currentSpeakerId,
      lastFeatures: features,
      speakerCount: isNew ? detector.speakerCount + 1 : detector.speakerCount,
      maxSpeakers: detector.maxSpeakers,
    },
    speakerId,
    isNewSpeaker: isNew,
    confidence,
  };
}

/**
 * Reset the speaker detector
 */
export function resetDetector(maxSpeakers: number = 5): SpeakerDetector {
  return createSpeakerDetector(maxSpeakers);
}

/**
 * Get speaker statistics
 */
export function getSpeakerStats(profiles: Map<number, VoiceProfile>): {
  speakerId: number;
  avgPitch: number;
  samples: number;
}[] {
  const stats: { speakerId: number; avgPitch: number; samples: number }[] = [];
  
  for (const [speakerId, profile] of profiles) {
    stats.push({
      speakerId,
      avgPitch: profile.avgPitch,
      samples: profile.samples,
    });
  }
  
  return stats.sort((a, b) => a.speakerId - b.speakerId);
}
