import type { AudioFeatures } from '../types';

/**
 * Calculate RMS (Root Mean Square) volume from audio data
 */
export function calculateRMS(dataArray: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < dataArray.length; i++) {
    sum += dataArray[i] * dataArray[i];
  }
  return Math.sqrt(sum / dataArray.length);
}

/**
 * Calculate Zero Crossing Rate - useful for voice detection
 */
export function calculateZeroCrossingRate(dataArray: ArrayLike<number>): number {
  let crossings = 0;
  for (let i = 1; i < dataArray.length; i++) {
    if ((dataArray[i] >= 0 && dataArray[i - 1] < 0) || 
        (dataArray[i] < 0 && dataArray[i - 1] >= 0)) {
      crossings++;
    }
  }
  return crossings / dataArray.length;
}

/**
 * Calculate Spectral Centroid - the "center of mass" of the spectrum
 * Higher values indicate brighter sounds
 */
export function calculateSpectralCentroid(
  frequencyData: ArrayLike<number>,
  sampleRate: number,
  fftSize: number
): number {
  let weightedSum = 0;
  let sum = 0;
  const binFrequency = sampleRate / fftSize;

  for (let i = 0; i < frequencyData.length; i++) {
    const magnitude = frequencyData[i];
    const frequency = i * binFrequency;
    weightedSum += magnitude * frequency;
    sum += magnitude;
  }

  return sum > 0 ? weightedSum / sum : 0;
}

/**
 * Estimate fundamental frequency (pitch) using autocorrelation
 */
export function estimatePitch(dataArray: ArrayLike<number>, sampleRate: number): number {
  // Simple autocorrelation-based pitch detection
  const minPeriod = Math.floor(sampleRate / 500); // Max 500 Hz
  const maxPeriod = Math.floor(sampleRate / 50);  // Min 50 Hz
  
  let bestCorrelation = 0;
  let bestPeriod = 0;

  for (let period = minPeriod; period < maxPeriod && period < dataArray.length; period++) {
    let correlation = 0;
    for (let i = 0; i < dataArray.length - period; i++) {
      correlation += dataArray[i] * dataArray[i + period];
    }
    
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestPeriod = period;
    }
  }

  return bestPeriod > 0 ? sampleRate / bestPeriod : 0;
}

/**
 * Calculate simplified MFCC-like features
 * These are mel-frequency band energies (not full MFCC but computationally lighter)
 */
export function calculateMelBandEnergies(
  frequencyData: ArrayLike<number>,
  sampleRate: number,
  numBands: number = 13
): number[] {
  const bands: number[] = new Array(numBands).fill(0);
  const maxFreq = sampleRate / 2;
  const binWidth = maxFreq / frequencyData.length;

  // Mel scale conversion helpers
  const freqToMel = (f: number) => 2595 * Math.log10(1 + f / 700);
  const melToFreq = (m: number) => 700 * (Math.pow(10, m / 2595) - 1);

  const minMel = freqToMel(20);
  const maxMel = freqToMel(maxFreq);
  const melStep = (maxMel - minMel) / (numBands + 1);

  for (let band = 0; band < numBands; band++) {
    const melStart = minMel + band * melStep;
    const melCenter = minMel + (band + 1) * melStep;
    const melEnd = minMel + (band + 2) * melStep;

    const freqStart = melToFreq(melStart);
    const freqCenter = melToFreq(melCenter);
    const freqEnd = melToFreq(melEnd);

    const binStart = Math.floor(freqStart / binWidth);
    const binCenter = Math.floor(freqCenter / binWidth);
    const binEnd = Math.min(Math.floor(freqEnd / binWidth), frequencyData.length - 1);

    let energy = 0;
    for (let bin = binStart; bin <= binEnd; bin++) {
      // Triangular filter
      let weight = 0;
      if (bin < binCenter) {
        weight = (bin - binStart) / (binCenter - binStart + 1);
      } else {
        weight = (binEnd - bin) / (binEnd - binCenter + 1);
      }
      energy += frequencyData[bin] * weight;
    }
    bands[band] = energy;
  }

  // Normalize
  const maxEnergy = Math.max(...bands, 1);
  return bands.map(e => e / maxEnergy);
}

/**
 * Extract all audio features from analyzer nodes
 */
export function extractAudioFeatures(
  analyser: AnalyserNode,
  timeDomainData: Float32Array<ArrayBuffer>,
  frequencyData: Uint8Array<ArrayBuffer>,
  sampleRate: number
): AudioFeatures {
  analyser.getFloatTimeDomainData(timeDomainData);
  analyser.getByteFrequencyData(frequencyData);

  const volume = calculateRMS(timeDomainData);
  const pitch = estimatePitch(timeDomainData, sampleRate);
  const spectralCentroid = calculateSpectralCentroid(frequencyData, sampleRate, analyser.fftSize);
  const zeroCrossingRate = calculateZeroCrossingRate(timeDomainData);
  const mfcc = calculateMelBandEnergies(frequencyData, sampleRate);

  // Create copies for visualization
  const waveformData = new Uint8Array(timeDomainData.length);
  for (let i = 0; i < timeDomainData.length; i++) {
    waveformData[i] = Math.floor((timeDomainData[i] + 1) * 128);
  }

  return {
    volume,
    pitch,
    spectralCentroid,
    zeroCrossingRate,
    mfcc,
    frequencyData: new Uint8Array(frequencyData),
    waveformData,
  };
}

/**
 * Check if the audio features indicate voice activity
 * SIMPLIFIED: Primarily volume-based for maximum reliability
 * Speaker differentiation is handled separately by the speaker detection module
 */
export function detectVoiceActivity(
  features: AudioFeatures,
  silenceThreshold: number = 0.008 // Very sensitive threshold
): boolean {
  // Primary detection: ANY sound above threshold is considered potential voice
  // This ensures we never miss actual speech
  // False positives are acceptable - speaker detection will filter further
  
  const hasSignificantVolume = features.volume > silenceThreshold;
  
  // For very low volumes, require some voice-like characteristic
  // For moderate volumes (>2%), just accept it as voice activity
  if (features.volume > 0.02) {
    return true;
  }
  
  // For quieter sounds, do a basic sanity check
  // Accept if pitch is in human range OR there's meaningful frequency content
  const hasPitch = features.pitch > 50 && features.pitch < 600;
  const hasFrequencyContent = features.spectralCentroid > 100;
  
  return hasSignificantVolume && (hasPitch || hasFrequencyContent);
}

/**
 * Normalize frequency data for visualization (0-1 range)
 */
export function normalizeFrequencyData(data: Uint8Array): number[] {
  const normalized: number[] = [];
  for (let i = 0; i < data.length; i++) {
    normalized.push(data[i] / 255);
  }
  return normalized;
}

/**
 * Downsample data for efficient visualization
 */
export function downsampleData(data: number[] | Uint8Array, targetLength: number): number[] {
  const result: number[] = [];
  const step = data.length / targetLength;
  
  for (let i = 0; i < targetLength; i++) {
    const start = Math.floor(i * step);
    const end = Math.floor((i + 1) * step);
    let sum = 0;
    for (let j = start; j < end && j < data.length; j++) {
      sum += data[j];
    }
    result.push(sum / (end - start));
  }
  
  return result;
}
