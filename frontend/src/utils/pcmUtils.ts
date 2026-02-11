/**
 * Resample float32 audio to target sample rate using linear interpolation.
 */
export function resampleTo(
  input: Float32Array,
  fromSampleRate: number,
  toSampleRate: number
): Float32Array {
  if (fromSampleRate === toSampleRate) return input;
  const outLength = Math.round((input.length * toSampleRate) / fromSampleRate);
  if (outLength <= 0) return new Float32Array(0);
  const result = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIndex = (i * fromSampleRate) / toSampleRate;
    const lo = Math.floor(srcIndex);
    const hi = Math.min(lo + 1, input.length - 1);
    const t = srcIndex - lo;
    result[i] = input[lo] * (1 - t) + input[hi] * t;
  }
  return result;
}

/**
 * Convert float32 [-1, 1] to Int16 and write into a Uint8Array (little-endian bytes).
 */
export function float32ToPcm16Bytes(float32: Float32Array): ArrayBuffer {
  const pcm16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16.buffer;
}
