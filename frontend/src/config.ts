/**
 * Backend WebSocket URL for speaker tracking.
 * Set VITE_WS_URL in .env (e.g. ws://localhost:8000/ws/session) to override.
 */
export function getBackendWsUrl(): string {
  const env = (typeof import.meta !== 'undefined' && (import.meta as { env?: { VITE_WS_URL?: string } }).env?.VITE_WS_URL) as string | undefined;
  if (env && typeof env === 'string') return env;
  const protocol = typeof window !== 'undefined' && window.location?.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = typeof window !== 'undefined' && window.location?.hostname ? window.location.hostname : 'localhost';
  const backendPort = '8000';
  return `${protocol}//${host}:${backendPort}/ws/session`;
}

export const TARGET_SAMPLE_RATE = 16000;
