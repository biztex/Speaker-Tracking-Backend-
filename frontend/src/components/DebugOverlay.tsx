import React from 'react';
import type { AudioFeatures } from '../types';

interface DebugOverlayProps {
  audioFeatures: AudioFeatures | null;
  isVoiceActive: boolean;
  currentSpeakerId: number | null;
  showDebug: boolean;
  onToggle: () => void;
}

const DebugOverlay: React.FC<DebugOverlayProps> = ({
  audioFeatures,
  isVoiceActive,
  currentSpeakerId,
  showDebug,
  onToggle,
}) => {
  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={onToggle}
        className="fixed bottom-4 right-4 z-50 px-3 py-2 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] text-xs font-medium text-[var(--color-text-secondary)] hover:bg-[var(--color-border)] transition-colors"
      >
        {showDebug ? 'Hide Debug' : 'Show Debug'}
      </button>

      {/* Debug Panel */}
      {showDebug && (
        <div className="fixed bottom-16 right-4 z-50 w-72 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)] shadow-xl overflow-hidden animate-fade-in">
          <div className="px-4 py-2 bg-[var(--color-bg-tertiary)] border-b border-[var(--color-border)]">
            <h4 className="text-xs font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
              Audio Debug
            </h4>
          </div>
          
          <div className="p-4 space-y-3 text-xs font-mono">
            {/* Voice Activity */}
            <div className="flex items-center justify-between">
              <span className="text-[var(--color-text-muted)]">Voice Active:</span>
              <span className={`font-bold ${isVoiceActive ? 'text-green-500' : 'text-red-500'}`}>
                {isVoiceActive ? 'YES' : 'NO'}
              </span>
            </div>

            {/* Current Speaker */}
            <div className="flex items-center justify-between">
              <span className="text-[var(--color-text-muted)]">Speaker ID:</span>
              <span className="text-[var(--color-text-primary)]">
                {currentSpeakerId !== null ? currentSpeakerId + 1 : '-'}
              </span>
            </div>

            <hr className="border-[var(--color-border)]" />

            {audioFeatures ? (
              <>
                {/* Volume */}
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[var(--color-text-muted)]">Volume:</span>
                    <span className={`font-medium ${
                      audioFeatures.volume > 0.02 
                        ? 'text-green-500' 
                        : audioFeatures.volume > 0.008 
                          ? 'text-yellow-500' 
                          : 'text-[var(--color-text-primary)]'
                    }`}>
                      {(audioFeatures.volume * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 bg-[var(--color-bg-tertiary)] rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-100 ${
                        audioFeatures.volume > 0.02 ? 'bg-green-500' : 'bg-yellow-500'
                      }`}
                      style={{ width: `${Math.min(audioFeatures.volume * 100 * 5, 100)}%` }}
                    />
                  </div>
                </div>

                {/* Pitch */}
                <div className="flex items-center justify-between">
                  <span className="text-[var(--color-text-muted)]">Pitch:</span>
                  <span className={`${
                    audioFeatures.pitch > 50 && audioFeatures.pitch < 600
                      ? 'text-green-500'
                      : 'text-[var(--color-text-muted)]'
                  }`}>
                    {audioFeatures.pitch > 0 ? `${Math.round(audioFeatures.pitch)} Hz` : '- Hz'}
                  </span>
                </div>

                {/* Spectral Centroid */}
                <div className="flex items-center justify-between">
                  <span className="text-[var(--color-text-muted)]">Spectral:</span>
                  <span className="text-[var(--color-text-primary)]">
                    {Math.round(audioFeatures.spectralCentroid)} Hz
                  </span>
                </div>

                {/* Zero Crossing Rate */}
                <div className="flex items-center justify-between">
                  <span className="text-[var(--color-text-muted)]">ZCR:</span>
                  <span className={`${
                    audioFeatures.zeroCrossingRate > 0.02 && audioFeatures.zeroCrossingRate < 0.3
                      ? 'text-green-500'
                      : 'text-[var(--color-text-muted)]'
                  }`}>
                    {audioFeatures.zeroCrossingRate.toFixed(3)}
                  </span>
                </div>

                <hr className="border-[var(--color-border)]" />

                {/* Thresholds info */}
                <div className="text-[var(--color-text-muted)] text-[10px] space-y-1">
                  <div>Volume threshold: 0.8% (or 2% auto-accept)</div>
                  <div>Pitch range: 50-600 Hz</div>
                  <div>Spectral min: 100 Hz</div>
                </div>
              </>
            ) : (
              <div className="text-center text-[var(--color-text-muted)] py-4">
                No audio data
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default DebugOverlay;
