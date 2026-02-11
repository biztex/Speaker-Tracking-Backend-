import React from 'react';
import type { Speaker } from '../types';

interface SpeakerListProps {
  speakers: Speaker[];
  currentSpeakerId: number | null;
  formatTime: (ms: number) => string;
}

const SpeakerList: React.FC<SpeakerListProps> = ({
  speakers,
  currentSpeakerId,
  formatTime,
}) => {
  // Calculate total speaking time
  const totalSpeakingTime = speakers.reduce((sum, s) => sum + s.totalTime, 0);

  // Calculate percentage for each speaker
  const getPercentage = (speakerTime: number): number => {
    if (totalSpeakingTime === 0) return 0;
    return Math.round((speakerTime / totalSpeakingTime) * 100);
  };

  if (speakers.length === 0) {
    return (
      <div className="w-full rounded-2xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)] p-6 shadow-lg">
        <div className="flex flex-col items-center justify-center gap-3 text-[var(--color-text-muted)]">
          <svg
            className="w-12 h-12 opacity-50"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
            />
          </svg>
          <p className="text-sm">No speakers detected yet</p>
          <p className="text-xs">Start a session to begin tracking</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full rounded-2xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)] overflow-hidden shadow-lg">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--color-border)]">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
            Speaker Statistics
          </h3>
          <span className="text-xs text-[var(--color-text-muted)]">
            {speakers.length} speaker{speakers.length !== 1 ? 's' : ''} detected
          </span>
        </div>
      </div>

      {/* Speaker list */}
      <div className="divide-y divide-[var(--color-border)]">
        {speakers.map((speaker) => {
          const percentage = getPercentage(speaker.totalTime);
          const isActive = speaker.id === currentSpeakerId;

          return (
            <div
              key={speaker.id}
              className={`
                px-4 py-3 transition-all duration-300
                ${isActive ? 'bg-[var(--color-bg-tertiary)]/50' : ''}
              `}
            >
              <div className="flex items-center gap-3">
                {/* Speaker indicator */}
                <div className="relative flex-shrink-0">
                  <div
                    className={`
                      w-10 h-10 rounded-full flex items-center justify-center
                      transition-all duration-300
                      ${isActive ? 'ring-2 ring-offset-2 ring-offset-[var(--color-bg-secondary)]' : ''}
                    `}
                    style={{
                      backgroundColor: speaker.color + '20',
                      borderColor: speaker.color,
                      // @ts-expect-error CSS custom property for ring color
                      '--tw-ring-color': speaker.color,
                    }}
                  >
                    <span
                      className="text-sm font-bold"
                      style={{ color: speaker.color }}
                    >
                      {speaker.id + 1}
                    </span>
                  </div>
                  
                  {/* Active pulse */}
                  {isActive && (
                    <span
                      className="absolute inset-0 rounded-full animate-ping opacity-30"
                      style={{ backgroundColor: speaker.color }}
                    />
                  )}
                </div>

                {/* Speaker info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-[var(--color-text-primary)] truncate">
                      {speaker.name}
                    </span>
                    {isActive && (
                      <span
                        className="px-1.5 py-0.5 rounded text-xs font-medium animate-pulse"
                        style={{
                          backgroundColor: speaker.color + '20',
                          color: speaker.color,
                        }}
                      >
                        Speaking
                      </span>
                    )}
                  </div>

                  {/* Progress bar */}
                  <div className="relative h-2 bg-[var(--color-bg-tertiary)] rounded-full overflow-hidden">
                    <div
                      className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
                      style={{
                        width: `${percentage}%`,
                        backgroundColor: speaker.color,
                      }}
                    />
                  </div>
                </div>

                {/* Time and percentage */}
                <div className="flex-shrink-0 text-right">
                  <div
                    className="text-sm font-mono font-semibold"
                    style={{ color: speaker.color }}
                  >
                    {formatTime(speaker.totalTime)}
                  </div>
                  <div className="text-xs text-[var(--color-text-muted)]">
                    {percentage}%
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary footer */}
      <div className="px-4 py-3 bg-[var(--color-bg-tertiary)]/30 border-t border-[var(--color-border)]">
        <div className="flex items-center justify-between text-xs text-[var(--color-text-muted)]">
          <span>Total speaking time</span>
          <span className="font-mono font-medium text-[var(--color-text-secondary)]">
            {formatTime(totalSpeakingTime)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default SpeakerList;
