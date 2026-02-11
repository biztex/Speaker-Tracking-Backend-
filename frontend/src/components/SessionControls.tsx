import React from 'react';
import type { SessionStatus } from '../types';

interface SessionControlsProps {
  status: SessionStatus;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  disabled?: boolean;
  error?: string | null;
}

const SessionControls: React.FC<SessionControlsProps> = ({
  status,
  onStart,
  onStop,
  onReset,
  disabled = false,
  error,
}) => {
  const isIdle = status === 'idle';
  const isRunning = status === 'running';
  const isStopped = status === 'stopped';

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Error message */}
      {error && (
        <div className="px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-sm animate-fade-in">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Control buttons */}
      <div className="flex items-center gap-3 sm:gap-4">
        {/* Start Button */}
        <button
          onClick={onStart}
          disabled={disabled || isRunning}
          className={`
            relative px-6 py-3 sm:px-8 sm:py-4 rounded-xl font-semibold text-sm sm:text-base
            transition-all duration-300 transform
            ${
              isRunning || disabled
                ? 'bg-[var(--color-bg-tertiary)] text-[var(--color-text-muted)] cursor-not-allowed'
                : 'bg-gradient-to-r from-green-500 to-emerald-600 text-white hover:shadow-lg hover:shadow-green-500/25 hover:-translate-y-0.5 active:translate-y-0'
            }
            focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-[var(--color-bg-primary)]
          `}
        >
          <span className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            Start
          </span>
          
          {/* Decorative pulse when idle */}
          {isIdle && !disabled && (
            <span className="absolute inset-0 rounded-xl bg-green-500/20 animate-pulse-ring -z-10" />
          )}
        </button>

        {/* Stop Button */}
        <button
          onClick={onStop}
          disabled={disabled || !isRunning}
          className={`
            px-6 py-3 sm:px-8 sm:py-4 rounded-xl font-semibold text-sm sm:text-base
            transition-all duration-300 transform
            ${
              !isRunning || disabled
                ? 'bg-[var(--color-bg-tertiary)] text-[var(--color-text-muted)] cursor-not-allowed'
                : 'bg-gradient-to-r from-red-500 to-rose-600 text-white hover:shadow-lg hover:shadow-red-500/25 hover:-translate-y-0.5 active:translate-y-0'
            }
            focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-[var(--color-bg-primary)]
          `}
        >
          <span className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
              />
            </svg>
            Stop
          </span>
        </button>

        {/* Reset Button (only show when stopped) */}
        {isStopped && (
          <button
            onClick={onReset}
            disabled={disabled}
            className="
              px-4 py-3 sm:px-6 sm:py-4 rounded-xl font-semibold text-sm sm:text-base
              bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]
              transition-all duration-300 transform
              hover:bg-[var(--color-border)] hover:-translate-y-0.5 active:translate-y-0
              focus:outline-none focus:ring-2 focus:ring-[var(--color-border)] focus:ring-offset-2 focus:ring-offset-[var(--color-bg-primary)]
              animate-fade-in
            "
          >
            <span className="flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Reset
            </span>
          </button>
        )}
      </div>

      {/* Help text */}
      <p className="text-xs text-[var(--color-text-muted)] text-center max-w-xs">
        {isIdle && 'Press Start to begin tracking speakers'}
        {isRunning && 'Listening for speakers...'}
        {isStopped && 'Session complete. Reset to start a new session.'}
      </p>
    </div>
  );
};

export default SessionControls;
