import React from 'react';
import type { SessionStatus } from '../types';

interface SessionTimerProps {
  time: number;
  status: SessionStatus;
  formatTime: (ms: number) => string;
}

const SessionTimer: React.FC<SessionTimerProps> = ({ time, status, formatTime }) => {
  const getStatusColor = () => {
    switch (status) {
      case 'running':
        return 'text-green-500';
      case 'stopped':
        return 'text-red-500';
      case 'paused':
        return 'text-amber-500';
      default:
        return 'text-[var(--color-text-muted)]';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'running':
        return 'Recording';
      case 'stopped':
        return 'Stopped';
      case 'paused':
        return 'Paused';
      default:
        return 'Ready';
    }
  };

  return (
    <div className="flex flex-col items-center gap-2">
      {/* Status indicator */}
      <div className="flex items-center gap-2">
        <span
          className={`w-2 h-2 rounded-full ${
            status === 'running'
              ? 'bg-green-500 animate-pulse'
              : status === 'stopped'
              ? 'bg-red-500'
              : 'bg-[var(--color-text-muted)]'
          }`}
        />
        <span className={`text-sm font-medium ${getStatusColor()}`}>
          {getStatusText()}
        </span>
      </div>

      {/* Timer display */}
      <div className="relative">
        <div
          className={`text-4xl sm:text-5xl lg:text-6xl font-mono font-bold tracking-wider ${
            status === 'running'
              ? 'text-[var(--color-text-primary)]'
              : 'text-[var(--color-text-secondary)]'
          }`}
        >
          {formatTime(time)}
        </div>
        
        {/* Glow effect when running */}
        {status === 'running' && (
          <div className="absolute inset-0 text-4xl sm:text-5xl lg:text-6xl font-mono font-bold tracking-wider text-blue-500/20 blur-lg -z-10">
            {formatTime(time)}
          </div>
        )}
      </div>

      {/* Session label */}
      <span className="text-xs text-[var(--color-text-muted)] uppercase tracking-widest">
        Session Duration
      </span>
    </div>
  );
};

export default SessionTimer;
