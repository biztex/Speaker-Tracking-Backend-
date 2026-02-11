import { useState, useRef, useCallback, useEffect } from 'react';

interface UseTimerReturn {
  time: number; // elapsed time in milliseconds
  isRunning: boolean;
  start: () => void;
  stop: () => void;
  reset: () => void;
  formatTime: (ms: number) => string;
}

export function useTimer(): UseTimerReturn {
  const [time, setTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const startTimeRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const accumulatedTimeRef = useRef(0);

  const updateTimer = useCallback(() => {
    if (startTimeRef.current !== null) {
      const elapsed = Date.now() - startTimeRef.current + accumulatedTimeRef.current;
      setTime(elapsed);
      animationFrameRef.current = requestAnimationFrame(updateTimer);
    }
  }, []);

  const start = useCallback(() => {
    if (!isRunning) {
      startTimeRef.current = Date.now();
      setIsRunning(true);
      animationFrameRef.current = requestAnimationFrame(updateTimer);
    }
  }, [isRunning, updateTimer]);

  const stop = useCallback(() => {
    if (isRunning && startTimeRef.current !== null) {
      accumulatedTimeRef.current += Date.now() - startTimeRef.current;
      startTimeRef.current = null;
      setIsRunning(false);
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    }
  }, [isRunning]);

  const reset = useCallback(() => {
    setTime(0);
    setIsRunning(false);
    startTimeRef.current = null;
    accumulatedTimeRef.current = 0;
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  const formatTime = useCallback((ms: number): string => {
    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) {
      return `${hours.toString().padStart(2, '0')}:${minutes
        .toString()
        .padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    return `${minutes.toString().padStart(2, '0')}:${seconds
      .toString()
      .padStart(2, '0')}`;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return {
    time,
    isRunning,
    start,
    stop,
    reset,
    formatTime,
  };
}

export default useTimer;
