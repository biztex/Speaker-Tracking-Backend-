import React, { useCallback, useState } from 'react';
import { ThemeProvider } from './contexts/ThemeContext';
import {
  Header,
  SessionTimer,
  SessionControls,
  Visualization,
  SpeakerList,
  SpeakerCountSelector,
  DebugOverlay,
} from './components';
import { useTimer, useBackendSession } from './hooks';

const AppContent: React.FC = () => {
  const [speakerCount, setSpeakerCount] = useState(2);
  const [showDebug, setShowDebug] = useState(false);
  
  const { time, start: startTimer, stop: stopTimer, reset: resetTimer, formatTime } = useTimer();
  
  const {
    status,
    speakers,
    currentSpeakerId,
    audioFeatures,
    error,
    start: startAudio,
    stop: stopAudio,
    reset: resetAudio,
    isVoiceActive,
  } = useBackendSession({ maxSpeakers: speakerCount });

  const isRunning = status === 'running';

  const handleStart = useCallback(async () => {
    await startAudio();
    startTimer();
  }, [startAudio, startTimer]);

  const handleStop = useCallback(() => {
    stopAudio();
    stopTimer();
  }, [stopAudio, stopTimer]);

  const handleReset = useCallback(() => {
    resetAudio();
    resetTimer();
  }, [resetAudio, resetTimer]);

  const handleSpeakerCountChange = useCallback((count: number) => {
    setSpeakerCount(count);
  }, []);

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)] text-[var(--color-text-primary)] transition-colors duration-300">
      <div className="flex flex-col min-h-screen">
        {/* Header */}
        <Header />

        {/* Main content */}
        <main className="flex-1 px-4 pb-8 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-4xl space-y-6 sm:space-y-8">
            {/* Timer Section */}
            <section className="flex flex-col items-center py-6 sm:py-8">
              <SessionTimer time={time} status={status} formatTime={formatTime} />
            </section>

            {/* Speaker Count Selector */}
            <section className="flex justify-center">
              <SpeakerCountSelector
                value={speakerCount}
                onChange={handleSpeakerCountChange}
                disabled={isRunning}
                min={2}
                max={5}
              />
            </section>

            {/* Controls Section */}
            <section className="flex justify-center">
              <SessionControls
                status={status}
                onStart={handleStart}
                onStop={handleStop}
                onReset={handleReset}
                error={error}
              />
            </section>

            {/* Visualization Section */}
            <section className="pt-4">
              <Visualization
                audioFeatures={audioFeatures}
                currentSpeakerId={currentSpeakerId}
                isActive={isVoiceActive && isRunning}
              />
            </section>

            {/* Speaker List Section */}
            <section>
              <SpeakerList
                speakers={speakers}
                currentSpeakerId={currentSpeakerId}
                formatTime={formatTime}
              />
            </section>

            {/* Footer */}
            <footer className="text-center py-6 text-xs text-[var(--color-text-muted)]">
              <p>Speaker Tracker v2.0 - Backend speaker tracking</p>
              <p className="mt-1 opacity-70">
                Audio is streamed to the backend for speaker recognition. No audio is recorded or stored.
              </p>
            </footer>
          </div>
        </main>
      </div>

      {/* Debug Overlay */}
      <DebugOverlay
        audioFeatures={audioFeatures}
        isVoiceActive={isVoiceActive}
        currentSpeakerId={currentSpeakerId}
        showDebug={showDebug}
        onToggle={() => setShowDebug(!showDebug)}
      />
    </div>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
};

export default App;
