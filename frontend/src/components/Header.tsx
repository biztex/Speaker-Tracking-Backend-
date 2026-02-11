import React from 'react';
import { useTheme } from '../contexts/ThemeContext';

const Header: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="w-full px-4 py-4 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-4xl flex items-center justify-between">
        {/* Logo and Title */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
              <svg
                className="w-5 h-5 sm:w-6 sm:h-6 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                />
              </svg>
            </div>
            {/* Decorative ring */}
            <div className="absolute -inset-1 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-600/20 blur-sm -z-10" />
          </div>
          <div>
            <h1 className="text-lg sm:text-xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
              Speaker Tracker
            </h1>
            <p className="text-xs text-[var(--color-text-muted)] hidden sm:block">
              Real-time voice analysis
            </p>
          </div>
        </div>

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="relative w-14 h-8 sm:w-16 sm:h-9 rounded-full bg-[var(--color-bg-tertiary)] transition-all duration-300 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-[var(--color-bg-primary)]"
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {/* Sun icon */}
          <span
            className={`absolute left-1 top-1 w-6 h-6 sm:w-7 sm:h-7 flex items-center justify-center transition-all duration-300 ${
              theme === 'light'
                ? 'opacity-100 transform translate-x-0'
                : 'opacity-0 transform -translate-x-2'
            }`}
          >
            <svg
              className="w-4 h-4 sm:w-5 sm:h-5 text-amber-500"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"
                clipRule="evenodd"
              />
            </svg>
          </span>

          {/* Moon icon */}
          <span
            className={`absolute right-1 top-1 w-6 h-6 sm:w-7 sm:h-7 flex items-center justify-center transition-all duration-300 ${
              theme === 'dark'
                ? 'opacity-100 transform translate-x-0'
                : 'opacity-0 transform translate-x-2'
            }`}
          >
            <svg
              className="w-4 h-4 sm:w-5 sm:h-5 text-blue-400"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
            </svg>
          </span>

          {/* Toggle ball */}
          <span
            className={`absolute top-1 w-6 h-6 sm:w-7 sm:h-7 rounded-full bg-white shadow-md transition-all duration-300 ${
              theme === 'dark'
                ? 'left-1 translate-x-0'
                : 'left-1 translate-x-6 sm:translate-x-7'
            }`}
          />
        </button>
      </div>
    </header>
  );
};

export default Header;
