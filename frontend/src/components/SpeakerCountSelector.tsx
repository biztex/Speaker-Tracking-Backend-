import React from 'react';

interface SpeakerCountSelectorProps {
  value: number;
  onChange: (count: number) => void;
  disabled?: boolean;
  min?: number;
  max?: number;
}

const SpeakerCountSelector: React.FC<SpeakerCountSelectorProps> = ({
  value,
  onChange,
  disabled = false,
  min = 2,
  max = 5,
}) => {
  const options = Array.from({ length: max - min + 1 }, (_, i) => min + i);

  return (
    <div className="flex items-center gap-3">
      <label
        htmlFor="speaker-count"
        className="text-sm font-medium text-[var(--color-text-secondary)]"
      >
        Number of Speakers:
      </label>
      <div className="relative">
        <select
          id="speaker-count"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
          className={`
            appearance-none
            px-4 py-2 pr-10
            rounded-xl
            bg-[var(--color-bg-secondary)]
            border border-[var(--color-border)]
            text-[var(--color-text-primary)]
            font-medium
            text-sm
            cursor-pointer
            transition-all duration-200
            ${disabled 
              ? 'opacity-50 cursor-not-allowed' 
              : 'hover:border-blue-500 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20'
            }
            focus:outline-none
          `}
        >
          {options.map((count) => (
            <option key={count} value={count}>
              {count} Speaker{count > 1 ? 's' : ''}
            </option>
          ))}
        </select>
        
        {/* Custom dropdown arrow */}
        <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
          <svg
            className={`w-4 h-4 transition-colors ${
              disabled ? 'text-[var(--color-text-muted)]' : 'text-[var(--color-text-secondary)]'
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </div>
      
      {disabled && (
        <span className="text-xs text-[var(--color-text-muted)] italic">
          (Stop session to change)
        </span>
      )}
    </div>
  );
};

export default SpeakerCountSelector;
