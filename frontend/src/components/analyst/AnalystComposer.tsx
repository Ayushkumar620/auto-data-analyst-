import React, { useRef, useEffect } from 'react';
import Spinner from '../ui/Spinner';

type AnalystComposerProps = {
  value: string;
  onChange: (val: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
  placeholder?: string;
};

export default function AnalystComposer({
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder = 'Ask your data anything (e.g. "Analyze revenue drivers", "Why?", "What if marketing increases by 15%?")...',
}: AnalystComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!disabled && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [disabled]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !disabled) {
        onSubmit();
      }
    }
  };

  return (
    <div
      style={{
        borderRadius: '16px',
        border: '1px solid rgba(226, 232, 240, 0.9)',
        backgroundColor: '#ffffff',
        boxShadow: '0 4px 20px rgba(15, 23, 42, 0.06)',
        padding: '0.65rem 0.85rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem',
      }}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={2}
        aria-label="Analytical command composer"
        style={{
          width: '100%',
          border: 'none',
          outline: 'none',
          resize: 'none',
          fontFamily: 'inherit',
          fontSize: '0.92rem',
          color: 'var(--ink)',
          lineHeight: '1.45',
          backgroundColor: 'transparent',
        }}
      />

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="muted" style={{ fontSize: '0.74rem' }}>
          Press <kbd style={{ fontFamily: 'var(--font-mono)', padding: '0.1rem 0.3rem', background: '#f1f5f9', borderRadius: '4px' }}>Enter</kbd> to send, <kbd style={{ fontFamily: 'var(--font-mono)', padding: '0.1rem 0.3rem', background: '#f1f5f9', borderRadius: '4px' }}>Shift+Enter</kbd> for new line
        </span>

        <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
          {value && (
            <button
              type="button"
              onClick={() => onChange('')}
              disabled={disabled}
              className="ghost-text-btn"
              style={{ fontSize: '0.78rem' }}
            >
              Clear
            </button>
          )}

          <button
            type="button"
            onClick={onSubmit}
            disabled={!value.trim() || disabled}
            className="primary-btn"
            style={{
              padding: '0.4rem 0.85rem',
              fontSize: '0.84rem',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
            }}
          >
            {disabled ? (
              <>
                <Spinner size={14} /> Analyzing…
              </>
            ) : (
              'Send ↵'
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

