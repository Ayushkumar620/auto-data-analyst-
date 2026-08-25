import React, { useState } from 'react';
import { IconAlertTriangle } from './Icons';

type ErrorStateProps = {
  message: string;
  detail?: string;
  onRetry?: () => void;
};

export default function ErrorState({ message, detail, onRetry }: ErrorStateProps) {
  const [showDetail, setShowDetail] = useState(false);

  return (
    <div className="error-state" role="alert">
      <div className="error-state-icon">
        <IconAlertTriangle size={28} aria-hidden />
      </div>
      <p className="error-state-msg">{message}</p>
      <div className="error-state-actions">
        {onRetry && (
          <button className="action-btn" onClick={onRetry} type="button">
            Try again
          </button>
        )}
        {detail && (
          <button
            className="ghost-text-btn"
            onClick={() => setShowDetail((s) => !s)}
            type="button"
            aria-expanded={showDetail}
          >
            {showDetail ? 'Hide details' : 'Show details'}
          </button>
        )}
      </div>
      {showDetail && detail && <pre className="error-detail">{detail}</pre>}
    </div>
  );
}
