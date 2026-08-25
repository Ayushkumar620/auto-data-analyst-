import React from 'react';
import { useNotification } from '../../context/NotificationContext';
import { IconCheck, IconAlertTriangle, IconInfo, IconX } from './Icons';
import type { NotificationType } from '../../context/NotificationContext';

type TypeConfig = {
  cls: string;
  Icon: React.ComponentType<{ size?: number; 'aria-hidden'?: boolean }>;
};

const typeConfig: Record<NotificationType, TypeConfig> = {
  success: { cls: 'toast-success', Icon: IconCheck },
  error: { cls: 'toast-error', Icon: IconAlertTriangle },
  warning: { cls: 'toast-warning', Icon: IconAlertTriangle },
  info: { cls: 'toast-info', Icon: IconInfo },
};

export default function NotificationToast() {
  const { notifications, dismiss } = useNotification();

  if (!notifications.length) return null;

  return (
    <div className="toast-container" role="region" aria-label="Notifications" aria-live="polite">
      {notifications.map((n) => {
        const { cls, Icon } = typeConfig[n.type];
        return (
          <div key={n.id} className={`toast ${cls}`} role="status">
            <Icon size={16} aria-hidden />
            <span className="toast-msg">{n.message}</span>
            <button
              className="toast-dismiss"
              onClick={() => dismiss(n.id)}
              aria-label="Dismiss notification"
              type="button"
            >
              <IconX size={14} aria-hidden />
            </button>
          </div>
        );
      })}
    </div>
  );
}
