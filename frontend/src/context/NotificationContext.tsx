import React, { createContext, useCallback, useContext, useMemo, useReducer } from 'react';

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export type Notification = {
  id: string;
  message: string;
  type: NotificationType;
};

type State = { notifications: Notification[] };
type Action = { type: 'ADD'; notification: Notification } | { type: 'REMOVE'; id: string };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'ADD':
      return { notifications: [...state.notifications, action.notification] };
    case 'REMOVE':
      return { notifications: state.notifications.filter((n) => n.id !== action.id) };
    default:
      return state;
  }
}

type ContextValue = {
  notifications: Notification[];
  notify: (message: string, type?: NotificationType) => void;
  dismiss: (id: string) => void;
};

const NotificationContext = createContext<ContextValue | undefined>(undefined);

export function NotificationProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, { notifications: [] });

  const notify = useCallback((message: string, type: NotificationType = 'info') => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    dispatch({ type: 'ADD', notification: { id, message, type } });
    setTimeout(() => dispatch({ type: 'REMOVE', id }), 4500);
  }, []);

  const dismiss = useCallback(
    (id: string) => dispatch({ type: 'REMOVE', id }),
    [],
  );

  const value = useMemo(
    () => ({ notifications: state.notifications, notify, dismiss }),
    [state.notifications, notify, dismiss],
  );

  return (
    <NotificationContext.Provider value={value}>{children}</NotificationContext.Provider>
  );
}

export function useNotification(): ContextValue {
  const ctx = useContext(NotificationContext);
  if (!ctx) throw new Error('useNotification must be used within NotificationProvider');
  return ctx;
}
