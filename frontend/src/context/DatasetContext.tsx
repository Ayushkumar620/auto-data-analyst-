import React, { createContext, useCallback, useContext, useMemo, useReducer } from 'react';
import type { DatasetProfile } from '../types';

type DatasetState = {
  profile: DatasetProfile | null;
  fileName: string | null;
};

type DatasetAction =
  | { type: 'SET_DATASET'; profile: DatasetProfile; fileName: string }
  | { type: 'CLEAR_DATASET' };

function reducer(state: DatasetState, action: DatasetAction): DatasetState {
  switch (action.type) {
    case 'SET_DATASET':
      return { profile: action.profile, fileName: action.fileName };
    case 'CLEAR_DATASET':
      return { profile: null, fileName: null };
    default:
      return state;
  }
}

type DatasetContextValue = DatasetState & {
  setDataset: (profile: DatasetProfile, fileName: string) => void;
  clearDataset: () => void;
};

const DatasetContext = createContext<DatasetContextValue | undefined>(undefined);

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, { profile: null, fileName: null });

  const setDataset = useCallback(
    (profile: DatasetProfile, fileName: string) =>
      dispatch({ type: 'SET_DATASET', profile, fileName }),
    [],
  );

  const clearDataset = useCallback(() => dispatch({ type: 'CLEAR_DATASET' }), []);

  const value = useMemo(
    () => ({ ...state, setDataset, clearDataset }),
    [state, setDataset, clearDataset],
  );

  return <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>;
}

export function useDataset(): DatasetContextValue {
  const ctx = useContext(DatasetContext);
  if (!ctx) throw new Error('useDataset must be used within DatasetProvider');
  return ctx;
}
