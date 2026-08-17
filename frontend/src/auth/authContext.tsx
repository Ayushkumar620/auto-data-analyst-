import React, { createContext, useCallback, useEffect, useMemo, useState } from 'react';
import {
  fetchCurrentUser,
  login as loginApi,
  register as registerApi,
  logout as logoutApi,
  type LoginRequest,
  type TokenResponse,
  type UserCreate,
  type UserOut,
} from '../services/authService';
import { getAuthToken, setAuthToken } from '../services/api';

type AuthContextValue = {
  user: UserOut | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (payload: LoginRequest) => Promise<void>;
  register: (payload: UserCreate) => Promise<void>;
  logout: () => void;
};

export const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserOut | null>(null);
  const [token, setToken] = useState<string | null>(getAuthToken());
  const [isLoading, setIsLoading] = useState(true);

  const hydrate = useCallback(async () => {
    if (!getAuthToken()) {
      setToken(null);
      setUser(null);
      setIsLoading(false);
      return;
    }
    try {
      const me = await fetchCurrentUser();
      setUser(me);
      setToken(getAuthToken());
    } catch {
      logoutApi();
      setUser(null);
      setToken(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  const login = useCallback(async (payload: LoginRequest) => {
    const res: TokenResponse = await loginApi(payload);
    setAuthToken(res.access_token);
    setToken(res.access_token);
    setUser(res.user);
  }, []);

  const register = useCallback(async (payload: UserCreate) => {
    const res: TokenResponse = await registerApi(payload);
    setAuthToken(res.access_token);
    setToken(res.access_token);
    setUser(res.user);
  }, []);

  const logout = useCallback(() => {
    logoutApi();
    setToken(null);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={useMemo(
        () => ({
          user,
          token,
          isAuthenticated: !!user && !!token,
          isLoading,
          login,
          register,
          logout,
        }),
        [user, token, isLoading, login, register, logout],
      )}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = React.useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
}