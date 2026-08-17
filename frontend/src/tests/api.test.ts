import { describe, it, expect } from 'vitest';
import { buildApiUrl, joinUrl, getAuthToken, setAuthToken } from '../services/api';

describe('api helpers', () => {
  it('joins absolute paths (with leading slash) to the base without double slashes', () => {
    expect(joinUrl('http://localhost:8000', '/api/v1/auth/login')).toBe(
      'http://localhost:8000/api/v1/auth/login',
    );
  });

  it('joins relative paths by inserting a separating slash', () => {
    expect(joinUrl('http://localhost:8000', 'api/v1/foo')).toBe('http://localhost:8000/api/v1/foo');
  });

  it('buildApiUrl produces a url ending with the requested path', () => {
    expect(buildApiUrl('/api/v1/auth/login').endsWith('/api/v1/auth/login')).toBe(true);
  });

  it('stores and clears auth tokens in localStorage', () => {
    setAuthToken('abc123');
    expect(getAuthToken()).toBe('abc123');
    setAuthToken(null);
    expect(getAuthToken()).toBeNull();
  });
});
