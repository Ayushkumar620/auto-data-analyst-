import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
      test: {
    environment: 'jsdom',
    globals: true,
    env: {
      VITE_API_BASE_URL: 'http://localhost:8000',
    },
    setupFiles: ['./src/tests/setup.ts'],
    include: ['src/tests/**/*.test.tsx', 'src/tests/**/*.test.ts'],
  },
});