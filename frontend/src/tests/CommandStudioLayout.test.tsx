vi.mock('plotly.js-dist-min', () => ({
  default: {
    newPlot: vi.fn(),
    Plots: { resize: vi.fn() },
  },
}));

import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import ChatPage from '../pages/ChatPage';
import { DatasetProvider } from '../context/DatasetContext';

describe('Command Studio Responsive Layout', () => {
  it('renders studio layout grid with left panel controls and right panel workspace', () => {
    const { container } = render(
      <BrowserRouter>
        <DatasetProvider>
          <ChatPage />
        </DatasetProvider>
      </BrowserRouter>,
    );

    // Verify grid layout class exists
    const grid = container.querySelector('.studio-layout-grid');
    expect(grid).toBeInTheDocument();

    const leftPanel = container.querySelector('.studio-left-panel');
    expect(leftPanel).toBeInTheDocument();

    const rightPanel = container.querySelector('.studio-right-panel');
    expect(rightPanel).toBeInTheDocument();

    // Verify left panel key sections
    expect(screen.getByText('Dataset Source')).toBeInTheDocument();
    expect(screen.getByText('Enter Command')).toBeInTheDocument();
    expect(screen.getByText('Quick Command Inspiration')).toBeInTheDocument();

    // Verify right panel empty state
    expect(screen.getByText('Ready for Analysis')).toBeInTheDocument();
  });
});
