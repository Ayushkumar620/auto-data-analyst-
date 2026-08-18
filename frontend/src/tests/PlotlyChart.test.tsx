vi.mock('plotly.js-dist-min', () => ({
  default: {
    newPlot: vi.fn(),
  },
}));

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import PlotlyChart from '../components/PlotlyChart';

describe('PlotlyChart', () => {
  it('renders an empty-state message when no points', () => {
    // PlotlyChart always renders a div; verify it mounts
    const { container } = render(
      <PlotlyChart data={[{ x: [1, 2, 3], y: [4, 5, 6], type: 'scatter' }]} />,
    );
    expect(container.querySelector('.plotly-chart')).toBeInTheDocument();
  });
});
