import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

export type ChartSpec = {
  data: unknown[];
  layout?: Record<string, unknown>;
};

export default function PlotlyChart({ data, layout }: ChartSpec) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.innerHTML = '';
      Plotly.newPlot(ref.current, data, {
        responsive: true,
        autosizable: true,
        margin: { t: 24, r: 12, b: 36, l: 48 },
        ...layout,
      });

      const handleResize = () => {
        if (ref.current) {
          Plotly.Plots.resize(ref.current);
        }
      };

      const resizeObserver = new ResizeObserver(() => {
        handleResize();
      });

      resizeObserver.observe(ref.current);
      window.addEventListener('resize', handleResize);

      return () => {
        resizeObserver.disconnect();
        window.removeEventListener('resize', handleResize);
      };
    }
  }, [data, layout]);

  return <div ref={ref} className="plotly-chart" style={{ width: '100%', maxWidth: '100%', minWidth: 0 }} />;
}