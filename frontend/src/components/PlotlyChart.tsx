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
        margin: { t: 24, r: 12, b: 36, l: 48 },
        ...layout,
      });
    }
  }, [data, layout]);

  return <div ref={ref} className="plotly-chart" />;
}