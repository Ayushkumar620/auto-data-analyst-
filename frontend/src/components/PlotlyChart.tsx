import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

export type ChartSpec = {
  data: Partial<Plotly.PlotData>[];
  layout?: Partial<Plotly.Layout>;
};

export default function PlotlyChart({ data, layout }: ChartSpec) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      // Clear previous render
      ref.current.innerHTML = '';
      Plotly.newPlot(ref.current, data as Plotly.PlotData[], {
        responsive: true,
        margin: { t: 24, r: 12, b: 36, l: 48 },
        ...layout,
      });
    }
  }, [data, layout]);

  return <div ref={ref} className="plotly-chart" />;
}