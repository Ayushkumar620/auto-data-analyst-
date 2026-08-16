import React from 'react';

type ForecastPoint = {
  date: string;
  prediction: number;
  lower: number;
  upper: number;
};

type ForecastChartProps = {
  points: ForecastPoint[];
  target: string;
};

function toPolyline(points: string[]): string {
  return points.join(' ');
}

export default function ForecastChart({ points, target }: ForecastChartProps) {
  if (!points.length) {
    return <p className="muted">No forecast points returned.</p>;
  }

  const width = 760;
  const height = 260;
  const paddingX = 36;
  const paddingTop = 18;
  const paddingBottom = 30;
  const chartWidth = width - paddingX * 2;
  const chartHeight = height - paddingTop - paddingBottom;

  const minValue = Math.min(...points.map((p) => p.lower));
  const maxValue = Math.max(...points.map((p) => p.upper));
  const span = Math.max(maxValue - minValue, 1);

  const xAt = (index: number): number => {
    if (points.length === 1) {
      return paddingX + chartWidth / 2;
    }
    return paddingX + (index / (points.length - 1)) * chartWidth;
  };

  const yAt = (value: number): number => {
    const normalized = (value - minValue) / span;
    return paddingTop + (1 - normalized) * chartHeight;
  };

  const predictionPath = toPolyline(points.map((point, index) => `${xAt(index)},${yAt(point.prediction)}`));
  const upperPath = points.map((point, index) => `${xAt(index)},${yAt(point.upper)}`);
  const lowerPath = points
    .map((point, index) => `${xAt(index)},${yAt(point.lower)}`)
    .reverse();
  const areaPath = toPolyline([...upperPath, ...lowerPath]);

  return (
    <div className="forecast-chart-shell">
      <svg className="forecast-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${target} forecast chart`}>
        <line x1={paddingX} y1={height - paddingBottom} x2={width - paddingX} y2={height - paddingBottom} stroke="#8b9d91" strokeWidth="1" />
        <line x1={paddingX} y1={paddingTop} x2={paddingX} y2={height - paddingBottom} stroke="#8b9d91" strokeWidth="1" />
        <polygon points={areaPath} fill="rgba(12, 125, 102, 0.2)" />
        <polyline points={predictionPath} fill="none" stroke="#0c7d66" strokeWidth="3" />

        {points.map((point, index) => (
          <g key={point.date}>
            <circle cx={xAt(index)} cy={yAt(point.prediction)} r={3} fill="#0c7d66" />
            <text x={xAt(index)} y={height - 10} textAnchor="middle" className="axis-label">
              {point.date.slice(5)}
            </text>
          </g>
        ))}

        <text x={paddingX + 8} y={paddingTop + 12} className="axis-label">
          max {maxValue.toFixed(2)}
        </text>
        <text x={paddingX + 8} y={height - paddingBottom - 6} className="axis-label">
          min {minValue.toFixed(2)}
        </text>
      </svg>
      <p className="muted">Solid line: prediction. Shaded area: confidence interval.</p>
    </div>
  );
}
