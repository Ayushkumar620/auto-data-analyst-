declare module 'plotly.js-dist-min' {
  const Plotly: {
    newPlot(div: HTMLElement, data: unknown[], layout?: Record<string, unknown>): unknown;
    react: unknown;
    d3: unknown;
    extendTraces: unknown;
    deleteTraces: unknown;
    moves: unknown;
  };
  export default Plotly;
}
