import React from 'react';
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import AnalysisResponseRenderer from '../components/analyst/AnalysisResponseRenderer';

describe('AnalysisResponseRenderer', () => {
  it('renders markdown table', () => {
    const markdown = `| date       | discount | product | region | sales | units |
| ---------- | -------- | ------- | ------ | ----- | ----- |
| 2025-01-01 | 5        | Laptop  | North  | 12000 | 10    |
| 2025-01-02 | 10       | Phone   | South  | 8000  | 20    |`;

    const { container } = render(<AnalysisResponseRenderer content={markdown} />);
    console.log('CONTAINER HTML:\n', container.innerHTML);
    const ths = container.querySelectorAll('th');
    console.log('TH COUNT:', ths.length);
    ths.forEach((th, i) => {
      console.log(`TH[${i}]:`, JSON.stringify(th.textContent));
    });
    const trs = container.querySelectorAll('tbody tr');
    console.log('TR COUNT:', trs.length);
  });
});
