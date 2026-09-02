import React from 'react';
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import AnalysisResponseRenderer from '../components/analyst/AnalysisResponseRenderer';

describe('AnalysisResponseRenderer - Markdown & Table Rendering', () => {
  it('fixes the header merging bug: renders separate <th> cells for each column', () => {
    const markdown = `| date       | discount | product | region | sales | units |
| ---------- | -------- | ------- | ------ | ----- | ----- |
| 2025-01-01 | 5        | Laptop  | North  | 12000 | 10    |
| 2025-01-02 | 10       | Phone   | South  | 8000  | 20    |`;

    const { container } = render(<AnalysisResponseRenderer content={markdown} />);

    // Must be real HTML table structure
    const table = container.querySelector('table.result-table');
    expect(table).not.toBeNull();
    expect(container.querySelector('thead')).not.toBeNull();
    expect(container.querySelector('tbody')).not.toBeNull();

    // Headers must be 6 distinct <th> elements, NOT merged into one string
    const ths = container.querySelectorAll('thead th');
    expect(ths.length).toBe(6);
    expect(ths[0].textContent).toBe('date');
    expect(ths[1].textContent).toBe('discount');
    expect(ths[2].textContent).toBe('product');
    expect(ths[3].textContent).toBe('region');
    expect(ths[4].textContent).toBe('sales');
    expect(ths[5].textContent).toBe('units');

    // Data rows and cells must be present and correctly ordered
    const rows = container.querySelectorAll('tbody tr');
    expect(rows.length).toBe(2);

    const firstRowCells = rows[0].querySelectorAll('td');
    expect(firstRowCells.length).toBe(6);
    expect(firstRowCells[0].textContent).toBe('2025-01-01');
    expect(firstRowCells[1].textContent).toBe('5');
    expect(firstRowCells[2].textContent).toBe('Laptop');
    expect(firstRowCells[3].textContent).toBe('North');
    expect(firstRowCells[4].textContent).toBe('12000');
    expect(firstRowCells[5].textContent).toBe('10');

    const secondRowCells = rows[1].querySelectorAll('td');
    expect(secondRowCells.length).toBe(6);
    expect(secondRowCells[0].textContent).toBe('2025-01-02');
    expect(secondRowCells[1].textContent).toBe('10');
    expect(secondRowCells[2].textContent).toBe('Phone');
    expect(secondRowCells[3].textContent).toBe('South');
    expect(secondRowCells[4].textContent).toBe('8000');
    expect(secondRowCells[5].textContent).toBe('20');
  });

  // TEST 1 — normal table with alignments
  it('TEST 1: renders normal table with column alignments', () => {
    const markdown = `| Name  | Age | City   |
| :---- | --: | :----: |
| Alice |  25 | Delhi  |
| Bob   |  30 | Mumbai |`;

    const { container } = render(<AnalysisResponseRenderer content={markdown} />);
    const ths = container.querySelectorAll('thead th');
    expect(ths.length).toBe(3);
    expect(ths[0].textContent).toBe('Name');
    expect(ths[1].textContent).toBe('Age');
    expect(ths[2].textContent).toBe('City');

    // Check alignments
    expect(ths[0]).toHaveStyle({ textAlign: 'left' });
    expect(ths[1]).toHaveStyle({ textAlign: 'right' });
    expect(ths[2]).toHaveStyle({ textAlign: 'center' });

    const rows = container.querySelectorAll('tbody tr');
    expect(rows.length).toBe(2);
    expect(rows[0].querySelectorAll('td')[1].textContent).toBe('25');
    expect(rows[0].querySelectorAll('td')[1]).toHaveStyle({ textAlign: 'right' });
    expect(rows[1].querySelectorAll('td')[2].textContent).toBe('Mumbai');
    expect(rows[1].querySelectorAll('td')[2]).toHaveStyle({ textAlign: 'center' });
  });

  // TEST 2 — wide table (at least 10 columns)
  it('TEST 2: renders wide table inside a horizontally scrollable container', () => {
    const markdown = `| Col1 | Col2 | Col3 | Col4 | Col5 | Col6 | Col7 | Col8 | Col9 | Col10 | Col11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 | A11 |
| B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 | B11 |`;

    const { container } = render(<AnalysisResponseRenderer content={markdown} />);
    const scrollContainer = container.querySelector('.table-responsive-container');
    expect(scrollContainer).not.toBeNull();
    expect(scrollContainer).toHaveStyle({ overflowX: 'auto' });

    const ths = container.querySelectorAll('thead th');
    expect(ths.length).toBe(11);
    expect(ths[0].textContent).toBe('Col1');
    expect(ths[10].textContent).toBe('Col11');

    const rows = container.querySelectorAll('tbody tr');
    expect(rows.length).toBe(2);
    expect(rows[0].querySelectorAll('td').length).toBe(11);
    expect(rows[1].querySelectorAll('td').length).toBe(11);
  });

  // TEST 3 — long column names
  it('TEST 3: renders long column names without collapsing or merging', () => {
    const markdown = `| customer_segment | total_operating_cost | average_conversion_rate |
| ---------------- | -------------------: | ----------------------: |
| Enterprise       |               150000 |                    4.52 |`;

    const { container } = render(<AnalysisResponseRenderer content={markdown} />);
    const ths = container.querySelectorAll('thead th');
    expect(ths.length).toBe(3);
    expect(ths[0].textContent).toBe('customer_segment');
    expect(ths[1].textContent).toBe('total_operating_cost');
    expect(ths[2].textContent).toBe('average_conversion_rate');

    const cells = container.querySelectorAll('tbody td');
    expect(cells.length).toBe(3);
    expect(cells[0].textContent).toBe('Enterprise');
    expect(cells[1].textContent).toBe('150000');
    expect(cells[2].textContent).toBe('4.52');
  });

  // TEST 4 — empty / null values
  it('TEST 4: renders empty/null values gracefully without misaligning rows', () => {
    const markdown = `| Product | Sales | Notes    |
| ------- | ----: | -------- |
| Laptop  | 12000 |          |
| Phone   |       | Returned |`;

    const { container } = render(<AnalysisResponseRenderer content={markdown} />);
    const rows = container.querySelectorAll('tbody tr');
    expect(rows.length).toBe(2);

    const row1 = rows[0].querySelectorAll('td');
    expect(row1.length).toBe(3);
    expect(row1[0].textContent).toBe('Laptop');
    expect(row1[1].textContent).toBe('12000');
    // Empty note cell renders placeholder dash
    expect(row1[2].textContent).toBe('—');

    const row2 = rows[1].querySelectorAll('td');
    expect(row2.length).toBe(3);
    expect(row2[0].textContent).toBe('Phone');
    // Empty sales cell renders placeholder dash
    expect(row2[1].textContent).toBe('—');
    expect(row2[2].textContent).toBe('Returned');
  });

  // TEST 5 — mixed Markdown (headings, text, tables, lists, code blocks)
  it('TEST 5: renders mixed Markdown properly without breaking layout', () => {
    const markdown = `# Analysis

Some explanatory text.

| Product | Sales |
| ------- | ----: |
| Laptop  | 12000 |
| Phone   |  8000 |

* Finding one
* Finding two

\`\`\`text
example code
\`\`\``;

    const { container } = render(<AnalysisResponseRenderer content={markdown} />);

    // Heading
    const h2 = container.querySelector('h2');
    expect(h2).not.toBeNull();
    expect(h2?.textContent).toBe('Analysis');

    // Paragraph
    expect(container.textContent).toContain('Some explanatory text.');

    // Table
    const table = container.querySelector('table');
    expect(table).not.toBeNull();
    const ths = container.querySelectorAll('thead th');
    expect(ths.length).toBe(2);
    expect(ths[0].textContent).toBe('Product');
    expect(ths[1].textContent).toBe('Sales');

    const tds = container.querySelectorAll('tbody td');
    expect(tds.length).toBe(4);
    expect(tds[0].textContent).toBe('Laptop');
    expect(tds[1].textContent).toBe('12000');

    // Bullet list
    const ul = container.querySelector('ul');
    expect(ul).not.toBeNull();
    const lis = ul?.querySelectorAll('li');
    expect(lis?.length).toBe(2);
    expect(lis?.[0].textContent).toBe('Finding one');
    expect(lis?.[1].textContent).toBe('Finding two');

    // Code block
    const pre = container.querySelector('pre');
    expect(pre).not.toBeNull();
    expect(pre?.textContent).toContain('example code');
  });

  it('renders tables without leading/trailing outer pipes', () => {
    const markdown = `Product | Quantity | Price
--- | :---: | ---:
Monitor | 25 | $320
Keyboard | 50 | $45`;

    const { container } = render(<AnalysisResponseRenderer content={markdown} />);
    const ths = container.querySelectorAll('thead th');
    expect(ths.length).toBe(3);
    expect(ths[0].textContent).toBe('Product');
    expect(ths[1].textContent).toBe('Quantity');
    expect(ths[2].textContent).toBe('Price');

    const rows = container.querySelectorAll('tbody tr');
    expect(rows.length).toBe(2);
    expect(rows[0].querySelectorAll('td')[0].textContent).toBe('Monitor');
  });

  it('handles inline formatting inside table cells and paragraphs', () => {
    const markdown = `| Feature | Status | Link |
| --- | --- | --- |
| **Speed** | *Active* | [Docs](https://example.com) |
| \`memory\` | Normal | \`ok\` |`;

    const { container } = render(<AnalysisResponseRenderer content={markdown} />);
    const strong = container.querySelector('strong');
    expect(strong?.textContent).toBe('Speed');

    const em = container.querySelector('em');
    expect(em?.textContent).toBe('Active');

    const link = container.querySelector('a');
    expect(link?.getAttribute('href')).toBe('https://example.com');
    expect(link?.textContent).toBe('Docs');

    const code = container.querySelector('code');
    expect(code?.textContent).toBe('memory');
  });

  it('renders GitHub alerts (> [!NOTE]) properly', () => {
    const markdown = `> [!NOTE]
> All statistical calculations use two-tailed hypothesis tests.`;

    const { container } = render(<AnalysisResponseRenderer content={markdown} />);
    expect(container.textContent).toContain('NOTE');
    expect(container.textContent).toContain('All statistical calculations use two-tailed hypothesis tests.');
  });
});
