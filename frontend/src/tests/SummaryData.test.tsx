import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import SummaryData from '../components/SummaryData';
import DatasetSummary from '../components/DatasetSummary';
import AnalysisResponseRenderer from '../components/analyst/AnalysisResponseRenderer';

describe('SummaryData Component', () => {
  const basicSalesData = {
    dataset_name: 'basic_sales_test.csv',
    rows: 10,
    columns: 6,
    nulls: 0,
    data_types: {
      date: 'string',
      product: 'string',
      region: 'string',
      sales: 'integer',
      units: 'integer',
      discount: 'integer',
    },
    preview: [
      { date: '2025-01-01', product: 'Laptop', region: 'North', sales: 12000, units: 10, discount: 5 },
      { date: '2025-01-02', product: 'Phone', region: 'South', sales: 8000, units: 20, discount: 10 },
      { date: '2025-01-03', product: 'Laptop', region: 'North', sales: 15000, units: 12, discount: 5 },
      { date: '2025-01-04', product: 'Tablet', region: 'East', sales: 6000, units: 15, discount: 8 },
      { date: '2025-01-05', product: 'Phone', region: 'South', sales: 9000, units: 22, discount: 10 },
      { date: '2025-01-06', product: 'Laptop', region: 'West', sales: 18000, units: 14, discount: 3 },
      { date: '2025-01-07', product: 'Tablet', region: 'East', sales: 7000, units: 17, discount: 8 },
      { date: '2025-01-08', product: 'Phone', region: 'North', sales: 11000, units: 25, discount: 5 },
      { date: '2025-01-09', product: 'Laptop', region: 'West', sales: 20000, units: 16, discount: 3 },
      { date: '2025-01-10', product: 'Tablet', region: 'South', sales: 7500, units: 18, discount: 7 },
    ],
  };

  it('renders SummaryData dashboard container with professional cards', () => {
    const { container } = render(<SummaryData data={basicSalesData} />);
    const dashboard = screen.getByTestId('summary-data-dashboard');
    expect(dashboard).toBeInTheDocument();
    expect(container.querySelector('.summary-box-card')).toBeInTheDocument();
    expect(container.querySelector('.column-types-card')).toBeInTheDocument();
    expect(container.querySelector('.data-preview-card')).toBeInTheDocument();
  });

  it('dynamically displays Rows, Columns, and Null Values in metric cards', () => {
    render(<SummaryData data={basicSalesData} />);
    const rowsEl = screen.getByTestId('summary-rows-value');
    const colsEl = screen.getByTestId('summary-columns-value');
    const nullsEl = screen.getByTestId('summary-nulls-value');

    expect(rowsEl.textContent).toBe('10');
    expect(colsEl.textContent).toBe('6');
    expect(nullsEl.textContent).toBe('0');
  });

  it('renders Columns & Types as a proper HTML table with separate cells', () => {
    const { container } = render(<SummaryData data={basicSalesData} />);
    const colTypesTable = container.querySelector('.column-types-card table');
    expect(colTypesTable).toBeInTheDocument();

    const ths = colTypesTable!.querySelectorAll('thead th');
    expect(ths.length).toBe(2);
    expect(ths[0].textContent?.trim()).toBe('Column');
    expect(ths[1].textContent?.trim()).toBe('Type');

    const rows = colTypesTable!.querySelectorAll('tbody tr');
    expect(rows.length).toBe(6);

    const firstRowTds = rows[0].querySelectorAll('td');
    expect(firstRowTds.length).toBe(2);
    expect(firstRowTds[0].textContent?.trim()).toBe('date');
    expect(firstRowTds[1].textContent?.trim()).toBe('string');

    // Verify discount is integer
    const discountRow = Array.from(rows).find((r) => r.textContent?.includes('discount'));
    expect(discountRow).toBeDefined();
    expect(discountRow!.textContent).toContain('integer');
  });

  it('renders Data Preview header with structured title and "First 10 rows" subtitle', () => {
    const { container } = render(<SummaryData data={basicSalesData} />);
    const header = container.querySelector('.data-preview-header');
    expect(header).toBeInTheDocument();

    const title = container.querySelector('.data-preview-title');
    expect(title).toBeInTheDocument();
    expect(title?.textContent).toBe('Data Preview');

    const subtitle = container.querySelector('.data-preview-subtitle');
    expect(subtitle).toBeInTheDocument();
    expect(subtitle?.textContent).toBe('First 10 rows');

    const countPill = container.querySelector('.data-preview-count-pill');
    expect(countPill).toBeInTheDocument();
    expect(countPill?.textContent).toContain('6 columns · 10 rows');
  });

  it('renders Data Preview as a proper HTML table with independent <th> cells and no merged header', () => {
    const { container } = render(<SummaryData data={basicSalesData} />);
    const previewTable = container.querySelector('.data-preview-card table');
    expect(previewTable).toBeInTheDocument();

    const thead = previewTable!.querySelector('thead');
    expect(thead).toBeInTheDocument();

    const ths = thead!.querySelectorAll('th');
    expect(ths.length).toBe(6);

    const headerNames = Array.from(ths).map((th) => th.textContent?.trim());
    expect(headerNames).toEqual(['date', 'product', 'region', 'sales', 'units', 'discount']);

    // Check that header is NOT merged
    const theadText = thead!.textContent?.replace(/\s+/g, '');
    expect(theadText).not.toBe('datediscountproductregionsalesunits');

    // Check rows count
    const tbodyRows = previewTable!.querySelectorAll('tbody tr');
    expect(tbodyRows.length).toBe(10);

    // Each row must have exactly 6 td elements
    tbodyRows.forEach((row) => {
      const tds = row.querySelectorAll('td');
      expect(tds.length).toBe(6);
    });
  });

  it('aligns numeric columns to the right and text/date columns to the left', () => {
    const { container } = render(<SummaryData data={basicSalesData} />);
    const previewTable = container.querySelector('.data-preview-card table');
    const ths = previewTable!.querySelectorAll('thead th');

    // date (text) -> left
    expect(ths[0]).toHaveStyle({ textAlign: 'left' });
    // product (text) -> left
    expect(ths[1]).toHaveStyle({ textAlign: 'left' });
    // sales (numeric) -> right
    expect(ths[3]).toHaveStyle({ textAlign: 'right' });
    // units (numeric) -> right
    expect(ths[4]).toHaveStyle({ textAlign: 'right' });

    // Verify tbody cells match alignment
    const firstRowTds = previewTable!.querySelectorAll('tbody tr')[0].querySelectorAll('td');
    expect(firstRowTds[0]).toHaveStyle({ textAlign: 'left' }); // date
    expect(firstRowTds[3]).toHaveStyle({ textAlign: 'right' }); // sales
  });

  it('renders null, undefined, and NaN as "—" while preserving 0, false, and empty string', () => {
    const dataWithSpecialValues = {
      dataset_name: 'test_special_values.csv',
      rows: 4,
      columns: 4,
      nulls: 3,
      data_types: { id: 'integer', label: 'string', count: 'integer', ratio: 'float' },
      preview: [
        { id: 1, label: 'Active', count: 0, ratio: 0.75 }, // valid 0 preserved
        { id: 2, label: null, count: 5, ratio: undefined }, // null and undefined -> '—'
        { id: 3, label: '', count: 12, ratio: NaN }, // empty string preserved, NaN -> '—'
        { id: 0, label: 'ZeroId', count: 0, ratio: 0 }, // 0 preserved
      ],
    };

    const { container } = render(<SummaryData data={dataWithSpecialValues} />);
    const previewTable = container.querySelector('.data-preview-card table');
    const rows = previewTable!.querySelectorAll('tbody tr');

    // Row 1: count is 0 -> should be rendered as "0"
    const row1Tds = rows[0].querySelectorAll('td');
    expect(row1Tds[2].textContent?.trim()).toBe('0');

    // Row 2: label is null -> "—", ratio is undefined -> "—"
    const row2Tds = rows[1].querySelectorAll('td');
    expect(row2Tds[1].textContent?.trim()).toBe('—');
    expect(row2Tds[3].textContent?.trim()).toBe('—');

    // Row 3: label is empty string -> "", ratio is NaN -> "—"
    const row3Tds = rows[2].querySelectorAll('td');
    expect(row3Tds[1].textContent?.trim()).toBe('');
    expect(row3Tds[3].textContent?.trim()).toBe('—');

    // Row 4: id 0 is preserved
    const row4Tds = rows[3].querySelectorAll('td');
    expect(row4Tds[0].textContent?.trim()).toBe('0');
  });

  it('supports wide datasets with 15 columns inside a horizontally scrollable container', () => {
    const wideColumns: Record<string, string> = {};
    const sampleRow: Record<string, unknown> = {};
    for (let i = 1; i <= 15; i++) {
      const colName = `feature_column_${i}`;
      wideColumns[colName] = i % 2 === 0 ? 'integer' : 'string';
      sampleRow[colName] = i % 2 === 0 ? i * 100 : `Value ${i}`;
    }

    const wideData = {
      dataset_name: 'wide_dataset.csv',
      rows: 50,
      columns: 15,
      nulls: 0,
      data_types: wideColumns,
      preview: [sampleRow],
    };

    const { container } = render(<SummaryData data={wideData} />);
    const previewContainer = container.querySelector('.data-preview-table-container');
    expect(previewContainer).toBeInTheDocument();
    expect(previewContainer).toHaveStyle({ overflowX: 'auto' });

    const ths = previewContainer!.querySelectorAll('th');
    expect(ths.length).toBe(15);
  });

  it('works with DatasetSummary component wrapper', () => {
    const profile = {
      dataset_name: 'basic_sales_test.csv',
      rows: 10,
      columns: 6,
      missing_values: 0,
      duplicates: 0,
      column_names: ['date', 'product', 'region', 'sales', 'units', 'discount'],
      data_types: {
        date: 'object',
        product: 'object',
        region: 'object',
        sales: 'int64',
        units: 'int64',
        discount: 'int64',
      },
      preview: basicSalesData.preview,
    };

    const { container } = render(<DatasetSummary profile={profile} />);
    expect(screen.getByTestId('summary-data-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('summary-rows-value').textContent).toBe('10');
    expect(screen.getByTestId('summary-columns-value').textContent).toBe('6');
    expect(screen.getByTestId('summary-nulls-value').textContent).toBe('0');
    expect(container.querySelector('.data-preview-card table')).toBeInTheDocument();
  });

  it('AnalysisResponseRenderer seamlessly renders SummaryData boxed layout when receiving summary markdown', () => {
    const summaryMarkdown = `📋 Summarydata

Rows 10
Columns 6
Nulls 0

#### Columns & Types

| Column | Type |
| date | string |
| discount | integer |
| product | string |
| region | string |
| sales | integer |
| units | integer |

#### Preview (first 10 rows)

| date | discount | product | region | sales | units |
| 2025-01-01 | 5 | Laptop | North | 12000 | 10 |
| 2025-01-02 | 10 | Phone | South | 8000 | 20 |
`;

    const { container } = render(<AnalysisResponseRenderer content={summaryMarkdown} />);
    expect(screen.getByTestId('summary-data-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('summary-rows-value').textContent).toBe('10');
    expect(screen.getByTestId('summary-columns-value').textContent).toBe('6');
    expect(screen.getByTestId('summary-nulls-value').textContent).toBe('0');

    // Confirm no merged header
    const previewThs = container.querySelectorAll('.data-preview-card thead th');
    expect(previewThs.length).toBe(6);
    expect(Array.from(previewThs).map((th) => th.textContent?.trim())).toEqual([
      'date',
      'discount',
      'product',
      'region',
      'sales',
      'units',
    ]);
  });

  it('AnalysisResponseRenderer preserves normal markdown rendering for regular text', () => {
    const regularMarkdown = `### Normal Analysis Heading

Here is an analysis with bullet points:
* Item 1
* Item 2

| Metric | Score |
| :--- | :---: |
| Accuracy | 0.94 |
`;

    const { container } = render(<AnalysisResponseRenderer content={regularMarkdown} />);
    expect(container.querySelector('h4')).toHaveTextContent('Normal Analysis Heading');
    expect(container.querySelectorAll('li').length).toBe(2);
    expect(container.querySelectorAll('th').length).toBe(2);
    // Not a summary dashboard
    expect(container.querySelector('[data-testid="summary-data-dashboard"]')).toBeNull();
  });
});
