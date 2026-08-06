import React from 'react';

type PreviewTableProps = {
  preview: Array<Record<string, unknown>>;
};

export default function PreviewTable({ preview }: PreviewTableProps) {
  if (!preview.length) {
    return <p>No preview available.</p>;
  }

  const columns = Object.keys(preview[0]);
  return (
    <table>
      <thead>
        <tr>
          {columns.map((column) => (
            <th key={column}>{column}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {preview.map((row, index) => (
          <tr key={index}>
            {columns.map((column) => (
              <td key={column}>{String(row[column] ?? '')}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
