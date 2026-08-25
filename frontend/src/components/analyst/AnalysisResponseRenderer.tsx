import React from 'react';

type AnalysisResponseRendererProps = {
  content: string;
};

export default function AnalysisResponseRenderer({ content }: AnalysisResponseRendererProps) {
  if (!content) return null;

  // Split lines to detect markdown tables, alerts, and paragraphs
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];

  let inTable = false;
  let tableHeader: string[] = [];
  let tableRows: string[][] = [];
  let inAlert: { type: string; lines: string[] } | null = null;
  let textBuffer: string[] = [];

  const flushTextBuffer = () => {
    if (textBuffer.length > 0) {
      const paragraph = textBuffer.join('\n').trim();
      if (paragraph) {
        elements.push(
          <div key={`p-${elements.length}`} style={{ marginBottom: '0.65rem', lineHeight: '1.55' }}>
            {renderInlineMarkdown(paragraph)}
          </div>,
        );
      }
      textBuffer = [];
    }
  };

  const flushTable = () => {
    if (inTable && tableHeader.length > 0) {
      elements.push(
        <div
          key={`table-${elements.length}`}
          style={{ overflowX: 'auto', margin: '0.85rem 0', borderRadius: '8px', border: '1px solid #e2e8f0' }}
        >
          <table className="result-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {tableHeader.map((h, i) => (
                  <th key={i} style={{ padding: '0.45rem 0.75rem', fontSize: '0.82rem', textAlign: 'left' }}>
                    {renderInlineMarkdown(h.trim())}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((row, rIdx) => (
                <tr key={rIdx}>
                  {row.map((cell, cIdx) => (
                    <td key={cIdx} style={{ padding: '0.45rem 0.75rem', fontSize: '0.82rem' }}>
                      {renderInlineMarkdown(cell.trim())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      inTable = false;
      tableHeader = [];
      tableRows = [];
    }
  };

  const flushAlert = () => {
    if (inAlert) {
      const alertType = inAlert.type.toUpperCase();
      let borderColor = 'var(--primary)';
      let bgColor = 'rgba(99, 102, 241, 0.08)';

      if (alertType === 'TIP') {
        borderColor = 'var(--accent)';
        bgColor = 'rgba(6, 182, 212, 0.08)';
      } else if (alertType === 'WARNING') {
        borderColor = '#f59e0b';
        bgColor = '#fffbeb';
      }

      elements.push(
        <div
          key={`alert-${elements.length}`}
          style={{
            margin: '0.75rem 0',
            padding: '0.65rem 0.85rem',
            borderRadius: '8px',
            borderLeft: `4px solid ${borderColor}`,
            backgroundColor: bgColor,
            fontSize: '0.84rem',
            lineHeight: '1.45',
          }}
        >
          <span style={{ fontWeight: 700, textTransform: 'uppercase', fontSize: '0.72rem', color: borderColor, display: 'block', marginBottom: '0.2rem' }}>
            {alertType}
          </span>
          {inAlert.lines.map((l, lIdx) => (
            <p key={lIdx} style={{ margin: '0.15rem 0' }}>
              {renderInlineMarkdown(l)}
            </p>
          ))}
        </div>,
      );
      inAlert = null;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Check for Github alert: > [!NOTE] or > [!TIP]
    const alertMatch = line.match(/^>\s*\[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]/i);
    if (alertMatch) {
      flushTextBuffer();
      flushTable();
      flushAlert();
      inAlert = { type: alertMatch[1], lines: [] };
      continue;
    }

    if (inAlert) {
      if (line.startsWith('>')) {
        inAlert.lines.push(line.replace(/^>\s*/, ''));
        continue;
      } else {
        flushAlert();
      }
    }

    // Check for Markdown table line: | ... |
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      flushTextBuffer();
      const cells = line.split('|').slice(1, -1);
      // Check if it's separator row | :--- | :--- |
      if (cells.every((c) => /^[\s:-]+$/.test(c))) {
        continue; // separator row
      }

      if (!inTable) {
        inTable = true;
        tableHeader = cells;
      } else {
        tableRows.push(cells);
      }
      continue;
    } else if (inTable) {
      flushTable();
    }

    // Normal markdown line
    textBuffer.push(line);
  }

  flushTextBuffer();
  flushTable();
  flushAlert();

  return <div className="analysis-response-body">{elements}</div>;
}

/** Helper to render bold (**text**), inline code (`code`), lists, and emojis */
function renderInlineMarkdown(text: string): React.ReactNode {
  // Check for headers
  if (text.startsWith('### ')) {
    return <h4 style={{ margin: '0.5rem 0 0.25rem', fontSize: '0.94rem', fontWeight: 600 }}>{text.slice(4)}</h4>;
  }
  if (text.startsWith('## ')) {
    return <h3 style={{ margin: '0.6rem 0 0.3rem', fontSize: '1.05rem', fontWeight: 600 }}>{text.slice(3)}</h3>;
  }
  if (text.startsWith('# ')) {
    return <h2 style={{ margin: '0.75rem 0 0.4rem', fontSize: '1.15rem', fontWeight: 700 }}>{text.slice(2)}</h2>;
  }

  // Split bold and code markers
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*.*?\*\*|`.*?`)/g;
  let lastIdx = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(text.substring(lastIdx, match.index));
    }
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(
        <strong key={`b-${match.index}`} style={{ fontWeight: 600, color: 'var(--ink)' }}>
          {token.slice(2, -2)}
        </strong>,
      );
    } else if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code
          key={`c-${match.index}`}
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.82rem',
            backgroundColor: 'rgba(99, 102, 241, 0.08)',
            color: 'var(--primary)',
            padding: '0.1rem 0.3rem',
            borderRadius: '4px',
          }}
        >
          {token.slice(1, -1)}
        </code>,
      );
    }
    lastIdx = regex.lastIndex;
  }

  if (lastIdx < text.length) {
    parts.push(text.substring(lastIdx));
  }

  return <>{parts}</>;
}
