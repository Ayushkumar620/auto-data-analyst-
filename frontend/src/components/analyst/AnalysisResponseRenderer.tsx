import React from 'react';
import SummaryData from '../SummaryData';

type AnalysisResponseRendererProps = {
  content: string;
};

type TableAlignment = 'left' | 'center' | 'right';

/**
 * Checks if a string qualifies as a Markdown table separator line.
 * e.g. | :--- | :---: | ---: | or --- | :--- | ---:
 */
function isTableSeparator(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed.includes('-')) return false;
  // Strip leading and trailing pipe if present
  const content = trimmed.replace(/^\|/, '').replace(/\|$/, '').trim();
  if (!content) return false;
  const parts = content.split('|');
  return parts.length > 0 && parts.every((p) => /^[\s:-]+$/.test(p) && p.includes('-'));
}

/**
 * Extracts column text alignment from a separator line.
 */
function parseAlignments(separatorLine: string): TableAlignment[] {
  const content = separatorLine.trim().replace(/^\|/, '').replace(/\|$/, '').trim();
  return content.split('|').map((part) => {
    const trimmed = part.trim();
    const startColon = trimmed.startsWith(':');
    const endColon = trimmed.endsWith(':');
    if (startColon && endColon) return 'center';
    if (endColon) return 'right';
    return 'left';
  });
}

/**
 * Splits a table row into individual cell strings.
 * Respects escaped pipes (\|) and strips leading/trailing table delimiters.
 */
function splitRowCells(line: string): string[] {
  let trimmed = line.trim();
  if (trimmed.startsWith('|')) {
    trimmed = trimmed.substring(1);
  }
  if (trimmed.endsWith('|') && !trimmed.endsWith('\\|')) {
    trimmed = trimmed.substring(0, trimmed.length - 1);
  }

  const rawCells: string[] = [];
  let current = '';
  let escaped = false;

  for (let i = 0; i < trimmed.length; i++) {
    const ch = trimmed[i];
    if (ch === '\\' && !escaped) {
      escaped = true;
      continue;
    }
    if (ch === '|' && !escaped) {
      rawCells.push(current.trim());
      current = '';
    } else {
      if (escaped) {
        current += '\\';
        escaped = false;
      }
      current += ch;
    }
  }
  rawCells.push(current.trim());
  return rawCells.map((c) => c.replace(/\\\|/g, '|'));
}

/**
 * Checks if a line is a candidate table row (contains | and is not a pure separator).
 */
function isTableRow(line: string): boolean {
  const trimmed = line.trim();
  return trimmed.includes('|') && !isTableSeparator(trimmed);
}

/**
 * Detects if a markdown text is a SummaryData report and parses it into structured summary props.
 */
function tryParseSummaryData(content: string) {
  if (!/(?:📋\s*Summary|Dataset Summary|Summarydata)/i.test(content)) {
    return null;
  }
  if (!/Rows/i.test(content) || !/Columns/i.test(content)) {
    return null;
  }

  // Extract rows
  const rowsMatch = content.match(/Rows[:\s]*(\d[\d,]*)/i);
  const colsMatch = content.match(/Columns[:\s]*(\d[\d,]*)/i);
  const nullsMatch =
    content.match(/Nulls[:\s]*(\d[\d,]*)/i) ||
    content.match(/Missing(?:\s*Values)?[:\s]*(\d[\d,]*)/i);

  if (!rowsMatch || !colsMatch) {
    return null;
  }

  const rows = parseInt(rowsMatch[1].replace(/,/g, ''), 10);
  const columns = parseInt(colsMatch[1].replace(/,/g, ''), 10);
  const nulls = nullsMatch ? parseInt(nullsMatch[1].replace(/,/g, ''), 10) : 0;

  // Extract dataset name if present
  const nameMatch = content.match(/📋\s*Summary\s*(?:data|:\s*([^\n\r]+))?/i);
  const datasetName =
    nameMatch && nameMatch[1] && nameMatch[1].trim().toLowerCase() !== 'data'
      ? nameMatch[1].trim()
      : '';

  // Extract Columns & Types table
  const colTypes: Record<string, string> = {};
  const colTypesMatch = content.match(
    /#{1,4}\s*Columns\s*&(?:\s*amp;)?\s*Types[\s\S]*?(?=#{1,4}\s*Preview|$)/i,
  );
  if (colTypesMatch) {
    const tableLines = colTypesMatch[0].split('\n');
    for (const l of tableLines) {
      if (isTableRow(l) && !isTableSeparator(l)) {
        const cells = splitRowCells(l);
        if (cells.length >= 2 && !/^(column|columntype)$/i.test(cells[0])) {
          colTypes[cells[0]] = cells[1];
        } else if (cells.length === 1 && !/^(column|columntype)$/i.test(cells[0])) {
          colTypes[cells[0]] = 'string';
        }
      }
    }
  }

  // Extract Preview table
  const preview: Array<Record<string, unknown>> = [];
  const previewMatch = content.match(/#{1,4}\s*Preview[^\n]*\n([\s\S]*)/i);
  if (previewMatch) {
    const previewLines = previewMatch[1].split('\n');
    let previewHeaders: string[] = [];
    for (let j = 0; j < previewLines.length; j++) {
      const pl = previewLines[j];
      if (isTableRow(pl)) {
        if (!isTableSeparator(pl)) {
          const cells = splitRowCells(pl);
          if (
            previewHeaders.length === 0 &&
            j + 1 < previewLines.length &&
            isTableSeparator(previewLines[j + 1])
          ) {
            previewHeaders = cells;
            j++; // skip separator
          } else if (previewHeaders.length > 0) {
            const rowObj: Record<string, unknown> = {};
            previewHeaders.forEach((h, idx) => {
              const val = cells[idx] ?? '';
              const num = Number(val);
              rowObj[h] = !isNaN(num) && val.trim() !== '' ? num : val;
            });
            preview.push(rowObj);
          }
        }
      } else if (pl.trim().startsWith('#') || (pl.trim() === '' && preview.length > 0)) {
        break;
      }
    }
  }

  return {
    dataset_name: datasetName,
    rows,
    columns,
    nulls,
    dtypes: colTypes,
    preview,
  };
}

export default function AnalysisResponseRenderer({ content }: AnalysisResponseRendererProps) {
  if (!content) return null;

  // Render structured SummaryData dashboard if content is a dataset summary report
  const summaryDataPayload = tryParseSummaryData(content);
  if (summaryDataPayload) {
    return (
      <div
        className="analysis-response-body"
        style={{
          width: '100%',
          maxWidth: '100%',
          minWidth: 0,
        }}
      >
        <SummaryData data={summaryDataPayload} />
      </div>
    );
  }

  // Normalize line breaks
  const normalized = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const lines = normalized.split('\n');
  const elements: React.ReactNode[] = [];

  let textBuffer: string[] = [];

  const flushTextBuffer = () => {
    if (textBuffer.length > 0) {
      const paragraph = textBuffer.join('\n').trim();
      if (paragraph) {
        elements.push(
          <div
            key={`p-${elements.length}`}
            style={{
              marginBottom: '0.65rem',
              lineHeight: '1.55',
              overflowWrap: 'anywhere',
              wordBreak: 'break-word',
              minWidth: 0,
            }}
          >
            {renderInlineMarkdown(paragraph)}
          </div>,
        );
      }
      textBuffer = [];
    }
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // 1. Fenced code blocks (```lang ... ```)
    if (line.trim().startsWith('```')) {
      flushTextBuffer();
      const lang = line.trim().slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      // Skip closing ``` if present
      if (i < lines.length && lines[i].trim().startsWith('```')) {
        i++;
      }
      elements.push(
        <div
          key={`code-${elements.length}`}
          style={{
            margin: '0.85rem 0',
            borderRadius: '8px',
            overflow: 'hidden',
            border: '1px solid #e2e8f0',
            backgroundColor: '#0f172a',
          }}
        >
          {lang && (
            <div
              style={{
                padding: '0.35rem 0.75rem',
                backgroundColor: '#1e293b',
                color: '#94a3b8',
                fontSize: '0.72rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              {lang}
            </div>
          )}
          <pre
            style={{
              margin: 0,
              padding: '0.75rem 1rem',
              overflowX: 'auto',
              color: '#f8fafc',
              fontFamily: 'var(--font-mono, monospace)',
              fontSize: '0.82rem',
              lineHeight: '1.5',
            }}
          >
            <code>{codeLines.join('\n')}</code>
          </pre>
        </div>,
      );
      continue;
    }

    // 2. Github Alert Blockquotes (> [!NOTE], > [!TIP], etc.)
    const alertMatch = line.match(/^>\s*\[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]/i);
    if (alertMatch) {
      flushTextBuffer();
      const alertType = alertMatch[1].toUpperCase();
      const alertLines: string[] = [];
      i++;
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        alertLines.push(lines[i].trim().replace(/^>\s*/, ''));
        i++;
      }

      let borderColor = 'var(--primary, #4f46e5)';
      let bgColor = 'rgba(99, 102, 241, 0.08)';

      if (alertType === 'TIP') {
        borderColor = 'var(--accent, #06b6d4)';
        bgColor = 'rgba(6, 182, 212, 0.08)';
      } else if (alertType === 'WARNING') {
        borderColor = '#f59e0b';
        bgColor = '#fffbeb';
      } else if (alertType === 'CAUTION') {
        borderColor = '#ef4444';
        bgColor = '#fef2f2';
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
          <span
            style={{
              fontWeight: 700,
              textTransform: 'uppercase',
              fontSize: '0.72rem',
              color: borderColor,
              display: 'block',
              marginBottom: '0.2rem',
            }}
          >
            {alertType}
          </span>
          {alertLines.map((l, lIdx) => (
            <p key={lIdx} style={{ margin: '0.15rem 0' }}>
              {renderInlineMarkdown(l)}
            </p>
          ))}
        </div>,
      );
      continue;
    }

    // Standard blockquote (> text)
    if (line.trim().startsWith('>')) {
      flushTextBuffer();
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quoteLines.push(lines[i].trim().replace(/^>\s*/, ''));
        i++;
      }
      elements.push(
        <blockquote
          key={`quote-${elements.length}`}
          style={{
            margin: '0.65rem 0',
            padding: '0.5rem 0.85rem',
            borderLeft: '3px solid #cbd5e1',
            backgroundColor: '#f8fafc',
            borderRadius: '0 6px 6px 0',
            color: 'var(--ink-secondary, #334155)',
            fontSize: '0.85rem',
            fontStyle: 'italic',
          }}
        >
          {quoteLines.map((ql, qIdx) => (
            <p key={qIdx} style={{ margin: '0.1rem 0' }}>
              {renderInlineMarkdown(ql)}
            </p>
          ))}
        </blockquote>,
      );
      continue;
    }

    // 3. Markdown Tables
    // A table begins when the current line is a table row and the next line is a separator row.
    if (isTableRow(line) && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      flushTextBuffer();
      const headers = splitRowCells(line);
      const alignments = parseAlignments(lines[i + 1]);
      const tableRows: string[][] = [];

      i += 2; // Advance past header and separator

      while (i < lines.length && isTableRow(lines[i])) {
        tableRows.push(splitRowCells(lines[i]));
        i++;
      }

      elements.push(
        <div
          key={`table-${elements.length}`}
          className="table-responsive-container"
          style={{
            overflowX: 'auto',
            width: '100%',
            maxWidth: '100%',
            minWidth: 0,
            margin: '0.85rem 0',
            borderRadius: '8px',
            border: '1px solid var(--border, #e2e8f0)',
            backgroundColor: 'var(--surface, #ffffff)',
            boxShadow: '0 1px 3px rgba(15, 23, 42, 0.04)',
          }}
        >
          <table
            className="result-table"
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '0.84rem',
              textAlign: 'left',
            }}
          >
            <thead>
              <tr style={{ backgroundColor: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                {headers.map((h, hIdx) => {
                  const align = alignments[hIdx] || 'left';
                  return (
                    <th
                      key={hIdx}
                      style={{
                        padding: '0.55rem 0.85rem',
                        fontSize: '0.80rem',
                        fontWeight: 600,
                        color: 'var(--ink-secondary, #334155)',
                        textAlign: align,
                        whiteSpace: 'nowrap',
                        letterSpacing: '0.02em',
                        borderRight: hIdx < headers.length - 1 ? '1px solid #f1f5f9' : undefined,
                      }}
                    >
                      {renderInlineMarkdown(h)}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((row, rIdx) => {
                const isEven = rIdx % 2 === 0;
                return (
                  <tr
                    key={rIdx}
                    style={{
                      backgroundColor: isEven ? '#ffffff' : '#fcfdfe',
                      borderBottom: '1px solid #f1f5f9',
                    }}
                  >
                    {headers.map((_, cIdx) => {
                      const cell = row[cIdx] ?? '';
                      const align = alignments[cIdx] || 'left';
                      const isEmpty = cell.trim() === '';
                      return (
                        <td
                          key={cIdx}
                          style={{
                            padding: '0.5rem 0.85rem',
                            fontSize: '0.82rem',
                            color: 'var(--ink, #0f172a)',
                            textAlign: align,
                            borderRight: cIdx < headers.length - 1 ? '1px solid #f8fafc' : undefined,
                            whiteSpace: cell.length > 50 ? 'normal' : 'nowrap',
                          }}
                        >
                          {isEmpty ? (
                            <span style={{ color: 'var(--muted, #94a3b8)', fontStyle: 'italic' }}>—</span>
                          ) : (
                            renderInlineMarkdown(cell)
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    // 4. Bullet lists (*, -, +)
    const bulletMatch = line.match(/^(\s*)([-*+])\s+(.*)$/);
    if (bulletMatch) {
      flushTextBuffer();
      const listItems: string[] = [bulletMatch[3]];
      i++;
      while (i < lines.length) {
        const nextMatch = lines[i].match(/^(\s*)([-*+])\s+(.*)$/);
        if (nextMatch) {
          listItems.push(nextMatch[3]);
          i++;
        } else {
          break;
        }
      }
      elements.push(
        <ul
          key={`ul-${elements.length}`}
          style={{
            margin: '0.45rem 0 0.65rem 1.3rem',
            padding: 0,
            fontSize: '0.88rem',
            lineHeight: '1.6',
            color: 'var(--ink, #0f172a)',
          }}
        >
          {listItems.map((item, lIdx) => (
            <li key={lIdx} style={{ marginBottom: '0.2rem' }}>
              {renderInlineMarkdown(item)}
            </li>
          ))}
        </ul>,
      );
      continue;
    }

    // 5. Numbered lists (1. , 2. )
    const numMatch = line.match(/^(\s*)(\d+)\.\s+(.*)$/);
    if (numMatch) {
      flushTextBuffer();
      const listItems: string[] = [numMatch[3]];
      i++;
      while (i < lines.length) {
        const nextMatch = lines[i].match(/^(\s*)(\d+)\.\s+(.*)$/);
        if (nextMatch) {
          listItems.push(nextMatch[3]);
          i++;
        } else {
          break;
        }
      }
      elements.push(
        <ol
          key={`ol-${elements.length}`}
          style={{
            margin: '0.45rem 0 0.65rem 1.3rem',
            padding: 0,
            fontSize: '0.88rem',
            lineHeight: '1.6',
            color: 'var(--ink, #0f172a)',
          }}
        >
          {listItems.map((item, lIdx) => (
            <li key={lIdx} style={{ marginBottom: '0.2rem' }}>
              {renderInlineMarkdown(item)}
            </li>
          ))}
        </ol>,
      );
      continue;
    }

    // 6. Markdown Headings (#, ##, ###, ####)
    if (line.startsWith('#### ')) {
      flushTextBuffer();
      elements.push(
        <h5 key={`h5-${elements.length}`} style={{ margin: '0.5rem 0 0.25rem', fontSize: '0.88rem', fontWeight: 600, color: 'var(--ink, #0f172a)' }}>
          {renderInlineMarkdown(line.slice(5).trim())}
        </h5>,
      );
      i++;
      continue;
    }
    if (line.startsWith('### ')) {
      flushTextBuffer();
      elements.push(
        <h4 key={`h4-${elements.length}`} style={{ margin: '0.6rem 0 0.3rem', fontSize: '0.96rem', fontWeight: 600, color: 'var(--ink, #0f172a)' }}>
          {renderInlineMarkdown(line.slice(4).trim())}
        </h4>,
      );
      i++;
      continue;
    }
    if (line.startsWith('## ')) {
      flushTextBuffer();
      elements.push(
        <h3 key={`h3-${elements.length}`} style={{ margin: '0.75rem 0 0.35rem', fontSize: '1.08rem', fontWeight: 700, color: 'var(--ink, #0f172a)' }}>
          {renderInlineMarkdown(line.slice(3).trim())}
        </h3>,
      );
      i++;
      continue;
    }
    if (line.startsWith('# ')) {
      flushTextBuffer();
      elements.push(
        <h2 key={`h2-${elements.length}`} style={{ margin: '0.9rem 0 0.45rem', fontSize: '1.22rem', fontWeight: 700, color: 'var(--ink, #0f172a)' }}>
          {renderInlineMarkdown(line.slice(2).trim())}
        </h2>,
      );
      i++;
      continue;
    }

    // 7. Normal markdown text line
    textBuffer.push(line);
    i++;
  }

  flushTextBuffer();

  return (
    <div
      className="analysis-response-body"
      style={{
        width: '100%',
        maxWidth: '100%',
        minWidth: 0,
        overflowWrap: 'anywhere',
        wordBreak: 'break-word',
      }}
    >
      {elements}
    </div>
  );
}

/**
 * Helper to render inline markdown:
 * - `code`
 * - **bold**
 * - *italic*
 * - _italic_ (boundary-guarded to preserve snake_case column names)
 * - [text](url) links
 */
function renderInlineMarkdown(text: string): React.ReactNode {
  if (!text) return null;

  // 1. `code`
  // 2. **bold**
  // 3. *italic*
  // 4. _italic_ (boundary-guarded to prevent stripping underscores from snake_case columns)
  // 5. [text](url)
  const tokenRegex =
    /(`[^`]+`|\*\*[^*]+\*\*|(?<!\*)\*[^*]+(?<!\*)\*|(?<=[\s(]|^)_[^_]+_(?=[\s).,;:!?]|$)|\[([^\]]+)\]\(([^)]+)\))/g;
  const parts: React.ReactNode[] = [];
  let lastIdx = 0;
  let match: RegExpExecArray | null;

  while ((match = tokenRegex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(text.substring(lastIdx, match.index));
    }
    const token = match[0];
    if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code
          key={`code-${match.index}`}
          style={{
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: '0.82rem',
            backgroundColor: 'rgba(99, 102, 241, 0.08)',
            color: 'var(--primary, #4f46e5)',
            padding: '0.1rem 0.35rem',
            borderRadius: '4px',
          }}
        >
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(
        <strong key={`bold-${match.index}`} style={{ fontWeight: 600, color: 'var(--ink, #0f172a)' }}>
          {token.slice(2, -2)}
        </strong>,
      );
    } else if (token.startsWith('*') && token.endsWith('*')) {
      parts.push(
        <em key={`em-${match.index}`} style={{ fontStyle: 'italic' }}>
          {token.slice(1, -1)}
        </em>,
      );
    } else if (token.startsWith('_') && token.endsWith('_')) {
      parts.push(
        <em key={`em-${match.index}`} style={{ fontStyle: 'italic' }}>
          {token.slice(1, -1)}
        </em>,
      );
    } else if (match[2] && match[3]) {
      parts.push(
        <a
          key={`a-${match.index}`}
          href={match[3]}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'var(--primary, #4f46e5)', textDecoration: 'underline' }}
        >
          {match[2]}
        </a>,
      );
    }
    lastIdx = tokenRegex.lastIndex;
  }

  if (lastIdx < text.length) {
    parts.push(text.substring(lastIdx));
  }

  return parts.length === 1 ? parts[0] : <>{parts}</>;
}
