import React, { useState, useEffect } from 'react';
import { scanDatasetPII, redactDataset, type PIIScanReport } from '../../services/enterpriseService';
import { useNotification } from '../../context/NotificationContext';
import { useDataset } from '../../context/DatasetContext';

type Props = {
  dataRows: Array<Record<string, any>>;
};

export default function PIIGuardBanner({ dataRows }: Props) {
  const { notify } = useNotification();
  const { profile, setDataset } = useDataset();

  const [report, setReport] = useState<PIIScanReport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [redacting, setRedacting] = useState<boolean>(false);

  useEffect(() => {
    if (dataRows && dataRows.length > 0) {
      scanDatasetPII(dataRows)
        .then(setReport)
        .catch(() => setReport(null));
    }
  }, [dataRows]);

  if (!report || !report.has_pii) return null;

  const handleRedact = async () => {
    setRedacting(true);
    try {
      const res = await redactDataset(dataRows);
      if (res.success && profile) {
        const updatedProfile = {
          ...profile,
          preview: res.rows,
        };
        setDataset(updatedProfile, profile.dataset_name);
        notify('PII columns successfully masked and sanitized for compliance.', 'success');
        setReport({ ...report, has_pii: false, total_pii_columns: 0, detections: [] });
      }
    } catch (e: any) {
      notify(e.message || 'Redaction failed', 'error');
    } finally {
      setRedacting(false);
    }
  };

  return (
    <div
      style={{
        padding: '0.85rem 1.25rem',
        borderRadius: '10px',
        backgroundColor: '#fffbeb',
        border: '1px solid #fef3c7',
        borderLeft: '4px solid #f59e0b',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '0.75rem',
        marginBottom: '1rem',
      }}
    >
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{ fontSize: '1rem' }}>🛡️</span>
          <span style={{ fontWeight: 700, fontSize: '0.88rem', color: '#92400e' }}>
            Enterprise Data Governance & PII Alert
          </span>
          <span
            style={{
              fontSize: '0.7rem',
              fontWeight: 700,
              padding: '0.1rem 0.4rem',
              borderRadius: '4px',
              backgroundColor: '#fef3c7',
              color: '#b45309',
            }}
          >
            {report.total_pii_columns} SENSITIVE COLUMN{report.total_pii_columns === 1 ? '' : 'S'} DETECTED
          </span>
        </div>

        <p style={{ margin: '0.2rem 0 0', fontSize: '0.8rem', color: '#78350f' }}>
          Detected fields:{' '}
          {report.detections.map((d) => (
            <strong key={d.column}>
              {d.column} ({d.pii_type.toUpperCase()}){' '}
            </strong>
          ))}
          — automatic masking recommended before external sharing.
        </p>
      </div>

      <button
        type="button"
        onClick={handleRedact}
        disabled={redacting}
        className="primary-btn"
        style={{
          padding: '0.35rem 0.9rem',
          fontSize: '0.8rem',
          backgroundColor: '#d97706',
          borderColor: '#b45309',
        }}
      >
        {redacting ? 'Masking PII Data…' : '🔒 Redact & Mask PII'}
      </button>
    </div>
  );
}
