import React, { useState, useEffect } from 'react';
import {
  listDBConnections,
  testDBConnection,
  createDBConnection,
  inspectConnectionTables,
  executeDBQuery,
  type DBConnection,
  type TableSchema,
  type QueryResult,
} from '../../services/enterpriseService';
import { useNotification } from '../../context/NotificationContext';
import { useDataset } from '../../context/DatasetContext';

type Props = {
  isOpen: boolean;
  onClose: () => void;
};

export default function DatabaseConnectorsModal({ isOpen, onClose }: Props) {
  const { notify } = useNotification();
  const { setDataset } = useDataset();

  const [connections, setConnections] = useState<DBConnection[]>([]);
  const [selectedConnId, setSelectedConnId] = useState<string>('');
  const [tables, setTables] = useState<TableSchema[]>([]);
  const [sqlQuery, setSqlQuery] = useState<string>('SELECT * FROM customer_transactions LIMIT 20');
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [showAddForm, setShowAddForm] = useState<boolean>(false);

  // New connection state
  const [name, setName] = useState('Production PostgreSQL');
  const [dbType, setDbType] = useState('postgresql');
  const [host, setHost] = useState('localhost');
  const [port, setPort] = useState(5432);
  const [database, setDatabase] = useState('analytics_db');
  const [username, setUsername] = useState('postgres');
  const [password, setPassword] = useState('');

  const loadConnections = async () => {
    try {
      const data = await listDBConnections();
      setConnections(data);
      if (data.length > 0 && !selectedConnId) {
        setSelectedConnId(data[0].connection_id);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadConnections();
    }
  }, [isOpen]);

  useEffect(() => {
    if (selectedConnId) {
      inspectConnectionTables(selectedConnId)
        .then(setTables)
        .catch(() => setTables([]));
    }
  }, [selectedConnId]);

  if (!isOpen) return null;

  const handleTestConnection = async () => {
    setLoading(true);
    try {
      const res = await testDBConnection({
        name,
        db_type: dbType,
        host,
        port,
        database,
        username,
        password,
      });
      if (res.success) {
        notify('Database connection verified successfully!', 'success');
      } else {
        notify(res.message, 'error');
      }
    } catch (e: any) {
      notify(e.message || 'Connection test failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateConnection = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const newConn = await createDBConnection({
        name,
        db_type: dbType,
        host,
        port,
        database,
        username,
        password,
      });
      notify(`Connection '${newConn.name}' registered.`, 'success');
      setShowAddForm(false);
      loadConnections();
      setSelectedConnId(newConn.connection_id);
    } catch (e: any) {
      notify(e.message || 'Failed to save connection', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleExecuteQuery = async () => {
    if (!selectedConnId || !sqlQuery.trim()) return;
    setLoading(true);
    try {
      const res = await executeDBQuery(selectedConnId, sqlQuery);
      setQueryResult(res);
      notify(`Executed query in ${res.execution_time_ms}ms (${res.total_rows} rows).`, 'info');
    } catch (e: any) {
      notify(e.message || 'Query execution failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleIngestDataset = () => {
    if (!queryResult || queryResult.rows.length === 0) return;
    const profile: any = {
      dataset_name: `DB_Query_${Date.now()}`,
      rows: queryResult.total_rows,
      columns: queryResult.columns.length,
      column_names: queryResult.columns,
      preview: queryResult.rows,
    };
    setDataset(profile, profile.dataset_name);
    notify(`Ingested ${queryResult.total_rows} rows into active Dataset Workspace!`, 'success');
    onClose();
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(4px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.5rem',
      }}
    >
      <div
        className="glass-card"
        style={{
          width: '100%',
          maxWidth: '900px',
          maxHeight: '90vh',
          overflowY: 'auto',
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          padding: '1.5rem 2rem',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 700 }}>
              🔌 Enterprise Live Database Connectors
            </h2>
            <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.82rem' }}>
              Query production data warehouses (PostgreSQL, Snowflake, MySQL, SQLite) and import live tables.
            </p>
          </div>
          <button type="button" onClick={onClose} className="ghost-text-btn" style={{ fontSize: '1.1rem' }}>
            ✕
          </button>
        </div>

        {/* Connection Selector */}
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
          <label style={{ fontSize: '0.84rem', fontWeight: 600 }}>Active Connection:</label>
          <select
            value={selectedConnId}
            onChange={(e) => setSelectedConnId(e.target.value)}
            className="horizon-input"
            style={{ padding: '0.35rem 0.65rem', minWidth: '220px' }}
          >
            {connections.map((c) => (
              <option key={c.connection_id} value={c.connection_id}>
                {c.name} ({c.db_type.toUpperCase()})
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => setShowAddForm(!showAddForm)}
            className="action-btn"
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem' }}
          >
            {showAddForm ? '✕ Close Form' : '+ Add Database'}
          </button>
        </div>

        {/* Add Connection Form */}
        {showAddForm && (
          <form onSubmit={handleCreateConnection} style={{ background: '#f8fafc', padding: '1rem', borderRadius: '8px', marginBottom: '1.25rem', border: '1px solid #e2e8f0' }}>
            <h4 style={{ margin: '0 0 0.75rem', fontSize: '0.92rem' }}>Configure Database Credentials</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.65rem' }}>
              <div>
                <label style={{ fontSize: '0.75rem', fontWeight: 600 }}>Connection Name</label>
                <input type="text" value={name} onChange={(e) => setName(e.target.value)} className="horizon-input" style={{ width: '100%', padding: '0.3rem' }} required />
              </div>
              <div>
                <label style={{ fontSize: '0.75rem', fontWeight: 600 }}>Database Type</label>
                <select value={dbType} onChange={(e) => setDbType(e.target.value)} className="horizon-input" style={{ width: '100%', padding: '0.3rem' }}>
                  <option value="postgresql">PostgreSQL</option>
                  <option value="snowflake">Snowflake</option>
                  <option value="mysql">MySQL</option>
                  <option value="sqlite">SQLite</option>
                </select>
              </div>
              <div>
                <label style={{ fontSize: '0.75rem', fontWeight: 600 }}>Host</label>
                <input type="text" value={host} onChange={(e) => setHost(e.target.value)} className="horizon-input" style={{ width: '100%', padding: '0.3rem' }} />
              </div>
              <div>
                <label style={{ fontSize: '0.75rem', fontWeight: 600 }}>Port</label>
                <input type="number" value={port} onChange={(e) => setPort(Number(e.target.value))} className="horizon-input" style={{ width: '100%', padding: '0.3rem' }} />
              </div>
              <div>
                <label style={{ fontSize: '0.75rem', fontWeight: 600 }}>Database Name / Path</label>
                <input type="text" value={database} onChange={(e) => setDatabase(e.target.value)} className="horizon-input" style={{ width: '100%', padding: '0.3rem' }} required />
              </div>
              <div>
                <label style={{ fontSize: '0.75rem', fontWeight: 600 }}>Username</label>
                <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} className="horizon-input" style={{ width: '100%', padding: '0.3rem' }} />
              </div>
              <div>
                <label style={{ fontSize: '0.75rem', fontWeight: 600 }}>Password</label>
                <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="horizon-input" style={{ width: '100%', padding: '0.3rem' }} />
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end', marginTop: '0.75rem' }}>
              <button type="button" onClick={handleTestConnection} className="action-btn" disabled={loading} style={{ padding: '0.3rem 0.8rem', fontSize: '0.8rem' }}>
                Test Connection
              </button>
              <button type="submit" className="primary-btn" disabled={loading} style={{ padding: '0.3rem 0.8rem', fontSize: '0.8rem' }}>
                Save Connection
              </button>
            </div>
          </form>
        )}

        {/* Remote Tables Quick View */}
        {tables.length > 0 && (
          <div style={{ marginBottom: '1rem' }}>
            <span className="muted" style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>Available Remote Tables:</span>
            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginTop: '0.3rem' }}>
              {tables.map((t) => (
                <button
                  key={t.table_name}
                  type="button"
                  onClick={() => setSqlQuery(`SELECT * FROM ${t.table_name} LIMIT 50`)}
                  className="action-btn"
                  style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                >
                  📋 {t.table_name} ({t.columns.length} cols)
                </button>
              ))}
            </div>
          </div>
        )}

        {/* SQL Editor */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ fontSize: '0.8rem', fontWeight: 600, display: 'block', marginBottom: '0.3rem' }}>
            Read-Only SQL Query
          </label>
          <textarea
            value={sqlQuery}
            onChange={(e) => setSqlQuery(e.target.value)}
            rows={3}
            className="horizon-input"
            style={{ width: '100%', fontFamily: 'var(--font-mono)', fontSize: '0.84rem', padding: '0.5rem' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.4rem' }}>
            <span className="muted" style={{ fontSize: '0.75rem' }}>
              🛡️ Read safety active: destructive SQL statements (DROP, DELETE, ALTER) are automatically blocked.
            </span>
            <button type="button" onClick={handleExecuteQuery} className="primary-btn" disabled={loading} style={{ padding: '0.35rem 0.9rem', fontSize: '0.82rem' }}>
              {loading ? 'Executing Query…' : '⚡ Run Query'}
            </button>
          </div>
        </div>

        {/* Results Preview */}
        {queryResult && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>
                Query Results ({queryResult.total_rows} rows · {queryResult.execution_time_ms}ms)
              </span>
              <button type="button" onClick={handleIngestDataset} className="primary-btn" style={{ padding: '0.3rem 0.8rem', fontSize: '0.8rem', background: '#059669' }}>
                📥 Ingest as Active Dataset
              </button>
            </div>

            <div style={{ overflowX: 'auto', maxHeight: '240px', border: '1px solid #e2e8f0', borderRadius: '6px' }}>
              <table style={{ width: '100%', fontSize: '0.78rem', borderCollapse: 'collapse' }}>
                <thead style={{ backgroundColor: '#f8fafc', position: 'sticky', top: 0 }}>
                  <tr>
                    {queryResult.columns.map((col) => (
                      <th key={col} style={{ padding: '0.4rem 0.6rem', borderBottom: '1px solid #cbd5e1', textAlign: 'left' }}>
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {queryResult.rows.slice(0, 20).map((row, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      {queryResult.columns.map((col) => (
                        <td key={col} style={{ padding: '0.35rem 0.6rem' }}>
                          {String(row[col] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
