import React, { useState } from 'react';
import UploadBox from '../components/UploadBox';
import DatasetSummary from '../components/DatasetSummary';
import ForecastChart from '../components/ForecastChart';
import PreviewTable from '../components/PreviewTable';
import {
  generateEda,
  generateForecast,
  generateInsights,
  generateReport,
  type ForecastResponse,
  type InsightsResponse,
  type ReportResponse,
} from '../services/analysisService';
import { uploadDataset } from '../services/uploadService';
import type { DatasetProfile } from '../types';
import './upload-page.css';

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [eda, setEda] = useState<Record<string, unknown> | null>(null);
  const [insights, setInsights] = useState<InsightsResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [working, setWorking] = useState('');
  const [error, setError] = useState('');
  const [horizon, setHorizon] = useState(6);

  const resetDerivedResults = () => {
    setEda(null);
    setInsights(null);
    setForecast(null);
    setReport(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please choose a file first.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const result = await uploadDataset(selectedFile);
      setProfile(result);
      resetDerivedResults();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const runEda = async () => {
    if (!selectedFile) {
      setError('Please choose a file first.');
      return;
    }

    setWorking('Generating EDA...');
    setError('');
    try {
      const result = await generateEda(selectedFile);
      setEda(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'EDA failed');
    } finally {
      setWorking('');
    }
  };

  const runInsights = async () => {
    if (!selectedFile) {
      setError('Please choose a file first.');
      return;
    }

    setWorking('Generating insights...');
    setError('');
    try {
      const result = await generateInsights(selectedFile);
      setInsights(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Insight generation failed');
    } finally {
      setWorking('');
    }
  };

  const runForecast = async () => {
    if (!selectedFile) {
      setError('Please choose a file first.');
      return;
    }

    setWorking('Generating forecast...');
    setError('');
    try {
      const result = await generateForecast(selectedFile, horizon);
      setForecast(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Forecast failed');
    } finally {
      setWorking('');
    }
  };

  const runReport = async () => {
    if (!selectedFile) {
      setError('Please choose a file first.');
      return;
    }

    setWorking('Generating report...');
    setError('');
    try {
      const result = await generateReport(selectedFile, 'pdf');
      setReport(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Report generation failed');
    } finally {
      setWorking('');
    }
  };

  const disabled = loading || Boolean(working);

  const insightList = insights?.insights ?? [];
  const forecastRows = forecast?.forecast ?? [];
  const edaSections = eda ? Object.keys(eda) : [];

  return (
    <div className="upload-page">
      <header className="hero-block">
        <p className="hero-eyebrow">Auto Data Analyst</p>
        <h1>Operational analytics cockpit</h1>
        <p>Upload once, run EDA, generate insights, forecast trends, and export a report from one workflow.</p>
      </header>

      <div className="layout-grid">
        <aside className="control-panel fade-in">
          <UploadBox onFileSelect={setSelectedFile} selectedFileName={selectedFile?.name} />

          <button className="primary-btn" onClick={handleUpload} disabled={disabled}>
            {loading ? 'Uploading...' : 'Upload Dataset'}
          </button>

          <div className="panel-section">
            <label htmlFor="forecast-horizon">Forecast horizon</label>
            <input
              id="forecast-horizon"
              className="horizon-input"
              type="number"
              min={1}
              max={24}
              value={horizon}
              onChange={(event) => setHorizon(Math.max(1, Math.min(24, Number(event.target.value) || 1)))}
            />
          </div>

          <div className="action-grid">
            <button className="action-btn" onClick={runEda} disabled={disabled || !profile}>
              Run EDA
            </button>
            <button className="action-btn" onClick={runInsights} disabled={disabled || !profile}>
              Generate Insights
            </button>
            <button className="action-btn" onClick={runForecast} disabled={disabled || !profile}>
              Run Forecast
            </button>
            <button className="action-btn" onClick={runReport} disabled={disabled || !profile}>
              Export Report
            </button>
          </div>

          {error ? <p className="status-banner status-error">{error}</p> : null}
          {working ? <p className="status-banner status-working">{working}</p> : null}
        </aside>

        <main className="results-panel fade-in delayed">
          {!profile ? (
            <div className="empty-state">
              <h3>No dataset loaded yet</h3>
              <p>Select a file and upload to begin analysis.</p>
            </div>
          ) : (
            <>
              <DatasetSummary profile={profile} />

              <section className="result-card">
                <h3>Preview</h3>
                <PreviewTable preview={profile.preview} />
              </section>

              {eda ? (
                <section className="result-card">
                  <h3>EDA modules complete</h3>
                  <div className="chip-row">
                    {edaSections.map((section) => (
                      <span key={section} className="data-chip">
                        {section}
                      </span>
                    ))}
                  </div>
                </section>
              ) : null}

              {insights ? (
                <section className="result-card">
                  <h3>Insights</h3>
                  {insightList.length ? (
                    <ul className="insight-list">
                      {insightList.map((insight, index) => (
                        <li key={`${insight.title}-${index}`}>
                          <strong>{insight.title}</strong>
                          <p>{insight.description}</p>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted">No insights generated.</p>
                  )}
                </section>
              ) : null}

              {forecast ? (
                <section className="result-card">
                  <h3>Forecast</h3>
                  <p className="forecast-meta">
                    Target: {forecast.target} | Model: {forecast.model} | Horizon: {forecast.horizon}
                  </p>
                  <ForecastChart points={forecastRows} target={forecast.target} />
                  {forecastRows.length ? (
                    <div className="table-shell">
                      <table className="result-table">
                        <thead>
                          <tr>
                            <th>Date</th>
                            <th>Prediction</th>
                            <th>Lower</th>
                            <th>Upper</th>
                          </tr>
                        </thead>
                        <tbody>
                          {forecastRows.map((point) => (
                            <tr key={point.date}>
                              <td>{point.date}</td>
                              <td>{point.prediction}</td>
                              <td>{point.lower}</td>
                              <td>{point.upper}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : null}
                </section>
              ) : null}

              {report ? (
                <section className="result-card">
                  <h3>Report export ready</h3>
                  <p>Report ID: {report.report_id}</p>
                  <a className="download-link" href={report.download_url}>
                    Download PDF report
                  </a>
                </section>
              ) : null}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
