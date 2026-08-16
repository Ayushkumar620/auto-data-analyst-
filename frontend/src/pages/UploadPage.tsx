import React, { useState } from 'react';
import UploadBox from '../components/UploadBox';
import DatasetSummary from '../components/DatasetSummary';
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
      const result = await generateForecast(selectedFile);
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
    <div>
      <h2>Upload Dataset</h2>
      <UploadBox onFileSelect={setSelectedFile} />
      <button onClick={handleUpload} disabled={disabled}>
        {loading ? 'Uploading...' : 'Upload'}
      </button>
      <button onClick={runEda} disabled={disabled || !profile}>
        Run EDA
      </button>
      <button onClick={runInsights} disabled={disabled || !profile}>
        Generate Insights
      </button>
      <button onClick={runForecast} disabled={disabled || !profile}>
        Forecast
      </button>
      <button onClick={runReport} disabled={disabled || !profile}>
        Generate Report
      </button>
      {error ? <p>{error}</p> : null}
      {working ? <p>{working}</p> : null}
      {profile ? (
        <div>
          <DatasetSummary profile={profile} />
          <PreviewTable preview={profile.preview} />

          {eda ? (
            <section>
              <h3>EDA Overview</h3>
              <p>Sections: {edaSections.join(', ')}</p>
            </section>
          ) : null}

          {insights ? (
            <section>
              <h3>Insights</h3>
              {insightList.length ? (
                <ul>
                  {insightList.map((insight, index) => (
                    <li key={`${insight.title}-${index}`}>
                      <strong>{insight.title}</strong>: {insight.description}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No insights generated.</p>
              )}
            </section>
          ) : null}

          {forecast ? (
            <section>
              <h3>Forecast</h3>
              <p>
                Target: {forecast.target} | Model: {forecast.model} | Horizon: {forecast.horizon}
              </p>
              {forecastRows.length ? (
                <table>
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
              ) : (
                <p>No forecast points returned.</p>
              )}
            </section>
          ) : null}

          {report ? (
            <section>
              <h3>Report</h3>
              <p>Report ID: {report.report_id}</p>
              <a href={report.download_url}>Download PDF report</a>
            </section>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
