import React, { useState } from 'react';
import UploadBox from '../components/UploadBox';
import DatasetSummary from '../components/DatasetSummary';
import PreviewTable from '../components/PreviewTable';
import { uploadDataset } from '../services/uploadService';
import type { DatasetProfile } from '../types';

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Upload Dataset</h2>
      <UploadBox onFileSelect={setSelectedFile} />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? 'Uploading...' : 'Upload'}
      </button>
      {error ? <p>{error}</p> : null}
      {profile ? (
        <div>
          <DatasetSummary profile={profile} />
          <PreviewTable preview={profile.preview} />
        </div>
      ) : null}
    </div>
  );
}
