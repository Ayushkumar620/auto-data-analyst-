import { useState } from 'react';
import { uploadDataset } from '../services/uploadService';
import type { DatasetProfile } from '../types';

export default function useUpload() {
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const upload = async (file: File) => {
    setLoading(true);
    setError('');

    try {
      const result = await uploadDataset(file);
      setProfile(result);
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { profile, loading, error, upload };
}
