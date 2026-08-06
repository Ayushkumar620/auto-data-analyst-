import React from 'react';

type UploadBoxProps = {
  onFileSelect: (file: File) => void;
};

export default function UploadBox({ onFileSelect }: UploadBoxProps) {
  return (
    <div>
      <h3>Upload Dataset</h3>
      <input
        type="file"
        accept=".csv,.xlsx,.xls"
        onChange={(event) => {
          if (event.target.files?.[0]) {
            onFileSelect(event.target.files[0]);
          }
        }}
      />
    </div>
  );
}
