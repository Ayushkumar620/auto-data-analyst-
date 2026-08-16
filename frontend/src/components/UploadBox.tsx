import React from 'react';

type UploadBoxProps = {
  onFileSelect: (file: File) => void;
  selectedFileName?: string;
};

export default function UploadBox({ onFileSelect, selectedFileName }: UploadBoxProps) {
  return (
    <div className="upload-box">
      <h3>Upload dataset</h3>
      <input
        className="upload-input"
        type="file"
        accept=".csv,.xlsx,.xls"
        onChange={(event) => {
          if (event.target.files?.[0]) {
            onFileSelect(event.target.files[0]);
          }
        }}
      />
      <p className="upload-hint">Supported: CSV, XLSX, XLS</p>
      {selectedFileName ? <p className="selected-file">Selected: {selectedFileName}</p> : null}
    </div>
  );
}
