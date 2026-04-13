import React, { useState, useRef } from 'react';

const UploadPanel = ({ onJobCreated, onProcessingComplete }) => {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | uploading | processing | completed | error
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const pollIntervalRef = useRef(null);

  const handleFileSelect = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    const ext = selected.name.toLowerCase();
    if (ext.endsWith('.pdf') || ext.endsWith('.rvt')) {
      setFile(selected);
      setError(null);
      setStatus('idle');
    } else {
      setError('Please select a valid PDF or RVT file.');
      setFile(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFileSelect({ target: { files: [dropped] } });
  };

  const handleUpload = async () => {
    if (!file) return;

    setStatus('uploading');
    setProgress(10);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const isRvt = file.name.toLowerCase().endsWith('.rvt');
      const endpoint = isRvt ? '/api/upload-rvt' : '/api/upload';

      const uploadRes = await fetch(endpoint, { method: 'POST', body: formData });
      if (!uploadRes.ok) throw new Error('Upload failed. Check file size and format.');

      const { job_id } = await uploadRes.json();
      onJobCreated(job_id);

      if (!isRvt) {
        setStatus('processing');
        setProgress(25);
        const processRes = await fetch(`/api/process/${job_id}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ project_name: file.name.replace(/\.pdf$/i, '') }),
        });
        if (!processRes.ok) throw new Error('Failed to start processing pipeline.');
      } else {
        setStatus('processing');
      }

      startPolling(job_id);
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  const startPolling = (id) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    pollIntervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${id}`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.progress) setProgress(data.progress);

        if (data.status === 'completed') {
          clearInterval(pollIntervalRef.current);
          setStatus('completed');
          onProcessingComplete(id, data.result, file?.name);
        } else if (data.status === 'failed') {
          clearInterval(pollIntervalRef.current);
          setStatus('error');
          setError(data.error || 'Pipeline processing failed.');
        }
      } catch { /* polling error — keep polling */ }
    }, 2000);
  };

  const isProcessing = status === 'uploading' || status === 'processing';

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className={`relative border-2 border-dashed rounded-xl flex flex-col items-center justify-center py-10 text-center transition-all cursor-pointer
          ${isProcessing
            ? 'border-blue-200 bg-blue-50/40 cursor-default'
            : file
              ? 'border-indigo-400 bg-indigo-50'
              : 'border-blue-300 bg-slate-50 hover:border-blue-500 hover:bg-blue-50'
          }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.rvt"
          onChange={handleFileSelect}
          className="hidden"
          disabled={isProcessing}
        />

        {file ? (
          <>
            <span className="text-4xl mb-3">📄</span>
            <p className="font-semibold text-slate-700 max-w-[220px] truncate">{file.name}</p>
            <p className="text-xs text-slate-400 mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            {!isProcessing && (
              <p className="text-xs text-blue-500 mt-2 font-medium">Click to change file</p>
            )}
          </>
        ) : (
          <>
            <span className="text-4xl mb-3 text-blue-300">⬆️</span>
            <p className="font-semibold text-slate-600">Drop PDF floor plan here</p>
            <p className="text-sm text-slate-400 mt-1">or click to browse · PDF or RVT</p>
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-xl px-4 py-3">
          <span className="text-lg shrink-0">⚠️</span>
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {/* Progress bar */}
      {isProcessing && (
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-slate-600 flex items-center gap-2">
              <span className="inline-block w-3 h-3 bg-blue-500 rounded-full pulse-ring"></span>
              {status === 'uploading' ? 'Uploading…' : 'Running AI pipeline…'}
            </span>
            <span className="text-sm font-bold text-blue-600">{progress}%</span>
          </div>
          <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-slate-400 text-center">
            Dual-track processing active · this may take 30–90 seconds
          </p>
        </div>
      )}

      {/* Generate button */}
      {file && status === 'idle' && (
        <button
          onClick={handleUpload}
          className="w-full py-3.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold rounded-xl shadow-md hover:shadow-lg transition-all hover:scale-[1.02] active:scale-100 flex items-center justify-center gap-2"
        >
          🚀 Generate 3D BIM Model
        </button>
      )}
    </div>
  );
};

export default UploadPanel;
