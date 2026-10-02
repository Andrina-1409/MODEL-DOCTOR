import React, { useState } from "react";

export default function LiveUpload({ onResult, onError, loading, setLoading }) {
  const [file, setFile] = useState(null);

  async function diagnose() {
    if (!file) return;
    setLoading(true);
    onError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch("/api/live/diagnose", { method: "POST", body });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || "Live diagnosis failed.");
      onResult(data);
    } catch (error) {
      onError(error.message);
    } finally {
      setLoading(false);
    }
  }

  return <section className="live-panel">
    <div>
      <span className="eyebrow">LIVE DIAGNOSIS</span>
      <h3>Upload a trained TinyCNN model</h3>
      <p>Give Model Doctor a compatible <code>.pt</code> or <code>.pth</code> checkpoint. It will load the model, run the diagnostic probes on the bundled MNIST diagnostic set, extract the 10 features, classify the failure mode, and generate Grad-CAM evidence.</p>
      <p className="scope-note"><strong>Supported live scope:</strong> TinyCNN + MNIST. The uploaded model must contain a compatible PyTorch state dict.</p>
    </div>
    <div className="live-controls">
      <input type="file" accept=".pt,.pth,.bin" onChange={e => setFile(e.target.files?.[0] || null)} />
      <button disabled={!file || loading} onClick={diagnose}>{loading ? "Diagnosing…" : "Diagnose uploaded model"}</button>
    </div>
    {file && <div className="selected-file">Selected: <strong>{file.name}</strong></div>}
  </section>;
}
