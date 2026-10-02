import React from "react";
import { useEffect, useState } from "react";
import { loadDiagnosis, loadModelIndex, loadProofCatalog } from "./api/client";
import Header from "./components/Header";
import ModelSelector from "./components/ModelSelector";
import DiagnosisCard from "./components/DiagnosisCard";
import ProbabilityBars from "./components/ProbabilityBars";
import FeatureTable from "./components/FeatureTable";
import GradcamGallery from "./components/GradcamGallery";
import SuggestedFix from "./components/SuggestedFix";
import GroundTruthBadge from "./components/GroundTruthBadge";
import ProofSelector from "./components/ProofSelector";
import LiveUpload from "./components/LiveUpload";

function Results({ result }) {
  const live = result.source === "live_upload";
  return <div className="grid">
    <DiagnosisCard diagnosis={result.diagnosis} confidence={result.confidence} />
    {live ? <div className="truth live-truth"><strong>LIVE MODEL</strong><br />Ground-truth fault is not assumed for an uploaded model.</div> : <GroundTruthBadge value={result.ground_truth} />}
    <ProbabilityBars probabilities={result.probabilities} />
    {live && <section className="card"><h3>Evaluation summary</h3><div className="metric-row"><span>Train-set accuracy</span><strong>{(result.train_accuracy * 100).toFixed(1)}%</strong></div><div className="metric-row"><span>MNIST test accuracy</span><strong>{(result.test_accuracy * 100).toFixed(1)}%</strong></div><div className="metric-row"><span>Model scope</span><strong>{result.supported_scope}</strong></div></section>}
    <FeatureTable features={result.features} />
    {live && <section className="card"><h3>Live diagnosis note</h3><p>{result.note}</p></section>}
    <GradcamGallery images={result.gradcam_images} />
    <div className="side"><SuggestedFix text={result.suggested_fix} /><div className="card"><h3>Corner attention</h3><p className="big">{(result.corner_attention * 100).toFixed(1)}%</p></div></div>
  </div>;
}

export default function App() {
  const [models, setModels] = useState([]);
  const [proofs, setProofs] = useState([]);
  const [proofId, setProofId] = useState("");
  const [modelId, setModelId] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState("proof");

  useEffect(() => {
    loadProofCatalog().then(data => {
      setProofs(data);
      if (data[0]) setProofId(data[0].proof_id);
    }).catch(e => setError(e.message));
    loadModelIndex().then(data => {
      setModels(data);
      if (data[0]) setModelId(data[0].model_id);
    }).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (mode !== "proof" || !modelId) return;
    loadDiagnosis(modelId).then(setResult).catch(e => setError(e.message));
  }, [modelId, mode]);

  useEffect(() => {
    if (mode !== "proof") return;
    const proof = proofs.find(item => item.proof_id === proofId);
    if (proof) setModelId(proof.model_id);
  }, [proofId, proofs, mode]);

  function handleLiveResult(data) {
    setMode("live");
    setResult(data);
    setError("");
  }

  return <main>
    <Header />

    <div className="mode-tabs">
      <button className={mode === "proof" ? "active" : ""} onClick={() => setMode("proof")}>Professor Proof Library</button>
      <button className={mode === "live" ? "active" : ""} onClick={() => setMode("live")}>Live Upload & Diagnose</button>
    </div>

    {mode === "live" ? <LiveUpload onResult={handleLiveResult} onError={setError} loading={loading} setLoading={setLoading} /> : <>
      <ProofSelector proofs={proofs} value={proofId} onChange={setProofId} />
      <div className="toolbar"><details><summary>Advanced: browse all 100 research models</summary><ModelSelector models={models} value={modelId} onChange={setModelId} /></details></div>
    </>}

    {error && <div className="error">{error}</div>}
    {result && <Results result={result} />}
  </main>;
}
