import React from "react";

export default function ProofSelector({ proofs, value, onChange }) {
  return <section className="proof-panel">
    <div>
      <span className="eyebrow">PROFESSOR DEMO LIBRARY</span>
      <h3>Choose a ready-to-run proof model</h3>
      <p>No internet search, dataset setup, or preprocessing is needed during the demo.</p>
    </div>
    <div className="proof-grid">
      {proofs.map(proof => <button key={proof.proof_id} className={`proof-card ${value === proof.proof_id ? "selected" : ""}`} onClick={() => onChange(proof.proof_id)}>
        <span className="proof-name">{proof.name}</span>
        <span className="proof-meta">{proof.architecture} · {proof.dataset}</span>
        <span className="proof-fault">{proof.fault.replaceAll("_", " ")}</span>
      </button>)}
    </div>
  </section>;
}
