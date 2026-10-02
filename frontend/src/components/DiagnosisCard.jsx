import React from "react";
export default function DiagnosisCard({ diagnosis, confidence }) {
  return <section className="card diagnosis"><span className="eyebrow">DIAGNOSIS</span><h2>{diagnosis.replace("_", " ")}</h2><div className="confidence">{(confidence * 100).toFixed(1)}% confidence</div></section>;
}
