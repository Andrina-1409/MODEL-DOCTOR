import React from "react";
export default function ProbabilityBars({ probabilities }) {
  return <section className="card"><h3>Diagnosis probabilities</h3>{Object.entries(probabilities).map(([label, value]) => <div className="prob" key={label}><div><span>{label.replace("_", " ")}</span><b>{(value * 100).toFixed(1)}%</b></div><div className="bar"><i style={{width: `${value * 100}%`}} /></div></div>)}</section>;
}
