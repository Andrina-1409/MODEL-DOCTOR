import React from "react";
export default function GroundTruthBadge({ value }) {
  return <div className="truth">Experiment ground truth: <strong>{value.replace("_", " ")}</strong></div>;
}
