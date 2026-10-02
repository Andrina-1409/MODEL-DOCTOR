import React from "react";
export default function ModelSelector({ models, value, onChange }) {
  return <label className="selector">Model<select value={value || ""} onChange={e => onChange(e.target.value)}><option value="" disabled>Select a model</option>{models.map(m => <option key={m.model_id} value={m.model_id}>{m.model_id} · {m.fault}</option>)}</select></label>;
}
