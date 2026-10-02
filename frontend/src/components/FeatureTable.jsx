import React from "react";
export default function FeatureTable({ features }) {
  return <section className="card"><h3>Diagnostic evidence</h3><table><tbody>{Object.entries(features).map(([key, value]) => <tr key={key}><td>{key}</td><td>{Number(value).toFixed(4)}</td></tr>)}</tbody></table></section>;
}
