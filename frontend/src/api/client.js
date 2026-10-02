export async function loadModelIndex() {
  const response = await fetch("/data/index.json");
  if (!response.ok) throw new Error("Could not load demo model index.");
  return response.json();
}

export async function loadProofCatalog() {
  const response = await fetch("/proof/catalog.json");
  if (!response.ok) throw new Error("Could not load the professor proof library.");
  return response.json();
}

export async function loadDiagnosis(modelId) {
  const response = await fetch(`/data/${modelId}.json`);
  if (!response.ok) throw new Error(`Could not load diagnosis for ${modelId}.`);
  return response.json();
}

export async function liveDiagnose(file) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/live/diagnose", { method: "POST", body });
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || "Live diagnosis failed.");
  return data;
}
