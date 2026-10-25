import axios from "axios";

const API = axios.create({ baseURL: "http://localhost:8000" });

// ─── Health ───────────────────────────────────────────────────────────────────
export const checkHealth = () => API.get("/api/health").then(r => r.data);

// ─── Workspaces ───────────────────────────────────────────────────────────────
export const getWorkspaces = () =>
  API.get("/api/workspaces").then(r => r.data);

export const getWorkspace = (id) =>
  API.get(`/api/workspaces/${id}`).then(r => r.data);

export const createWorkspace = (body) =>
  API.post("/api/workspaces", body).then(r => r.data);

export const updateWorkspace = (id, body) =>
  API.patch(`/api/workspaces/${id}`, body).then(r => r.data);

export const deleteWorkspace = (id) =>
  API.delete(`/api/workspaces/${id}`);

// ─── Papers ───────────────────────────────────────────────────────────────────
export const getPapers = (workspaceId) =>
  API.get(`/api/workspaces/${workspaceId}/papers`).then(r => r.data);

export const getPaper = (paperId) =>
  API.get(`/api/papers/${paperId}`).then(r => r.data);

export const uploadPapers = (workspaceId, files) => {
  const form = new FormData();
  files.forEach(f => form.append("files", f));
  return API.post(`/api/workspaces/${workspaceId}/papers`, form).then(r => r.data);
};

// Legacy single upload (backward compat)
export const uploadPaper = (file) => {
  const form = new FormData();
  form.append("file", file);
  return API.post("/api/papers/upload", form).then(r => r.data);
};

export const getAnalysis = (paperId) =>
  API.get(`/api/papers/${paperId}/analysis`).then(r => r.data);

export const reanalyzePaper = (paperId) =>
  API.post(`/api/papers/${paperId}/analyze`).then(r => r.data);

export const deletePaper = (paperId) =>
  API.delete(`/api/papers/${paperId}`);

export const verifyField = (fieldId, status) =>
  API.patch(`/api/fields/${fieldId}/verify`, { verification_status: status }).then(r => r.data);

export const updateScreeningDecision = (paperId, decision) =>
  API.patch(`/api/papers/${paperId}/screening-decision`, { decision }).then(r => r.data);

// ─── Screening ────────────────────────────────────────────────────────────────
export const screenPapers = (workspaceId, query) =>
  API.post(`/api/workspaces/${workspaceId}/screen`, { query }).then(r => r.data);

// ─── Comparison ───────────────────────────────────────────────────────────────
export const getComparison = (workspaceId) =>
  API.get(`/api/workspaces/${workspaceId}/comparison`).then(r => r.data);

export const exportCSV = (workspaceId) => {
  window.open(`http://localhost:8000/api/workspaces/${workspaceId}/export.csv`, "_blank");
};

// ─── Insights ─────────────────────────────────────────────────────────────────
export const getInsights = (workspaceId) =>
  API.get(`/api/workspaces/${workspaceId}/insights`).then(r => r.data);

// ─── Collection Research Brief ────────────────────────────────────────────────
export const getResearchBrief = (workspaceId) =>
  API.get(`/api/workspaces/${workspaceId}/research-brief`).then(r => r.data);

export const analyzeCollection = (workspaceId) =>
  API.post(`/api/workspaces/${workspaceId}/analyze-collection`).then(r => r.data);

// ─── Cross-Paper Q&A ──────────────────────────────────────────────────────────
export const askCollection = (workspaceId, question, top_k = 5) =>
  API.post(`/api/workspaces/${workspaceId}/ask`, { question, top_k }).then(r => r.data);

// ─── Related Papers & Academic Discovery ──────────────────────────────────────
export const getRelatedPapers = (workspaceId, focalPaperId = null) => {
  const params = focalPaperId ? `?focal_paper_id=${focalPaperId}` : "";
  return API.get(`/api/workspaces/${workspaceId}/related-papers${params}`).then(r => r.data);
};

// ─── Test connection (legacy) ─────────────────────────────────────────────────
export const testBackendConnection = () =>
  API.get("/api/test").then(r => r.data);