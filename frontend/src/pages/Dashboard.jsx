import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { getWorkspaces, createWorkspace, deleteWorkspace, uploadPaper } from "../services/api";
import { formatDate } from "../utils/helpers";

export default function Dashboard() {
  const navigate = useNavigate();
  const singleFileRef = useRef(null);

  const [workspaces, setWorkspaces]         = useState([]);
  const [loading, setLoading]               = useState(true);
  const [creating, setCreating]             = useState(false);
  const [showForm, setShowForm]             = useState(false);
  const [form, setForm]                     = useState({ title: "", description: "", research_question: "" });
  const [error, setError]                   = useState("");
  
  // Single paper mode state
  const [singleUploading, setSingleUploading] = useState(false);
  const [singleError, setSingleError]         = useState("");

  useEffect(() => { load(); }, []);

  async function load() {
    try {
      setLoading(true);
      const data = await getWorkspaces();
      setWorkspaces(data);
    } catch {
      setError("Could not load workspaces. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate(e) {
    e.preventDefault();
    if (!form.title.trim()) return;
    try {
      setCreating(true);
      const newWs = await createWorkspace(form);
      setForm({ title: "", description: "", research_question: "" });
      setShowForm(false);
      navigate(`/workspace/${newWs.id}`);
    } catch {
      setError("Failed to create workspace.");
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(id, e) {
    e.stopPropagation();
    if (!confirm("Delete this workspace and all its papers?")) return;
    try {
      await deleteWorkspace(id);
      await load();
    } catch {
      setError("Failed to delete workspace.");
    }
  }

  async function handleSinglePaperUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      setSingleUploading(true);
      setSingleError("");
      const res = await uploadPaper(file);
      if (res && res.paper_id) {
        navigate(`/paper/${res.paper_id}`);
      } else {
        setSingleError("Upload completed but no paper ID returned.");
      }
    } catch (err) {
      setSingleError(err?.response?.data?.detail || "Failed to upload single paper.");
    } finally {
      setSingleUploading(false);
      if (singleFileRef.current) singleFileRef.current.value = "";
    }
  }

  return (
    <div className="page-container">
      {/* Hero */}
      <div className="dashboard-hero">
        <div className="hero-pill"><span className="hero-pill-dot" /> NLP Research Analysis Platform · M.Sc. IT</div>
        <h1>Scientific <span className="gradient-text">Literature</span><br />Mining System</h1>
        <p className="hero-description">
          A dual-mode research intelligence platform for investigation across scientific publications.
          Analyze individual papers or synthesize multi-paper collections with evidence-backed extraction,
          cross-paper question answering, and comparative matrices.
        </p>
        <div className="hero-stats">
          <div className="hero-stat"><span className="hero-stat-value">T5</span><span className="hero-stat-label">Summarization Model</span></div>
          <div className="hero-stat-divider" />
          <div className="hero-stat"><span className="hero-stat-value">TF-IDF</span><span className="hero-stat-label">Passage Retrieval</span></div>
          <div className="hero-stat-divider" />
          <div className="hero-stat"><span className="hero-stat-value">6 Facets</span><span className="hero-stat-label">Information Extraction</span></div>
          <div className="hero-stat-divider" />
          <div className="hero-stat"><span className="hero-stat-value">Dual-Mode</span><span className="hero-stat-label">Paper vs Collection</span></div>
        </div>
      </div>

      {/* ─── DUAL-MODE PRIMARY WORKFLOW SELECTOR ─── */}
      <div className="dual-mode-banner">
        <h2 className="section-title" style={{ marginBottom: 8, fontSize: "1.25rem" }}>
          Select Research Workflow
        </h2>
        <p style={{ color: "var(--clr-text-2)", marginBottom: 20, fontSize: "0.95rem" }}>
          Choose between analyzing a standalone document or investigating a research question across a multi-paper collection.
        </p>

        <div className="dual-mode-grid">
          {/* OPTION 1: MODE A - INDIVIDUAL PAPER ANALYSIS */}
          <div className="mode-selection-card">
            <div className="mode-card-badge">Mode A · Document Level</div>
            <div className="mode-card-icon">📄</div>
            <h3 className="mode-card-title">Analyze an Individual Paper</h3>
            <p className="mode-card-desc">
              Upload a single scientific PDF to generate a 4-part T5 abstractive summary (Overview, Method, Results, Conclusion),
              extract structured scientific entities, and audit ground-truth page evidence.
            </p>
            <div style={{ marginTop: "auto" }}>
              <label className="primary-button" style={{ display: "inline-flex", cursor: singleUploading ? "wait" : "pointer" }}>
                <input
                  ref={singleFileRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  style={{ display: "none" }}
                  disabled={singleUploading}
                  onChange={handleSinglePaperUpload}
                />
                {singleUploading ? "⏳ Processing PDF…" : "Upload Single PDF →"}
              </label>
              {singleError && <div className="error-message" style={{ marginTop: 10 }}>{singleError}</div>}
            </div>
          </div>

          {/* OPTION 2: MODE B - RESEARCH COLLECTION ANALYSIS */}
          <div className="mode-selection-card highlight">
            <div className="mode-card-badge primary">Mode B · Primary Capability</div>
            <div className="mode-card-icon">🔬</div>
            <h3 className="mode-card-title">Create or Open a Research Workspace</h3>
            <p className="mode-card-desc">
              Formulate a research inquiry across multiple publications. Synthesize a combined 9-part research brief,
              ask questions across the collection with passage citations, and contrast methods in a comparison matrix.
            </p>
            <div style={{ marginTop: "auto" }}>
              <button
                id="create-ws-quick-btn"
                className="primary-button"
                onClick={() => setShowForm(true)}
              >
                + New Research Workspace
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ─── WORKSPACES CATALOG ─── */}
      <div className="section-header-row" style={{ marginTop: 32 }}>
        <div>
          <h2 className="section-title">Active Research Collections</h2>
          <p style={{ color: "var(--clr-text-2)", fontSize: "0.9rem" }}>
            Persistent workspaces with stored papers, extraction records, and synthesized briefs.
          </p>
        </div>
        <button id="new-workspace-btn" className="secondary-button" onClick={() => setShowForm(v => !v)}>
          {showForm ? "✕ Close Form" : "+ Create Workspace"}
        </button>
      </div>

      {error && <div className="error-message"><span>⚠️</span><span>{error}</span></div>}

      {/* Create workspace form */}
      {showForm && (
        <form className="create-workspace-form" onSubmit={handleCreate} id="create-workspace-form">
          <h3 className="form-title">Create Research Workspace</h3>
          <p style={{ color: "var(--clr-text-2)", fontSize: "0.88rem", marginBottom: 16 }}>
            Define the scope and central research question for this collection of papers.
          </p>
          <div className="form-group">
            <label className="form-label">Workspace Title *</label>
            <input
              id="ws-title"
              className="form-input"
              placeholder="e.g. Transformer-Based Summarization Systems"
              value={form.title}
              onChange={e => setForm(p => ({ ...p, title: e.target.value }))}
              required
            />
          </div>
          <div className="form-group">
            <label className="form-label">Research Question</label>
            <input
              id="ws-question"
              className="form-input"
              placeholder="e.g. Which methods achieve the best ROUGE scores on CNN/DailyMail and what limitations recur?"
              value={form.research_question}
              onChange={e => setForm(p => ({ ...p, research_question: e.target.value }))}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Description / Scope</label>
            <input
              id="ws-desc"
              className="form-input"
              placeholder="Optional notes on targeted research collection"
              value={form.description}
              onChange={e => setForm(p => ({ ...p, description: e.target.value }))}
            />
          </div>
          <div className="form-actions">
            <button type="submit" id="create-ws-submit" className="primary-button" disabled={creating}>
              {creating ? "Creating Workspace…" : "Create & Enter Workspace →"}
            </button>
          </div>
        </form>
      )}

      {/* Workspace list */}
      {loading ? (
        <div className="loading-container">
          <div className="loading-spinner-wrapper"><div className="loading-spinner" /></div>
          <p className="loading-subtitle">Loading research workspaces…</p>
        </div>
      ) : workspaces.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">🔬</div>
          <h3>No research workspaces yet</h3>
          <p>Create a workspace to begin uploading papers and investigating your research question.</p>
        </div>
      ) : (
        <div className="workspace-grid">
          {workspaces.map(ws => (
            <div
              key={ws.id}
              className="workspace-card"
              id={`ws-${ws.id}`}
              onClick={() => navigate(`/workspace/${ws.id}`)}
            >
              <div className="ws-card-header">
                <div className="ws-icon">📚</div>
                <button
                  className="ws-delete-btn"
                  onClick={e => handleDelete(ws.id, e)}
                  title="Delete workspace"
                >
                  ✕
                </button>
              </div>
              <h3 className="ws-title">{ws.title}</h3>
              {ws.research_question ? (
                <p className="ws-question">🎯 "{ws.research_question}"</p>
              ) : (
                <p className="ws-question" style={{ fontStyle: "italic", color: "var(--clr-text-3)" }}>
                  No research question specified
                </p>
              )}
              <div className="ws-meta">
                <span className="ws-paper-count">{ws.paper_count} paper{ws.paper_count !== 1 ? "s" : ""}</span>
                <span className="ws-date">{formatDate(ws.created_at)}</span>
              </div>
              <div className="ws-card-footer">
                <span className="ws-open-hint">Open Workspace & Brief →</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
