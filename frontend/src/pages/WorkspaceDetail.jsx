import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import {
  getWorkspace, updateWorkspace, getPapers, uploadPapers,
  deletePaper, updateScreeningDecision, screenPapers,
  getComparison, exportCSV, getInsights,
  getResearchBrief, analyzeCollection, askCollection, getRelatedPapers
} from "../services/api";
import { formatDate, formatSize, STATUS_META, DECISION_META } from "../utils/helpers";

export default function WorkspaceDetail() {
  const { workspaceId } = useParams();
  const navigate        = useNavigate();
  const fileRef         = useRef(null);

  const [workspace, setWorkspace]         = useState(null);
  const [papers, setPapers]               = useState([]);
  const [loading, setLoading]             = useState(true);
  const [uploading, setUploading]         = useState(false);
  const [uploadResults, setUploadResults] = useState([]);
  
  // Workspace editing
  const [editingWs, setEditingWs]         = useState(false);
  const [editTitle, setEditTitle]         = useState("");
  const [editQuestion, setEditQuestion]   = useState("");
  const [savingWs, setSavingWs]           = useState(false);

  // Active tab: "brief" | "qa" | "papers" | "comparison" | "screening" | "insights" | "recommendations"
  const [activeTab, setActiveTab]         = useState("brief");

  // Research Brief state
  const [brief, setBrief]                 = useState(null);
  const [briefLoading, setBriefLoading]   = useState(false);
  const [briefError, setBriefError]       = useState("");
  const [evidenceModal, setEvidenceModal] = useState(null);
  const [expandedSections, setExpandedSections] = useState({});

  // Q&A state
  const [qaQuestion, setQaQuestion]       = useState("");
  const [qaLoading, setQaLoading]         = useState(false);
  const [qaResult, setQaResult]           = useState(null);
  const [qaError, setQaError]             = useState("");

  // Search & Screening state
  const [query, setQuery]                 = useState("");
  const [screening, setScreening]         = useState(false);
  const [screenResults, setScreenResults] = useState(null);

  // Comparison & Insights data
  const [comparisonData, setComparisonData] = useState(null);
  const [insightsData, setInsightsData]     = useState(null);
  const [compFilter, setCompFilter]         = useState("");

  // Recommendations state
  const [recommendations, setRecommendations] = useState(null);
  const [recLoading, setRecLoading]           = useState(false);

  const [error, setError]                 = useState("");

  useEffect(() => {
    loadAll();
  }, [workspaceId]);

  // Polling papers while any are still processing
  useEffect(() => {
    const processing = papers.some(p =>
      p.analysis_status === "pending" || p.analysis_status === "processing"
    );
    if (!processing) return;
    const timer = setTimeout(() => {
      loadPapers();
      loadComparison();
      loadInsights();
      loadBrief(false);
    }, 4000);
    return () => clearTimeout(timer);
  }, [papers]);

  async function loadAll() {
    try {
      setLoading(true);
      const [ws, ps] = await Promise.all([
        getWorkspace(workspaceId),
        getPapers(workspaceId),
      ]);
      setWorkspace(ws);
      setPapers(ps);
      setEditTitle(ws.title || "");
      setEditQuestion(ws.research_question || "");
      if (ws.research_question) setQuery(ws.research_question);

      // Load supporting data in parallel
      loadBrief(false);
      loadComparison();
      loadInsights();
    } catch {
      setError("Could not load workspace. Is the backend server running?");
    } finally {
      setLoading(false);
    }
  }

  async function loadPapers() {
    try {
      const ps = await getPapers(workspaceId);
      setPapers(ps);
    } catch {}
  }

  async function loadBrief(forceReanalyze = false) {
    try {
      setBriefLoading(true);
      setBriefError("");
      const data = forceReanalyze
        ? await analyzeCollection(workspaceId)
        : await getResearchBrief(workspaceId);
      setBrief(data);
    } catch {
      setBriefError("Could not generate or retrieve collection research brief.");
    } finally {
      setBriefLoading(false);
    }
  }

  async function loadComparison() {
    try {
      const cd = await getComparison(workspaceId);
      setComparisonData(cd);
    } catch {}
  }

  async function loadInsights() {
    try {
      const ins = await getInsights(workspaceId);
      setInsightsData(ins);
    } catch {}
  }

  async function loadRecommendations() {
    try {
      setRecLoading(true);
      const data = await getRelatedPapers(workspaceId);
      setRecommendations(data);
    } catch {}
    finally {
      setRecLoading(false);
    }
  }

  async function handleSaveWorkspace(e) {
    e.preventDefault();
    try {
      setSavingWs(true);
      const updated = await updateWorkspace(workspaceId, {
        title: editTitle,
        research_question: editQuestion,
      });
      setWorkspace(updated);
      setEditingWs(false);
      await loadBrief(true);
    } catch {
      setError("Failed to update workspace metadata.");
    } finally {
      setSavingWs(false);
    }
  }

  async function handleUpload(e) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    try {
      setUploading(true);
      setError("");
      setUploadResults([]);
      const data = await uploadPapers(workspaceId, files);
      setUploadResults(data.uploaded || []);
      await loadPapers();
      await loadComparison();
      await loadInsights();
    } catch (err) {
      setError(err?.response?.data?.detail || "Upload failed.");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function handleDelete(paperId) {
    if (!confirm("Delete this paper and its analysis from the workspace?")) return;
    try {
      await deletePaper(paperId);
      await loadPapers();
      await loadComparison();
      await loadInsights();
      await loadBrief(true);
    } catch {
      setError("Delete failed.");
    }
  }

  async function handleDecision(paperId, decision) {
    try {
      await updateScreeningDecision(paperId, decision);
      await loadPapers();
    } catch {
      setError("Failed to update decision.");
    }
  }

  async function handleScreen(e) {
    e.preventDefault();
    if (!query.trim()) return;
    try {
      setScreening(true);
      setError("");
      const data = await screenPapers(workspaceId, query);
      setScreenResults(data);
      setActiveTab("screening");
    } catch {
      setError("Screening failed. Ensure papers are uploaded.");
    } finally {
      setScreening(false);
    }
  }

  async function handleQA(e) {
    if (e) e.preventDefault();
    if (!qaQuestion.trim()) return;
    try {
      setQaLoading(true);
      setQaError("");
      const res = await askCollection(workspaceId, qaQuestion.trim());
      setQaResult(res);
    } catch {
      setQaError("Could not retrieve answer from collection. Please verify papers are processed.");
    } finally {
      setQaLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-container">
          <div className="loading-spinner" />
          <p style={{ marginTop: 14, color: "var(--clr-text-2)" }}>Loading workspace & research brief…</p>
        </div>
      </div>
    );
  }

  if (!workspace) {
    return (
      <div className="page-container">
        <div className="error-message">⚠️ Workspace not found.</div>
        <button className="secondary-button" onClick={() => navigate("/")}>← Back to Dashboard</button>
      </div>
    );
  }

  const statusMeta = (s) => STATUS_META[s] || { label: s, cls: "status-pending" };
  const completedPapers = papers.filter(p => p.analysis_status === "completed");

  return (
    <div className="page-container">
      {/* Top back navigation */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <Link to="/" className="back-link">← All Workspaces & Workflows</Link>
        <span className="hero-pill" style={{ margin: 0 }}>Mode B · Research Collection</span>
      </div>

      {/* Workspace Header Box */}
      <div className="ws-detail-header">
        <div className="ws-detail-info" style={{ width: "100%" }}>
          {!editingWs ? (
            <>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16 }}>
                <div>
                  <h1 className="ws-detail-title">{workspace.title}</h1>
                  {workspace.research_question ? (
                    <p className="ws-detail-question">🎯 Research Question: "{workspace.research_question}"</p>
                  ) : (
                    <p className="ws-detail-question" style={{ fontStyle: "italic", color: "var(--clr-text-3)" }}>
                      🎯 No research question set. Click edit to define your research question.
                    </p>
                  )}
                  {workspace.description && (
                    <p className="ws-detail-desc">{workspace.description}</p>
                  )}
                </div>
                <button
                  className="secondary-button sm"
                  onClick={() => setEditingWs(true)}
                  style={{ whiteSpace: "nowrap" }}
                >
                  ✏️ Edit Scope
                </button>
              </div>
              <div className="ws-detail-meta" style={{ marginTop: 12 }}>
                <span><strong>{papers.length}</strong> total paper{papers.length !== 1 ? "s" : ""}</span>
                <span>·</span>
                <span><strong>{completedPapers.length}</strong> analyzed</span>
                <span>·</span>
                <span>Created {formatDate(workspace.created_at)}</span>
              </div>
            </>
          ) : (
            <form onSubmit={handleSaveWorkspace} style={{ width: "100%" }}>
              <div className="form-group">
                <label className="form-label">Workspace Title</label>
                <input
                  className="form-input"
                  value={editTitle}
                  onChange={e => setEditTitle(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">Research Question</label>
                <input
                  className="form-input"
                  value={editQuestion}
                  onChange={e => setEditQuestion(e.target.value)}
                  placeholder="e.g. Which methods achieve the best results and what limitations recur?"
                />
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button type="submit" className="primary-button sm" disabled={savingWs}>
                  {savingWs ? "Saving…" : "Save Changes"}
                </button>
                <button
                  type="button"
                  className="secondary-button sm"
                  onClick={() => setEditingWs(false)}
                >
                  Cancel
                </button>
              </div>
            </form>
          )}
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      {/* ─── DEDICATED WORKSPACE NAVIGATION TABS ─── */}
      <div className="main-tabs">
        <button
          className={`main-tab-btn ${activeTab === "brief" ? "active" : ""}`}
          onClick={() => {
            setActiveTab("brief");
            loadBrief(false);
          }}
        >
          📑 Collection Research Brief
        </button>
        <button
          className={`main-tab-btn ${activeTab === "qa" ? "active" : ""}`}
          onClick={() => setActiveTab("qa")}
        >
          💬 Ask the Collection
        </button>
        <button
          className={`main-tab-btn ${activeTab === "papers" ? "active" : ""}`}
          onClick={() => setActiveTab("papers")}
        >
          📄 Papers & Upload <span className="tab-badge">{papers.length}</span>
        </button>
        <button
          className={`main-tab-btn ${activeTab === "comparison" ? "active" : ""}`}
          onClick={() => {
            setActiveTab("comparison");
            loadComparison();
          }}
        >
          📊 Comparison Matrix
        </button>
        <button
          className={`main-tab-btn ${activeTab === "screening" ? "active" : ""}`}
          onClick={() => setActiveTab("screening")}
        >
          🔍 Relevance Screening {screenResults ? <span className="tab-badge">{screenResults.results.length}</span> : ""}
        </button>
        <button
          className={`main-tab-btn ${activeTab === "insights" ? "active" : ""}`}
          onClick={() => {
            setActiveTab("insights");
            loadInsights();
          }}
        >
          💡 Collection Insights
        </button>
        <button
          className={`main-tab-btn ${activeTab === "recommendations" ? "active" : ""}`}
          onClick={() => {
            setActiveTab("recommendations");
            loadRecommendations();
          }}
        >
          🔗 Related Papers & Discovery
        </button>
      </div>

      {/* ════════════════════════════════════════════════════════════
          TAB 1: COLLECTION RESEARCH BRIEF (MODULE C)
          ════════════════════════════════════════════════════════════ */}
      {activeTab === "brief" && (
        <div className="tab-content-container">
          <div className="section-header-row">
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <h2 className="section-title">📑 Multi-Paper Research Synthesis</h2>
                <span className="navbar-badge" style={{ backgroundColor: "rgba(99, 102, 241, 0.15)", color: "#4f46e5" }}>
                  Evidence-Grounded Cross-Paper Engine
                </span>
              </div>
              <p className="page-subtitle">
                Synthesized scientific evidence across {completedPapers.length} analyzed papers, contrasting architectural paradigms, pretraining objectives, benchmarks, and author-stated constraints.
              </p>
            </div>
            <button
              className="secondary-button sm"
              onClick={() => loadBrief(true)}
              disabled={briefLoading}
            >
              {briefLoading ? "⏳ Synthesizing Brief…" : "⚡ Re-analyze Collection"}
            </button>
          </div>

          {briefLoading && !brief && (
            <div className="loading-container">
              <div className="loading-spinner" />
              <p style={{ marginTop: 12, color: "var(--clr-text-2)" }}>Synthesizing collection research brief…</p>
            </div>
          )}

          {briefError && <div className="error-message">{briefError}</div>}

          {brief && (
            <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              {/* Secondary Metadata Strip (Small and Secondary) */}
              {brief.coverage_stats && (
                <div className="compact-stats-strip">
                  <span className="compact-stat-item"><strong>{brief.paper_count}</strong> Analyzed Papers</span>
                  <span className="compact-stat-dot">•</span>
                  <span className="compact-stat-item"><strong>{brief.coverage_stats.total_pages || 0}</strong> Screened Pages</span>
                  <span className="compact-stat-dot">•</span>
                  <span className="compact-stat-item"><strong>{brief.coverage_stats.methods_cataloged || 0}</strong> Distinct Architectures</span>
                  <span className="compact-stat-dot">•</span>
                  <span className="compact-stat-item"><strong>{brief.coverage_stats.datasets_cataloged || 0}</strong> Benchmarks</span>
                  <span className="compact-stat-dot">•</span>
                  <span className="compact-stat-item"><strong>{brief.coverage_stats.total_entities || 0}</strong> Extracted Entities</span>
                  <span className="compact-stat-dot">•</span>
                  <span className="compact-stat-pill">100% Page Verified</span>
                </div>
              )}

              {/* 1. EXECUTIVE RESEARCH SYNTHESIS */}
              {brief.executive_synthesis && (
                <div className="executive-synthesis-card">
                  <div className="executive-synthesis-header">
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span className="exec-icon">✨</span>
                      <div>
                        <h3 className="exec-title">Executive Research Synthesis</h3>
                        <div className="exec-inquiry-tag">
                          Workspace Target Inquiry: <strong>{brief.research_question || workspace?.title || "Scientific Literature Mining"}</strong>
                        </div>
                      </div>
                    </div>
                  </div>
                  <p className="exec-narrative">{brief.executive_synthesis}</p>
                </div>
              )}

              {/* 2. SUBSTANTIVE KEY FINDINGS ACROSS PAPERS */}
              {brief.key_findings && brief.key_findings.length > 0 && (
                <div className="findings-section-container">
                  <div className="section-subheading-row">
                    <div>
                      <h3 className="findings-group-title">🔍 Key Findings Across Papers</h3>
                      <p className="findings-group-desc">
                        Substantive cross-paper findings explaining major architectural paradigms, meaningful formulation differences, trade-offs, and evidence-backed conclusions.
                      </p>
                    </div>
                  </div>

                  <div className="key-findings-grid">
                    {brief.key_findings.map((kf, kfi) => (
                      <div key={kfi} className="key-finding-card">
                        <div className="kf-number-badge">Finding {kfi + 1}</div>
                        <h4 className="kf-title">{kf.title}</h4>
                        <p className="kf-synthesis">{kf.synthesis}</p>

                        {kf.trade_offs && (
                          <div className="kf-tradeoff-box">
                            <div className="kf-tradeoff-label">⚖️ Methodological Trade-off:</div>
                            <div className="kf-tradeoff-text">{kf.trade_offs}</div>
                          </div>
                        )}

                        {kf.evidence_citations && kf.evidence_citations.length > 0 && (
                          <div className="kf-evidence-row">
                            <span className="kf-evidence-label">Audited Evidence Passages:</span>
                            <div className="kf-citations-wrap">
                              {kf.evidence_citations.map((cit, citi) => (
                                <button
                                  key={citi}
                                  type="button"
                                  className="evidence-cite-pill"
                                  title="Click to inspect exact passage and PDF page"
                                  onClick={() => setEvidenceModal({
                                    title: cit.paper_title || cit.filename,
                                    filename: cit.filename,
                                    page_number: cit.page_number,
                                    passage: cit.verbatim_passage,
                                    context: kf.title,
                                  })}
                                >
                                  📄 {cit.filename.replace('.pdf','')} · p.{cit.page_number}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 3. CROSS-PAPER COMPARISON MATRIX */}
              {brief.cross_paper_matrix && brief.cross_paper_matrix.length > 0 && (
                <div className="comparison-matrix-container">
                  <div className="section-subheading-row">
                    <div>
                      <h3 className="findings-group-title">📊 Cross-Paper Comparative Analysis</h3>
                      <p className="findings-group-desc">
                        Direct comparison of each paper's research problem, primary contribution, architectural approach, experimental setting, key findings, and documented limitations.
                      </p>
                    </div>
                  </div>

                  <div className="table-responsive-wrapper">
                    <table className="cross-paper-table">
                      <thead>
                        <tr>
                          <th style={{ width: "16%" }}>Paper & Architecture</th>
                          <th style={{ width: "18%" }}>Research Problem</th>
                          <th style={{ width: "18%" }}>Primary Contribution</th>
                          <th style={{ width: "18%" }}>Experimental Setting</th>
                          <th style={{ width: "14%" }}>Key Findings</th>
                          <th style={{ width: "16%" }}>Documented Limitations</th>
                        </tr>
                      </thead>
                      <tbody>
                        {brief.cross_paper_matrix.map((row, ri) => (
                          <tr key={ri}>
                            <td>
                              <div className="matrix-paper-cell">
                                <Link to={`/paper/${row.paper_id}`} className="matrix-paper-title">
                                  📄 {row.title || row.filename}
                                </Link>
                                <span className="matrix-arch-pill">{row.approach_architecture.split(' (')[0]}</span>
                              </div>
                            </td>
                            <td>
                              <div className="matrix-cell-text">{row.research_problem}</div>
                            </td>
                            <td>
                              <div className="matrix-cell-text">{row.primary_contribution}</div>
                            </td>
                            <td>
                              <div className="matrix-setting-text">{row.experimental_setting}</div>
                            </td>
                            <td>
                              <div className="matrix-cell-text">{row.major_findings}</div>
                            </td>
                            <td>
                              <div className="matrix-limitation-text">{row.documented_limitations}</div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 4. POTENTIAL RESEARCH OPPORTUNITIES (INFERRED GAPS) */}
              {brief.research_opportunities && brief.research_opportunities.length > 0 && (
                <div className="opportunities-container">
                  <div className="section-subheading-row">
                    <div>
                      <h3 className="findings-group-title">💡 Potential Research Opportunities & Inferred Gaps</h3>
                      <p className="findings-group-desc">
                        Plausible research gaps inferred through cross-paper synthesis. Inferred hypotheses are strictly distinguished from author-established facts.
                      </p>
                    </div>
                  </div>

                  <div className="opportunities-grid">
                    {brief.research_opportunities.map((opp, oppi) => (
                      <div key={oppi} className="opportunity-card">
                        <div className="opp-header">
                          <h4 className="opp-title">{opp.title}</h4>
                          <span className="opp-caution-badge">{opp.caution_label}</span>
                        </div>
                        <div className="opp-hypothesis-box">
                          <span className="opp-label">Inferred Hypothesis:</span>
                          <p className="opp-text">{opp.hypothesis}</p>
                        </div>
                        <div className="opp-rationale-box">
                          <span className="opp-label">Evidence-Backed Rationale:</span>
                          <p className="opp-text">{opp.collection_rationale}</p>
                        </div>
                        {opp.evidence_citations && opp.evidence_citations.length > 0 && (
                          <div className="opp-evidence-row">
                            <span className="kf-evidence-label">Supporting Literature Passages:</span>
                            <div className="kf-citations-wrap">
                              {opp.evidence_citations.map((c, ci) => (
                                <button
                                  key={ci}
                                  type="button"
                                  className="evidence-cite-pill"
                                  title="Inspect supporting passage in source PDF"
                                  onClick={() => setEvidenceModal({
                                    title: c.filename,
                                    filename: c.filename,
                                    page_number: c.page_number,
                                    passage: c.passage,
                                    context: opp.title,
                                  })}
                                >
                                  📄 {c.filename.replace('.pdf','')} · p.{c.page_number}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 5. NINE DETAILED CORE SECTIONS */}
              <div className="core-sections-container">
                <div className="section-subheading-row">
                  <div>
                    <h3 className="findings-group-title">📚 Detailed Core Research Dimensions</h3>
                    <p className="findings-group-desc">
                      Rigorous 9-section academic synthesis with clean dataset classification, contextualized results, verified limitations, and evidence provenance.
                    </p>
                  </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  {brief.sections && brief.sections.map((sec, idx) => {
                    const isExpanded = expandedSections[sec.section_id] !== false; // default expanded
                    return (
                      <div key={idx} className="brief-section-card" id={`brief-sec-${sec.section_id}`}>
                        <div
                          className="brief-sec-header"
                          style={{ cursor: "pointer" }}
                          onClick={() => setExpandedSections(prev => ({ ...prev, [sec.section_id]: !isExpanded }))}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ fontSize: "1.1rem" }}>{isExpanded ? "▾" : "▸"}</span>
                            <h3 className="brief-sec-title">{sec.title}</h3>
                          </div>
                          {sec.papers_cited && sec.papers_cited.length > 0 && (
                            <div className="brief-citations">
                              <span style={{ fontSize: "0.8rem", color: "var(--clr-text-3)", marginRight: 4 }}>Sources:</span>
                              {sec.papers_cited.map((pname, pi) => (
                                <span key={pi} className="citation-pill" title={pname}>
                                  📄 {pname.replace('.pdf', '')}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>

                        {isExpanded && (
                          <div style={{ marginTop: 14 }}>
                            <p className="brief-sec-content">{sec.content}</p>

                            {sec.key_points && sec.key_points.length > 0 && (
                              <div className="brief-key-points">
                                <div style={{ fontWeight: 600, fontSize: "0.85rem", color: "var(--clr-text-2)", marginBottom: 8 }}>
                                  Key Extracted Findings & Data:
                                </div>
                                <ul className="brief-points-list">
                                  {sec.key_points.map((pt, pidx) => (
                                    <li key={pidx}>{pt}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {sec.evidence_items && sec.evidence_items.length > 0 && (
                              <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 6 }}>
                                <span style={{ fontSize: "0.8rem", color: "var(--clr-text-3)", marginRight: 4 }}>Evidence Provenance:</span>
                                {sec.evidence_items.map((ev, evi) => (
                                  <button
                                    key={evi}
                                    type="button"
                                    className="evidence-cite-pill"
                                    onClick={() => setEvidenceModal({
                                      title: ev.filename,
                                      filename: ev.filename,
                                      page_number: ev.page_number,
                                      passage: ev.snippet,
                                      context: `${sec.title} — ${ev.label}`,
                                    })}
                                  >
                                    📍 {ev.filename.replace('.pdf','')} · p.{ev.page_number}
                                  </button>
                                ))}
                              </div>
                            )}

                            {sec.caveats && (
                              <div className="brief-caveat-box">
                                <span style={{ fontWeight: 600 }}>Provenance & Verification Note:</span> {sec.caveats}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* EVIDENCE NAVIGATION MODAL */}
              {evidenceModal && (
                <div className="evidence-modal-backdrop" onClick={() => setEvidenceModal(null)}>
                  <div className="evidence-modal-card" onClick={e => e.stopPropagation()}>
                    <div className="evidence-modal-header">
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: "1.3rem" }}>🔍</span>
                        <div>
                          <h3 className="evidence-modal-title">Evidence & Provenance Passage</h3>
                          <div className="evidence-modal-subtitle">
                            {evidenceModal.title || evidenceModal.filename} · PDF Page {evidenceModal.page_number}
                          </div>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="evidence-modal-close"
                        onClick={() => setEvidenceModal(null)}
                      >
                        ✕
                      </button>
                    </div>

                    <div className="evidence-modal-body">
                      {evidenceModal.context && (
                        <div className="evidence-modal-context">
                          <strong>Finding / Context:</strong> {evidenceModal.context}
                        </div>
                      )}

                      <div className="evidence-quote-box">
                        <div className="evidence-quote-label">Supporting Verbatim Passage:</div>
                        <p className="evidence-quote-text">"{evidenceModal.passage}"</p>
                      </div>

                      <div className="evidence-modal-footer">
                        <span className="provenance-pill">
                          📍 1-Indexed PDF Page {evidenceModal.page_number}
                        </span>
                        {(() => {
                          const matchedPaper = papers.find(p => p.filename === evidenceModal.filename);
                          if (matchedPaper) {
                            return (
                              <Link
                                to={`/paper/${matchedPaper.id}`}
                                className="primary-button sm"
                                onClick={() => setEvidenceModal(null)}
                              >
                                Open in Paper Detail →
                              </Link>
                            );
                          }
                          return null;
                        })()}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════
          TAB 2: ASK THE COLLECTION (MODULE E)
          ════════════════════════════════════════════════════════════ */}
      {activeTab === "qa" && (
        <div className="tab-content-container">
          <div className="screening-banner-box">
            <div className="screening-banner-header">
              <div className="screening-banner-title">
                <span>💬</span>
                <span>Ask the Collection — Evidence-Grounded Cross-Paper Q&A</span>
              </div>
              <span className="navbar-badge">TF-IDF Passage Retrieval</span>
            </div>

            <p style={{ color: "var(--clr-text-2)", fontSize: "0.92rem", marginBottom: 14 }}>
              Query all uploaded publications simultaneously. The system retrieves the most relevant page passages
              across papers and synthesizes a grounded answer with clickable citations.
            </p>

            <form onSubmit={handleQA} className="screen-input-row">
              <input
                id="qa-input-box"
                className="form-input screen-query-input"
                placeholder="Ask any research question (e.g. Which datasets are used? What limitations recur? Which papers discuss attention?)..."
                value={qaQuestion}
                onChange={e => setQaQuestion(e.target.value)}
              />
              <button
                type="submit"
                className="primary-button"
                disabled={qaLoading || !qaQuestion.trim()}
              >
                {qaLoading ? "Retrieving Evidence…" : "Ask Question"}
              </button>
            </form>

            {/* Quick example queries */}
            <div style={{ marginTop: 12, display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--clr-text-3)" }}>Suggested questions:</span>
              {[
                "Which datasets are used across these studies?",
                "What evaluation metrics are reported?",
                "What limitations and constraints do the authors report?",
                "Which methods and model architectures are compared?",
              ].map((qText, qi) => (
                <button
                  key={qi}
                  type="button"
                  className="quick-chip-btn"
                  onClick={() => {
                    setQaQuestion(qText);
                  }}
                >
                  {qText}
                </button>
              ))}
            </div>
          </div>

          {qaError && <div className="error-message">{qaError}</div>}

          {qaLoading && (
            <div className="loading-container">
              <div className="loading-spinner" />
              <p style={{ marginTop: 12, color: "var(--clr-text-2)" }}>
                Scanning page texts, ranking evidence passages, and synthesizing grounded answer…
              </p>
            </div>
          )}

          {qaResult && (
            <div className="qa-result-box" style={{ marginTop: 24 }}>
              <div className="qa-result-header">
                <div>
                  <span style={{ fontSize: "0.85rem", color: "var(--clr-text-3)" }}>Inquiry:</span>
                  <h3 style={{ fontSize: "1.1rem", color: "var(--clr-text-1)", marginTop: 2 }}>
                    "{qaResult.question}"
                  </h3>
                </div>
                <span className={`status-badge ${qaResult.is_sufficient ? "status-completed" : "status-failed"}`}>
                  {qaResult.is_sufficient ? `Evidence Found · ${qaResult.papers_consulted} Paper(s)` : "Insufficient Evidence"}
                </span>
              </div>

              {/* Synthesized Answer */}
              <div className="qa-answer-card">
                <div style={{ fontWeight: 600, color: "var(--clr-text-1)", marginBottom: 8, fontSize: "0.95rem" }}>
                  Synthesized Answer:
                </div>
                <div className="qa-answer-text" style={{ whiteSpace: "pre-line" }}>
                  {qaResult.answer}
                </div>
              </div>

              {/* Retrieved Passages and Citations */}
              {qaResult.citations && qaResult.citations.length > 0 && (
                <div style={{ marginTop: 20 }}>
                  <h4 style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--clr-text-2)", marginBottom: 12 }}>
                    Supporting Source Evidence ({qaResult.citations.length} passages):
                  </h4>
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {qaResult.citations.map((c, ci) => (
                      <div key={ci} className="evidence-passage-card">
                        <div className="evidence-passage-header">
                          <Link to={`/paper/${c.paper_id}`} className="evidence-paper-link">
                            📄 <strong>{c.filename}</strong> (Page {c.page_number})
                          </Link>
                          <span className="similarity-badge">
                            Relevance: {c.relevance_score}
                          </span>
                        </div>
                        <p className="evidence-passage-text">
                          "{c.text}"
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════
          TAB 3: PAPERS & UPLOAD (MODULE A)
          ════════════════════════════════════════════════════════════ */}
      {activeTab === "papers" && (
        <div className="tab-content-container">
          <label className="upload-panel">
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,application/pdf"
              multiple
              className="upload-file-input"
              onChange={handleUpload}
            />
            <div className="upload-title">
              {uploading ? "⏳ Uploading & extracting pages…" : "📁 Click or Drag PDF Papers Here to Upload"}
            </div>
            <div className="upload-subtitle">
              Upload multiple scientific PDFs (max 50 MB each). Duplicate files are detected via SHA-256 hash.
            </div>
          </label>

          {uploadResults.length > 0 && (
            <div className="upload-results" style={{ marginBottom: 20 }}>
              {uploadResults.map((r, i) => (
                <div key={i} className={`upload-result-item ${r.status}`}>
                  <span>{r.status === "uploaded" ? "✅" : r.status === "duplicate" ? "⚠️ Duplicate:" : "❌"}</span>
                  <span style={{ fontWeight: 600 }}>{r.filename}</span>
                  <span style={{ color: "var(--clr-text-3)", marginLeft: "auto" }}>{r.detail || r.status}</span>
                </div>
              ))}
            </div>
          )}

          {/* Papers Table */}
          {papers.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📭</div>
              <h3>No papers uploaded yet</h3>
              <p>Upload PDFs above to begin multi-paper research synthesis.</p>
            </div>
          ) : (
            <div className="table-wrapper">
              <table className="papers-table">
                <thead>
                  <tr>
                    <th>Publication / Document</th>
                    <th>Pages & Words</th>
                    <th>Analysis Status</th>
                    <th>Relevance</th>
                    <th>Screening</th>
                    <th style={{ textAlign: "right" }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {papers.map(p => {
                    const sm = statusMeta(p.analysis_status);
                    const dm = DECISION_META[p.screening_decision] || { label: p.screening_decision, cls: "dec-unreviewed" };
                    return (
                      <tr key={p.id}>
                        <td>
                          <div className="paper-name-cell">
                            <span className="pdf-icon">📄</span>
                            <div>
                              <Link to={`/paper/${p.id}`} className="paper-link" title="Open individual paper analysis">
                                {p.filename}
                              </Link>
                              <div className="paper-sub-meta">
                                {formatSize(p.file_size_bytes)} · Added {formatDate(p.created_at)}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td>
                          <div style={{ fontSize: "0.88rem" }}>
                            <strong>{p.page_count}</strong> pages
                            <div style={{ color: "var(--clr-text-3)", fontSize: "0.78rem" }}>
                              {p.word_count ? `${p.word_count.toLocaleString()} words` : "Pending extraction"}
                            </div>
                          </div>
                        </td>
                        <td>
                          <span className={`status-badge ${sm.cls}`}>{sm.label}</span>
                        </td>
                        <td>
                          {p.relevance_score > 0 ? (
                            <span className="score-pill">{(p.relevance_score * 100).toFixed(1)}%</span>
                          ) : (
                            <span style={{ color: "var(--clr-text-3)", fontSize: "0.85rem" }}>—</span>
                          )}
                        </td>
                        <td>
                          <select
                            className={`decision-select ${dm.cls}`}
                            value={p.screening_decision}
                            onChange={e => handleDecision(p.id, e.target.value)}
                          >
                            <option value="unreviewed">Unreviewed</option>
                            <option value="relevant">Relevant</option>
                            <option value="needs_review">Needs Review</option>
                            <option value="irrelevant">Irrelevant</option>
                          </select>
                        </td>
                        <td style={{ textAlign: "right" }}>
                          <div style={{ display: "inline-flex", gap: 8 }}>
                            <Link to={`/paper/${p.id}`} className="secondary-button sm">
                              Inspect Analysis →
                            </Link>
                            <button
                              className="ws-delete-btn"
                              onClick={() => handleDelete(p.id)}
                              title="Delete paper"
                            >
                              ✕
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════
          TAB 4: CROSS-PAPER COMPARISON MATRIX (MODULE F)
          ════════════════════════════════════════════════════════════ */}
      {activeTab === "comparison" && (
        <div className="tab-content-container">
          <div className="section-header-row">
            <div>
              <h2 className="section-title">📊 Cross-Paper Comparison Matrix</h2>
              <p className="page-subtitle">
                Side-by-side comparison of extracted tasks, methods, datasets, metrics, and limitations across publications.
              </p>
            </div>
            <button className="primary-button sm" onClick={() => exportCSV(workspaceId)}>
              📥 Export CSV (RFC 4180)
            </button>
          </div>

          <div style={{ marginBottom: 16 }}>
            <input
              className="form-input"
              style={{ maxWidth: 400 }}
              placeholder="Filter comparison table by method, dataset, metric..."
              value={compFilter}
              onChange={e => setCompFilter(e.target.value)}
            />
          </div>

          {!comparisonData ? (
            <div className="loading-container">
              <div className="loading-spinner" />
              <p style={{ marginTop: 12 }}>Loading comparison data…</p>
            </div>
          ) : comparisonData.rows.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📊</div>
              <h3>No comparison data available</h3>
              <p>Upload and process papers to populate the comparison matrix.</p>
            </div>
          ) : (
            <div className="table-wrapper">
              <table className="papers-table" style={{ fontSize: "0.85rem" }}>
                <thead>
                  <tr>
                    <th>Paper / Publication</th>
                    <th>Research Objective (Task)</th>
                    <th>Method / Architecture</th>
                    <th>Dataset / Benchmark</th>
                    <th>Metric & Result</th>
                    <th>Reported Limitation</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonData.rows
                    .filter(r => {
                      if (!compFilter.trim()) return true;
                      const q = compFilter.toLowerCase();
                      return (
                        r.filename.toLowerCase().includes(q) ||
                        r.task.toLowerCase().includes(q) ||
                        r.method.toLowerCase().includes(q) ||
                        r.dataset.toLowerCase().includes(q) ||
                        r.metric.toLowerCase().includes(q) ||
                        r.limitation.toLowerCase().includes(q)
                      );
                    })
                    .map(r => (
                      <tr key={r.paper_id}>
                        <td>
                          <Link to={`/paper/${r.paper_id}`} style={{ fontWeight: 600, color: "var(--clr-primary)" }}>
                            📄 {r.filename}
                          </Link>
                        </td>
                        <td>{r.task}</td>
                        <td><strong>{r.method}</strong></td>
                        <td>{r.dataset}</td>
                        <td>
                          <div><strong>{r.metric}</strong></div>
                          <div style={{ color: "var(--clr-text-2)" }}>{r.result}</div>
                        </td>
                        <td style={{ color: "var(--clr-text-2)" }}>{r.limitation}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════
          TAB 5: SEARCH & SCREENING (MODULE A/D)
          ════════════════════════════════════════════════════════════ */}
      {activeTab === "screening" && (
        <div className="tab-content-container">
          <div className="screening-banner-box">
            <div className="screening-banner-header">
              <div className="screening-banner-title">
                <span>🔍</span>
                <span>TF-IDF Relevance Screening</span>
              </div>
              <span className="navbar-badge">Lexical Retrieval</span>
            </div>
            <form onSubmit={handleScreen} className="screen-input-row">
              <input
                className="form-input screen-query-input"
                placeholder="Enter query to rank papers (e.g. self-attention, BLEU score, robotic grasping)..."
                value={query}
                onChange={e => setQuery(e.target.value)}
              />
              <button type="submit" className="primary-button" disabled={screening || !query.trim()}>
                {screening ? "Ranking…" : "Run Screening"}
              </button>
            </form>
          </div>

          {screenResults && (
            <div style={{ marginTop: 24 }}>
              <h3 style={{ fontSize: "1.05rem", fontWeight: 600, marginBottom: 16 }}>
                Ranked Screening Results for "{screenResults.query}":
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                {screenResults.results.map((res, ri) => (
                  <div key={ri} className="screen-result-card">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <Link to={`/paper/${res.paper_id}`} style={{ fontWeight: 600, fontSize: "1rem", color: "var(--clr-primary)" }}>
                        #{ri + 1} · 📄 {res.filename}
                      </Link>
                      <span className="score-pill">{(res.score * 100).toFixed(1)}% Match</span>
                    </div>
                    {res.snippet && (
                      <p style={{ fontSize: "0.9rem", color: "var(--clr-text-2)", fontStyle: "italic", margin: 0 }}>
                        "{res.snippet}" (Page {res.page_number})
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════
          TAB 6: COLLECTION INSIGHTS (MODULE H)
          ════════════════════════════════════════════════════════════ */}
      {activeTab === "insights" && (
        <div className="tab-content-container">
          <div className="section-header-row">
            <div>
              <h2 className="section-title">💡 Collection Insights</h2>
              <p className="page-subtitle">
                Aggregated statistics and frequency distributions generated from real stored extraction records.
              </p>
            </div>
          </div>

          {!insightsData ? (
            <div className="loading-container">
              <div className="loading-spinner" />
              <p style={{ marginTop: 12 }}>Loading collection insights…</p>
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20 }}>
              {/* Methods chart */}
              <div className="insights-card">
                <h3 className="insights-card-title">Top Extracted Methods & Models</h3>
                {insightsData.top_methods.length === 0 ? (
                  <p style={{ color: "var(--clr-text-3)", fontSize: "0.88rem" }}>No methods extracted yet.</p>
                ) : (
                  <div className="frequency-list">
                    {insightsData.top_methods.map((item, ii) => (
                      <div key={ii} className="frequency-row">
                        <span className="freq-label">{item.value}</span>
                        <div className="freq-bar-wrap">
                          <div
                            className="freq-bar"
                            style={{ width: `${Math.min(100, (item.count / insightsData.paper_count) * 100)}%` }}
                          />
                        </div>
                        <span className="freq-count">{item.count}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Datasets chart */}
              <div className="insights-card">
                <h3 className="insights-card-title">Top Extracted Datasets & Benchmarks</h3>
                {insightsData.top_datasets.length === 0 ? (
                  <p style={{ color: "var(--clr-text-3)", fontSize: "0.88rem" }}>No datasets extracted yet.</p>
                ) : (
                  <div className="frequency-list">
                    {insightsData.top_datasets.map((item, ii) => (
                      <div key={ii} className="frequency-row">
                        <span className="freq-label">{item.value}</span>
                        <div className="freq-bar-wrap">
                          <div
                            className="freq-bar"
                            style={{ width: `${Math.min(100, (item.count / insightsData.paper_count) * 100)}%`, backgroundColor: "#10b981" }}
                          />
                        </div>
                        <span className="freq-count">{item.count}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Metrics chart */}
              <div className="insights-card">
                <h3 className="insights-card-title">Evaluation Metrics Distribution</h3>
                {insightsData.top_metrics.length === 0 ? (
                  <p style={{ color: "var(--clr-text-3)", fontSize: "0.88rem" }}>No metrics extracted yet.</p>
                ) : (
                  <div className="frequency-list">
                    {insightsData.top_metrics.map((item, ii) => (
                      <div key={ii} className="frequency-row">
                        <span className="freq-label">{item.value}</span>
                        <div className="freq-bar-wrap">
                          <div
                            className="freq-bar"
                            style={{ width: `${Math.min(100, (item.count / insightsData.paper_count) * 100)}%`, backgroundColor: "#f59e0b" }}
                          />
                        </div>
                        <span className="freq-count">{item.count}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Missing Fields / Coverage Gaps */}
              <div className="insights-card">
                <h3 className="insights-card-title">Extraction Coverage & Gaps</h3>
                <p style={{ fontSize: "0.85rem", color: "var(--clr-text-3)", marginBottom: 12 }}>
                  Count of papers missing explicit extraction in each category:
                </p>
                {insightsData.missing_fields && (
                  <div className="frequency-list">
                    {Object.entries(insightsData.missing_fields).map(([ft, count], fi) => (
                      <div key={fi} className="frequency-row">
                        <span className="freq-label">{ft}</span>
                        <span className="freq-count" style={{ color: count > 0 ? "#ef4444" : "var(--clr-text-3)" }}>
                          {count} paper{count !== 1 ? "s" : ""} missing
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════
          TAB 7: RELATED PAPERS & DISCOVERY (MODULE G)
          ════════════════════════════════════════════════════════════ */}
      {activeTab === "recommendations" && (
        <div className="tab-content-container">
          <div className="section-header-row">
            <div>
              <h2 className="section-title">🔗 Related Paper Discovery & Recommendations</h2>
              <p className="page-subtitle">
                Two-stage discovery: (1) Relevance ranking within this collection; (2) External academic discovery via Semantic Scholar.
              </p>
            </div>
            <button className="secondary-button sm" onClick={loadRecommendations} disabled={recLoading}>
              {recLoading ? "Searching…" : "🔄 Refresh Discovery"}
            </button>
          </div>

          {recLoading && (
            <div className="loading-container">
              <div className="loading-spinner" />
              <p style={{ marginTop: 12 }}>Consulting local collection index and Semantic Scholar…</p>
            </div>
          )}

          {recommendations && (
            <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
              {/* Stage 1: Internal Recommendations */}
              <div>
                <h3 style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--clr-text-1)", marginBottom: 12 }}>
                  Stage 1: Internal Collection Alignment
                </h3>
                {recommendations.internal_results.length === 0 ? (
                  <div className="empty-state" style={{ padding: 24 }}>
                    <p>Upload at least 2 papers to see internal cross-paper recommendations.</p>
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {recommendations.internal_results.map((rec, ri) => (
                      <div key={ri} className="rec-card">
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                          <Link to={`/paper/${rec.paper_id}`} style={{ fontWeight: 600, color: "var(--clr-primary)", fontSize: "1rem" }}>
                            📄 {rec.title}
                          </Link>
                          <span className="navbar-badge">Internal Collection</span>
                        </div>
                        <p style={{ fontSize: "0.85rem", color: "var(--clr-text-3)", margin: "4px 0 8px 0" }}>
                          {rec.relevance_reason}
                        </p>
                        <p style={{ fontSize: "0.9rem", color: "var(--clr-text-2)", margin: 0 }}>
                          {rec.abstract}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Stage 2: External Semantic Scholar Discovery */}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <h3 style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--clr-text-1)", margin: 0 }}>
                    Stage 2: External Academic Discovery (Semantic Scholar)
                  </h3>
                  <span className={`status-badge ${recommendations.external_status === "success" ? "status-completed" : "status-pending"}`}>
                    Status: {recommendations.external_status}
                  </span>
                </div>

                {recommendations.external_status === "unavailable" && (
                  <div className="brief-caveat-box">
                    <span>Notice:</span> External Semantic Scholar API is currently offline or unreachable without internet connectivity. Internal collection discovery remains fully functional.
                  </div>
                )}

                {recommendations.external_status === "rate_limited" && (
                  <div className="brief-caveat-box">
                    <span>Notice:</span> Semantic Scholar free public rate limit reached. External search will resume shortly.
                  </div>
                )}

                {recommendations.external_results && recommendations.external_results.length > 0 ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    {recommendations.external_results.map((ext, ei) => (
                      <div key={ei} className="rec-card">
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                          <h4 style={{ margin: 0, fontSize: "1rem", color: "var(--clr-text-1)" }}>
                            {ext.url ? (
                              <a href={ext.url} target="_blank" rel="noopener noreferrer" style={{ color: "var(--clr-primary)", textDecoration: "none" }}>
                                🌐 {ext.title} ↗
                              </a>
                            ) : (
                              ext.title
                            )}
                          </h4>
                          {ext.year && <span className="citation-pill">{ext.year}</span>}
                        </div>
                        <p style={{ fontSize: "0.85rem", color: "var(--clr-text-3)", margin: "4px 0" }}>
                          {ext.authors} {ext.citation_count ? `· ${ext.citation_count} citations` : ""}
                        </p>
                        <p style={{ fontSize: "0.9rem", color: "var(--clr-text-2)", margin: "6px 0" }}>
                          {ext.abstract}
                        </p>
                        {ext.open_access_pdf && (
                          <div style={{ marginTop: 8 }}>
                            <a
                              href={ext.open_access_pdf}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="secondary-button sm"
                              style={{ display: "inline-flex", textDecoration: "none" }}
                            >
                              📥 Open Access PDF ↗
                            </a>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  recommendations.external_status === "success" && (
                    <p style={{ color: "var(--clr-text-3)", fontSize: "0.9rem" }}>No external publications found matching this inquiry.</p>
                  )
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
