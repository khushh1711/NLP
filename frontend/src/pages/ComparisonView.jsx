import { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { getComparison, exportCSV } from "../services/api";

export default function ComparisonView() {
  const { workspaceId } = useParams();
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState("");
  const [filters, setFilters] = useState({ method: "", dataset: "", metric: "" });
  const [evidence, setEvidence] = useState(null); // {label, text}

  useEffect(() => { load(); }, [workspaceId]);

  async function load() {
    try { const d = await getComparison(workspaceId); setData(d); }
    catch { setError("Failed to load comparison."); }
    finally { setLoading(false); }
  }

  if (loading) return (
    <div className="page-container">
      <div className="loading-container">
        <div className="loading-spinner-wrapper"><div className="loading-spinner" /></div>
        <p className="loading-subtitle">Building comparison table…</p>
      </div>
    </div>
  );

  const rows = (data?.rows || []).filter(row => {
    if (filters.method  && !row.method.toLowerCase().includes(filters.method.toLowerCase()))  return false;
    if (filters.dataset && !row.dataset.toLowerCase().includes(filters.dataset.toLowerCase())) return false;
    if (filters.metric  && !row.metric.toLowerCase().includes(filters.metric.toLowerCase()))   return false;
    return true;
  });

  return (
    <div className="page-container">
      <Link to={`/workspace/${workspaceId}`} className="back-link">← Workspace</Link>

      <div className="section-header-row">
        <h1 className="section-title">📊 Cross-Paper Comparison</h1>
        <button className="primary-button" onClick={() => exportCSV(workspaceId)}>
          ⬇ Export CSV
        </button>
      </div>

      <p className="page-subtitle">
        Each row represents one paper. Extracted fields are sourced from stored analysis records.
        Click 📎 to view the supporting evidence passage.
      </p>

      {error && <div className="error-message"><span>⚠️</span><span>{error}</span></div>}

      {/* Filters */}
      <div className="filter-bar comparison-filters">
        <input className="form-input filter-input" placeholder="Filter by method…"
          value={filters.method} onChange={e => setFilters(p => ({...p, method: e.target.value}))} />
        <input className="form-input filter-input" placeholder="Filter by dataset…"
          value={filters.dataset} onChange={e => setFilters(p => ({...p, dataset: e.target.value}))} />
        <input className="form-input filter-input" placeholder="Filter by metric…"
          value={filters.metric} onChange={e => setFilters(p => ({...p, metric: e.target.value}))} />
        {(filters.method || filters.dataset || filters.metric) && (
          <button className="secondary-button sm"
            onClick={() => setFilters({ method: "", dataset: "", metric: "" })}>Clear</button>
        )}
      </div>

      {/* Evidence modal */}
      {evidence && (
        <div className="evidence-modal-overlay" onClick={() => setEvidence(null)}>
          <div className="evidence-modal" onClick={e => e.stopPropagation()}>
            <div className="evidence-modal-header">
              <span className="evidence-modal-label">{evidence.label}</span>
              <button className="evidence-modal-close" onClick={() => setEvidence(null)}>✕</button>
            </div>
            <blockquote className="evidence-modal-text">"{evidence.text}"</blockquote>
          </div>
        </div>
      )}

      {rows.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>No data to compare yet</h3>
          <p>Upload and analyze papers in the workspace first.</p>
        </div>
      ) : (
        <div className="comparison-table-wrapper">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Paper</th>
                <th>🎯 Task</th>
                <th>⚙️ Method</th>
                <th>📦 Dataset</th>
                <th>📏 Metric</th>
                <th>📊 Result</th>
                <th>⚠️ Limitation</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(row => (
                <tr key={row.paper_id} id={`cmp-${row.paper_id}`}>
                  <td className="cmp-filename">
                    <Link to={`/paper/${row.paper_id}`} className="paper-link">
                      {row.filename}
                    </Link>
                    <span className={`status-badge status-sm ${row.analysis_status === "completed" ? "status-done" : "status-pending"}`}>
                      {row.analysis_status}
                    </span>
                  </td>
                  {["task","method","dataset","metric","result","limitation"].map(field => (
                    <td key={field} className={row[field] === "Not found" ? "cmp-not-found" : ""}>
                      <div className="cmp-cell">
                        <span>{row[field]?.slice(0, 80) || "Not found"}</span>
                        {row.evidence?.[field] && (
                          <button className="evidence-btn" title="View evidence"
                            onClick={() => setEvidence({
                              label: `${field.toUpperCase()} evidence — ${row.filename}`,
                              text: row.evidence[field]
                            })}>
                            📎
                          </button>
                        )}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="disclaimer-banner" style={{marginTop:24}}>
        <span className="disclaimer-icon">⚠️</span>
        <p className="disclaimer-text">
          <strong>Note:</strong> Extracted values are from pattern-based analysis and require human verification.
          Numeric results from different papers may not be directly comparable (different datasets,
          experimental conditions, or evaluation protocols).
        </p>
      </div>
    </div>
  );
}
