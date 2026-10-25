import { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { getInsights } from "../services/api";

function Bar({ value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="bar-wrapper">
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="bar-count">{value}</span>
    </div>
  );
}

function FreqList({ items, color, emptyMsg }) {
  if (!items || items.length === 0) return <p className="insight-empty">{emptyMsg}</p>;
  const max = items[0]?.count || 1;
  return (
    <ul className="freq-list">
      {items.map((item, i) => (
        <li key={i} className="freq-item">
          <span className="freq-label" title={item.value}>
            {item.value.length > 40 ? item.value.slice(0, 40) + "…" : item.value}
          </span>
          <Bar value={item.count} max={max} color={color} />
        </li>
      ))}
    </ul>
  );
}

export default function InsightsView() {
  const { workspaceId } = useParams();
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState("");

  useEffect(() => { load(); }, [workspaceId]);

  async function load() {
    try { const d = await getInsights(workspaceId); setData(d); }
    catch { setError("Failed to load insights."); }
    finally { setLoading(false); }
  }

  if (loading) return (
    <div className="page-container">
      <div className="loading-container">
        <div className="loading-spinner-wrapper"><div className="loading-spinner" /></div>
        <p className="loading-subtitle">Computing insights…</p>
      </div>
    </div>
  );

  return (
    <div className="page-container">
      <Link to={`/workspace/${workspaceId}`} className="back-link">← Workspace</Link>

      <h1 className="section-title" style={{marginBottom:8}}>💡 Collection Insights</h1>
      <p className="page-subtitle">
        Statistics derived from extracted fields in the current workspace.
        All counts reflect stored records — not hardcoded values.
      </p>

      {error && <div className="error-message"><span>⚠️</span><span>{error}</span></div>}

      {data && (
        <>
          {/* Summary stats */}
          <div className="document-grid" style={{marginBottom:28}}>
            <div className="info-card">
              <span className="info-label">Total Papers</span>
              <span className="info-value accent">{data.paper_count}</span>
            </div>
            <div className="info-card">
              <span className="info-label">Analyzed</span>
              <span className="info-value accent">{data.analyzed_count}</span>
            </div>
            <div className="info-card">
              <span className="info-label">Relevant</span>
              <span className="info-value accent">{data.screening_summary?.relevant || 0}</span>
            </div>
            <div className="info-card">
              <span className="info-label">Needs Review</span>
              <span className="info-value accent">{data.screening_summary?.needs_review || 0}</span>
            </div>
          </div>

          {/* Frequency charts */}
          <div className="insights-grid">
            <div className="insight-card">
              <h3 className="insight-title">⚙️ Top Methods / Models</h3>
              <FreqList items={data.top_methods} color="var(--clr-accent)"
                emptyMsg="No methods extracted yet." />
            </div>
            <div className="insight-card">
              <h3 className="insight-title">📦 Top Datasets</h3>
              <FreqList items={data.top_datasets} color="var(--clr-cyan)"
                emptyMsg="No datasets extracted yet." />
            </div>
            <div className="insight-card">
              <h3 className="insight-title">📏 Top Metrics</h3>
              <FreqList items={data.top_metrics} color="var(--clr-success)"
                emptyMsg="No metrics extracted yet." />
            </div>
            <div className="insight-card">
              <h3 className="insight-title">🎯 Research Tasks</h3>
              <FreqList items={data.top_tasks} color="#fb923c"
                emptyMsg="No task descriptions extracted yet." />
            </div>
          </div>

          {/* Missing fields */}
          {data.missing_fields && Object.keys(data.missing_fields).length > 0 && (
            <div className="insight-card" style={{marginTop:16}}>
              <h3 className="insight-title">📋 Missing Field Coverage</h3>
              <p className="insight-note">
                Papers where each field type was not extracted.
                A missing field may reflect extraction limitations, reporting variation,
                or the paper's scope — not necessarily a research gap.
              </p>
              <div className="missing-fields-grid">
                {Object.entries(data.missing_fields).map(([ft, count]) => (
                  <div key={ft} className="missing-field-item">
                    <span className="missing-field-type">{ft}</span>
                    <span className="missing-field-count">
                      {count} / {data.paper_count} papers missing
                    </span>
                    <div className="bar-track" style={{width:"100%",marginTop:4}}>
                      <div className="bar-fill" style={{
                        width: `${data.paper_count > 0 ? (count/data.paper_count)*100 : 0}%`,
                        background: count > data.paper_count * 0.7 ? "#f43f5e" : "#f59e0b",
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {data.note && (
            <div className="disclaimer-banner" style={{marginTop:20}}>
              <span className="disclaimer-icon">ℹ️</span>
              <p className="disclaimer-text">{data.note}</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
