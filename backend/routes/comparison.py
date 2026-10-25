"""
routes/comparison.py — Cross-paper comparison table and CSV export.
Synthesizes scientific facets across papers with multi-entity formatting.
"""

import csv
import io
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from database import get_db, rows_to_list
from schemas import ComparisonRowOut

router = APIRouter(tags=["comparison"])
logger = logging.getLogger(__name__)


def _best_task(fields: list[dict]) -> tuple[str, str]:
    """Return a clean concise task statement and evidence."""
    tasks = [f for f in fields if f["field_type"] == "TASK"]
    if not tasks:
        return "Not found", ""
    # Prefer a task that isn't excessively long
    tasks.sort(key=lambda t: len(t.get("value", "")))
    best = tasks[0]
    return best.get("value", "Not found"), best.get("evidence_text", "")


def _best_methods(fields: list[dict]) -> tuple[str, str]:
    """Return primary model architecture(s)."""
    methods = [f for f in fields if f["field_type"] == "METHOD"]
    if not methods:
        return "Not found", ""
    # Filter out purely generic words if specific models exist
    specific = [m for m in methods if m.get("value") not in ["Fine-tuning", "Pre-training", "Transfer Learning", "Zero-shot"]]
    top_candidates = specific if specific else methods
    
    # Pick top 1 or top 2
    vals = [m.get("value") for m in top_candidates[:2]]
    combined = " / ".join(vals)
    ev = top_candidates[0].get("evidence_text", "")
    return combined, ev


def _best_datasets(fields: list[dict]) -> tuple[str, str]:
    """Return top benchmark datasets."""
    datasets = [f for f in fields if f["field_type"] == "DATASET"]
    if not datasets:
        return "Not found", ""
    # Take up to 2 key datasets
    vals = []
    seen = set()
    for d in datasets:
        v = d.get("value")
        if v and v.lower() not in seen and v.lower() != "arxiv":
            seen.add(v.lower())
            vals.append(v)
        if len(vals) >= 2:
            break
    
    combined = ", ".join(vals) if vals else "Not found"
    ev = datasets[0].get("evidence_text", "")
    return combined, ev


def _best_metrics(fields: list[dict]) -> tuple[str, str]:
    """Return key metrics."""
    metrics = [f for f in fields if f["field_type"] == "METRIC"]
    if not metrics:
        return "Not found", ""
    vals = []
    seen = set()
    for m in metrics:
        v = m.get("value")
        if v and v.lower() not in seen:
            seen.add(v.lower())
            vals.append(v)
        if len(vals) >= 2:
            break
    combined = ", ".join(vals) if vals else "Not found"
    ev = metrics[0].get("evidence_text", "")
    return combined, ev


def _best_result(fields: list[dict]) -> tuple[str, str]:
    """Return key quantitative result."""
    results = [f for f in fields if f["field_type"] == "RESULT"]
    if not results:
        return "Not found", ""
    # Pick the first concise result
    res = results[0]
    val = res.get("value", "")
    ev = res.get("evidence_text", "")
    return val if val else "Not found", ev


def _best_limitation(fields: list[dict]) -> tuple[str, str]:
    """Return key limitation or constraint."""
    limits = [f for f in fields if f["field_type"] == "LIMITATION"]
    if not limits:
        return "Not found", ""
    lim = limits[0]
    val = lim.get("value", "")
    ev = lim.get("evidence_text", "")
    return val if val else "Not found", ev


@router.get("/api/workspaces/{workspace_id}/comparison")
def get_comparison(workspace_id: str):
    """
    Build a cross-paper comparison table from stored extracted fields.
    Each row = one paper; columns = TASK, METHOD, DATASET, METRIC, RESULT, LIMITATION.
    """
    with get_db() as conn:
        ws = conn.execute(
            "SELECT id FROM workspaces WHERE id=?", (workspace_id,)
        ).fetchone()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found.")

        papers = conn.execute(
            """SELECT id, filename, analysis_status, screening_decision
               FROM papers WHERE workspace_id=? ORDER BY created_at""",
            (workspace_id,),
        ).fetchall()

    rows = []
    with get_db() as conn:
        for paper in papers:
            fields = rows_to_list(conn.execute(
                "SELECT field_type, value, evidence_text FROM extracted_fields WHERE paper_id=?",
                (paper["id"],),
            ).fetchall())

            evidence = {}
            task,    ev_task   = _best_task(fields)
            method,  ev_method = _best_methods(fields)
            dataset, ev_ds     = _best_datasets(fields)
            metric,  ev_metric = _best_metrics(fields)
            result,  ev_result = _best_result(fields)
            limit,   ev_limit  = _best_limitation(fields)

            if ev_task:   evidence["task"]       = ev_task
            if ev_method: evidence["method"]     = ev_method
            if ev_ds:     evidence["dataset"]    = ev_ds
            if ev_metric: evidence["metric"]     = ev_metric
            if ev_result: evidence["result"]     = ev_result
            if ev_limit:  evidence["limitation"] = ev_limit

            rows.append({
                "paper_id":           paper["id"],
                "filename":           paper["filename"],
                "analysis_status":    paper["analysis_status"],
                "screening_decision": paper["screening_decision"],
                "task":               task,
                "method":             method,
                "dataset":            dataset,
                "metric":             metric,
                "result":             result,
                "limitation":         limit,
                "evidence":           evidence,
            })

    return {"workspace_id": workspace_id, "rows": rows}


@router.get("/api/workspaces/{workspace_id}/export.csv")
def export_csv(workspace_id: str):
    """Export comparison table as a downloadable CSV file."""
    comparison = get_comparison(workspace_id)
    rows = comparison["rows"]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Filename", "Task", "Method / Model", "Benchmark Dataset",
        "Metric", "Result", "Limitation", "Analysis Status"
    ])
    for row in rows:
        writer.writerow([
            row["filename"], row["task"], row["method"],
            row["dataset"], row["metric"], row["result"],
            row["limitation"], row["analysis_status"],
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="comparison_{workspace_id[:8]}.csv"'},
    )
