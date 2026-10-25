"""
routes/insights.py — Pattern analysis and collection-level statistics.
All counts are derived from real database records.
"""
import logging
from collections import Counter
from fastapi import APIRouter, HTTPException
from database import get_db, rows_to_list

router = APIRouter(tags=["insights"])
logger = logging.getLogger(__name__)


@router.get("/api/workspaces/{workspace_id}/insights")
def get_insights(workspace_id: str):
    """
    Return collection-level statistics derived from extracted fields.
    Clearly communicates the size and composition of the collection.
    """
    with get_db() as conn:
        ws = conn.execute(
            "SELECT id FROM workspaces WHERE id=?", (workspace_id,)
        ).fetchone()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found.")

        papers = rows_to_list(conn.execute(
            "SELECT id, analysis_status, screening_decision FROM papers WHERE workspace_id=?",
            (workspace_id,),
        ).fetchall())

        paper_ids = [p["id"] for p in papers]

        if not paper_ids:
            return {
                "paper_count": 0, "analyzed_count": 0,
                "top_methods": [], "top_datasets": [],
                "top_metrics": [], "top_tasks": [],
                "missing_fields": {}, "screening_summary": {},
                "note": "No papers in this workspace.",
            }

        placeholders = ",".join("?" * len(paper_ids))
        fields = rows_to_list(conn.execute(
            f"SELECT paper_id, field_type, value FROM extracted_fields WHERE paper_id IN ({placeholders})",
            paper_ids,
        ).fetchall())

    # Count frequencies per field type
    def top_n(field_type: str, n: int = 8):
        vals = [f["value"] for f in fields if f["field_type"] == field_type]
        return [{"value": v, "count": c} for v, c in Counter(vals).most_common(n)]

    # Which papers are missing each field type?
    field_types = ["TASK", "METHOD", "DATASET", "METRIC", "RESULT", "LIMITATION"]
    papers_with_field: dict[str, set] = {ft: set() for ft in field_types}
    for f in fields:
        ft = f["field_type"]
        if ft in papers_with_field:
            papers_with_field[ft].add(f["paper_id"])

    total = len(paper_ids)
    missing_fields = {
        ft: total - len(papers_with_field[ft]) for ft in field_types
    }

    # Screening decision summary
    screening_summary: dict[str, int] = Counter(
        p["screening_decision"] for p in papers
    )

    analyzed_count = sum(1 for p in papers if p["analysis_status"] == "completed")

    return {
        "paper_count":       total,
        "analyzed_count":    analyzed_count,
        "top_methods":       top_n("METHOD"),
        "top_datasets":      top_n("DATASET"),
        "top_metrics":       top_n("METRIC"),
        "top_tasks":         top_n("TASK", 4),
        "missing_fields":    missing_fields,
        "screening_summary": dict(screening_summary),
        "note": (
            "Counts reflect extracted records in the current collection. "
            "A missing field may indicate extraction limitations or reporting differences, "
            "not necessarily a research gap."
        ),
    }
