"""
routes/screening.py — Research-question-based paper ranking.
"""
import logging
from fastapi import APIRouter, HTTPException

from database import get_db, rows_to_list
from schemas import ScreeningRequest, ScreeningResultOut, ScreeningDecisionUpdate
from modules.relevance import rank_papers_by_relevance

router = APIRouter(tags=["screening"])
logger = logging.getLogger(__name__)


@router.post("/api/workspaces/{workspace_id}/screen")
def screen_papers(workspace_id: str, body: ScreeningRequest):
    """
    Rank papers in the workspace by TF-IDF cosine similarity to the query.
    Returns papers sorted by relevance score with a supporting snippet.
    """
    with get_db() as conn:
        ws = conn.execute(
            "SELECT id FROM workspaces WHERE id=?", (workspace_id,)
        ).fetchone()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found.")

        papers = conn.execute(
            "SELECT id, filename FROM papers WHERE workspace_id=?", (workspace_id,)
        ).fetchall()

    if not papers:
        return {"query": body.query, "results": [],
                "note": "No papers in this workspace yet."}

    # Build per-paper page-text lists from DB
    paper_docs = []
    with get_db() as conn:
        for paper in papers:
            pages = conn.execute(
                "SELECT page_number, text FROM paper_pages WHERE paper_id=? ORDER BY page_number",
                (paper["id"],),
            ).fetchall()
            paper_docs.append({
                "paper_id":   paper["id"],
                "filename":   paper["filename"],
                "page_texts": [(r["page_number"], r["text"]) for r in pages],
            })

    # Papers whose text hasn't been extracted yet can still be returned
    # (they'll get a score of 0 and an empty snippet)
    ranked = rank_papers_by_relevance(body.query, paper_docs)

    # Persist/update screening results
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        for r in ranked:
            conn.execute(
                """INSERT INTO screening_results(paper_id, query, score, snippet, page_number, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (r["paper_id"], body.query, r["score"],
                 r["snippet"], r["page_number"], now),
            )

    return {"query": body.query, "results": ranked}


@router.patch("/api/papers/{paper_id}/screening-decision")
def update_screening_decision(paper_id: str, body: ScreeningDecisionUpdate):
    """Update a paper's manual relevance decision."""
    allowed = {"relevant", "irrelevant", "needs_review", "unreviewed"}
    if body.decision not in allowed:
        raise HTTPException(
            status_code=400, detail=f"Decision must be one of {allowed}"
        )
    with get_db() as conn:
        row = conn.execute("SELECT id FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found.")
        conn.execute(
            "UPDATE papers SET screening_decision=? WHERE id=?",
            (body.decision, paper_id),
        )
    return {"paper_id": paper_id, "decision": body.decision}
