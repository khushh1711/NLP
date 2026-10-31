"""
routes/collection_analysis.py — Collection-wide research synthesis endpoints.

Supports:
- POST /api/workspaces/{workspace_id}/analyze-collection
- GET  /api/workspaces/{workspace_id}/research-brief
"""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from database import get_db, rows_to_list, row_to_dict
from schemas import CollectionBriefOut
from modules.collection_synthesis import generate_collection_brief

router = APIRouter(prefix="/api/workspaces", tags=["collection-analysis"])
logger = logging.getLogger(__name__)


def _fetch_workspace_data(workspace_id: str):
    """Retrieve all data needed for collection synthesis from the DB."""
    with get_db() as conn:
        ws = conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found.")
        
        papers = conn.execute(
            "SELECT * FROM papers WHERE workspace_id = ? ORDER BY created_at ASC",
            (workspace_id,)
        ).fetchall()
        
        paper_ids = [p["id"] for p in papers]
        if not paper_ids:
            return ws, [], [], [], []
            
        placeholders = ",".join("?" for _ in paper_ids)
        
        fields = conn.execute(
            f"SELECT * FROM extracted_fields WHERE paper_id IN ({placeholders})",
            paper_ids
        ).fetchall()
        
        summaries = conn.execute(
            f"SELECT * FROM summaries WHERE paper_id IN ({placeholders})",
            paper_ids
        ).fetchall()

        pages = conn.execute(
            f"SELECT paper_id, page_number, text FROM paper_pages WHERE paper_id IN ({placeholders}) ORDER BY paper_id, page_number ASC",
            paper_ids
        ).fetchall()
        
    return ws, rows_to_list(papers), rows_to_list(fields), rows_to_list(summaries), rows_to_list(pages)


@router.post("/{workspace_id}/analyze-collection", response_model=CollectionBriefOut)
def analyze_collection(workspace_id: str):
    """
    Synthesize research brief across all analyzed papers in the workspace.
    Caches result in collection_briefs table.
    """
    ws, papers, fields, summaries, pages = _fetch_workspace_data(workspace_id)
    rq = ws["research_question"] if ws else ""
    
    brief = generate_collection_brief(
        workspace_id=workspace_id,
        research_question=rq,
        papers=papers,
        extracted_fields=fields,
        summaries=summaries,
        pages=pages,
    )
    
    now = datetime.utcnow().isoformat()
    brief["created_at"] = now
    brief["updated_at"] = now
    
    brief_json = json.dumps(brief)
    paper_count = brief["paper_count"]
    
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO collection_briefs
               (workspace_id, research_question, brief_json, paper_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (workspace_id, rq, brief_json, paper_count, now, now)
        )
        
    return brief


@router.get("/{workspace_id}/research-brief", response_model=CollectionBriefOut)
def get_research_brief(workspace_id: str):
    """
    Retrieve cached research brief. If no cached brief exists, generates one on-demand.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM collection_briefs WHERE workspace_id = ?",
            (workspace_id,)
        ).fetchone()
        
    if row and row["brief_json"]:
        try:
            brief_data = json.loads(row["brief_json"])
            brief_data["created_at"] = row["created_at"]
            brief_data["updated_at"] = row["updated_at"]
            return brief_data
        except Exception as exc:
            logger.warning("Corrupt cached brief JSON: %s. Regenerating.", exc)
            
    # Generate on-demand if not cached
    return analyze_collection(workspace_id)
