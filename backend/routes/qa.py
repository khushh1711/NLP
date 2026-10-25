"""
routes/qa.py — Cross-paper question-answering endpoints.

Endpoint:
- POST /api/workspaces/{workspace_id}/ask
"""

import logging
from fastapi import APIRouter, HTTPException

from database import get_db, rows_to_list
from schemas import QARequest, QAAnswerOut
from modules.qa_engine import answer_collection_question

router = APIRouter(prefix="/api/workspaces", tags=["qa"])
logger = logging.getLogger(__name__)


@router.post("/{workspace_id}/ask", response_model=QAAnswerOut)
def ask_collection(workspace_id: str, body: QARequest):
    """
    Perform evidence-based passage retrieval and answer synthesis
    across all papers in the specified workspace.
    """
    with get_db() as conn:
        ws = conn.execute("SELECT id FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found.")
            
        rows = conn.execute(
            """SELECT pp.paper_id, p.filename, pp.page_number, pp.text
               FROM paper_pages pp
               JOIN papers p ON p.id = pp.paper_id
               WHERE p.workspace_id = ? AND p.analysis_status = 'completed'
               ORDER BY pp.paper_id, pp.page_number""",
            (workspace_id,)
        ).fetchall()
        
    pages_data = rows_to_list(rows)
    
    result = answer_collection_question(
        question=body.question,
        pages_data=pages_data,
        top_k=body.top_k,
    )
    return result
