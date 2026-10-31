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

        papers_rows = conn.execute(
            """SELECT id, filename, page_count, word_count
               FROM papers
               WHERE workspace_id = ? AND analysis_status = 'completed'
               ORDER BY created_at ASC""",
            (workspace_id,)
        ).fetchall()

        fields_rows = conn.execute(
            """SELECT ef.paper_id, p.filename, ef.field_type, ef.value, ef.normalized_value,
                      ef.evidence_text, ef.page_number, ef.verification_status
               FROM extracted_fields ef
               JOIN papers p ON p.id = ef.paper_id
               WHERE p.workspace_id = ? AND p.analysis_status = 'completed'
               ORDER BY ef.paper_id, ef.field_type""",
            (workspace_id,)
        ).fetchall()

        summaries_rows = conn.execute(
            """SELECT s.paper_id, p.filename, s.overview, s.method, s.results, s.conclusion
               FROM summaries s
               JOIN papers p ON p.id = s.paper_id
               WHERE p.workspace_id = ? AND p.analysis_status = 'completed'""",
            (workspace_id,)
        ).fetchall()
        
    pages_data = rows_to_list(rows)
    papers_data = rows_to_list(papers_rows)
    fields_data = rows_to_list(fields_rows)
    summaries_data = rows_to_list(summaries_rows)
    
    result = answer_collection_question(
        question=body.question,
        pages_data=pages_data,
        top_k=body.top_k,
        extracted_fields=fields_data,
        summaries=summaries_data,
        papers=papers_data,
    )
    return result
