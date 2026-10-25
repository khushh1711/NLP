"""
routes/workspaces.py — Workspace CRUD endpoints.
"""
import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from database import get_db, rows_to_list, row_to_dict
from schemas import WorkspaceCreate, WorkspaceUpdate, WorkspaceOut

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])
logger = logging.getLogger(__name__)


@router.post("", response_model=WorkspaceOut, status_code=201)
def create_workspace(body: WorkspaceCreate):
    ws_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO workspaces(id, title, description, research_question, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ws_id, body.title.strip(), body.description.strip(),
             body.research_question.strip(), now, now),
        )
    return {**body.model_dump(), "id": ws_id, "created_at": now,
            "updated_at": now, "paper_count": 0}


@router.get("", response_model=list[WorkspaceOut])
def list_workspaces():
    with get_db() as conn:
        rows = conn.execute(
            """SELECT w.*,
                      (SELECT COUNT(*) FROM papers p WHERE p.workspace_id = w.id) AS paper_count
               FROM workspaces w ORDER BY w.created_at DESC"""
        ).fetchall()
    return rows_to_list(rows)


@router.get("/{workspace_id}", response_model=WorkspaceOut)
def get_workspace(workspace_id: str):
    with get_db() as conn:
        row = conn.execute(
            """SELECT w.*,
                      (SELECT COUNT(*) FROM papers p WHERE p.workspace_id = w.id) AS paper_count
               FROM workspaces w WHERE w.id = ?""",
            (workspace_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return row_to_dict(row)


@router.patch("/{workspace_id}", response_model=WorkspaceOut)
def update_workspace(workspace_id: str, body: WorkspaceUpdate):
    with get_db() as conn:
        ws = conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found.")
        
        updates = []
        params = []
        if body.title is not None:
            updates.append("title = ?")
            params.append(body.title.strip())
        if body.description is not None:
            updates.append("description = ?")
            params.append(body.description.strip())
        if body.research_question is not None:
            updates.append("research_question = ?")
            params.append(body.research_question.strip())
            # Invalidate cached collection brief when research question changes
            conn.execute("DELETE FROM collection_briefs WHERE workspace_id = ?", (workspace_id,))
            
        if updates:
            now = datetime.utcnow().isoformat()
            updates.append("updated_at = ?")
            params.append(now)
            params.append(workspace_id)
            conn.execute(
                f"UPDATE workspaces SET {', '.join(updates)} WHERE id = ?",
                params,
            )
        
        row = conn.execute(
            """SELECT w.*,
                      (SELECT COUNT(*) FROM papers p WHERE p.workspace_id = w.id) AS paper_count
               FROM workspaces w WHERE w.id = ?""",
            (workspace_id,),
        ).fetchone()
        return row_to_dict(row)


@router.delete("/{workspace_id}", status_code=204)
def delete_workspace(workspace_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))

