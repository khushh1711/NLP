"""
routes/papers.py — Paper upload, listing, analysis, and management endpoints.
"""
import uuid
import logging
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form

from config import STORAGE_DIR, MAX_FILE_SIZE_BYTES, ALLOWED_EXTENSIONS
from database import get_db, rows_to_list, row_to_dict, compute_file_hash
from schemas import PaperOut, AnalysisOut, SummaryOut, ExtractedFieldOut, FieldVerifyUpdate
from pdf_processor import extract_text_from_pdf, extract_pages_from_pdf
from summarizer import summarize_text
from modules.information_extraction import extract_all_fields

router = APIRouter(tags=["papers"])
logger = logging.getLogger(__name__)


# ─── Helper ──────────────────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """Keep only safe characters in a filename."""
    import re
    safe = re.sub(r'[^\w\-_. ]', '_', name)
    return safe[:200]


def _run_full_analysis(paper_id: str, pdf_path: Path) -> None:
    """
    Background task: extract page texts, run summariser, run extraction.
    Writes results to DB; errors are logged but do not crash the server.
    """
    try:
        with get_db() as conn:
            conn.execute(
                "UPDATE papers SET analysis_status='processing', updated_at=? WHERE id=?",
                (datetime.utcnow().isoformat(), paper_id),
            )

        pdf_bytes = pdf_path.read_bytes()
        page_texts = extract_pages_from_pdf(pdf_bytes)          # [(page_num, text), ...]
        full_text  = "\n".join(t for _, t in page_texts)

        # ── Store per-page text ──────────────────────────────────────────────
        with get_db() as conn:
            conn.execute("DELETE FROM paper_pages WHERE paper_id = ?", (paper_id,))
            conn.executemany(
                "INSERT INTO paper_pages(paper_id, page_number, text) VALUES (?,?,?)",
                [(paper_id, pn, pt) for pn, pt in page_texts],
            )

        # ── Summarisation ───────────────────────────────────────────────────
        summary = summarize_text(full_text)
        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO summaries
                   (paper_id, overview, method, results, conclusion, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (paper_id, summary.get("overview",""), summary.get("method",""),
                 summary.get("results",""), summary.get("conclusion",""), now),
            )

        # ── Information extraction ───────────────────────────────────────────
        fields = extract_all_fields(page_texts)
        with get_db() as conn:
            conn.execute("DELETE FROM extracted_fields WHERE paper_id = ?", (paper_id,))
            conn.executemany(
                """INSERT INTO extracted_fields
                   (paper_id, field_type, value, normalized_value, evidence_text,
                    page_number, extraction_method, verification_status)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [(paper_id, f["field_type"], f["value"], f.get("normalized_value",""),
                  f["evidence_text"], f["page_number"], f["extraction_method"],
                  f["verification_status"]) for f in fields],
            )

        # ── Update paper stats ───────────────────────────────────────────────
        word_count = len(full_text.split())
        char_count = len(full_text)
        page_count = len(page_texts)
        with get_db() as conn:
            conn.execute(
                """UPDATE papers SET
                   analysis_status='completed', word_count=?, char_count=?,
                   page_count=?, updated_at=?
                   WHERE id=?""",
                (word_count, char_count, page_count,
                 datetime.utcnow().isoformat(), paper_id),
            )
            # Invalidate cached collection brief so it refreshes with the new paper's data
            p_row = conn.execute("SELECT workspace_id FROM papers WHERE id=?", (paper_id,)).fetchone()
            if p_row:
                conn.execute("DELETE FROM collection_briefs WHERE workspace_id=?", (p_row["workspace_id"],))
        logger.info("Analysis complete for paper %s", paper_id)

    except Exception as exc:
        logger.error("Analysis failed for paper %s: %s", paper_id, exc)
        with get_db() as conn:
            conn.execute(
                "UPDATE papers SET analysis_status='failed', updated_at=? WHERE id=?",
                (datetime.utcnow().isoformat(), paper_id),
            )


# ─── Upload into workspace ───────────────────────────────────────────────────

@router.post("/api/workspaces/{workspace_id}/papers", status_code=201)
async def upload_papers(
    workspace_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    # Verify workspace exists
    with get_db() as conn:
        ws = conn.execute("SELECT id FROM workspaces WHERE id=?", (workspace_id,)).fetchone()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    results = []
    for file in files:
        result = await _process_single_upload(workspace_id, file, background_tasks)
        results.append(result)
    return {"uploaded": results}


async def _process_single_upload(workspace_id, file, background_tasks):
    """Validate, save, and queue analysis for one PDF."""
    if not file.filename:
        return {"filename": "", "status": "error", "detail": "Empty filename."}

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {"filename": file.filename, "status": "error",
                "detail": "Only PDF files are accepted."}

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        return {"filename": file.filename, "status": "error", "detail": "File is empty."}
    if len(pdf_bytes) > MAX_FILE_SIZE_BYTES:
        return {"filename": file.filename, "status": "error",
                "detail": "File exceeds the 50 MB size limit."}

    file_hash = compute_file_hash(pdf_bytes)

    # Duplicate check within this workspace
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id, filename FROM papers WHERE workspace_id=? AND file_hash=?",
            (workspace_id, file_hash),
        ).fetchone()
    if existing:
        return {"filename": file.filename, "status": "duplicate",
                "detail": f"Already uploaded as '{existing['filename']}'.",
                "paper_id": existing["id"]}

    # Save file
    paper_id = str(uuid.uuid4())
    safe_name = _sanitize_filename(file.filename)
    dest = STORAGE_DIR / f"{paper_id}_{safe_name}"
    dest.write_bytes(pdf_bytes)

    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO papers
               (id, workspace_id, filename, file_path, file_hash, file_size_bytes,
                upload_status, analysis_status, created_at, updated_at)
               VALUES (?,?,?,?,?,?,'uploaded','pending',?,?)""",
            (paper_id, workspace_id, file.filename, str(dest),
             file_hash, len(pdf_bytes), now, now),
        )

    background_tasks.add_task(_run_full_analysis, paper_id, dest)
    return {"filename": file.filename, "status": "uploaded", "paper_id": paper_id}


# ─── Legacy single-file upload (preserves existing endpoint) ─────────────────

@router.post("/api/papers/upload")
async def upload_paper_legacy(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Original single-file upload endpoint — preserved for backward compatibility.
    Creates a default workspace named 'Default' if none exists, then delegates.
    """
    with get_db() as conn:
        default_ws = conn.execute(
            "SELECT id FROM workspaces WHERE title='Default Workspace' LIMIT 1"
        ).fetchone()

    if not default_ws:
        ws_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            conn.execute(
                """INSERT INTO workspaces(id,title,description,research_question,created_at,updated_at)
                   VALUES (?,'Default Workspace','Auto-created workspace','',?,?)""",
                (ws_id, now, now),
            )
    else:
        ws_id = default_ws["id"]

    result = await _process_single_upload(ws_id, file, background_tasks)

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["detail"])

    paper_id = result.get("paper_id")

    # Return immediately with partial data (analysis runs in background)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "PDF uploaded. Analysis running in background.",
        "paper_id": paper_id,
        "workspace_id": ws_id,
        "status": result["status"],
    }


# ─── List papers in workspace ────────────────────────────────────────────────

@router.get("/api/workspaces/{workspace_id}/papers", response_model=list[PaperOut])
def list_papers(workspace_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM papers WHERE workspace_id=? ORDER BY created_at DESC",
            (workspace_id,),
        ).fetchall()
    return rows_to_list(rows)


# ─── Get single paper ────────────────────────────────────────────────────────

@router.get("/api/papers/{paper_id}", response_model=PaperOut)
def get_paper(paper_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return row_to_dict(row)


# ─── Get paper analysis ──────────────────────────────────────────────────────

@router.get("/api/papers/{paper_id}/analysis", response_model=AnalysisOut)
def get_analysis(paper_id: str):
    with get_db() as conn:
        paper = conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found.")
        summary = conn.execute(
            "SELECT * FROM summaries WHERE paper_id=?", (paper_id,)
        ).fetchone()
        fields = conn.execute(
            "SELECT * FROM extracted_fields WHERE paper_id=? ORDER BY field_type, page_number",
            (paper_id,),
        ).fetchall()

    return {
        "paper": row_to_dict(paper),
        "summary": row_to_dict(summary) if summary else None,
        "extracted_fields": rows_to_list(fields),
    }


# ─── Re-trigger analysis ─────────────────────────────────────────────────────

@router.post("/api/papers/{paper_id}/analyze", status_code=202)
def reanalyze_paper(paper_id: str, background_tasks: BackgroundTasks):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, file_path FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Paper not found.")
    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")
    background_tasks.add_task(_run_full_analysis, paper_id, path)
    return {"message": "Analysis queued.", "paper_id": paper_id}


# ─── Evidence endpoint ───────────────────────────────────────────────────────

@router.get("/api/papers/{paper_id}/evidence")
def get_evidence(paper_id: str):
    with get_db() as conn:
        fields = conn.execute(
            """SELECT field_type, value, evidence_text, page_number, verification_status
               FROM extracted_fields WHERE paper_id=? ORDER BY field_type""",
            (paper_id,),
        ).fetchall()
    return rows_to_list(fields)


# ─── Verify a field ──────────────────────────────────────────────────────────

@router.patch("/api/fields/{field_id}/verify")
def verify_field(field_id: int, body: FieldVerifyUpdate):
    allowed = {"verified", "incorrect", "needs_review"}
    if body.verification_status not in allowed:
        raise HTTPException(status_code=400, detail=f"Status must be one of {allowed}")
    with get_db() as conn:
        conn.execute(
            "UPDATE extracted_fields SET verification_status=? WHERE id=?",
            (body.verification_status, field_id),
        )
    return {"field_id": field_id, "verification_status": body.verification_status}


# ─── Delete paper ────────────────────────────────────────────────────────────

@router.delete("/api/papers/{paper_id}", status_code=204)
def delete_paper(paper_id: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT workspace_id, file_path FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found.")
        workspace_id = row["workspace_id"]
        # Delete DB record (cascades to pages, summary, fields, screening)
        conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
        # Invalidate cached collection brief for this workspace
        conn.execute("DELETE FROM collection_briefs WHERE workspace_id=?", (workspace_id,))

    # Remove physical file
    try:
        fp = Path(row["file_path"])
        if fp.exists():
            fp.unlink()
    except Exception as exc:
        logger.warning("Could not delete file %s: %s", row["file_path"], exc)
