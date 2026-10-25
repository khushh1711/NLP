"""
main.py — FastAPI application entry point for the Scientific Literature Mining System.

Start with:
    .\\venv\\Scripts\\uvicorn.exe main:app --reload --port 8000
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS
from database import init_db

# Route modules
from routes.workspaces          import router as ws_router
from routes.papers              import router as papers_router
from routes.screening           import router as screening_router
from routes.comparison          import router as comparison_router
from routes.insights            import router as insights_router
from routes.collection_analysis import router as collection_router
from routes.qa                  import router as qa_router
from routes.recommendations     import router as recommendations_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Scientific Literature Mining System API",
    description=(
        "Backend for the M.Sc. IT NLP project: workspace management, "
        "PDF processing, TF-IDF relevance screening, scientific information "
        "extraction, cross-paper comparison, collection research brief synthesis, "
        "evidence-grounded Q&A, and T5 summarisation."
    ),
    version="2.0.0",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    """Initialise database tables on first run."""
    init_db()
    logger.info("Scientific Literature Mining System backend ready.")

# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(ws_router)
app.include_router(papers_router)
app.include_router(screening_router)
app.include_router(comparison_router)
app.include_router(insights_router)
app.include_router(collection_router)
app.include_router(qa_router)
app.include_router(recommendations_router)

# ─── Core routes ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    from database import get_connection
    db_ok = False
    try:
        conn = get_connection()
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception:
        pass

    # Check if summariser model is loaded
    model_ok = False
    try:
        from summarizer import model
        model_ok = model is not None
    except Exception:
        pass

    return {"status": "ok", "db_connected": db_ok, "model_loaded": model_ok}


@app.get("/")
def root():
    return {"message": "Scientific Literature Mining System API v2.0 is running."}


# ─── Legacy test route (backward compat) ─────────────────────────────────────

@app.get("/api/test")
def test_connection():
    return {"message": "React and FastAPI are connected successfully!"}