"""
config.py — Central configuration for the Scientific Literature Mining System backend.
All paths, limits, and settings are defined here so nothing is scattered across modules.
"""

import os
from pathlib import Path


# ─── Directory layout ─────────────────────────────────────────────────────────

# Root of the backend directory (the folder that contains this file)
BACKEND_DIR = Path(__file__).parent.resolve()

# Where uploaded PDFs are stored persistently
STORAGE_DIR = BACKEND_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database file
DATABASE_PATH = BACKEND_DIR / "literature_mining.db"

# ─── Upload limits ────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB   = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_CONTENT_TYPES = {"application/pdf"}
ALLOWED_EXTENSIONS    = {".pdf"}

# ─── Summarisation model ──────────────────────────────────────────────────────

SUMMARIZER_MODEL_NAME = "Bashaarat1/t5-small-arxiv-summarizer"

# ─── Extraction / relevance ───────────────────────────────────────────────────

# Maximum characters passed per chunk to the summarizer
SUMMARIZER_MAX_CHARS = 3000

# TF-IDF: number of top matching snippets returned per screening query
TFIDF_TOP_K_SNIPPETS = 5

# ─── API ──────────────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]
