"""
database.py — SQLite setup and schema for the Scientific Literature Mining System.

Uses Python's built-in sqlite3 module only (no extra ORM dependency).
All tables are created on first run; existing tables are left untouched.
"""

import sqlite3
import hashlib
import logging
from pathlib import Path
from contextlib import contextmanager

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


# ─── Schema version ──────────────────────────────────────────────────────────
# Increment when the schema changes so we can detect mismatches.
SCHEMA_VERSION = 1


# ─── DDL statements ──────────────────────────────────────────────────────────

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version     INTEGER PRIMARY KEY,
        applied_at  TEXT    DEFAULT (datetime('now'))
    )
    """,

    # Research workspaces
    """
    CREATE TABLE IF NOT EXISTS workspaces (
        id              TEXT    PRIMARY KEY,
        title           TEXT    NOT NULL,
        description     TEXT    DEFAULT '',
        research_question TEXT  DEFAULT '',
        created_at      TEXT    DEFAULT (datetime('now')),
        updated_at      TEXT    DEFAULT (datetime('now'))
    )
    """,

    # Uploaded papers
    """
    CREATE TABLE IF NOT EXISTS papers (
        id              TEXT    PRIMARY KEY,
        workspace_id    TEXT    NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
        filename        TEXT    NOT NULL,
        file_path       TEXT    NOT NULL,
        file_hash       TEXT    NOT NULL,
        file_size_bytes INTEGER DEFAULT 0,
        page_count      INTEGER DEFAULT 0,
        word_count      INTEGER DEFAULT 0,
        char_count      INTEGER DEFAULT 0,
        upload_status   TEXT    DEFAULT 'uploaded',
        analysis_status TEXT    DEFAULT 'pending',
        screening_decision TEXT DEFAULT 'unreviewed',
        relevance_score REAL    DEFAULT 0.0,
        created_at      TEXT    DEFAULT (datetime('now')),
        updated_at      TEXT    DEFAULT (datetime('now'))
    )
    """,

    # Per-page text (preserved page numbers for evidence)
    """
    CREATE TABLE IF NOT EXISTS paper_pages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id    TEXT    NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        page_number INTEGER NOT NULL,
        text        TEXT    NOT NULL DEFAULT ''
    )
    """,

    # Generated summaries (four-field output from T5)
    """
    CREATE TABLE IF NOT EXISTS summaries (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id    TEXT    UNIQUE NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        overview    TEXT    DEFAULT '',
        method      TEXT    DEFAULT '',
        results     TEXT    DEFAULT '',
        conclusion  TEXT    DEFAULT '',
        created_at  TEXT    DEFAULT (datetime('now'))
    )
    """,

    # Extracted structured scientific fields
    """
    CREATE TABLE IF NOT EXISTS extracted_fields (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id            TEXT    NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        field_type          TEXT    NOT NULL,
        value               TEXT    NOT NULL DEFAULT '',
        normalized_value    TEXT    DEFAULT '',
        evidence_text       TEXT    DEFAULT '',
        page_number         INTEGER DEFAULT 0,
        extraction_method   TEXT    DEFAULT 'pattern',
        verification_status TEXT    DEFAULT 'needs_review',
        created_at          TEXT    DEFAULT (datetime('now'))
    )
    """,

    # Screening results: per-query relevance scores and matching snippets
    """
    CREATE TABLE IF NOT EXISTS screening_results (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id        TEXT    NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        query           TEXT    NOT NULL,
        score           REAL    DEFAULT 0.0,
        snippet         TEXT    DEFAULT '',
        page_number     INTEGER DEFAULT 0,
        created_at      TEXT    DEFAULT (datetime('now'))
    )
    """,

    # Collection research briefs (cross-paper synthesized analysis)
    """
    CREATE TABLE IF NOT EXISTS collection_briefs (
        workspace_id        TEXT PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
        research_question   TEXT DEFAULT '',
        brief_json          TEXT NOT NULL,
        paper_count         INTEGER DEFAULT 0,
        created_at          TEXT DEFAULT (datetime('now')),
        updated_at          TEXT DEFAULT (datetime('now'))
    )
    """,

    # Indexes for common access patterns
    "CREATE INDEX IF NOT EXISTS idx_papers_workspace ON papers(workspace_id)",
    "CREATE INDEX IF NOT EXISTS idx_pages_paper     ON paper_pages(paper_id, page_number)",
    "CREATE INDEX IF NOT EXISTS idx_fields_paper    ON extracted_fields(paper_id)",
    "CREATE INDEX IF NOT EXISTS idx_fields_type     ON extracted_fields(field_type)",
    "CREATE INDEX IF NOT EXISTS idx_screening_paper ON screening_results(paper_id)",
]


# ─── Connection factory ───────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with foreign-key enforcement enabled."""
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row          # access columns by name
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # better concurrent read performance
    return conn


@contextmanager
def get_db():
    """Context manager: yields a connection and commits on success, rolls back on error."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Initialisation ──────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables and indexes if they do not already exist."""
    with get_db() as conn:
        for statement in _DDL:
            conn.execute(statement)

        # Record schema version if not already present
        existing = conn.execute(
            "SELECT version FROM schema_version WHERE version = ?",
            (SCHEMA_VERSION,)
        ).fetchone()

        if not existing:
            conn.execute(
                "INSERT INTO schema_version(version) VALUES (?)",
                (SCHEMA_VERSION,)
            )

    logger.info("Database initialised at %s (schema v%d)", DATABASE_PATH, SCHEMA_VERSION)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def compute_file_hash(file_bytes: bytes) -> str:
    """Return a SHA-256 hex digest of the given bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(row) if row else {}


def rows_to_list(rows) -> list[dict]:
    """Convert a list of sqlite3.Row objects to a list of dicts."""
    return [dict(r) for r in rows]
