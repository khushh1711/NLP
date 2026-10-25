"""
tests/test_backend.py — Automated backend tests.

Run from the backend directory with the venv active:
    .\\venv\\Scripts\\python.exe -m pytest tests/ -v

Tests cover: DB init, PDF extraction, workspace CRUD,
paper upload/duplicate detection, relevance ranking, extraction, comparison.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import io
from pathlib import Path


# ─── Database tests ───────────────────────────────────────────────────────────

def test_database_initialises(tmp_path, monkeypatch):
    """DB init must create all tables without error."""
    monkeypatch.setattr("config.DATABASE_PATH", tmp_path / "test.db")
    from database import init_db, get_db
    import importlib, database
    importlib.reload(database)
    database.init_db()
    with database.get_db() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
    for expected in ["workspaces", "papers", "paper_pages", "summaries",
                     "extracted_fields", "screening_results"]:
        assert expected in tables, f"Table '{expected}' missing"


def test_file_hash_deterministic():
    """Same bytes must always produce the same hash."""
    from database import compute_file_hash
    data = b"hello world"
    assert compute_file_hash(data) == compute_file_hash(data)
    assert compute_file_hash(data) != compute_file_hash(b"different")


# ─── PDF processor tests ──────────────────────────────────────────────────────

def _make_minimal_pdf() -> bytes:
    """
    Return the raw bytes of a minimal valid single-page PDF.
    Built without any external library so this test has no extra dependency.
    """
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"
        b">>>>/Contents 4 0 R>>endobj\n"
        b"4 0 obj<</Length 44>>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000058 00000 n \n"
        b"0000000115 00000 n \n0000000266 00000 n \n"
        b"trailer<</Size 5/Root 1 0 R>>\nstartxref\n360\n%%EOF"
    )
    return content


def test_pdf_extraction_returns_pages():
    """extract_pages_from_pdf must return a non-empty list."""
    from pdf_processor import extract_pages_from_pdf
    pages = extract_pages_from_pdf(_make_minimal_pdf())
    assert isinstance(pages, list)
    assert len(pages) >= 1
    assert all(isinstance(pn, int) and isinstance(pt, str) for pn, pt in pages)


def test_pdf_extraction_page_numbers_are_one_indexed():
    from pdf_processor import extract_pages_from_pdf
    pages = extract_pages_from_pdf(_make_minimal_pdf())
    page_numbers = [pn for pn, _ in pages]
    assert 1 in page_numbers, "Page numbers must be 1-indexed"


def test_invalid_pdf_returns_empty():
    from pdf_processor import extract_pages_from_pdf
    pages = extract_pages_from_pdf(b"not a pdf at all")
    assert pages == []


# ─── Relevance ranking tests ──────────────────────────────────────────────────

def test_relevance_ranking_sorts_by_score():
    from modules.relevance import rank_papers_by_relevance
    papers = [
        {"paper_id": "A", "filename": "bert.pdf",
         "page_texts": [(1, "BERT is a bidirectional Transformer for language understanding.")]},
        {"paper_id": "B", "filename": "cooking.pdf",
         "page_texts": [(1, "This paper discusses pasta recipes and Italian cuisine.")]},
    ]
    results = rank_papers_by_relevance("Transformer language model BERT", papers)
    assert len(results) == 2
    # BERT paper should rank first
    assert results[0]["paper_id"] == "A"
    assert results[0]["score"] >= results[1]["score"]


def test_relevance_empty_papers():
    from modules.relevance import rank_papers_by_relevance
    assert rank_papers_by_relevance("any query", []) == []


def test_relevance_returns_snippet():
    from modules.relevance import rank_papers_by_relevance
    papers = [{"paper_id": "X", "filename": "x.pdf",
               "page_texts": [(1, "We propose a novel attention mechanism for text summarization.")]}]
    results = rank_papers_by_relevance("attention summarization", papers)
    assert results[0]["snippet"] != ""


# ─── Information extraction tests ────────────────────────────────────────────

def test_extraction_finds_bert():
    from modules.information_extraction import extract_methods
    page_texts = [(1, "We fine-tuned BERT on the SQuAD dataset and evaluated using F1 score.")]
    results = extract_methods(page_texts)
    values = [r["value"] for r in results]
    assert "BERT" in values


def test_extraction_finds_squad():
    from modules.information_extraction import extract_datasets
    page_texts = [(1, "Experiments were conducted on the SQuAD benchmark.")]
    results = extract_datasets(page_texts)
    values = [r["value"] for r in results]
    assert any("SQuAD" in v for v in values)


def test_extraction_finds_f1():
    from modules.information_extraction import extract_metrics
    page_texts = [(2, "The model achieves 91.5 F1 on the test set.")]
    results = extract_metrics(page_texts)
    values = [r["value"] for r in results]
    assert any("F1" in v or "f1" in v.lower() for v in values)


def test_extraction_empty_input():
    from modules.information_extraction import extract_all_fields
    assert extract_all_fields([]) == []


def test_extraction_attaches_evidence():
    from modules.information_extraction import extract_datasets
    sentence = "We trained the model on the CNN/DailyMail dataset for summarization."
    results = extract_datasets([(3, sentence)])
    assert len(results) > 0
    assert results[0]["evidence_text"] != ""
    assert results[0]["page_number"] == 3


def test_extraction_verification_status_default():
    from modules.information_extraction import extract_methods
    results = extract_methods([(1, "We use BERT for classification.")])
    for r in results:
        assert r["verification_status"] == "needs_review"


# ─── Module C: Collection Synthesis tests ─────────────────────────────────────

def test_collection_synthesis_empty_papers():
    from modules.collection_synthesis import generate_collection_brief
    brief = generate_collection_brief("ws-1", "Summarization methods", [], [], [])
    assert brief["paper_count"] == 0
    assert len(brief["sections"]) == 1
    assert "empty_state" in brief["sections"][0]["section_id"]


def test_collection_synthesis_single_paper():
    from modules.collection_synthesis import generate_collection_brief
    papers = [{"id": "p1", "filename": "single.pdf", "analysis_status": "completed", "page_count": 5, "word_count": 1200}]
    brief = generate_collection_brief("ws-1", "Summarization methods", papers, [], [])
    assert brief["paper_count"] == 1
    assert "single_paper_note" in brief["sections"][0]["section_id"]
    assert "at least 2" in brief["sections"][0]["content"]


def test_collection_synthesis_multi_paper_generates_9_sections():
    from modules.collection_synthesis import generate_collection_brief
    papers = [
        {"id": "p1", "filename": "attention.pdf", "analysis_status": "completed", "page_count": 10, "word_count": 4000},
        {"id": "p2", "filename": "bert.pdf", "analysis_status": "completed", "page_count": 12, "word_count": 5000},
    ]
    fields = [
        {"paper_id": "p1", "field_type": "METHOD", "value": "Transformer", "page_number": 1, "extraction_method": "pattern", "verification_status": "verified"},
        {"paper_id": "p1", "field_type": "DATASET", "value": "WMT 2014", "page_number": 6, "extraction_method": "pattern", "verification_status": "verified"},
        {"paper_id": "p1", "field_type": "METRIC", "value": "BLEU", "page_number": 8, "extraction_method": "pattern", "verification_status": "verified"},
        {"paper_id": "p1", "field_type": "RESULT", "value": "28.4 BLEU", "page_number": 8, "extraction_method": "pattern", "verification_status": "verified"},
        {"paper_id": "p2", "field_type": "METHOD", "value": "BERT", "page_number": 1, "extraction_method": "pattern", "verification_status": "verified"},
        {"paper_id": "p2", "field_type": "DATASET", "value": "GLUE", "page_number": 4, "extraction_method": "pattern", "verification_status": "verified"},
    ]
    brief = generate_collection_brief("ws-1", "Compare Transformer architectures", papers, fields, [])
    assert brief["paper_count"] == 2
    assert len(brief["sections"]) == 9
    sec_ids = [s["section_id"] for s in brief["sections"]]
    for expected_id in ["overview", "themes", "methods", "datasets", "results", "limitations", "comparison", "gaps", "coverage"]:
        assert expected_id in sec_ids


# ─── Module E: Collection Q&A tests ───────────────────────────────────────────

def test_collection_qa_finds_evidence_and_cites():
    from modules.qa_engine import answer_collection_question
    pages_data = [
        {"paper_id": "p1", "filename": "attention.pdf", "page_number": 6,
         "text": "The Transformer model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, setting a new state of the art."},
        {"paper_id": "p2", "filename": "bert.pdf", "page_number": 4,
         "text": "BERT achieves an 80.5 score on the GLUE benchmark, outperforming existing multi-task baselines."},
    ]
    res = answer_collection_question("What BLEU score is reported on WMT 2014?", pages_data)
    assert res["is_sufficient"] is True
    assert "attention.pdf" in res["answer"]
    assert len(res["citations"]) > 0
    assert res["citations"][0]["filename"] == "attention.pdf"
    assert res["citations"][0]["page_number"] == 6


def test_collection_qa_rejects_insufficient_evidence():
    from modules.qa_engine import answer_collection_question
    pages_data = [
        {"paper_id": "p1", "filename": "attention.pdf", "page_number": 1,
         "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."},
    ]
    res = answer_collection_question("What is the quantum superposition of an electron in magnetic resonance?", pages_data)
    assert res["is_sufficient"] is False
    assert "not contain enough evidence" in res["answer"]
    assert len(res["citations"]) == 0


# ─── Run directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
