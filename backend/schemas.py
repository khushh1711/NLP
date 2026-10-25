"""
schemas.py — Pydantic v2 request/response models for the API.

All API input validation and output serialisation flows through these models.
"""

from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


# ─── Workspace ───────────────────────────────────────────────────────────────

class WorkspaceCreate(BaseModel):
    title: str             = Field(..., min_length=1, max_length=200)
    description: str       = Field("", max_length=1000)
    research_question: str = Field("", max_length=500)


class WorkspaceUpdate(BaseModel):
    title: Optional[str]             = Field(None, min_length=1, max_length=200)
    description: Optional[str]       = Field(None, max_length=1000)
    research_question: Optional[str] = Field(None, max_length=500)


class WorkspaceOut(BaseModel):
    id:                str
    title:             str
    description:       str
    research_question: str
    created_at:        str
    updated_at:        str
    paper_count:       int = 0


# ─── Paper ───────────────────────────────────────────────────────────────────

class PaperOut(BaseModel):
    id:                 str
    workspace_id:       str
    filename:           str
    file_size_bytes:    int
    page_count:         int
    word_count:         int
    char_count:         int
    upload_status:      str
    analysis_status:    str
    screening_decision: str
    relevance_score:    float
    created_at:         str
    updated_at:         str


# ─── Summary ─────────────────────────────────────────────────────────────────

class SummaryOut(BaseModel):
    paper_id:   str
    overview:   str = ""
    method:     str = ""
    results:    str = ""
    conclusion: str = ""
    created_at: str = ""


# ─── Extracted field ─────────────────────────────────────────────────────────

class ExtractedFieldOut(BaseModel):
    id:                 int
    paper_id:           str
    field_type:         str
    value:              str
    normalized_value:   str
    evidence_text:      str
    page_number:        int
    extraction_method:  str
    verification_status: str
    created_at:         str


class FieldVerifyUpdate(BaseModel):
    verification_status: str  # "verified" | "incorrect" | "needs_review"


# ─── Screening ───────────────────────────────────────────────────────────────

class ScreeningRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


class ScreeningResultOut(BaseModel):
    paper_id:   str
    filename:   str
    score:      float
    snippet:    str
    page_number: int


class ScreeningDecisionUpdate(BaseModel):
    decision: str  # "relevant" | "irrelevant" | "needs_review" | "unreviewed"


# ─── Comparison ──────────────────────────────────────────────────────────────

class ComparisonRowOut(BaseModel):
    paper_id:   str
    filename:   str
    task:       str = "Not found"
    method:     str = "Not found"
    dataset:    str = "Not found"
    metric:     str = "Not found"
    result:     str = "Not found"
    limitation: str = "Not found"
    evidence:   dict = Field(default_factory=dict)


# ─── Insights ────────────────────────────────────────────────────────────────

class FrequencyItem(BaseModel):
    value: str
    count: int


class InsightsOut(BaseModel):
    paper_count:        int
    analyzed_count:     int
    top_methods:        List[FrequencyItem]
    top_datasets:       List[FrequencyItem]
    top_metrics:        List[FrequencyItem]
    top_tasks:          List[FrequencyItem]
    missing_fields:     dict   # field_type -> count of papers missing it
    screening_summary:  dict   # decision -> count


# ─── Analysis response ───────────────────────────────────────────────────────

class AnalysisOut(BaseModel):
    paper:          PaperOut
    summary:        Optional[SummaryOut]
    extracted_fields: List[ExtractedFieldOut]


# ─── Health ──────────────────────────────────────────────────────────────────

class HealthOut(BaseModel):
    status:       str
    db_connected: bool
    model_loaded: bool


# ─── Collection Research Brief ────────────────────────────────────────────────

class BriefSection(BaseModel):
    section_id:   str
    title:        str
    content:      str
    papers_cited: List[str] = Field(default_factory=list)
    key_points:   List[str] = Field(default_factory=list)
    caveats:      Optional[str] = None


class CollectionBriefOut(BaseModel):
    workspace_id:      str
    research_question: str
    paper_count:       int
    sections:          List[BriefSection]
    coverage_stats:    dict
    created_at:        str
    updated_at:        str


# ─── Cross-Paper Q&A ──────────────────────────────────────────────────────────

class QARequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    top_k:    int = Field(5, ge=1, le=10)


class QAPassageCitation(BaseModel):
    paper_id:        str
    filename:        str
    page_number:     int
    text:            str
    relevance_score: float


class QAAnswerOut(BaseModel):
    question:          str
    answer:            str
    is_sufficient:     bool
    citations:         List[QAPassageCitation]
    papers_consulted:  int
    summary_of_methods: List[str] = Field(default_factory=list)


# ─── Related Papers & Discovery ───────────────────────────────────────────────

class RelatedPaperItem(BaseModel):
    title:            str
    authors:          str = ""
    year:             Optional[int] = None
    abstract:         str = ""
    source:           str  # "internal_collection" | "semantic_scholar"
    url:              Optional[str] = None
    relevance_reason: str = ""
    paper_id:         Optional[str] = None
    citation_count:   Optional[int] = None
    open_access_pdf:  Optional[str] = None


class RelatedPapersOut(BaseModel):
    workspace_id:       str
    research_question:  str
    internal_results:   List[RelatedPaperItem]
    external_results:   List[RelatedPaperItem]
    external_status:    str  # "success" | "unavailable" | "rate_limited"

