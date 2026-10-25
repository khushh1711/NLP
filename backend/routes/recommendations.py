"""
routes/recommendations.py — Multi-stage related paper discovery and recommendations.

Supports:
- Stage 1: Internal collection recommendations based on research question and TF-IDF similarity.
- Stage 2: External academic paper discovery via Semantic Scholar API with timeout and graceful offline fallback.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import httpx

from database import get_db, rows_to_list, row_to_dict
from schemas import RelatedPapersOut, RelatedPaperItem
from modules.relevance import rank_papers_by_relevance

router = APIRouter(prefix="/api/workspaces", tags=["recommendations"])
logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


@router.get("/{workspace_id}/related-papers", response_model=RelatedPapersOut)
async def get_related_papers(
    workspace_id: str,
    focal_paper_id: Optional[str] = Query(None, description="Paper currently being viewed, to exclude from internal recommendations"),
):
    with get_db() as conn:
        ws = conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found.")
        
        rq = ws["research_question"] or ws["title"] or "natural language processing"
        
        # Get papers in workspace with summaries
        papers = conn.execute(
            """SELECT p.id, p.filename, s.overview, s.method
               FROM papers p
               LEFT JOIN summaries s ON s.paper_id = p.id
               WHERE p.workspace_id = ? AND p.analysis_status = 'completed'""",
            (workspace_id,)
        ).fetchall()
        
    papers_list = rows_to_list(papers)
    
    # ── Stage 1: Internal recommendations ────────────────────────────────────
    internal_items = []
    if len(papers_list) > 1:
        # Fetch page texts for papers
        paper_pages_map = {}
        with get_db() as conn:
            for p in papers_list:
                rows = conn.execute(
                    "SELECT page_number, text FROM paper_pages WHERE paper_id = ? ORDER BY page_number",
                    (p["id"],)
                ).fetchall()
                paper_pages_map[p["id"]] = [(r["page_number"], r["text"]) for r in rows]

        ranking_input = [
            {
                "paper_id": p["id"],
                "filename": p["filename"],
                "page_texts": paper_pages_map.get(p["id"]) or [(1, f"{p.get('overview','') or ''} {p.get('method','') or ''}")]
            }
            for p in papers_list
        ]
        ranked = rank_papers_by_relevance(rq, ranking_input)
        for r in ranked:
            if focal_paper_id and r["paper_id"] == focal_paper_id:
                continue
            if r["score"] > 0:
                internal_items.append(
                    RelatedPaperItem(
                        title=r["filename"],
                        authors="Local Collection Paper",
                        year=None,
                        abstract=r.get("snippet", "") or "Extracted summary from local workspace paper.",
                        source="internal_collection",
                        url=None,
                        relevance_reason=f"TF-IDF relevance score {round(r['score'], 3)} to inquiry \"{rq[:60]}\"",
                        paper_id=r["paper_id"],
                        citation_count=None,
                    )
                )

    # ── Stage 2: External Discovery via Semantic Scholar API ─────────────────
    external_items = []
    external_status = "unavailable"
    
    try:
        search_query = rq.strip()
        if len(search_query) > 120:
            search_query = " ".join(search_query.split()[:12])  # Keep query concise for API
            
        params = {
            "query": search_query,
            "limit": 5,
            "fields": "title,authors,year,abstract,openAccessPdf,citationCount,url",
        }
        headers = {
            "User-Agent": "ScientificLiteratureMiningSystem/2.0 (Academic Research; MSc IT NLP)",
        }
        
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(SEMANTIC_SCHOLAR_SEARCH_URL, params=params, headers=headers)
            
            if resp.status_code == 200:
                data = resp.json()
                papers_data = data.get("data", [])
                for item in papers_data:
                    title = item.get("title") or "Untitled Publication"
                    authors_list = [a.get("name") for a in item.get("authors", []) if a.get("name")]
                    if len(authors_list) > 3:
                        authors_str = ", ".join(authors_list[:3]) + " et al."
                    else:
                        authors_str = ", ".join(authors_list) or "Unknown Authors"
                        
                    pdf_info = item.get("openAccessPdf")
                    pdf_url = pdf_info.get("url") if isinstance(pdf_info, dict) else None
                    
                    abstract = item.get("abstract") or "No abstract preview provided by Semantic Scholar."
                    if len(abstract) > 300:
                        abstract = abstract[:297] + "..."
                        
                    external_items.append(
                        RelatedPaperItem(
                            title=title,
                            authors=authors_str,
                            year=item.get("year"),
                            abstract=abstract,
                            source="semantic_scholar",
                            url=item.get("url"),
                            relevance_reason="Matched Semantic Scholar index for research topic",
                            paper_id=None,
                            citation_count=item.get("citationCount"),
                            open_access_pdf=pdf_url,
                        )
                    )
                external_status = "success" if external_items else "no_results"
            elif resp.status_code == 429:
                external_status = "rate_limited"
                logger.info("Semantic Scholar API rate limited.")
            else:
                external_status = "unavailable"
                logger.warning("Semantic Scholar returned status %d", resp.status_code)
                
    except Exception as exc:
        logger.info("External academic discovery offline or timed out: %s", exc)
        external_status = "unavailable"

    return {
        "workspace_id": workspace_id,
        "research_question": rq,
        "internal_results": internal_items,
        "external_results": external_items,
        "external_status": external_status,
    }
