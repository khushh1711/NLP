"""
qa_engine.py — Evidence-grounded collection-wide question answering system.

Uses lightweight TF-IDF and cosine similarity passage retrieval over page-level text.
Never hallucinates citations or non-existent facts.
Provides honest rejection when evidence is insufficient.
"""

import re
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.08  # Below this, return insufficient evidence response


def _chunk_page_text(text: str, words_per_passage: int = 150, overlap: int = 40) -> List[str]:
    """Split page text into overlapping multi-sentence passage chunks."""
    words = text.split()
    if len(words) <= words_per_passage:
        clean = " ".join(words).strip()
        return [clean] if clean else []
    
    passages = []
    step = max(1, words_per_passage - overlap)
    for i in range(0, len(words), step):
        chunk = words[i:i + words_per_passage]
        if len(chunk) < 30 and passages:
            break
        passages.append(" ".join(chunk))
    return passages


def answer_collection_question(
    question: str,
    pages_data: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Given a research question and all pages across workspace papers,
    retrieves the top evidence passages and constructs a grounded response.
    """
    clean_q = question.strip()
    if not clean_q or not pages_data:
        return {
            "question": clean_q,
            "answer": "The current collection does not contain enough evidence to answer this question reliably.",
            "is_sufficient": False,
            "citations": [],
            "papers_consulted": 0,
            "summary_of_methods": [],
        }

    # ── Build passage corpus ──────────────────────────────────────────────────
    passages_meta = []  # list of dicts: {paper_id, filename, page_number, text}
    corpus = []

    for item in pages_data:
        pid = item["paper_id"]
        fname = item["filename"]
        pnum = item["page_number"]
        ptext = item.get("text", "")
        
        chunks = _chunk_page_text(ptext)
        for c in chunks:
            # Skip noise or references headers
            if len(c) < 50:
                continue
            passages_meta.append({
                "paper_id": pid,
                "filename": fname,
                "page_number": pnum,
                "text": c,
            })
            corpus.append(c)

    if not corpus:
        return {
            "question": clean_q,
            "answer": "The current collection does not contain enough evidence to answer this question reliably.",
            "is_sufficient": False,
            "citations": [],
            "papers_consulted": 0,
            "summary_of_methods": [],
        }

    # ── TF-IDF Retrieval ──────────────────────────────────────────────────────
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        q_vec = vectorizer.transform([clean_q])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    except Exception as exc:
        logger.error("TF-IDF vectorization error: %s", exc)
        return {
            "question": clean_q,
            "answer": "Error performing passage retrieval on collection text.",
            "is_sufficient": False,
            "citations": [],
            "papers_consulted": 0,
            "summary_of_methods": [],
        }

    # ── Check relevance threshold ─────────────────────────────────────────────
    best_indices = np.argsort(scores)[::-1]
    top_score = scores[best_indices[0]] if len(best_indices) > 0 else 0.0

    if top_score < SIMILARITY_THRESHOLD:
        return {
            "question": clean_q,
            "answer": "The current collection does not contain enough evidence to answer this question reliably. "
                      "None of the retrieved passages met the relevance threshold for this specific query.",
            "is_sufficient": False,
            "citations": [],
            "papers_consulted": 0,
            "summary_of_methods": [],
        }

    # ── Gather top-K distinct citations ───────────────────────────────────────
    citations = []
    seen_texts = set()
    papers_consulted_set = set()

    for idx in best_indices:
        if len(citations) >= top_k:
            break
        sc = float(scores[idx])
        if sc < SIMILARITY_THRESHOLD:
            break
            
        meta = passages_meta[idx]
        text_preview = meta["text"][:120]
        if text_preview in seen_texts:
            continue
        seen_texts.add(text_preview)
        
        papers_consulted_set.add(meta["paper_id"])
        citations.append({
            "paper_id": meta["paper_id"],
            "filename": meta["filename"],
            "page_number": meta["page_number"],
            "text": meta["text"],
            "relevance_score": round(sc, 4),
        })

    # ── Synthesize grounded answer ────────────────────────────────────────────
    # Group evidence sentences by paper
    grouped_by_paper = {}
    for c in citations:
        fname = c["filename"]
        if fname not in grouped_by_paper:
            grouped_by_paper[fname] = []
        grouped_by_paper[fname].append(c)

    answer_lines = []
    answer_lines.append(f"Based on the analysis of {len(papers_consulted_set)} paper(s) in this collection:")

    for fname, p_citations in grouped_by_paper.items():
        top_cit = p_citations[0]
        # Extract best sentence from the passage that matches query terms
        sentences = re.split(r'(?<=[.!?])\s+', top_cit["text"])
        best_sentence = top_cit["text"][:250] + "..."
        
        q_tokens = set(re.findall(r'\w+', clean_q.lower())) - {"which", "what", "how", "the", "in", "and", "is", "are", "of", "to"}
        if q_tokens:
            best_match_count = -1
            for s in sentences:
                s_tokens = set(re.findall(r'\w+', s.lower()))
                match_count = len(q_tokens & s_tokens)
                if match_count > best_match_count and len(s.strip()) > 30:
                    best_match_count = match_count
                    best_sentence = s.strip()

        answer_lines.append(f"• **{fname}** (Page {top_cit['page_number']}): \"{best_sentence}\"")

    full_answer = "\n\n".join(answer_lines)

    return {
        "question": clean_q,
        "answer": full_answer,
        "is_sufficient": True,
        "citations": citations,
        "papers_consulted": len(papers_consulted_set),
        "summary_of_methods": list(grouped_by_paper.keys()),
    }
