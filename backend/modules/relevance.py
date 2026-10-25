"""
modules/relevance.py — TF-IDF-based research-question relevance ranking.

Uses scikit-learn's TfidfVectorizer and cosine_similarity.
Returns per-paper relevance scores and the best matching text snippet.

Why TF-IDF instead of a neural embedding model?
  • No additional model download required (scikit-learn already installed).
  • Runs fast on CPU with no GPU requirement.
  • Fully explainable: the score comes directly from term overlap.
  • Appropriate for a lexical retrieval baseline.

Limitation: TF-IDF measures lexical overlap, not semantic similarity.
Papers that discuss the same concept using different vocabulary may score low.
A future upgrade could replace or supplement this with a sentence-embedding model.
"""

import re
import logging
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ─── Sentence splitting ───────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple punctuation rules.
    No NLTK or spaCy required.
    """
    # Split on '. ', '! ', '? ' followed by a capital letter or number
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _clean(text: str) -> str:
    """Remove excessive whitespace and control characters."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─── Snippet extraction ───────────────────────────────────────────────────────

def _find_best_snippet(query: str, page_texts: List[Tuple[int, str]]) -> Tuple[str, int]:
    """
    Given a query and a list of (page_number, page_text) pairs,
    find the single sentence that best matches the query using TF-IDF.

    Returns (snippet, page_number).
    """
    if not page_texts:
        return ("", 0)

    # Build a flat list of (page_number, sentence)
    candidates: List[Tuple[int, str]] = []
    for page_num, page_text in page_texts:
        for sentence in _split_sentences(page_text):
            if len(sentence.split()) >= 6:
                candidates.append((page_num, sentence))

    if not candidates:
        # Fallback: return first 300 chars of the first page
        first_page_num, first_page_text = page_texts[0]
        return (_clean(first_page_text[:300]), first_page_num)

    sentences_only = [c[1] for c in candidates]

    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        # Build corpus: query + all sentences
        corpus = [query] + sentences_only
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Similarity of query (index 0) against all sentences
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        best_idx = int(similarities.argmax())
        best_page_num, best_sentence = candidates[best_idx]
        return (_clean(best_sentence), best_page_num)

    except Exception as exc:
        logger.warning("Snippet extraction failed: %s", exc)
        first_page_num, first_page_text = page_texts[0]
        return (_clean(first_page_text[:300]), first_page_num)


# ─── Main ranking function ────────────────────────────────────────────────────

def rank_papers_by_relevance(
    query: str,
    papers: List[dict],
) -> List[dict]:
    """
    Rank papers by TF-IDF cosine similarity to the research query.

    Parameters
    ----------
    query : str
        The user's research question or search string.
    papers : list of dict
        Each dict must contain:
          - "paper_id"   : str
          - "filename"   : str
          - "page_texts" : list of (page_number: int, text: str) tuples

    Returns
    -------
    list of dict (sorted descending by score), each containing:
      - paper_id, filename, score (float 0-1), snippet (str), page_number (int)
    """
    if not papers or not query.strip():
        return []

    # Build one document per paper (concatenation of all page texts)
    documents: List[str] = []
    for p in papers:
        full_text = " ".join(text for _, text in p.get("page_texts", []))
        documents.append(_clean(full_text) if full_text.strip() else p["filename"])

    try:
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
        )
        corpus = [query] + documents
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Score each paper against the query
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    except Exception as exc:
        logger.error("TF-IDF vectorisation failed: %s", exc)
        scores = [0.0] * len(papers)

    results = []
    for i, paper in enumerate(papers):
        score = float(scores[i])
        snippet, page_num = _find_best_snippet(query, paper.get("page_texts", []))
        results.append({
            "paper_id":   paper["paper_id"],
            "filename":   paper["filename"],
            "score":      round(score, 4),
            "snippet":    snippet,
            "page_number": page_num,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
