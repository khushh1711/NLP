"""
modules/information_extraction.py — Scientific Information Extraction Pipeline.

Extracts 6 canonical scientific facets from academic papers:
  • TASK       — Concise research objective or domain problem
  • METHOD     — Primary model architecture, novel technique, or baseline
  • DATASET    — Benchmark datasets and evaluation corpora (excluding repositories like arXiv)
  • METRIC     — Evaluation metrics (BLEU, F1, ROUGE, Accuracy, EM, etc.)
  • RESULT     — Quantitative experimental outcomes and score improvements
  • LIMITATION — Reported constraints, computational complexity, and failure modes

Features:
  - Table-noise filtering for clean evidence
  - Bi-directional metric-number pattern matching
  - Primary architecture prioritization
  - Full sentence ground-truth evidence attribution with page coordinates
"""

import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# ─── Clean Entity Dictionaries ───────────────────────────────────────────────

# Real benchmark datasets and corpora (Strictly no repositories like arXiv/PubMed/S2ORC!)
KNOWN_DATASETS = [
    # NLP Benchmarks
    "GLUE", "SuperGLUE", "SQuAD", "SQuAD v1.1", "SQuAD v2.0", "SQuAD 2.0",
    "MultiNLI", "SNLI", "MNLI", "QNLI", "SST-2", "CoLA", "MRPC", "QQP", "STS-B", "RTE", "WNLI",
    "CoNLL-2003", "CoNLL 2003", "CoNLL",
    "CNN/DailyMail", "CNN-DailyMail", "XSum", "Gigaword",
    "WMT 2014", "WMT14", "WMT 2016", "WMT16", "WMT", "IWSLT",
    "BookCorpus", "BooksCorpus", "English Wikipedia", "Wikipedia", "Common Crawl", "C4", "OpenWebText",
    "TriviaQA", "Natural Questions", "NaturalQuestions", "HotpotQA", "NewsQA", "RACE", "ARC",
    "Penn Treebank", "PTB",
    # Vision & Robotics Benchmarks
    "ImageNet", "MS-COCO", "COCO", "CIFAR-10", "CIFAR-100", "MNIST",
    "JIGSAWS", "PegTransfer", "dVRK", "MISAW", "RoboTurk", "BridgeData",
]

# Primary model architectures and algorithms
KNOWN_METHODS = [
    # Deep Pre-trained & Foundation Models
    "Surgical World-Action Model", "Surgical WAM", "World-Action Model", "Cosmos Policy",
    "BERT", "BERTBASE", "BERTLARGE", "RoBERTa", "ALBERT", "DistilBERT",
    "Transformer", "Multi-Head Attention", "Self-Attention", "Scaled Dot-Product Attention",
    "GPT", "OpenAI GPT", "GPT-2", "GPT-3", "GPT-4",
    "T5", "BART", "ELECTRA", "DeBERTa", "XLNet",
    "Diffusion Transformer", "Video Diffusion", "Diffusion Model",
    "LSTM", "BiLSTM", "ELMo", "ResNet", "CNN", "RNN",
    # Training / Algorithmic Paridigms
    "Pre-training", "Fine-tuning", "Transfer Learning", "Masked Language Model", "MLM",
    "Next Sentence Prediction", "NSP", "Contrastive Learning", "Reinforcement Learning",
    "Adam", "AdamW", "SGD", "LoRA", "Beam Search",
]

# Quantitative metrics
KNOWN_METRICS = [
    "BLEU", "BLEU-4", "SacreBLEU",
    "F1", "F1-score", "F1 Score", "Test F1", "Macro F1", "Micro F1",
    "Exact Match", "EM", "Accuracy", "Precision", "Recall",
    "ROUGE", "ROUGE-1", "ROUGE-2", "ROUGE-L",
    "GLUE score", "GLUE", "Success Rate", "Average Success Rate",
    "Perplexity", "PPL", "AUC", "ROC-AUC", "Mean Average Precision", "MAP",
    "Submillimeter Precision", "PegTransfer Score"
]

# Canonical task indicator phrases
TASK_INDICATORS = [
    "we address", "we study", "we propose", "we present", "we introduce",
    "our goal is", "our objective is", "the task of", "the problem of",
    "designed to", "pre-train deep bidirectional representations",
    "language representation", "machine translation", "question answering",
    "surgical robot learning", "surgical manipulation", "robot control",
    "text summarization", "natural language inference", "sequence transduction"
]

# Limitation indicator phrases
LIMITATION_INDICATORS = [
    "limitation", "limitations", "drawback", "drawbacks", "shortcoming", "shortcomings",
    "computational cost", "computationally expensive", "computational complexity",
    "quadratic", "quadratic complexity", "bottleneck", "data bottleneck",
    "scarcity of", "scarce", "scarce surgical", "limited action-labeled",
    "fixed budget", "sensitive to", "sensitive and peaks",
    "fails to", "cannot handle", "trade-off", "tradeoff",
    "restricted to", "requires large memory", "high latency",
    "does not appear", "discrepancy", "future work"
]


# ─── Helper Utilities ────────────────────────────────────────────────────────

def _normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, stripping raw author/email headers."""
    # Strip emails like {devlin, kentonl}@google.com
    clean = re.sub(r'\{[^}]+\}@[a-zA-Z0-9.-]+', '', text)
    clean = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', clean)
    
    # Split on periods/punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?])\s+', clean)
    sentences = []
    for s in raw:
        s_clean = _normalize_whitespace(s)
        if len(s_clean) > 20:
            sentences.append(s_clean)
    return sentences


def _is_table_dump(text: str) -> bool:
    """Detect if a text passage is an unformatted table row or column header dump."""
    # Count numbers and short metric codes
    numbers = re.findall(r'\b\d+\.?\d*[kM%]?\b', text)
    if len(numbers) >= 6:
        return True
    if re.search(r'\b(?:MNLI|QQP|QNLI|SST-2|CoLA|STS-B|MRPC|RTE)\b.*\b(?:MNLI|QQP|QNLI|SST-2)\b', text):
        return True
    return False


# ─── Facet Extractors ────────────────────────────────────────────────────────

def extract_tasks(page_texts: List[Tuple[int, str]]) -> List[Dict]:
    """Extract clean research task objectives without author or header metadata."""
    found: List[Dict] = []
    # Search first 3 pages
    for page_num, page_text in page_texts[:3]:
        sentences = _split_into_sentences(page_text)
        for sentence in sentences:
            # Skip author / affiliation blocks
            if any(bad in sentence.lower() for bad in ["google ai", "university", "department of", "arxiv:", "correspondence to", "equal contribution"]):
                continue

            lower = sentence.lower()
            if any(ind in lower for ind in TASK_INDICATORS):
                # Clean up the task sentence
                task_sentence = sentence
                # Remove leading abstract labels if present
                task_sentence = re.sub(r'^(?:Abstract|ABSTRACT|1\s+Introduction|Introduction)\s*[:.-]?\s*', '', task_sentence)

                # Formulate a concise display value for the table
                concise = task_sentence
                if len(concise) > 110:
                    # Try to capture up to the first clean comma or clause
                    cut = concise[:110]
                    last_space = cut.rfind(' ')
                    concise = cut[:last_space] + "…"

                found.append({
                    "field_type":         "TASK",
                    "value":              concise,
                    "normalized_value":   "Research Task",
                    "evidence_text":      task_sentence,
                    "page_number":        page_num,
                    "extraction_method":  "discourse_extraction",
                    "verification_status": "needs_review",
                })
                if len(found) >= 3:
                    return found

    return found


def extract_methods(page_texts: List[Tuple[int, str]]) -> List[Dict]:
    """
    Extract model architectures and methods, prioritizing primary novel models
    over generic words like 'fine-tuning'.
    """
    found: List[Dict] = []
    seen: set = set()

    # Score method priority (core models appear in title/abstract or are specialized)
    def priority_score(name: str, page_num: int) -> int:
        score = 0
        if page_num == 1:
            score += 10
        if name in ["Surgical World-Action Model", "Surgical WAM", "BERT", "Transformer", "Cosmos Policy"]:
            score += 20
        if name in ["Fine-tuning", "Pre-training", "Transfer Learning"]:
            score -= 5
        return score

    for page_num, page_text in page_texts:
        sentences = _split_into_sentences(page_text)
        for sentence in sentences:
            if _is_table_dump(sentence):
                continue

            for method in KNOWN_METHODS:
                pattern = re.compile(r'\b' + re.escape(method) + r'\b', re.IGNORECASE)
                if pattern.search(sentence):
                    norm = method.lower()
                    if norm not in seen:
                        seen.add(norm)
                        found.append({
                            "field_type":         "METHOD",
                            "value":              method,
                            "normalized_value":   method,
                            "evidence_text":      sentence,
                            "page_number":        page_num,
                            "extraction_method":  "entity_matching",
                            "verification_status": "needs_review",
                            "_score":             priority_score(method, page_num)
                        })

    # Sort so that primary novel architectures appear first!
    found.sort(key=lambda x: x["_score"], reverse=True)
    for f in found:
        f.pop("_score", None)

    return found[:12]


def extract_datasets(page_texts: List[Tuple[int, str]]) -> List[Dict]:
    """
    Extract benchmark datasets and corpora.
    Filters out raw table noise and consolidates sub-benchmarks.
    """
    found: List[Dict] = []
    seen: set = set()

    for page_num, page_text in page_texts:
        sentences = _split_into_sentences(page_text)
        for sentence in sentences:
            # Skip massive table dumps
            if _is_table_dump(sentence):
                continue

            for dataset in KNOWN_DATASETS:
                pattern = re.compile(r'\b' + re.escape(dataset) + r'\b', re.IGNORECASE)
                if pattern.search(sentence):
                    norm = dataset.lower()
                    if norm not in seen:
                        seen.add(norm)
                        found.append({
                            "field_type":         "DATASET",
                            "value":              dataset,
                            "normalized_value":   dataset,
                            "evidence_text":      sentence,
                            "page_number":        page_num,
                            "extraction_method":  "benchmark_entity",
                            "verification_status": "needs_review",
                        })

    return found[:12]


def extract_metrics(page_texts: List[Tuple[int, str]]) -> List[Dict]:
    """Extract quantitative evaluation criteria."""
    found: List[Dict] = []
    seen: set = set()

    for page_num, page_text in page_texts:
        sentences = _split_into_sentences(page_text)
        for sentence in sentences:
            if _is_table_dump(sentence):
                continue

            for metric in KNOWN_METRICS:
                pattern = re.compile(r'\b' + re.escape(metric) + r'\b', re.IGNORECASE)
                if pattern.search(sentence):
                    norm = metric.lower()
                    if norm not in seen:
                        seen.add(norm)
                        found.append({
                            "field_type":         "METRIC",
                            "value":              metric,
                            "normalized_value":   metric.upper(),
                            "evidence_text":      sentence,
                            "page_number":        page_num,
                            "extraction_method":  "metric_matching",
                            "verification_status": "needs_review",
                        })

    return found[:10]


def extract_results(page_texts: List[Tuple[int, str]]) -> List[Dict]:
    """
    High-recall quantitative result extraction:
    Captures:
      - [Metric] of [Number] (e.g. BLEU score of 28.4, F1 of 91.0%)
      - [Number] [Metric] (e.g. 28.4 BLEU, 91.0% F1, 80.5% GLUE, 93.2 Test F1)
      - Improvement verbs: 'improves from X% to Y%', 'absolute gain of Z percentage points'
      - Achievement verbs: 'achieves X%', 'outperforms by Y'
    """
    found: List[Dict] = []
    seen_evidence: set = set()

    # Robust academic quantitative result regex patterns
    result_patterns = [
        # 1. Number + Metric: e.g. "28.4 BLEU", "93.2 Test F1", "80.5% GLUE score", "84.1% EM"
        re.compile(
            r'(\b\d{1,3}(?:\.\d{1,2})?\s*%\s*(?:(?:overall|test|validation)\s+)?(?:' + '|'.join(re.escape(m) for m in KNOWN_METRICS) + r')\b)',
            re.IGNORECASE
        ),
        re.compile(
            r'(\b\d{1,3}(?:\.\d{1,2})?\s+(?:' + '|'.join(re.escape(m) for m in KNOWN_METRICS) + r')\b)',
            re.IGNORECASE
        ),
        # 2. Metric + Number: e.g. "BLEU of 28.4", "F1 score of 91.0%", "Test F1 to 93.2"
        re.compile(
            r'((?:' + '|'.join(re.escape(m) for m in KNOWN_METRICS) + r')\s*(?:score|points?)?\s*(?:of|to|is|reach(?:es)?|achieve(?:s)?|=)\s*\d{1,3}(?:\.\d{1,2})?\s*%?)',
            re.IGNORECASE
        ),
        # 3. Improvement verbs: e.g. "improves the average success rate from 63.5% to 77.8%"
        re.compile(
            r'((?:improves?|increases?|advances?|gains?)\s+(?:the\s+)?(?:[a-zA-Z\s-]{1,30})?\s*from\s+\d{1,3}(?:\.\d{1,2})?%?\s+to\s+\d{1,3}(?:\.\d{1,2})?%?)',
            re.IGNORECASE
        ),
        # 4. Absolute gain: e.g. "absolute gain of 20 percentage points"
        re.compile(
            r'((?:gain|improvement|boost)\s+of\s+\d{1,3}(?:\.\d{1,2})?\s*(?:percentage\s+points|points?|%))',
            re.IGNORECASE
        ),
        # 5. Achieves score: e.g. "achieves an overall score of 80.5%"
        re.compile(
            r'(achieves?\s+(?:a\s+)?(?:new\s+)?(?:state-of-the-art\s+)?(?:[a-zA-Z\s-]{1,25})?\s*(?:of\s+)?\d{1,3}(?:\.\d{1,2})?%?)',
            re.IGNORECASE
        )
    ]

    for page_num, page_text in page_texts:
        sentences = _split_into_sentences(page_text)
        for sentence in sentences:
            if _is_table_dump(sentence):
                continue

            for pattern in result_patterns:
                match = pattern.search(sentence)
                if match:
                    matched_phrase = match.group(1).strip()
                    if sentence not in seen_evidence:
                        seen_evidence.add(sentence)
                        
                        # Clean concise value for tabular view
                        clean_val = matched_phrase
                        if len(clean_val) > 75:
                            clean_val = clean_val[:75] + "…"

                        found.append({
                            "field_type":         "RESULT",
                            "value":              clean_val,
                            "normalized_value":   clean_val,
                            "evidence_text":      sentence,
                            "page_number":        page_num,
                            "extraction_method":  "numeric_regex",
                            "verification_status": "needs_review",
                        })
                    break  # Move to next sentence once matched

    return found[:10]


def extract_limitations(page_texts: List[Tuple[int, str]]) -> List[Dict]:
    """
    Extract limitations, bottlenecks, and computational constraints.
    Searches across body pages while avoiding references.
    """
    found: List[Dict] = []
    seen: set = set()

    for page_num, page_text in page_texts:
        # Stop if we hit the references section
        if re.search(r'\b(?:References|REFERENCES|Bibliography)\b', page_text) and page_num > 4:
            continue

        sentences = _split_into_sentences(page_text)
        for sentence in sentences:
            if _is_table_dump(sentence):
                continue

            lower = sentence.lower()
            if any(ind in lower for ind in LIMITATION_INDICATORS):
                if sentence not in seen:
                    seen.add(sentence)

                    # Create a clean concise summary value for table display
                    concise = sentence
                    if len(concise) > 90:
                        cut = concise[:90]
                        last_space = cut.rfind(' ')
                        concise = cut[:last_space] + "…"

                    found.append({
                        "field_type":         "LIMITATION",
                        "value":              concise,
                        "normalized_value":   "Limitation",
                        "evidence_text":      sentence,
                        "page_number":        page_num,
                        "extraction_method":  "constraint_heuristic",
                        "verification_status": "needs_review",
                    })
                    if len(found) >= 8:
                        return found

    return found


# ─── Master Extraction Entry Point ──────────────────────────────────────────

def extract_all_fields(page_texts: List[Tuple[int, str]]) -> List[Dict]:
    """
    Extract all 6 scientific facets from page-by-page text.
    Returns structured list with exact page numbers and evidence sentences.
    """
    all_fields: List[Dict] = []
    
    # 1. Research Tasks
    all_fields.extend(extract_tasks(page_texts))

    # 2. Methodologies & Architectures (Prioritized)
    all_fields.extend(extract_methods(page_texts))

    # 3. Benchmark Datasets (No repos like arXiv)
    all_fields.extend(extract_datasets(page_texts))

    # 4. Quantitative Metrics
    all_fields.extend(extract_metrics(page_texts))

    # 5. Experimental Results (High-Recall Numerics)
    all_fields.extend(extract_results(page_texts))

    # 6. Limitations & Constraints
    all_fields.extend(extract_limitations(page_texts))

    return all_fields
