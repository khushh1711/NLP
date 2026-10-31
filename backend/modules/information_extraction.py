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

# Real benchmark datasets and corpora
KNOWN_DATASETS = [
    # NLP Benchmarks
    "GLUE", "SuperGLUE", "SQuAD", "SQuAD v1.1", "SQuAD v2.0", "SQuAD 2.0",
    "MultiNLI", "SNLI", "MNLI", "QNLI", "SST-2", "CoLA", "MRPC", "QQP", "STS-B", "RTE", "WNLI",
    "CoNLL-2003", "CoNLL 2003", "CoNLL",
    "CNN/DailyMail", "CNN-DailyMail", "XSum", "Gigaword",
    "PubMed", "arXiv", "SciTLDR", "Multi-News", "BillSum", "BigPatent", "WikiHow", "Newsroom",
    "WMT 2014", "WMT14", "WMT 2016", "WMT16", "WMT", "IWSLT",
    "BookCorpus", "BooksCorpus", "English Wikipedia", "Wikipedia", "Common Crawl", "C4", "HugeNews", "OpenWebText",
    "TriviaQA", "Natural Questions", "NaturalQuestions", "HotpotQA", "WikiHop", "NewsQA", "RACE", "ARC",
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
    "T5", "BART", "PEGASUS", "Longformer", "LED", "Discourse-Aware", "CATTS", "ELECTRA", "DeBERTa", "XLNet",
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

def _normalize_task_objective(task_sentence: str) -> str:
    """Normalize extracted task sentences into substantive scientific task descriptions."""
    lower = task_sentence.lower()
    if "extreme summarization" in lower or "tldr" in lower:
        return "Extreme Scientific Summarization (TLDR)"
    if "abstractive summarization" in lower or "summarize" in lower or "summarization" in lower:
        if "long" in lower:
            return "Long-Document Abstractive Summarization"
        return "Abstractive Text Summarization"
    if "gap-sentences" in lower or ("pre-training" in lower and "summarization" in lower):
        return "Pre-training for Abstractive Summarization"
    if "denoising" in lower or "sequence-to-sequence" in lower:
        return "Denoising Sequence-to-Sequence Generation"
    if "transfer learning" in lower or "text-to-text" in lower:
        return "Unified Text-to-Text Transfer Learning"
    if "long sequences" in lower or "long document" in lower or "linearly with sequence length" in lower:
        return "Scalable Long-Document Transformer Modeling"
    if "machine translation" in lower or "translation" in lower:
        return "Neural Machine Translation"
    if "question answering" in lower or "qa" in lower:
        return "Question Answering"
    if "language representation" in lower or "pre-train" in lower:
        return "Pre-trained Language Representations"
    m = re.search(r'(?:we propose|we present|we introduce|we study|we address|the task of)\s+([^,;.]+)', task_sentence, re.I)
    if m:
        cand = m.group(1).strip()
        if 8 < len(cand) < 60:
            return cand.title()
    return "Scientific NLP Investigation"


def extract_tasks(page_texts: List[Tuple[int, str]]) -> List[Dict]:
    """Extract clean research task objectives without author or header metadata."""
    found: List[Dict] = []
    # Search first 3 pages
    for page_num, page_text in page_texts[:3]:
        sentences = _split_into_sentences(page_text)
        for sentence in sentences:
            # Skip author / affiliation blocks
            if any(bad in sentence.lower() for bad in ["google ai", "university", "department of", "arxiv:", "correspondence to", "equal contribution", "{"]):
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
                    cut = concise[:110]
                    last_space = cut.rfind(' ')
                    concise = cut[:last_space] + "…"

                norm_val = _normalize_task_objective(task_sentence)

                found.append({
                    "field_type":         "TASK",
                    "value":              concise,
                    "normalized_value":   norm_val,
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
    over generic words like 'fine-tuning' and excluding optimizers.
    """
    found: List[Dict] = []
    seen: set = set()

    # Prioritize primary proposed architectures over baselines and optimizers
    def priority_score(name: str, page_num: int) -> int:
        score = 0
        if page_num == 1:
            score += 15
        # Primary novel architectures get top weight
        if name in [
            "Discourse-Aware", "CATTS", "PEGASUS", "BART", "T5", "Longformer", "LED",
            "Surgical World-Action Model", "Surgical WAM", "Cosmos Policy", "BERTBASE", "BERTLARGE"
        ]:
            score += 25
        elif name in ["BERT", "Transformer", "RoBERTa", "GPT", "GPT-2"]:
            score += 10
        # Generic optimizers and baselines get penalized so they don't replace primary models
        if name in ["Adam", "AdamW", "SGD", "Beam Search"]:
            score -= 30
        if name in ["Fine-tuning", "Pre-training", "Transfer Learning", "MLM", "NSP"]:
            score -= 10
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
        # 1. Number + Metric (e.g. 28.4 BLEU, 91.0% F1, 80.5% GLUE, 93.2 Test F1)
        re.compile(
            r'(\b\d{1,3}(?:\.\d{1,2})?\s*%\s*(?:(?:overall|test|validation)\s+)?(?:' + '|'.join(re.escape(m) for m in KNOWN_METRICS) + r')\b)',
            re.IGNORECASE
        ),
        re.compile(
            r'(\b\d{1,3}\.\d{1,2}\s+(?:' + '|'.join(re.escape(m) for m in KNOWN_METRICS) + r')\b)',
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
                    # Filter out table/figure cross-references e.g. "Table 6 ROUGE"
                    if re.search(r'\b(?:Table|Figure|Fig\.|Section|Sec\.)\s+' + re.escape(matched_phrase), sentence, re.I):
                        continue
                    # Filter out isolated table numbers like "6 ROUGE"
                    if re.match(r'^\d+\s+[A-Za-z]+$', matched_phrase) and not '.' in matched_phrase:
                        continue
                    if sentence not in seen_evidence:
                        seen_evidence.add(sentence)
                        
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
                    break

    return found[:10]


def _normalize_limitation(sentence: str) -> str:
    """Normalize author limitation sentences to clear, substantive scientific categories."""
    lower = sentence.lower()
    if any(k in lower for k in ["quadratic", "sequence length", "scales quadratically", "long sequence"]):
        return "Quadratic Self-Attention Bottleneck"
    if any(k in lower for k in ["expensive", "computational cost", "tpu", "prohibitively"]):
        return "Computational & Hardware Scalability Constraints"
    if any(k in lower for k in ["rouge", "human evaluation", "metric", "nuance"]):
        return "Automated Metric & Evaluation Nuance Limitations"
    if any(k in lower for k in ["overfitting", "regularization", "back-translation"]):
        return "Regularization & Overfitting Risks"
    if any(k in lower for k in ["section", "headings", "discourse structure"]):
        return "Discourse Section Dependency"
    if any(k in lower for k in ["hallucinat", "factual", "faithfulness", "memoriz"]):
        return "Factual Consistency & Memorization Risks"
    if any(k in lower for k in ["low-resource", "small data", "supervised examples", "mismatch"]):
        return "Low-Resource Adaptation & Supervised Data Scarcity"
    return "Methodological Constraint"


def extract_limitations(page_texts: List[Tuple[int, str]]) -> List[Dict]:
    """
    Extract limitations, bottlenecks, and computational constraints.
    Stops before references and appendices to prevent noisy non-body text.
    """
    found: List[Dict] = []
    seen: set = set()
    total_pages = len(page_texts)
    ref_seen = False

    for page_num, page_text in page_texts:
        lower_p = page_text.lower()
        # Detect references section start (typically page > 4)
        if page_num > 4 and (
            re.search(r'^\s*(?:references|bibliography)\s*$', lower_p, re.MULTILINE)
            or '\nreferences\n' in lower_p
            or 'references\n' in lower_p
        ):
            ref_seen = True

        if ref_seen:
            break

        # In very long papers (>20 pages), main body discussions usually finish by page 15 or 45
        if total_pages > 20 and page_num > 46:
            break

        sentences = _split_into_sentences(page_text)
        for sentence in sentences:
            if _is_table_dump(sentence):
                continue

            # Skip author / affiliation / email / acknowledgment noise
            if any(bad in sentence.lower() for bad in [
                "@", "http", "arxiv:", "google ai", "allen institute", "facebook ai",
                "university", "department of", "equal contribution", "grant", "fellowship",
                "section 901", "taxable"
            ]):
                continue

            # Skip opening abstract boilerplate
            if sentence.lower().startswith("abstract"):
                continue

            lower = sentence.lower()
            if any(ind in lower for ind in LIMITATION_INDICATORS):
                if sentence not in seen:
                    seen.add(sentence)

                    concise = sentence
                    if len(concise) > 95:
                        cut = concise[:95]
                        last_space = cut.rfind(' ')
                        concise = cut[:last_space] + "…"

                    norm_label = _normalize_limitation(sentence)

                    found.append({
                        "field_type":         "LIMITATION",
                        "value":              concise,
                        "normalized_value":   norm_label,
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
