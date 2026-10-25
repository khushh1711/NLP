# 🔬 Scientific Literature Mining System

> **A dual-mode Natural Language Processing (NLP) research analysis platform for multi-paper academic investigation, evidence-grounded research synthesis, structured scientific entity extraction, and cross-paper comparative reasoning.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Model-t5--small--arxiv--summarizer-FFD21E.svg)](https://huggingface.co/Bashaarat1/t5-small-arxiv-summarizer)
[![Tests](https://img.shields.io/badge/Pytest-19%2F19%20Passing-brightgreen.svg)]()
[![Evaluation](https://img.shields.io/badge/SciNLP%2FSciER-F1%3A%200.642-success.svg)]()

---

## 📖 Project Identity & Academic Context

* **Degree:** M.Sc. Information Technology
* **Subject:** Natural Language Processing (NLP)
* **Application Type:** Full-Stack Academic Literature Mining & Research Synthesis Platform
* **Domain:** Artificial Intelligence, Machine Learning, and NLP Scientific Publications

### Why a Literature Mining System rather than only a Paper Summarizer?
A generic paper summarizer processes disconnected PDFs one by one in isolation. While useful, it leaves researchers with the burdensome task of manually reconciling competing claims, comparing discordant evaluation benchmarks, cross-referencing datasets, and synthesizing findings. 

The **Scientific Literature Mining System** upgrades this experience into a **Dual-Mode Research Analysis Platform**:
1. **Mode A — Individual Paper Analysis:** Retains single-paper ingestion, character/word counts, 4-part T5 scientific abstractive summaries (*Overview, Method, Results, Conclusion*), structured scientific facets, and human verification audit controls.
2. **Mode B — Research Collection Analysis (Primary New Capability):** Centers the research workflow around a user's central research inquiry. Batch-analyzes collections of scientific papers, generates a synthesized 9-part research brief, powers natural language cross-paper question answering with passage retrieval and citations `[Paper, Page X]`, tabulates comparative matrices, computes collection-level insights, and discovers related internal and external publications.

---

## 🌟 Key Functional Modules

* **📑 Module C — Collection-Wide Scientific Brief:** Synthesizes evidence across all papers in a workspace into 9 structured dimensions:
  1. *Research Question Overview & Scope*
  2. *Common Research Themes & Objectives*
  3. *Methods, Architectures & Algorithms*
  4. *Datasets & Benchmarks (Shared vs. Unique)*
  5. *Evaluation Metrics & Experimental Findings (Context-Preserving)*
  6. *Reported Limitations & Constraints*
  7. *Agreements, Differences & Methodological Contrast*
  8. *Potential Research Gaps (Clearly flagged for human review)*
  9. *Evidence Coverage & Audit Provenance*
* **💬 Module E — Ask the Collection (Evidence-Grounded Cross-Paper Q&A):** Natural language question answering across all uploaded publications simultaneously. Divides page text into overlapping passages, ranks them via TF-IDF cosine similarity, synthesizes grounded multi-paper answers, and honestly rejects queries when insufficient evidence exists (*"The current collection does not contain enough evidence to answer this question reliably."*).
* **📄 Module A — Research Workspaces & Management:** Dedicated workspace containers with title, description, and research question. Supports multi-PDF drag-and-drop batch upload, 1-indexed page text persistence, SHA-256 cryptographic deduplication, and cascading invalidation when papers are modified or deleted.
* **🏷️ Module D — Structured Scientific Information Extraction:** Pattern-directed extraction across 6 key facets:
  * **Research Tasks** (Abstract/intro problem extraction, email/affiliation filtering)
  * **Methods & Architectures** (Novel models prioritized over generic terms: *Transformer, BERT, RoBERTa, Self-Attention*)
  * **Datasets & Benchmarks** (Excludes repository names like *arXiv/PubMed*; identifies real experimental corpora: *GLUE, SQuAD, WMT 2014, MNLI*)
  * **Evaluation Metrics & Results** (Bidirectional matcher capturing `Metric + Number` and `Number + Metric`: *28.4 BLEU, 80.5% accuracy, 91.4 F1*)
  * **Reported Limitations** (Hardware requirements, quadratic complexity, pretraining costs)
* **📊 Module F — Cross-Paper Comparison Matrix:** Side-by-side comparative table filterable by method, dataset, or metric, with one-click RFC 4180 compliant CSV export.
* **💡 Module H — Collection Insights:** Visual frequency distributions of top methods, datasets, evaluation metrics, and missing field coverage generated dynamically from stored SQLite records.
* **🔗 Module G — Related Paper Discovery:** Two-stage discovery:
  * *Stage 1 (Internal):* TF-IDF relevance ranking of other papers in the collection against the inquiry.
  * *Stage 2 (External):* Live integration with Semantic Scholar API (`https://api.semanticscholar.org/graph/v1/paper/search`) with graceful offline fallback and open-access PDF links.
* **🛡️ Human-in-the-Loop (HITL) Audit Controls:** Every extracted entity connects to its verbatim source passage and exact PDF page number with interactive verification controls (*Verified, Needs Review, Incorrect*).

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     React 19 Frontend (Vite)                            │
│  • Dual-Mode Dashboard: Single Document vs Multi-Paper Workspace        │
│  • Collection Research Brief: 9 structured synthesized dimensions       │
│  • Ask the Collection: Cross-paper Q&A with clickable page citations    │
│  • Comparison Matrix: Tabular cross-paper synthesis & CSV export        │
│  • Collection Insights: Stored frequency charts & extraction coverage   │
│  • Related Papers: Internal collection ranking + Semantic Scholar       │
│  • Paper Detail: 4-part T5 summary, entity pills & HITL audit buttons   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP REST (Port 8000)
┌────────────────────────────────────▼────────────────────────────────────┐
│                    FastAPI Backend Architecture                         │
│  • /api/workspaces (CRUD & PATCH scope)                                 │
│  • /api/workspaces/{id}/papers (Batch upload & SHA-256 deduplication)   │
│  • /api/workspaces/{id}/analyze-collection & /research-brief            │
│  • /api/workspaces/{id}/ask (Cross-paper passage retrieval & Q&A)       │
│  • /api/workspaces/{id}/comparison & /export.csv                        │
│  • /api/workspaces/{id}/screen (TF-IDF relevance ranking)               │
│  • /api/workspaces/{id}/insights (Aggregated stored statistics)         │
│  • /api/workspaces/{id}/related-papers (Multi-stage discovery)          │
│  • /api/papers/{id}/analysis & /api/fields/{id}/verify                  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                    NLP Processing & Storage Layer                       │
│  • PyMuPDF (fitz): Page-by-page text parsing with 1-indexed association │
│  • Hugging Face Transformers: Bashaarat1/t5-small-arxiv-summarizer     │
│  • Scikit-Learn: Sublinear TF-IDF Vectorizer & Cosine Similarity        │
│  • Pattern IE Engine: Bidirectional entity & metric regex extraction    │
│  • SQLite 3 Database (literature_mining.db with WAL mode & Foreign Keys)│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Non-Negotiable Resource & Design Constraints

* **Hardware:** Local CPU execution on Windows 11; no GPU requirement.
* **Storage Footprint:** Designed for constrained environments (~5 GB disk free); strictly avoids downloading models larger than 300 MB.
* **Memory Management:** Neural T5 model loaded exactly once per backend process, never re-initialized on every API request.
* **Zero Paid Dependencies:** No OpenAI or proprietary API keys required. No external vector databases (Pinecone, Weaviate) or heavy database engines required.
* **Aesthetic Standard:** Strictly styled in high-contrast academic light theme (slate `#0f172a`, pure white `#ffffff`, subtle borders `#e2e8f0`) tailored for faculty review and viva demonstrations.

---

## 🚀 Setup & Execution Instructions (Windows PowerShell)

### 1. Prerequisites
* Python 3.10 – 3.12
* Node.js v18+

### 2. Backend Setup
Open a PowerShell terminal:
```powershell
# Navigate to the backend directory
cd C:\Users\DELL\Desktop\NLP\Scientific-Literature-Mining-System\backend

# Activate the existing virtual environment (or create with python -m venv venv)
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Run full automated test suite (19 unit & module tests)
.\venv\Scripts\python.exe -m pytest tests/test_backend.py -v

# Run reproducible academic benchmark evaluation
.\venv\Scripts\python.exe evaluation/evaluate_system.py

# Launch FastAPI backend server
.\venv\Scripts\uvicorn.exe main:app --port 8000
```
> Backend API: `http://localhost:8000`  
> Interactive OpenAPI / Swagger Docs: `http://localhost:8000/docs`

### 3. Frontend Setup
Open a second PowerShell terminal:
```powershell
# Navigate to the frontend directory
cd C:\Users\DELL\Desktop\NLP\Scientific-Literature-Mining-System\frontend

# Install node dependencies
npm install

# Test production build
npm run build

# Start Vite development server
npm run dev
```
> Access the web application at `http://localhost:5173`.

---

## 📊 Academic Evaluation & Benchmark Results

The system was evaluated against a curated gold-standard benchmark subset mapped to the **SciNLP / SciER** schema:

### 1. Information Extraction (Micro-Averaged Benchmark)
| Facet Type | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Methods & Architectures** | **1.000** | 0.857 | **0.923** | `High Precision` |
| **Datasets & Benchmarks** | **1.000** | 0.571 | **0.727** | `High Precision` |
| **Evaluation Metrics** | 0.800 | 0.800 | **0.800** | `Balanced` |
| **Quantitative Results** | **1.000** | 0.500 | **0.667** | `High Precision` |
| **Overall Extraction** | **0.810** (81.0%) | 0.531 (53.1%) | **0.642** (64.2%) | `Benchmark Report Generated` |

### 2. Relevance Screening Benchmark
* **Mean Reciprocal Rank (MRR):** `1.000` (100% of target documents ranked at position #1)
* **Top-1 Accuracy:** `100.0%`

### 3. Collection Q&A & Insufficient Evidence Rejection
* **Sufficiency Detection Accuracy:** `100.0%` (Correctly answered in-domain questions; successfully rejected all out-of-domain queries without hallucinating).
* **In-Domain Citation Rate:** `100.0%` (All answered questions link directly to verified source document names and 1-indexed page numbers).

---

## 🔬 Academic Honesty & Technical Disclaimers

1. **Pretrained Model Attribution:** We do not claim to have trained the underlying T5 model from scratch. We utilize the publicly available `Bashaarat1/t5-small-arxiv-summarizer` checkpoint and adapt it with custom discourse segmentation and reference truncation.
2. **Lexical vs. Semantic Retrieval:** Relevance screening and passage retrieval utilize TF-IDF and cosine similarity. While highly efficient, explainable, and local, this measures lexical overlap rather than deep semantic entailment.
3. **Pattern-Based Extraction:** Information extraction is performed using high-precision regex and context-directed heuristic rules. It does not pretend to be a heavy deep-learning NER transformer.
4. **Potential Research Gaps:** Unreported metrics or missing methods across papers are identified algorithmically as candidate literature gaps. They are explicitly flagged as requiring human scientific verification and must not be cited as definitive proof of novelty.

---

## 📚 References & Dependencies

* **Neural Summarization Model:** Bashaarat, *t5-small-arxiv-summarizer*, Hugging Face Model Hub, 2023. [https://huggingface.co/Bashaarat1/t5-small-arxiv-summarizer](https://huggingface.co/Bashaarat1/t5-small-arxiv-summarizer)
* **PyMuPDF:** Artifex Software, *PyMuPDF (fitz) PDF processing engine for Python*, 2024.
* **Scikit-Learn:** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011.
* **SciNLP Benchmark:** AKADDC, *SciNLP: A Benchmark for Scientific Entity and Relation Extraction*, 2023. [https://github.com/AKADDC/SciNLP](https://github.com/AKADDC/SciNLP)
* **SciER Dataset:** EMNLP 2024 Findings on Scientific Information Extraction. [https://aclanthology.org/2024.emnlp-main.726/](https://aclanthology.org/2024.emnlp-main.726/)
* **Semantic Scholar Academic Graph API:** Allen Institute for AI. [https://api.semanticscholar.org/api-docs/](https://api.semanticscholar.org/api-docs/)
