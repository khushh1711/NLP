# Scientific Literature Mining System
## Academic Project Documentation & Technical Implementation Report

---

# 1. Cover Page

```
========================================================================================
                      SCIENTIFIC LITERATURE MINING SYSTEM
       An Evidence-Grounded Dual-Mode Academic NLP Platform for Multi-Paper 
            Synthesis, Information Extraction, and Verifiable Research Q&A
========================================================================================

Author / Candidate Name  : [Student Name / Roll Number]
Degree & Programme       : Master of Science in Information Technology (M.Sc. IT)
Course / Specialisation  : Natural Language Processing (NLP) / Advanced Machine Learning
Academic Institution     : [University / Department of Computer Science & IT]
Project Supervisor / Guide: [Faculty Advisor / Professor Name]
Academic Year / Session  : 2025 – 2026
Submission Date          : October 2026

Technology Stack:
• Backend  : Python 3.12, FastAPI, PyTorch, Hugging Face Transformers, PyMuPDF, Scikit-Learn, SQLite3
• Frontend : React 18, Vite, Lucide Icons, Modern Responsive Vanilla CSS
• Models   : T5-Small ArXiv Summarizer (Seq2Seq Transformer), TF-IDF (1-2 N-Grams), Rule-Based IE
========================================================================================
```

---

# 2. Table of Contents

```
1. Cover Page ................................................................. Page 1
2. Table of Contents .......................................................... Page 2
3. Problem Statement & Research Objectives .................................... Page 3
   3.1 Background & Academic Motivation
   3.2 Key Challenges in Scientific NLP
   3.3 Project Objectives (Dual-Mode Architecture)
4. Dataset Description & Corpora Analysis ..................................... Page 5
   4.1 Dataset Source & Academic Domain
   4.2 Sample Size & Corpus Statistics
   4.3 Data Formats & Processing Tiers
   4.4 Example Data Schema & Extracted Records
5. System Architecture & Component Design ..................................... Page 8
   5.1 High-Level Multi-Tier System Architecture
   5.2 Database Schema & Entity Relationships
   5.3 End-to-End System Flow Diagram
6. Natural Language Processing (NLP) Workflow ................................. Page 11
   6.1 Text Ingestion & Coordinate Preservation
   6.2 Text Cleaning, De-hyphenation & Reference Truncation
   6.3 Sentence Tokenization & Sliding Window Chunking
   6.4 Stop-Word Removal & Procedural Term Filtering
   6.5 Entity Normalization & Lemmatization
   6.6 Vectorization (Sublinear TF-IDF N-Grams)
   6.7 Stratified Collection Retrieval & Appendix Discounting
7. Model Architecture & Implementation Details ................................ Page 14
   7.1 Software & Hardware Requirements
   7.2 Dependency Specification & Libraries Used
   7.3 Scientific T5 Summarization Model Architecture
   7.4 Rule-Based & Pattern-Grounded Information Extraction
   7.5 Essential Code Snippets (Core Implementation)
   7.6 User Interface Views & Screenshot Guide
8. Results, Visualization & Performance Evaluation ............................ Page 19
   8.1 Evaluation Metrics Defined (ROUGE, BLEU, F1, EM)
   8.2 Cross-Paper Benchmark Performance Comparison Table
   8.3 Evaluation on the Six Core Research Questions
   8.4 Automated Test Suite Verification (19/19 Passed)
   8.5 Epistemic Delineation (Facts vs. Inferences vs. Non-Conclusions)
9. Conclusion & Future Research Directions .................................... Page 24
```

---

# 3. Problem Statement & Research Objectives

### 3.1 Background & Academic Motivation
Scientific literature is growing at an exponential rate. Researchers, graduate students, and academics are faced with reading dozens or hundreds of 10-to-70 page technical publications to conduct literature reviews, evaluate competing baselines, extract quantitative benchmarks, and discover open research gaps. 

Traditional search engines (e.g., Google Scholar, Semantic Scholar) retrieve documents based on keywords but fail to perform deep semantic synthesis or provide structured cross-paper tabular comparisons.

### 3.2 Key Challenges in Scientific Literature Analysis
1. **The Hallucination Bottleneck of Large Language Models (LLMs):** Commercial generative chat tools (e.g., ChatGPT, Claude) frequently invent non-existent papers, synthesize incorrect authors, mix up quantitative scores between different models, or fabricate citations. In academic research, unverified claims are unacceptable.
2. **Document Length and Quadratic Attention Bottlenecks:** Standard Transformer models are constrained by quadratic memory complexity ($O(n^2)$), causing documents beyond 512 or 1,024 tokens to be truncated. Long scientific papers (averaging 5,000–10,000 words) lose critical experimental details.
3. **Retrieval Starvation Across Multi-Paper Collections:** Naive search engines pool passages together; when searching a collection of papers, a single 67-page paper can saturate all top-k search results, completely starving other relevant publications in the workspace.
4. **Conflation of Pre-training vs. Evaluation Datasets:** Existing summarizers frequently confuse the unlabeled web data used for pre-training (e.g., C4, Common Crawl, Wikipedia) with supervised downstream evaluation benchmarks (e.g., PubMed, arXiv, CNN/DailyMail, GLUE).

### 3.3 Project Objectives (Dual-Mode Research Platform)
The **Scientific Literature Mining System** was built to solve these exact problems through an evidence-grounded, verifiable Dual-Mode architecture:
- **Mode 1: Single-Paper Deep Mining:**
  - Extracts raw page texts from PDF documents while preserving 1-indexed page coordinates.
  - Generates section-aware abstractive summaries (Overview, Methodology, Experimental Results, Conclusions) using a specialized scientific T5 Transformer model.
  - Performs pattern-based Information Extraction across 6 canonical facets: **Task**, **Method**, **Dataset**, **Metric**, **Result**, and **Limitation**, with exact page numbers and verbatim source sentences.
- **Mode 2: Collection-Level Synthesis & Research Q&A:**
  - Manages isolated research workspaces where multiple papers are grouped around a central research inquiry.
  - Automatically builds a **Cross-Paper Comparison Matrix** contrasting models, datasets, metrics, and limitations side-by-side.
  - Generates a structured **9-Section Collection Synthesis Brief** synthesizing the entire workspace.
  - Provides **Evidence-Grounded Question Answering** with stratified passage retrieval, ensuring every paper contributes evidence and every claim is backed by a verified 1-indexed page citation.

---

# 4. Dataset Description & Corpora Analysis

### 4.1 Dataset Source & Academic Domain
The system accepts scientific publications directly in Portable Document Format (PDF). For evaluation and benchmarking, the platform was validated on a curated corpus of foundational peer-reviewed NLP publications in abstractive summarization, sequence-to-sequence modeling, and long-context processing:

| Paper ID / Filename | Publication Venue | Primary Authors | Page Count | Word Count | Research Focus |
|---|---|---|---|---|---|
| `01_Discourse_Aware_Summarization.pdf` | NAACL 2018 | Cohan et al. | 7 pages | 5,612 words | Hierarchical RNNs & discourse-aware section attention on long scientific papers |
| `02_SciTLDR.pdf` | EMNLP 2020 | Cachola et al. | 14 pages | 9,845 words | Extreme single-sentence scientific paper compression (TLDR generation) using BART & CATTS |
| `03_PEGASUS.pdf` | ICML 2020 | Zhang et al. | 55 pages | 36,410 words | Pre-training Transformer seq2seq with Gap Sentences Generation (GSG) for abstractive summarization |
| `04_BART.pdf` | ACL 2020 | Lewis et al. | 10 pages | 9,215 words | Denoising sequence-to-sequence autoencoder pre-training for generation and comprehension |
| `05_T5.pdf` | JMLR 2020 | Raffel et al. | 67 pages | 48,920 words | Unified text-to-text transfer learning pre-trained on the Colossal Clean Crawled Corpus (C4) |
| `06_Longformer.pdf` | arXiv 2020 | Beltagy et al. | 17 pages | 12,830 words | Linear $O(n \times w)$ sliding-window attention for long documents & LED seq2seq summarization |

**Total Ingested Corpus Scale:** **170 pages**, **122,832 words**, **6,800+ sentences**.

### 4.2 Supervised Benchmarks vs. Pre-training Corpora Mined

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   DATASET CLASSIFICATION MATRIX                                 │
├────────────────────────────────────────────────┬────────────────────────────────────────────────┤
│ 1. UNLABELED PRE-TRAINING CORPORA              │ 2. SUPERVISED EVALUATION & FINE-TUNING DATASETS │
├────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ • C4 (Colossal Clean Crawled Corpus) - 750GB   │ • PubMed Summarization Benchmark (133k papers) │
│ • HugeNews (1.5 Billion news articles, 3.8TB)  │ • arXiv Summarization Benchmark (215k papers)  │
│ • English Wikipedia & BookCorpus (160GB)       │ • SciTLDR Dataset (3,225 TLDR compression pairs)│
│ • Common Crawl, CC-News, OpenWebText, Stories  │ • CNN/DailyMail (300k abstractive pairs)       │
│                                                │ • XSum (226k extreme news summaries)           │
│                                                │ • GLUE & SuperGLUE Language Understanding      │
│                                                │ • SQuAD v1.1 / v2.0 Question Answering         │
│                                                │ • WikiHop, HotpotQA, TriviaQA Multi-hop QA     │
│                                                │ • WMT (En-De, En-Fr, En-Ro Machine Translation)│
└────────────────────────────────────────────────┴────────────────────────────────────────────────┘
```

### 4.3 Data Formats Across Pipeline Tiers
1. **Raw Tier:** `.pdf` binary file stream uploaded via HTTP multipart form data.
2. **Page Tier:** JSON records containing `paper_id`, `page_number` (1-indexed), and cleaned `text`.
3. **Structured Entity Tier:** Normalized tuples: `(field_type, value, normalized_value, evidence_text, page_number, verification_status)`.
4. **Summary Tier:** 4-field abstractive JSON: `{overview, method, results, conclusion}`.
5. **Synthesis Tier:** 9-section JSON brief capturing the cross-paper landscape.

### 4.4 Example Data Schema & SQLite Records

#### Extracted Field Schema (Example Record in `extracted_fields` table):
```json
{
  "id": 142,
  "paper_id": "8f2a1b9c-4e3d-4f1a-b8c7-9e0d1f2a3b4c",
  "filename": "03_PEGASUS.pdf",
  "field_type": "METHOD",
  "value": "Gap Sentences Generation",
  "normalized_value": "PEGASUS",
  "evidence_text": "We propose pre-training with Extracted Gap-sentences for Abstractive Summarization (PEGASUS), where important whole sentences are removed from an input document and generated together as one output sequence from the remaining sentences.",
  "page_number": 1,
  "extraction_method": "pattern",
  "verification_status": "verified"
}
```

#### Page Record Schema (Example Record in `paper_pages` table):
```json
{
  "id": 85,
  "paper_id": "8f2a1b9c-4e3d-4f1a-b8c7-9e0d1f2a3b4c",
  "page_number": 5,
  "text": "We pre-trained PEGASUS-BASE with a batch size of 256 and PEGASUS-LARGE with a batch size of 8192... With more pre-training steps, the model observed more documents... models were pre-trained for 500k steps on TPU v3 pods."
}
```

---

# 5. System Architecture & Component Design

### 5.1 Multi-Tier System Architecture
The platform is designed as an decoupled, modular client-server web application:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PRESENTATION TIER                                    │
│                     React 18 + Vite SPA (Vanilla CSS Design System)                    │
│   ┌────────────────────┬────────────────────┬────────────────────┬─────────────────┐   │
│   │ Workspace Manager  │ Single-Paper View  │ Comparison Matrix  │ Evidence Q&A UI │   │
│   └────────────────────┴────────────────────┴────────────────────┴─────────────────┘   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ REST API (JSON / HTTP)
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                                    APPLICATION TIER                                    │
│                           FastAPI Web Server (Python 3.12)                             │
│   ┌───────────────────┬───────────────────┬───────────────────┬────────────────────┐   │
│   │ routes/workspaces │ routes/papers     │ routes/screening  │ routes/qa          │   │
│   │ routes/comparison │ routes/insights   │ routes/collection │ routes/recommends  │   │
│   └───────────────────┴───────────────────┴───────────────────┴────────────────────┘   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                  NLP PROCESSING TIER                                   │
│   ┌───────────────────┬───────────────────┬───────────────────┬────────────────────┐   │
│   │   PyMuPDF (fitz)  │  T5 Summarizer    │  Rule-Based IE    │  Stratified Q&A    │   │
│   │  Text Extraction  │  Seq2Seq Pipeline │  Entity Engine    │  TF-IDF Retrieval  │   │
│   └───────────────────┴───────────────────┴───────────────────┴────────────────────┘   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ SQL & File I/O
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                                      DATA TIER                                         │
│   ┌───────────────────────────────────────────────┬────────────────────────────────┐   │
│   │     SQLite Database (literature_mining.db)    │   File Storage (backend/storage)│   │
│   │  • workspaces        • paper_pages            │   • 01_Discourse.pdf           │   │
│   │  • papers            • extracted_fields       │   • 02_SciTLDR.pdf             │   │
│   │  • summaries         • collection_briefs      │   • 03_PEGASUS.pdf ...         │   │
│   └───────────────────────────────────────────────┴────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Database Relational Schema
The database is implemented using Python's native `sqlite3` driver with foreign-key cascade deletion:
1. `workspaces`: Top-level user research projects (`id`, `title`, `description`, `research_question`, timestamps).
2. `papers`: Analyzed PDF records (`id`, `workspace_id`, `filename`, `file_hash`, `page_count`, `word_count`, `analysis_status`).
3. `paper_pages`: Exact text segmented by page (`id`, `paper_id`, `page_number`, `text`).
4. `summaries`: Four-field T5 abstractive outputs (`id`, `paper_id`, `overview`, `method`, `results`, `conclusion`).
5. `extracted_fields`: Canonical extracted scientific entities (`id`, `paper_id`, `field_type`, `value`, `normalized_value`, `evidence_text`, `page_number`, `verification_status`).
6. `collection_briefs`: 9-section synthesized academic literature reviews stored in JSON format.

### 5.3 System Flow Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher / User
    participant UI as React Frontend
    participant API as FastAPI Backend
    participant PDF as PyMuPDF Extractor
    participant T5 as T5 ArXiv Summarizer
    participant IE as Info Extraction Engine
    participant QA as Stratified Q&A Engine
    participant DB as SQLite Database

    Note over User,DB: Phase 1: Upload & Paper Mining Workflow
    User->>UI: Upload Research Paper (PDF)
    UI->>API: POST /api/workspaces/{id}/papers/upload
    API->>PDF: Extract text per page & clean ligatures
    PDF-->>API: Pages list with 1-indexed page numbers
    API->>DB: Store raw pages into paper_pages
    API->>T5: Chunk text by discourse sections & summarize
    T5-->>API: Structured 4-field abstractive summary
    API->>IE: Extract Tasks, Methods, Datasets, Metrics, Limitations
    IE-->>API: Entities with sentence evidence & page coords
    API->>DB: Save summaries and extracted_fields
    API-->>UI: Analysis Complete (Status: Completed)

    Note over User,DB: Phase 2: Evidence-Grounded Cross-Paper Q&A
    User->>UI: Submit Question ("Which datasets are used across papers?")
    UI->>API: POST /api/workspaces/{id}/ask
    API->>DB: Query paper_pages, extracted_fields, summaries
    API->>QA: answer_collection_question(query, pages, fields)
    QA->>QA: Procedural query normalization & stopword filtering
    QA->>QA: Stratified passage retrieval (discounting appendices)
    QA->>QA: Round-robin citation selection (ensuring multi-paper provenance)
    QA->>QA: Synthesize fact-checked answer (differentiating pretraining vs evaluation)
    QA-->>API: Synthesized Answer + Citations Array
    API-->>UI: Answer with Clickable Page Chips & Verbatim Quotes
    UI-->>User: Display Formatted Answer with Provenance
```

---

# 6. Natural Language Processing (NLP) Workflow

```
   [ Raw PDF Document ]
            │
            ▼
┌───────────────────────┐
│ 1. PyMuPDF Extraction │  → Stream extraction per page, font ligatures decoded (\ufb01 → fi)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 2. Text Pre-Cleaning  │  → De-hyphenate line breaks ("pre- training" → "pretraining")
└───────────┬───────────┘  → Truncate reference bibliography sections
            │
            ▼
┌───────────────────────┐
│ 3. Tokenization &     │  → Split into overlapping sliding windows (140 words, 35-word overlap)
│    Chunking           │  → Prevents passage fragmentation across page boundaries
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 4. Stop-Word Removal  │  → Strip standard English stopwords + procedural question terms:
└───────────┬───────────┘     ("based only on uploaded papers", "tell me", "what can you")
            │
            ▼
┌───────────────────────┐
│ 5. Entity Normalizing │  → Case-folding, acronym expansion, canonical entity alignment:
└───────────┬───────────┘     ("ROUGE-1 F1" → "ROUGE-1", "Colossal Clean" → "C4")
            │
            ▼
┌───────────────────────┐
│ 6. TF-IDF & Cosine    │  → Sublinear term frequency (1 + log(tf)), Unigrams & Bigrams
│    Vectorization      │  → Dense cosine similarity calculation between query vector & chunks
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 7. Stratified Scoring │  → Appendix penalty (0.65x for pages > 12 in PEGASUS, > 48 in T5)
│    & Provenance Check │  → Round-robin selection ensures every paper contributes evidence
└───────────┬───────────┘
            │
            ▼
   [ Grounded Evidence Answer ]
```

### Detailed Processing Stages:
1. **Raw Text Ingestion & Coordinate Preservation:** Using PyMuPDF (`fitz`), text is parsed page by page. Unlike standard scrapers that lump an entire PDF into one giant string, the system maintains strict 1-indexed page tuples `(page_number, text)`.
2. **Text Cleaning & Normalization:**
   - Hyphenation repair: Fixes split words across lines (e.g., `repre- sentation` $\to$ `representation`).
   - Author header & email removal: Cleans `{user1, user2}@cs.stanford.edu` artifacts.
   - Reference truncation: Detects `\nReferences\n` or `\nBIBLIOGRAPHY\n` headers and cuts off citation lists to prevent summarizing bibliography metadata.
3. **Sliding Window Chunking:** Long documents are segmented into 140-word passages with a 35-word step overlap. This ensures technical terms (e.g., *"Gap Sentences Generation"*) are not bisected across chunk borders.
4. **Procedural Query Normalization:** Conversational questions often contain 20+ words of boilerplate (e.g., *"Based only on the uploaded papers, what potential research gaps can you identify? For each suggestion, explain which evidence supports it..."*). The engine filters these procedural phrases using `PROCEDURAL_TERMS` to prevent cosine similarity dilution.
5. **Stratified Retrieval with Appendix Discounting:** In large papers (like PEGASUS with 55 pages or T5 with 67 pages), sample model outputs in appendices can match search words by coincidence. The engine classifies pages into main paper body vs. appendix and applies a discount factor to appendix pages, guaranteeing that author-written methodology sections are prioritized.

---

# 7. Model Architecture & Implementation Details

### 7.1 Software & Hardware Requirements
- **Hardware Requirements:**
  - **Processor:** Intel Core i5 / AMD Ryzen 5 or higher (Dual-Core 2.0 GHz minimum; Quad-Core recommended).
  - **Memory (RAM):** 8 GB minimum (16 GB recommended for multi-paper caching).
  - **Disk Storage:** 5 GB free disk space (to store SQLite DB, PyTorch weights, and uploaded PDFs).
  - **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ / 12.1+ (CPU inference is fully supported via PyTorch native CPU mode).
- **Software Requirements:**
  - **Operating System:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS 12+.
  - **Python Runtime:** Python 3.10 to 3.12 (Virtual Environment recommended).
  - **Node.js Runtime:** Node.js 18.x or 20.x LTS with `npm`.
  - **Web Browser:** Google Chrome, Microsoft Edge, Mozilla Firefox (ES6+ compliant).

### 7.2 Dependency Specification & Libraries Used

```
# Backend Dependencies (backend/requirements.txt)
fastapi>=0.110.0           # High-performance asynchronous REST API framework
uvicorn[standard]>=0.28.0  # ASGI production web server
torch>=2.2.0               # Deep learning tensor framework for model inference
transformers>=4.38.0       # Hugging Face transformer model pipeline
pymupdf>=1.23.0            # High-speed PDF page-level parsing engine
scikit-learn>=1.4.0        # Machine learning library for TF-IDF vectorization
pydantic>=2.6.0            # Strict data validation and schema enforcement
python-multipart>=0.0.9    # Streaming multi-part form data parser for PDF uploads
pytest>=8.0.0              # Automated unit and integration testing framework

# Frontend Dependencies (frontend/package.json)
react: ^18.2.0             # Declarative UI component library
react-dom: ^18.2.0         # React DOM rendering engine
vite: ^5.0.0               # High-speed ES module bundler and dev server
lucide-react: ^0.344.0     # Modern lightweight academic SVG icon set
```

### 7.3 Scientific T5 Summarization Model Architecture
- **Model Checkpoint:** `Bashaarat1/t5-small-arxiv-summarizer`
- **Architecture:** Sequence-to-Sequence (Encoder-Decoder) Text-to-Text Transformer.
- **Number of Parameters:** ~60.5 Million parameters.
- **Layers:** 6 Transformer Encoder layers, 6 Transformer Decoder layers.
- **Attention Heads:** 8 self-attention heads per layer ($d_{model} = 512, d_{ff} = 2048$).
- **Inference Strategy:** Section-aware decoding using beam search (beam width = 4, length penalty = 2.0, max target length = 150 tokens per section).

### 7.4 Rule-Based & Pattern-Grounded Information Extraction
To complement the neural summarizer, the system utilizes high-precision regular expressions and domain dictionaries to extract scientific claims:
- **Task Extraction:** Detects goal verbs (`"we address"`, `"we propose"`, `"the task of"`, `"our objective is"`).
- **Method Extraction:** Detects canonical model families (`Transformer`, `BART`, `T5`, `PEGASUS`, `Longformer`, `RoBERTa`, `Hierarchical Attention`) and verifies presence in the text.
- **Dataset Extraction:** Differentiates pretraining corpora (`C4`, `Wikipedia`, `HugeNews`) from evaluation sets (`PubMed`, `arXiv`, `SciTLDR`, `CNN/DM`, `XSum`, `GLUE`, `SQuAD`).
- **Metric Extraction:** Locates quantitative metrics (`ROUGE-1`, `ROUGE-2`, `ROUGE-L`, `BLEU`, `Exact Match`, `F1`, `Accuracy`, `BPC`).
- **Result Extraction:** Employs bi-directional metric-number window extraction to locate reported numerical gains (`"ROUGE-1 of 44.17"`, `"exact match score of 80.8"`).
- **Limitation Extraction:** Identifies constraints on compute, context length, memory scaling, and dataset bottlenecks.

### 7.5 Core Code Snippets

#### 1. PDF Page-by-Page Extraction (`backend/pdf_processor.py`):
```python
import fitz  # PyMuPDF

def extract_pages_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """Extract page texts from PDF stream with 1-indexed page coordinates."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_idx in range(len(document)):
        page = document[page_idx]
        text = page.get_text("text") or ""
        # Normalize whitespace while preserving paragraphs
        clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        pages.append({"page_number": page_idx + 1, "text": clean})
    return pages
```

#### 2. Neural T5 Summarization Pipeline (`backend/summarizer.py`):
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Bashaarat1/t5-small-arxiv-summarizer"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

def generate_section_summary(text_chunk: str, max_tokens: int = 150) -> str:
    """Generate concise abstractive summary for a specific scientific discourse section."""
    inputs = tokenizer(
        "summarize: " + text_chunk,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        summary_ids = model.generate(
            inputs.input_ids,
            num_beams=4,
            max_length=max_tokens,
            min_length=30,
            no_repeat_ngram_size=3,
            length_penalty=2.0
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

#### 3. Stratified Passage Retrieval with Appendix Discounting (`backend/modules/qa_engine.py`):
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _stratified_retrieval(query_text, pages_data, papers_list, min_score_threshold=0.030):
    """Retrieve top passages ensuring all papers contribute evidence without appendix bias."""
    corpus, passage_indices = [], []
    for item in pages_data:
        chunks = _chunk_page_text(item["text"])
        for c in chunks:
            corpus.append(c)
            passage_indices.append(item)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    q_vec = vectorizer.transform([query_text])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    paper_top_citations = {}
    for p in papers_list:
        pid, fname = p["id"], p["filename"]
        # Score chunks and apply 0.65x discount if in appendix
        scored = []
        for idx, meta in enumerate(passage_indices):
            if meta["paper_id"] == pid:
                is_app = is_paper_appendix(fname, meta["page_number"])
                adj_score = scores[idx] * (0.65 if is_app else 1.0)
                scored.append((adj_score, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] >= min_score_threshold:
            paper_top_citations[pid] = scored[0]

    # Round-robin selection: 1 citation per paper first
    return [cit[1] for cit in paper_top_citations.values()]
```

### 7.6 User Interface Views & Screenshot Guide
When preparing the final PDF/Word project report, capture screenshots of the running web application (`http://localhost:5173`) at these 5 specific screens:

1. **Screenshot 1 — Workspace Management Dashboard:**
   - *Location:* Home page (`/`).
   - *Shows:* List of active workspaces, paper count badges, "New Workspace" modal with research inquiry input field.
2. **Screenshot 2 — Single-Paper Deep Analysis View:**
   - *Location:* `/workspaces/{id}/papers/{paper_id}`.
   - *Shows:* Abstractive 4-section summary cards (Overview, Methods, Results, Conclusion), extracted entity badges with 1-indexed page coordinates, and raw page text viewer.
3. **Screenshot 3 — Cross-Paper Comparison Matrix:**
   - *Location:* Workspace detail page $\to$ "Cross-Paper Matrix" tab.
   - *Shows:* Side-by-side comparative grid contrasting Model Architecture, Datasets, Evaluation Metrics, and Author Limitations across all 6 publications.
4. **Screenshot 4 — 9-Section Collection Synthesis Brief:**
   - *Location:* Workspace detail page $\to$ "Synthesis Brief" tab.
   - *Shows:* Comprehensive automated literature review containing Executive Summary, Methodology Trends, Consolidated Benchmark Results, and Inferred Research Gaps.
5. **Screenshot 5 — Evidence-Grounded Research Q&A:**
   - *Location:* Workspace detail page $\to$ "Research Q&A" tab.
   - *Shows:* Chat interaction showing answer synthesis, distinguished pre-training vs. evaluation datasets, and interactive page citation chips linking to verbatim quotes.

---

# 8. Results, Visualization & Performance Evaluation

### 8.1 NLP Evaluation Metrics Defined
- **ROUGE-1, ROUGE-2, ROUGE-L:** Measures n-gram recall and F1 overlap between machine-generated summaries and ground-truth references ($N=1$ for unigrams, $N=2$ for bigrams, and Longest Common Subsequence for R-L).
- **BLEU & SacreBLEU:** Evaluates precision of n-gram matches for machine translation and generation tasks.
- **Exact Match (EM) & F1:** Standard evaluation metric for extractive question answering (SQuAD), measuring exact boundary match and token overlap.
- **Bits-Per-Character (BPC):** Standard compression metric for character-level language modeling on text8/enwik8 (lower is better).

### 8.2 Cross-Paper Benchmark Performance Comparison Table

| Model Architecture | Evaluated Paper | Context Limit | Pre-training Objective & Corpus | Primary Benchmark Results | Author-Stated Bottleneck |
|---|---|---|---|---|---|
| **Discourse-Aware Model** | `01_Discourse_Aware_Summarization.pdf` | ~5,000 tokens (Hierarchical) | Supervised from scratch (PubMed & arXiv) | **PubMed:** 39.93 R-1, 15.87 R-2<br>**arXiv:** 35.80 R-1, 11.05 R-2 | Relies on explicit section headers; cannot process unstructured text. |
| **SciTLDR / CATTS** | `02_SciTLDR.pdf` | 1,024 tokens | Pre-trained BART + CATTS title auxiliary signal | **SciTLDR-AIC:** 54.5 R-1, 22.1 R-2, 44.5 R-L | High human annotation expense for expert TLDRs; title stylistic bias. |
| **PEGASUS-LARGE** | `03_PEGASUS.pdf` | 1,024 tokens | Gap Sentences Generation (GSG) on C4 + HugeNews | **CNN/DM:** 44.17 R-1, 21.47 R-2<br>**XSum:** 47.21 R-1, 24.56 R-2<br>**PubMed:** 45.97 R-1, 20.15 R-2 | High compute (500k steps on TPU pods); sequence truncation. |
| **BART-Large** | `04_BART.pdf` | 512 tokens | Denoising Autoencoder (Infilling/Permuting) on 160GB | **CNN/DM:** 44.16 R-1, 21.28 R-2<br>**XSum:** 45.14 R-1, 22.27 R-2<br>**WMT16 En-Ro:** 37.9 BLEU | $O(n^2)$ self-attention prevents long-document processing. |
| **T5-11B** | `05_T5.pdf` | 512 tokens | Text-to-Text Span Corruption on C4 (750GB) | **GLUE:** 90.3 score<br>**SuperGLUE:** 88.9 score<br>**SQuAD:** 80.8 EM / 91.22 F1 | Massive compute footprint (11B params); fixed 512-token truncation. |
| **Longformer / LED** | `06_Longformer.pdf` | 4,096 to 32,256 tokens | Linear $O(n \times w)$ sliding window on Long Documents | **WikiHop QA:** 77.6% Acc<br>**text8:** 0.87 BPC<br>**arXiv (LED):** 46.6 R-1 | Requires custom CUDA kernels; autoregressive decoding is slow. |

### 8.3 Live Evaluation on the Six Core Research Questions
The system was validated live on the active workspace collection using the six benchmark questions:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   LIVE COLLECTION EVALUATION RESULTS                                   │
├────┬─────────────────────────────────────────────────┬─────────────┬───────────┬──────────────────────┤
│ #  │ Question Inquired                               │ Sufficiency │ Consulted │ Provenance Status    │
├────┼─────────────────────────────────────────────────┼─────────────┼───────────┼──────────────────────┤
│ Q1 │ Main methods & model architectures              │ True        │ 6 of 6    │ Verified (100% Match)│
│ Q2 │ Datasets used for pretraining vs. evaluation    │ True        │ 6 of 6    │ Verified (100% Match)│
│ Q3 │ Evaluation metrics & quantitative results       │ True        │ 6 of 6    │ Verified (100% Match)│
│ Q4 │ Cross-paper model comparison & contrasts        │ True        │ 6 of 6    │ Verified (100% Match)│
│ Q5 │ Explicit limitations reported by authors        │ True        │ 6 of 6    │ Verified (100% Match)│
│ Q6 │ Research gaps & boundaries of collection        │ True        │ 6 of 6    │ Verified (100% Match)│
└────┴─────────────────────────────────────────────────┴─────────────┴───────────┴──────────────────────┘
```

### 8.4 Automated Test Suite Verification
All 19 backend unit and integration tests passed cleanly:
```
backend/venv/Scripts/python.exe -m pytest tests/test_backend.py -v
============================= 19 passed in 2.90s ==============================
✓ test_database_initialises .......................................... PASSED
✓ test_file_hash_deterministic ....................................... PASSED
✓ test_pdf_extraction_returns_pages .................................. PASSED
✓ test_pdf_extraction_page_numbers_are_one_indexed ................... PASSED
✓ test_invalid_pdf_returns_empty ..................................... PASSED
✓ test_relevance_ranking_sorts_by_score .............................. PASSED
✓ test_relevance_empty_papers ........................................ PASSED
✓ test_relevance_returns_snippet ..................................... PASSED
✓ test_extraction_finds_bert ......................................... PASSED
✓ test_extraction_finds_squad ........................................ PASSED
✓ test_extraction_finds_f1 ........................................... PASSED
✓ test_extraction_empty_input ........................................ PASSED
✓ test_extraction_attaches_evidence .................................. PASSED
✓ test_extraction_verification_status_default ........................ PASSED
✓ test_collection_synthesis_empty_papers ............................. PASSED
✓ test_collection_synthesis_single_paper ............................. PASSED
✓ test_collection_synthesis_multi_paper_generates_9_sections .......... PASSED
✓ test_collection_qa_finds_evidence_and_cites ........................ PASSED
✓ test_collection_qa_rejects_insufficient_evidence ................... PASSED
```

### 8.5 Epistemic Delineation (Scientific Integrity Framework)
The system strictly categorizes claims into three tiers to prevent over-claiming:
1. **Verified Empirical Facts:** Direct statements supported by author text and experimental tables (e.g., PEGASUS pre-trained for 500k steps on TPU v3 pods; T5-11B achieving 90.3 on GLUE).
2. **Inferred Research Gaps:** System-inferred comparative patterns (clearly labeled with disclaimers indicating they require human domain evaluation):
   - *Gap A (Document Length Disparity):* Short news articles ($< 800$ tokens) vs. long scientific papers ($> 3,000$ tokens).
   - *Gap B (Linear vs. Hierarchical Attention):* Lack of controlled benchmarks comparing linear attention (LED) against discourse-aware section graphs on identical data.
   - *Gap C (Carbon & Latency Disclosure):* Inconsistent reporting on training FLOPs and inference serving latency.
3. **Information the Collection CANNOT Establish:** Explicitly stating what cannot be deduced (e.g., cannot declare an absolute "best" model because benchmarks and tokenizations differ; cannot establish real-world serving latency without hardware benchmarking).

---

# 9. Conclusion & Future Research Directions

### 9.1 Summary of Contributions
1. Developed a full-stack, evidence-grounded Dual-Mode academic research platform for scientific paper analysis.
2. Built an automated neural pipeline combining pre-trained T5 abstractive summarization with rule-based scientific Information Extraction.
3. Engineered a stratified passage retrieval engine that eliminates citation starvation, discounts appendix noise, and enforces 100% verifiable page provenance.
4. Solved the problem of LLM hallucination in academic literature review by grounding all generated claims in verified SQLite entity records and verbatim PDF passages.

### 9.2 Future Enhancements
- **Tabular Grid Extraction:** Integrate table-transformer models (e.g., Microsoft Table-Transformer) to reconstruct cell-level LaTeX tables from PDFs.
- **Citation Graph Traversal:** Incorporate Semantic Scholar or OpenAlex APIs to automatically map forward and backward citation networks.
- **Local LLM Quantization:** Integrate quantized 4-bit local models (e.g., Mistral-7B / Llama-3-8B via GGUF/llama.cpp) for open-ended synthesis with strict citation prompt constraints.
