"""
summarizer.py — Scientific document summarization with section awareness.
Uses Hugging Face T5 with discourse extraction, bibliography filtering,
and grammatical post-processing.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Bashaarat1/t5-small-arxiv-summarizer"

print("Loading scientific T5 summarization model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
print("Scientific T5 summarization model loaded successfully.")


def clean_text(text: str) -> str:
    """Clean PDF extracted text while preserving punctuation."""
    if not text:
        return ""

    # Fix hyphenated line breaks (e.g., "repre- sentation" -> "representation")
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # Remove author emails, e.g. {devlin, mingweichang}@google.com
    text = re.sub(r'\{[^}]+\}@[a-zA-Z0-9.-]+', '', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def strip_references(text: str) -> str:
    """
    Truncate text before the References or Bibliography section
    to prevent summarizing citation lists.
    """
    ref_match = re.search(
        r'\n\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|Literature Cited)\s*\n',
        text
    )
    if ref_match and ref_match.start() > 2000:
        return text[:ref_match.start()].strip()
    
    # Alternative heuristic: find "References" as a standalone line or prominent heading
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        clean_line = line.strip()
        if clean_line in ["References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY"] and idx > 100:
            return "\n".join(lines[:idx]).strip()

    return text


def post_process_summary(text: str) -> str:
    """
    Clean up T5 generated text:
    - Capitalize first letter of each sentence
    - Remove dangling truncated trailing sentences
    - Strip inline math garbage
    """
    if not text:
        return ""

    text = text.strip()

    # Remove raw math equation tokens like "snull = S", "maxjiSTi + ETj"
    text = re.sub(r'\b[a-z]{1,4}\s*=\s*[a-zA-Z0-9_+^/*-]+\b', '', text)
    text = re.sub(r'\b\d{1,2}The [A-Z]', 'The ', text)  # remove footnote numbers glued to words

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove trailing dangling clauses (e.g., "...in this paper, we present a new method for implementing pre-")
    # If the text does not end with a sentence terminator, trim back to the last one.
    last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_punct > 40:
        text = text[:last_punct + 1]

    # Sentence capitalization
    sentences = re.split(r'(?<=[.!?])\s+', text)
    capitalized = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Capitalize first character
        s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
        capitalized.append(s)

    return " ".join(capitalized).strip()


def generate_summary(text: str, min_length: int = 35, max_length: int = 90) -> str:
    """Generate one summary from a text segment using T5."""
    if not text or len(text.split()) < 30:
        return ""

    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    raw_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    return raw_summary


def extract_section_by_keywords(text: str, keywords: list[str], fallback_start_pct: float, fallback_end_pct: float) -> str:
    """
    Find text belonging to a specific section by heading keywords,
    or fall back to document percentage slice.
    """
    for kw in keywords:
        pattern = re.compile(
            rf'(?:^|\n)\s*(?:\d+[\.\s]+)?(?:{re.escape(kw)})\b[^\n]*\n',
            re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            start_pos = match.end()
            # Grab next 1200 characters (approx 200 words) after heading
            segment = text[start_pos:start_pos + 1600].strip()
            if len(segment.split()) >= 40:
                return clean_text(segment)

    # Fallback to percentage slice of clean non-reference text
    words = text.split()
    total_words = len(words)
    start_idx = int(total_words * fallback_start_pct)
    end_idx = int(total_words * fallback_end_pct)
    return clean_text(" ".join(words[start_idx:end_idx]))


def summarize_text(text: str) -> dict:
    """
    Produces a 4-part structured scientific summary:
    - Overview (Objective, Problem formulation)
    - Method (Model, Architecture, Pipeline)
    - Results (Benchmark performance, Quantitative evaluation)
    - Conclusion (Key takeaways, synthesis - NOT references)
    """
    if not text or len(text.strip()) < 80:
        return {
            "overview": "",
            "method": "",
            "results": "",
            "conclusion": ""
        }

    # 1. Truncate text before References/Bibliography!
    body_text = strip_references(text)
    body_text = clean_text(body_text)

    # 2. Extract targeted sections
    # Overview: Abstract or Introduction
    overview_text = extract_section_by_keywords(
        body_text,
        keywords=["Abstract", "Introduction", "Overview"],
        fallback_start_pct=0.0,
        fallback_end_pct=0.18
    )

    # Method: Methodology, Architecture, Model, Proposed Approach
    method_text = extract_section_by_keywords(
        body_text,
        keywords=["Methodology", "Method", "Model Architecture", "Proposed Method", "Approach", "Architecture"],
        fallback_start_pct=0.18,
        fallback_end_pct=0.48
    )

    # Results: Experiments, Results, Evaluation, Performance
    results_text = extract_section_by_keywords(
        body_text,
        keywords=["Results", "Experimental Results", "Experiments", "Evaluation", "Quantitative Results"],
        fallback_start_pct=0.48,
        fallback_end_pct=0.80
    )

    # Conclusion: Conclusion, Discussion, Summary (strictly BEFORE references!)
    conclusion_text = extract_section_by_keywords(
        body_text,
        keywords=["Conclusion", "Conclusions", "Discussion", "Concluding Remarks", "Summary"],
        fallback_start_pct=0.80,
        fallback_end_pct=1.00
    )

    # 3. Generate section summaries
    overview_raw = generate_summary(overview_text, min_length=40, max_length=95)
    method_raw   = generate_summary(method_text, min_length=40, max_length=95)
    results_raw  = generate_summary(results_text, min_length=40, max_length=95)
    concl_raw    = generate_summary(conclusion_text, min_length=35, max_length=90)

    # 4. Clean, capitalize, and format
    overview_clean = post_process_summary(overview_raw)
    method_clean   = post_process_summary(method_raw)
    results_clean  = post_process_summary(results_raw)
    concl_clean    = post_process_summary(concl_raw)

    return {
        "overview":   overview_clean,
        "method":     method_clean,
        "results":    results_clean,
        "conclusion": concl_clean
    }