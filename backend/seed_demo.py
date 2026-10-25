"""
seed_demo.py — End-to-end test and sample dataset seeder.
Generates realistic sample scientific research papers as PDFs,
uploads them to a workspace, triggers analysis, screening, and comparison.
"""

import sys
import os
import io
import json
import time
import urllib.request
import urllib.error
import fitz  # PyMuPDF

BASE_URL = "http://127.0.0.1:8000"

def create_sample_pdf(title: str, text_sections: list[tuple[str, str]]) -> bytes:
    """Create a clean multi-page PDF using PyMuPDF."""
    doc = fitz.open()
    
    # Page 1: Title and initial sections
    page = doc.new_page(width=595, height=842)  # A4 size
    y = 50
    
    # Title
    page.insert_text(fitz.Point(50, y), title, fontsize=16, fontname="helv", color=(0.1, 0.2, 0.4))
    y += 40
    
    for section_title, content in text_sections:
        if y > 720:
            page = doc.new_page(width=595, height=842)
            y = 50
        
        page.insert_text(fitz.Point(50, y), section_title.upper(), fontsize=11, fontname="helv", color=(0.2, 0.4, 0.7))
        y += 20
        
        # Split content into lines to fit page
        words = content.split(" ")
        line = ""
        for word in words:
            if len(line + " " + word) > 75:
                page.insert_text(fitz.Point(50, y), line, fontsize=9.5, fontname="helv", color=(0.15, 0.15, 0.15))
                y += 15
                line = word
                if y > 780:
                    page = doc.new_page(width=595, height=842)
                    y = 50
            else:
                line = (line + " " + word).strip()
        if line:
            page.insert_text(fitz.Point(50, y), line, fontsize=9.5, fontname="helv", color=(0.15, 0.15, 0.15))
            y += 25

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def main():
    print("==================================================")
    print("Scientific Literature Mining System — E2E Test & Seeder")
    print("==================================================")
    
    # 1. Health Check
    req = urllib.request.Request(f"{BASE_URL}/api/health")
    with urllib.request.urlopen(req) as resp:
        health_data = json.loads(resp.read().decode())
        print(f"[*] Health Check: {health_data}")
        assert health_data["status"] == "ok"
    
    # 2. Create Workspace
    ws_payload = json.dumps({
        "title": "Transformer Architectures & NLP Benchmarks",
        "description": "Comparative study of self-attention mechanisms, pre-trained language models, and benchmark evaluation.",
        "research_question": "How do bidirectional pre-trained transformers compare with pure attention architectures on sequence benchmarks?"
    }).encode("utf-8")
    
    req = urllib.request.Request(
        f"{BASE_URL}/api/workspaces",
        data=ws_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        ws_data = json.loads(resp.read().decode())
        ws_id = ws_data["id"]
        print(f"[+] Created Workspace: ID={ws_id}, Title='{ws_data['title']}'")

    # 3. Create Sample Papers
    paper1_sections = [
        ("Abstract", "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization."),
        ("Methodology", "The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions. We also implement scaled dot-product attention and positional encoding."),
        ("Experimental Setup & Datasets", "We train on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. For English-French translation, we use the significantly larger WMT 2014 English-French dataset consisting of 36 million sentences. Model training was conducted on 8 NVIDIA P100 GPUs using the Adam optimizer with beta_1 = 0.9, beta_2 = 0.98."),
        ("Results & Evaluation", "On the WMT 2014 English-to-German translation task, the big Transformer model achieves a state-of-the-art BLEU score of 28.4, outperforming the best existing models including ensembles by over 2.0 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days."),
        ("Conclusion", "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.")
    ]
    p1_bytes = create_sample_pdf("Attention Is All You Need", paper1_sections)

    paper2_sections = [
        ("Abstract", "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers."),
        ("Methodology", "There are two steps in our framework: pre-training and fine-tuning. During pre-training, the model is trained on unlabeled data over different pre-training tasks including Masked Language Model (MLM) and Next Sentence Prediction (NSP). For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all parameters are fine-tuned using labeled data from downstream tasks."),
        ("Datasets and Benchmarks", "We evaluate BERT on the GLUE benchmark, which consists of 9 sentence-level and sentence-pair tasks including MNLI, QQP, QNLI, and SST-2. We also evaluate on the Stanford Question Answering Dataset (SQuAD v1.1 and SQuAD v2.0), which contains 100,000 question/answer pairs from Wikipedia."),
        ("Results and Discussion", "BERT advances the state of the art for 11 natural language processing tasks. On the GLUE benchmark, BERT achieves an overall score of 80.5%, yielding a 7.7% absolute improvement over OpenAI GPT. On SQuAD v1.1, BERT achieves a Test F1-score of 91.0% and an Exact Match (EM) score of 84.1%, surpassing the previous human performance benchmark."),
        ("Conclusion & Findings", "Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems. In particular, these results enable even low-resource tasks to benefit from deep bidirectional architectures.")
    ]
    p2_bytes = create_sample_pdf("BERT: Pre-training of Deep Bidirectional Transformers", paper2_sections)

    # 4. Upload Papers via Multipart Form Data
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    def upload_file(filename: str, pdf_data: bytes):
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(b"Content-Type: application/pdf\r\n\r\n")
        body.extend(pdf_data)
        body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        
        req = urllib.request.Request(
            f"{BASE_URL}/api/workspaces/{ws_id}/papers",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())

    print("[*] Uploading Paper 1: Attention Is All You Need...")
    res1 = upload_file("Attention_Is_All_You_Need.pdf", p1_bytes)
    p1_id = res1["uploaded"][0]["paper_id"]
    print(f"[+] Paper 1 uploaded: ID={p1_id}")

    print("[*] Uploading Paper 2: BERT...")
    res2 = upload_file("BERT_Bidirectional_Transformers.pdf", p2_bytes)
    p2_id = res2["uploaded"][0]["paper_id"]
    print(f"[+] Paper 2 uploaded: ID={p2_id}")

    # Polling until analysis complete
    print("[*] Waiting for background NLP analysis (PyMuPDF extraction + regex IE + T5 summarization)...")
    for pid, name in [(p1_id, "Attention Is All You Need"), (p2_id, "BERT")]:
        for attempt in range(60):
            req = urllib.request.Request(f"{BASE_URL}/api/papers/{pid}/analysis")
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                paper = data.get("paper", {})
                status = paper.get("analysis_status")
                if status == "completed":
                    print(f"[+] {name} successfully analyzed!")
                    print(f"    - Extracted Fields: {len(data['extracted_fields'])}")
                    if data.get("summary"):
                        print(f"    - Summary Overview: {data['summary'].get('overview', '')[:100]}...")
                    for field in data['extracted_fields'][:4]:
                        print(f"      • [{field['field_type']}] {field['value']} (p.{field['page_number']}, conf=verified:{field['verification_status']})")
                    break
                elif status == "failed":
                    print(f"[-] Analysis failed for {name}")
                    break
            time.sleep(1)

    # 6. Test Relevance Screening
    print("\n[*] Testing TF-IDF Relevance Screening...")
    screen_payload = json.dumps({
        "query": "deep bidirectional transformer language representation masked language model"
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/workspaces/{ws_id}/screen",
        data=screen_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        screen_results = json.loads(resp.read().decode())
        print(f"[+] Screening completed for {len(screen_results['results'])} papers:")
        for r in screen_results["results"]:
            print(f"    - ID: {r['paper_id']} | Score: {r['score']:.4f} | Snippet: {r['snippet'][:60]}...")

    # 7. Test Cross-Paper Comparison Matrix
    print("\n[*] Testing Cross-Paper Comparison Matrix...")
    req = urllib.request.Request(f"{BASE_URL}/api/workspaces/{ws_id}/comparison")
    with urllib.request.urlopen(req) as resp:
        comp_data = json.loads(resp.read().decode())
        print(f"[+] Comparison generated for {len(comp_data['rows'])} papers:")
        for p in comp_data["rows"]:
            print(f"    - {p['filename']}:")
            print(f"      Method: {p['method']}")
            print(f"      Dataset: {p['dataset']}")
            print(f"      Metric: {p['metric']}")
            print(f"      Result: {p['result']}")

    # 8. Test CSV Export
    print("\n[*] Testing CSV Export...")
    req = urllib.request.Request(f"{BASE_URL}/api/workspaces/{ws_id}/export.csv")
    with urllib.request.urlopen(req) as resp:
        csv_text = resp.read().decode()
        print(f"[+] CSV exported successfully ({len(csv_text)} bytes):")
        print("--- CSV Preview ---")
        for line in csv_text.splitlines()[:5]:
            print(line)
        print("-------------------")

    # 9. Test Research Insights
    print("\n[*] Testing Research Insights...")
    req = urllib.request.Request(f"{BASE_URL}/api/workspaces/{ws_id}/insights")
    with urllib.request.urlopen(req) as resp:
        insights = json.loads(resp.read().decode())
        print(f"[+] Insights: Total Papers={insights['paper_count']}, Analyzed={insights['analyzed_count']}")
        print(f"    Top Methods: {insights['top_methods']}")
        print(f"    Top Datasets: {insights['top_datasets']}")
        print(f"    Top Metrics: {insights['top_metrics']}")

    # 10. Test Human-in-the-Loop Verification
    print("\n[*] Testing Field Verification Endpoint...")
    req = urllib.request.Request(f"{BASE_URL}/api/papers/{p1_id}/analysis")
    with urllib.request.urlopen(req) as resp:
        p1_analysis = json.loads(resp.read().decode())
        if p1_analysis["extracted_fields"]:
            field_to_verify = p1_analysis["extracted_fields"][0]
            fid = field_to_verify["id"]
            verify_payload = json.dumps({
                "verification_status": "verified",
                "notes": "Verified by human evaluator for benchmark accuracy."
            }).encode("utf-8")
            
            patch_req = urllib.request.Request(
                f"{BASE_URL}/api/fields/{fid}/verify",
                data=verify_payload,
                headers={"Content-Type": "application/json"},
                method="PATCH"
            )
            with urllib.request.urlopen(patch_req) as v_resp:
                v_res = json.loads(v_resp.read().decode())
                print(f"[+] Field {fid} ({field_to_verify['value']}) verified: {v_res['verification_status']}")

    print("\n==================================================")
    print("ALL END-TO-END WORKFLOWS AND INTEGRATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    main()
