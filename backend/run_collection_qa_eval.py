import requests
import json
import sqlite3

BASE_URL = "http://127.0.0.1:8000"
WORKSPACE_ID = "386e0060-79f9-4f5a-9503-0b08125c8d2e"
DB_PATH = "literature_mining.db"

QUESTIONS = [
    "What are the main methods and model architectures used across these papers?",
    "Which datasets and benchmarks are used across these papers?",
    "What evaluation metrics and quantitative results are reported?",
    "How do the models compare with each other across the collection?",
    "What limitations and constraints do the authors report across these papers?",
    "Based only on the uploaded papers, what potential research gaps or underexplored areas can you identify? For each suggestion, explain which evidence supports it and what cannot be concluded from this collection alone."
]

def verify_citation_in_db(conn, filename, page_number, text_snippet):
    # Check that text_snippet actually occurs in paper_pages for that filename and page_number
    # Use first 40 chars of snippet
    clean_snippet = text_snippet[:40].strip()
    row = conn.execute("""
        SELECT pp.text 
        FROM paper_pages pp
        JOIN papers p ON p.id = pp.paper_id
        WHERE p.filename = ? AND pp.page_number = ?
    """, (filename, page_number)).fetchone()
    
    if not row:
        return False, "Page not found in DB"
    
    # Check presence of a substantial substring
    words = [w for w in clean_snippet.split() if len(w) > 3]
    if not words:
        return True, "Verified"
    
    found_words = sum(1 for w in words if w.lower() in row[0].lower())
    match_ratio = found_words / len(words)
    if match_ratio >= 0.7:
        return True, f"Verified ({match_ratio*100:.0f}% match)"
    return False, f"Low match: {match_ratio*100:.0f}%"

def run_evaluation():
    conn = sqlite3.connect(DB_PATH)
    results = []

    print("=" * 80)
    print("RUNNING 6-QUESTION RESEARCH BENCHMARK ON COLLECTION")
    print(f"Workspace: {WORKSPACE_ID}")
    print("=" * 80)

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/6] Query: {q}")
        payload = {"question": q, "top_k": 6}
        resp = requests.post(f"{BASE_URL}/api/workspaces/{WORKSPACE_ID}/ask", json=payload)
        
        if resp.status_code != 200:
            print(f"ERROR: HTTP {resp.status_code} - {resp.text}")
            continue
            
        data = resp.json()
        citations = data.get("citations", [])
        
        # Verify provenance of citations against DB
        verified_cits = []
        for c in citations:
            fname = c.get("filename")
            pnum = c.get("page_number")
            text = c.get("text", "")
            is_valid, reason = verify_citation_in_db(conn, fname, pnum, text)
            verified_cits.append({
                "filename": fname,
                "page_number": pnum,
                "score": c.get("relevance_score"),
                "valid": is_valid,
                "reason": reason,
                "snippet": text[:140].strip()
            })
            
        res_summary = {
            "index": i,
            "question": q,
            "is_sufficient": data.get("is_sufficient"),
            "papers_consulted": data.get("papers_consulted"),
            "summary_of_methods": data.get("summary_of_methods"),
            "answer": data.get("answer"),
            "citations_count": len(citations),
            "citations": verified_cits
        }
        results.append(res_summary)
        
        print(f"  Sufficiency: {data.get('is_sufficient')}")
        print(f"  Papers Consulted: {data.get('papers_consulted')}")
        print(f"  Retrieved Papers: {data.get('summary_of_methods')}")
        print(f"  Total Citations: {len(citations)}")
        for vc in verified_cits:
            print(f"    - {vc['filename']} (p. {vc['page_number']}) [score: {vc['score']}] -> Provenance: {vc['reason']}")

    conn.close()
    
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\nBenchmark evaluation complete. Results written to benchmark_results.json")

if __name__ == "__main__":
    run_evaluation()
