import json
import urllib.request

WORKSPACE_ID = "386e0060-79f9-4f5a-9503-0b08125c8d2e"

QUESTIONS = [
    "What are the main methods and model architectures used across these papers?",
    "Which datasets and benchmarks are used across these papers?",
    "What evaluation metrics and quantitative results are reported?",
    "How do the models compare with each other across the collection?",
    "What limitations and constraints do the authors report across these papers?",
    "Based only on the uploaded papers, what potential research gaps or underexplored areas can you identify? For each suggestion, explain which evidence supports it and what cannot be concluded from this collection alone.",
]

print(f"=== TESTING EXISTING QA ON WORKSPACE: {WORKSPACE_ID} ===")
for i, q in enumerate(QUESTIONS, 1):
    print(f"\n[{i}/6] QUESTION: {q}")
    req = urllib.request.Request(
        f"http://localhost:8000/api/workspaces/{WORKSPACE_ID}/ask",
        data=json.dumps({"question": q, "top_k": 5}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read().decode('utf-8'))
        print(f"Sufficient: {data.get('is_sufficient')}")
        print(f"Papers consulted: {data.get('papers_consulted')}")
        citations = data.get('citations', [])
        print(f"Citations count: {len(citations)}")
        for c in citations:
            print(f"  - {c.get('filename')} (p.{c.get('page_number')}) [score: {c.get('relevance_score')}]: {c.get('text')[:80]}...")
        print("Answer preview:\n", data.get('answer')[:300])
    except Exception as e:
        print("Error:", e)
