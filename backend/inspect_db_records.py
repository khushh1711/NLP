import sqlite3

conn = sqlite3.connect('literature_mining.db')
conn.row_factory = sqlite3.Row

papers = conn.execute(
    "SELECT id, filename, page_count, word_count, analysis_status FROM papers WHERE workspace_id = '386e0060-79f9-4f5a-9503-0b08125c8d2e'"
).fetchall()

for p in papers:
    msg = f"\n=== Paper: {p['filename']} (ID: {p['id']}, Pages: {p['page_count']}, Words: {p['word_count']}) ==="
    print(msg)
    
    # Check summaries
    s = conn.execute("SELECT overview, method, results, conclusion FROM summaries WHERE paper_id = ?", (p['id'],)).fetchone()
    if s:
        print(f"  Overview: {s['overview'][:80]}...")
        print(f"  Method:   {s['method'][:80]}...")
        print(f"  Results:  {s['results'][:80]}...")
        print(f"  Concl:    {s['conclusion'][:80]}...")
    
    # Check extracted fields
    fields = conn.execute(
        "SELECT field_type, value, normalized_value, page_number, evidence_text FROM extracted_fields WHERE paper_id = ? ORDER BY field_type",
        (p['id'],)
    ).fetchall()
    print(f"  Total Extracted Fields: {len(fields)}")
    by_type = {}
    for f in fields:
        ft = f['field_type']
        if ft not in by_type:
            by_type[ft] = []
        by_type[ft].append(f"{f['value']} (p.{f['page_number']})")
    for ft, vals in by_type.items():
        line = f"    - {ft} ({len(vals)}): {', '.join(vals[:8])}"
        print(line.encode('ascii', errors='replace').decode())
