import sys
import json
import sqlite3
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'backend')

from database import get_db, rows_to_list
from modules.collection_synthesis import generate_collection_brief

ws_id = "386e0060-79f9-4f5a-9503-0b08125c8d2e"

with get_db() as conn:
    ws = conn.execute("SELECT * FROM workspaces WHERE id = ?", (ws_id,)).fetchone()
    papers = conn.execute("SELECT * FROM papers WHERE workspace_id = ? ORDER BY filename ASC", (ws_id,)).fetchall()
    paper_ids = [p["id"] for p in papers]
    placeholders = ",".join("?" for _ in paper_ids)
    fields = conn.execute(f"SELECT * FROM extracted_fields WHERE paper_id IN ({placeholders})", paper_ids).fetchall()
    summaries = conn.execute(f"SELECT * FROM summaries WHERE paper_id IN ({placeholders})", paper_ids).fetchall()
    pages = conn.execute(f"SELECT paper_id, page_number, text FROM paper_pages WHERE paper_id IN ({placeholders}) ORDER BY paper_id, page_number ASC", paper_ids).fetchall()

print(f"Loaded workspace '{ws['title']}': {len(papers)} papers, {len(fields)} fields, {len(summaries)} summaries, {len(pages)} pages.")

rq = ws["research_question"] if ws and ws["research_question"] else ws["title"]

brief = generate_collection_brief(
    workspace_id=ws_id,
    research_question=rq,
    papers=rows_to_list(papers),
    extracted_fields=rows_to_list(fields),
    summaries=rows_to_list(summaries),
    pages=rows_to_list(pages),
)

now = datetime.utcnow().isoformat()
brief["created_at"] = now
brief["updated_at"] = now

brief_json = json.dumps(brief)
paper_count = brief["paper_count"]

with get_db() as conn:
    conn.execute(
        """INSERT OR REPLACE INTO collection_briefs
           (workspace_id, research_question, brief_json, paper_count, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (ws_id, rq, brief_json, paper_count, now, now)
    )

print("\nSUCCESS! Collection Research Brief regenerated and cached in DB.")
print("\n=== EXECUTIVE RESEARCH SYNTHESIS ===")
print(brief.get("executive_synthesis"))

print(f"\n=== KEY FINDINGS ACROSS PAPERS ({len(brief.get('key_findings', []))}) ===")
for kf in brief.get("key_findings", []):
    print(f"\n--- {kf['title']} ---")
    print("Synthesis:", kf["synthesis"])
    print("Trade-offs:", kf["trade_offs"])
    print(f"Evidence Citations ({len(kf['evidence_citations'])}):")
    for cit in kf["evidence_citations"]:
        print(f"  * {cit['filename']} (p.{cit['page_number']}): {cit['verbatim_passage'][:100]}...")

print(f"\n=== CROSS-PAPER COMPARISON MATRIX ({len(brief.get('cross_paper_matrix', []))} papers) ===")
for row in brief.get("cross_paper_matrix", []):
    print(f"\nPaper: {row['filename']} ({row['title'][:50]})")
    print(f"  Architecture: {row['approach_architecture']}")
    print(f"  Problem:      {row['research_problem'][:100]}...")
    print(f"  Contribution: {row['primary_contribution'][:100]}...")
    print(f"  Setting:      {row['experimental_setting']}")
    print(f"  Findings:     {row['major_findings']}")
    print(f"  Limitations:  {row['documented_limitations'][:100]}...")

print(f"\n=== POTENTIAL RESEARCH OPPORTUNITIES ({len(brief.get('research_opportunities', []))}) ===")
for opp in brief.get("research_opportunities", []):
    print(f"\nOpportunity: {opp['title']}")
    print(f"Caution:     {opp['caution_label']}")
    print(f"Hypothesis:  {opp['hypothesis']}")
    print(f"Rationale:   {opp['collection_rationale']}")
