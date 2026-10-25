"""
reprocess_all.py — Re-run full analysis on all papers in the database
using the updated section-aware summarizer and high-recall information extractor.
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_db, rows_to_list
from routes.papers import _run_full_analysis

def main():
    print("=" * 60)
    print("Re-processing all papers with enhanced extraction & summarization...")
    print("=" * 60)

    with get_db() as conn:
        papers = rows_to_list(conn.execute("SELECT id, filename, file_path FROM papers").fetchall())

    print(f"Found {len(papers)} papers to re-analyze:")
    for p in papers:
        pid = p["id"]
        fname = p["filename"]
        fpath = Path(p["file_path"])
        print(f"\n[*] Processing: {fname} (ID={pid})...")
        if not fpath.exists():
            print(f"[-] File not found on disk: {fpath}")
            continue

        _run_full_analysis(pid, fpath)
        print(f"[+] Successfully re-analyzed {fname}")

    print("\n" + "=" * 60)
    print("ALL PAPERS RE-ANALYZED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
