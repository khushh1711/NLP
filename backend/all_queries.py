import sqlite3

conn = sqlite3.connect('literature_mining.db')
conn.row_factory = sqlite3.Row

rows = conn.execute("SELECT DISTINCT query FROM screening_results").fetchall()
print("All distinct queries in screening_results:")
for r in rows:
    print("-", r['query'])
