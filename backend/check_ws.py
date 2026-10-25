import sqlite3

conn = sqlite3.connect('literature_mining.db')
conn.row_factory = sqlite3.Row

rows = conn.execute("SELECT * FROM papers WHERE workspace_id = '386e0060-79f9-4f5a-9503-0b08125c8d2e'").fetchall()
for r in rows:
    print(r['id'], r['filename'], r['created_at'], r['analysis_status'], r['page_count'])

ws = conn.execute("SELECT * FROM workspaces WHERE id = '386e0060-79f9-4f5a-9503-0b08125c8d2e'").fetchone()
print("\nWorkspace Details:")
print(dict(ws))
