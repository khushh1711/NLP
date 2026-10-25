import json

path = r'C:/Users/DELL/.gemini/antigravity-ide/brain/23b3c588-0f4b-4b89-a272-9afe72ad5b7b/.system_generated/logs/transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if '01_Discourse' in line or 'SciTLDR' in line or '386e0060' in line:
            data = json.loads(line)
            c = str(data.get('content', ''))
            print(f"=== Line {i} ({data.get('type')}) ===")
            print(c[:400].encode('ascii', errors='replace').decode())
            print("\n")
