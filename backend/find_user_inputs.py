import json

path = r'C:/Users/DELL/.gemini/antigravity-ide/brain/23b3c588-0f4b-4b89-a272-9afe72ad5b7b/.system_generated/logs/transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        content = str(data.get('content', ''))
        # Search for typical question phrases or question marks in USER_INPUT
        if data.get('type') == 'USER_INPUT':
            print(f"=== USER_INPUT at line {i} ===")
            print(content[:500])
            print("\n")
