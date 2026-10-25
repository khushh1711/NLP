import json

path = r'C:/Users/DELL/.gemini/antigravity-ide/brain/23b3c588-0f4b-4b89-a272-9afe72ad5b7b/.system_generated/logs/transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == 56:
            line56 = json.loads(line)
            content = line56.get('content', '')
            print("Length of line 56:", len(content))
            for chunk in content.split('\n\n'):
                if any(term in chunk.lower() for term in ['scitldr', 'pegasus', 'bart', 'longformer', 't5', 'six', 'question 1', 'question 2', 'discourse']):
                    print("--- CHUNK ---")
                    print(chunk[:1000])
