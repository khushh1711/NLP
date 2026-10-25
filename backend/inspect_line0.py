import json

path = r'C:/Users/DELL/.gemini/antigravity-ide/brain/23b3c588-0f4b-4b89-a272-9afe72ad5b7b/.system_generated/logs/transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    line0 = json.loads(f.readline())
    content = line0.get('content', '')
    print("Length of line 0:", len(content))
    # Look for question or test cases in line 0
    for chunk in content.split('\n\n'):
        if any(term in chunk.lower() for term in ['question', 'scitldr', 'pegasus', 'bart', 'longformer', 't5', 'discourse']):
            print("--- CHUNK ---")
            print(chunk[:1000])
