import os
import glob

print("Scanning files for test questions...")
for root, dirs, files in os.walk("."):
    if "venv" in root or "node_modules" in root or ".git" in root or ".pytest_cache" in root:
        continue
    for f in files:
        if f.endswith(('.py', '.json', '.md', '.txt')):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8') as fp:
                    content = fp.read()
                    if "six" in content.lower() and "question" in content.lower():
                        print(f"Match in {path}")
            except Exception as e:
                pass
