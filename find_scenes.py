import json
import re
import sys

TERMS = sys.argv[1:]
if not TERMS:
    sys.exit('Usage: python find_scenes.py word1 "a phrase" ...')
PATTERNS = [re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) for t in TERMS]

seen = set()
with open("data/processed/scenes.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        text = r["text"]
        if all(p.search(text) for p in PATTERNS):
            scene = r["id"].rsplit("_p", 1)[0]
            if scene in seen:
                continue
            seen.add(scene)
            m = PATTERNS[0].search(text)
            start = max(m.start() - 100, 0)
            snippet = text[start:m.end() + 150].replace("\n", " ")
            print(f"{scene} | {r['title']} ({r['year']}) | {r['character']} | {r['gender']} {r['generation']}")
            print("    ..." + snippet + "...")
            if len(seen) >= 20:
                break
print(f"\n{len(seen)} scenes shown (max 20)")