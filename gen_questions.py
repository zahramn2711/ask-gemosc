import json
import random
import time

from scene_copies import copies_of, load_index
from src.generate import call_llm

N = 15          # how many candidate questions to draft
random.seed(8)  # change the number for a new batch

PROMPT = """You write test questions for a search engine over movie scripts.
You get one scene excerpt. Write ONE question that this excerpt clearly answers.
Rules:
- Use your own words. Do not copy distinctive phrases or rare words from the excerpt.
- Do not mention the film title or any character name.
- The question must be answerable from this excerpt alone.
- Reply with the question only, in English."""

index = load_index()
by_id = index[0]

with open("eval/questions.jsonl", encoding="utf-8") as f:
    used = {s for line in f if line.strip() for s in json.loads(line)["relevant"]}

pool = [r for r in by_id.values()
        if r["id"].endswith("_p0")
        and r["id"].rsplit("_p", 1)[0] not in used
        and 60 <= len(r["text"].split()) <= 250
        and r["text"].count(":") >= 3]

with open("eval/candidates.jsonl", "w", encoding="utf-8") as out:
    for i, r in enumerate(random.sample(pool, N), 1):
        q = call_llm(r["text"], PROMPT, temperature=0.7).strip()
        relevant = copies_of(index, r["id"])
        out.write(json.dumps({"q": q, "relevant": relevant}, ensure_ascii=False) + "\n")
        print(f"\n[{i}] {q}")
        print(f"    {r['title']} ({r['year']}) - {r['heading']}")
        print(f"    {r['text'][:300]}")
        time.sleep(5)
print("\nCandidates saved to eval/candidates.jsonl")