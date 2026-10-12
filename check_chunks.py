import json, random
import pandas as pd

rows = [json.loads(l) for l in open("data/processed/scenes.jsonl", encoding="utf-8")]
df = pd.DataFrame(rows)
print(len(df), "chunks from", df.title.nunique(), "films")
print(df.text.str.split().str.len().describe())          # words per chunk
print(df.gender.value_counts(), df.generation.value_counts(), sep="\n")
for r in random.sample(rows, 5):
    print("\n---", r["title"], r["year"], "|", r["heading"], "|", r["gender"], r["generation"])
    print(r["text"][:400])