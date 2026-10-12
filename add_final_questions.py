import json

PATH = "eval/questions.jsonl"

with open(PATH, encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

# (words that identify the English question, Italian version)
ITALIAN = [
    ("company representative make to the staff",
     "Cosa comunica un dirigente ai dipendenti sul futuro della rivista e dei loro posti di lavoro?"),
    ("suggest doing to their bosses",
     "Cosa propone un personaggio di fare ai propri capi dopo essere stato tradito?"),
    ("customer accidentally activate",
     "Quale oggetto accende per sbaglio un cliente mentre fa acquisti in un negozio?"),
    ("skier preparing to achieve",
     "Quale obiettivo sportivo vuole raggiungere una sciatrice prima di iscriversi a giurisprudenza?"),
]

UNANSWERABLE = [
    "What does Jack promise Rose while they are freezing in the ocean after the Titanic sinks?",
    "How does Darth Vader reveal to Luke that he is his father?",
    "What does Michael Corleone tell his brother Fredo after learning about his betrayal?",
]

new = []
for key, italian in ITALIAN:
    matches = [it for it in items if key in it["q"]]
    if len(matches) != 1:
        raise SystemExit(f"Expected 1 English question containing '{key}', found {len(matches)}")
    new.append({"q": italian, "relevant": matches[0]["relevant"]})
for q in UNANSWERABLE:
    new.append({"q": q, "relevant": []})

existing = {it["q"] for it in items}
new = [it for it in new if it["q"] not in existing]   # safe to run twice

with open(PATH, "w", encoding="utf-8") as f:
    for it in items + new:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"Added {len(new)} questions, total {len(items) + len(new)}")