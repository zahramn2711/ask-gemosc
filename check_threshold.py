from src.evaluate import QS
from src.rerank import retrieve_reranked
from src.retrieve import parse_filters

for item in QS:
    chunks = retrieve_reranked(item["q"], k=5, **parse_filters(item["q"]))
    best = min(c["distance"] for c in chunks)
    kind = "answerable  " if item["relevant"] else "UNANSWERABLE"
    print(f"{best:.3f} | {kind} | {item['q'][:60]}")