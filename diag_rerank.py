import contextlib
import io

from src.evaluate import retrieval_metrics
from src.rerank import retrieve_reranked

for k in (5, 20):
    with contextlib.redirect_stdout(io.StringIO()):   # hide MISS lines
        hit, mrr = retrieval_metrics(retriever=retrieve_reranked, k=k)
    print(f"Rerank Hit@{k}: {hit:.2f}  MRR {mrr:.2f}")