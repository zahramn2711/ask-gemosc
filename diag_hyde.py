import contextlib
import io

from src.evaluate import retrieval_metrics
from src.hyde import retrieve_hyde

for k in (5, 20, 100):
    with contextlib.redirect_stdout(io.StringIO()):   # hide MISS lines
        hit, mrr = retrieval_metrics(retriever=retrieve_hyde, k=k)
    print(f"HyDE Hit@{k}: {hit:.2f}  MRR {mrr:.2f}")