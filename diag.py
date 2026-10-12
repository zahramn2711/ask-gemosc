import contextlib
import io

from src.evaluate import retrieval_metrics

for k in (5, 20, 50, 100):
    with contextlib.redirect_stdout(io.StringIO()):   # hide the MISS lines
        hit, mrr = retrieval_metrics(k=k)
    print(f"Hit@{k}: {hit:.2f}")