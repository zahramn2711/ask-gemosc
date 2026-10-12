from sentence_transformers import CrossEncoder

from src.retrieve import retrieve

_reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")   # multilingual


def retrieve_reranked(question, k=5, fetch=50, **filters):
    """Get `fetch` candidates with embeddings, re-score them with the cross-encoder, keep the best k."""
    candidates = retrieve(question, k=fetch, **filters)
    if not candidates:
        return []
    scores = _reranker.predict([(question, c["text"]) for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [candidates[i] for i in order[:k]]