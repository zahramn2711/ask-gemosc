import time
from functools import lru_cache

import numpy as np

from src.generate import call_llm
from src.retrieve import _col, _model

HYDE_PROMPT = """You write short movie-script excerpts.
Given a question about a scene, write a plausible 80-120 word excerpt from that scene:
character names in capitals followed by a colon, their lines, and short action lines.
Write in English even if the question is in another language.
Reply with the excerpt only."""


@lru_cache(maxsize=None)
def fake_excerpt(question):
    """Ask Gemini once per question; repeated calls reuse the same excerpt."""
    time.sleep(4)   # stay under free-tier rate limits
    return call_llm(question, HYDE_PROMPT, temperature=0)


def retrieve_hyde(question, k=5, **filters):
    q = _model.encode(["query: " + question], normalize_embeddings=True)[0]
    h = _model.encode(["passage: " + fake_excerpt(question)], normalize_embeddings=True)[0]
    v = q + h
    v = (v / np.linalg.norm(v)).tolist()

    conditions = [{key: value} for key, value in filters.items() if value is not None]
    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif conditions:
        where = {"$and": conditions}

    res = _col.query(query_embeddings=[v], n_results=k, where=where)
    return [
        {"id": i, "text": d, "meta": m, "distance": dist}
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                 res["metadatas"][0], res["distances"][0])
    ]