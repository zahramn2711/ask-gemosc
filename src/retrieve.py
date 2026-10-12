import os
import re

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
COLLECTION = os.environ.get("COLLECTION", "scenes_base")
_model = SentenceTransformer(MODEL)
_col = chromadb.PersistentClient(path="chroma_db").get_or_create_collection(COLLECTION)

# Generation labels exactly as they appear in the GeMoSC table (English + Italian)
GENERATIONS = {
    "Gen Z": r"\bgen(?:eration|erazione)?[\s-]*z\b|\bzoomers?\b",
    "Millennial": r"\bmillennials?\b|\bgen(?:eration|erazione)?[\s-]*y\b",
    "Gen X": r"\bgen(?:eration|erazione)?[\s-]*x\b",
    "Boomer": r"\b(?:baby[\s-]*)?boomers?\b",
    "Silent": r"\bsilent generation\b|\bgenerazione silenziosa\b",
    "Lost Gen": r"\blost generation\b|\bgenerazione perduta\b",
}
FEMALE = r"\b(?:women|woman|female|females|girls?|donne|donna|femminil\w*)\b"
MALE = r"\b(?:men|man|male|males|boys?|uomini|uomo|maschil\w*)\b"


def _load_all(batch=5000):
    """Load every chunk's id, text, metadata and vector into memory for exact search."""
    ids, docs, metas, vecs = [], [], [], []
    for offset in range(0, _col.count(), batch):
        got = _col.get(include=["embeddings", "documents", "metadatas"],
                       limit=batch, offset=offset)
        ids += got["ids"]
        docs += got["documents"]
        metas += got["metadatas"]
        vecs.append(np.asarray(got["embeddings"], dtype=np.float32))
    return ids, docs, metas, np.vstack(vecs)


_IDS, _DOCS, _METAS, _VECS = _load_all()


def parse_filters(question):
    """Find generation, gender and year mentioned in the question."""
    filters = {}
    for label, pattern in GENERATIONS.items():
        if re.search(pattern, question, re.IGNORECASE):
            filters["generation"] = label
            break
    female = re.search(FEMALE, question, re.IGNORECASE)
    male = re.search(MALE, question, re.IGNORECASE)
    if female and not male:
        filters["gender"] = "female"
    elif male and not female:
        filters["gender"] = "male"
    m = re.search(r"\b(200[89]|201[0-7])\b", question)   # corpus spans 2008-2017
    if m:
        filters["year"] = int(m.group(1))
    return filters


def retrieve(question, k=5, **filters):
    """Exact search: compare the question with every chunk (no approximation)."""
    q = _model.encode(["query: " + question], normalize_embeddings=True)[0]
    scores = _VECS @ q                       # cosine similarity, since all vectors are normalized
    active = {key: value for key, value in filters.items() if value is not None}
    if active:
        keep = np.array([all(m.get(key) == value for key, value in active.items())
                         for m in _METAS])
        scores = np.where(keep, scores, -np.inf)
    top = np.argsort(-scores)[:k]
    return [
        {"id": _IDS[i], "text": _DOCS[i], "meta": _METAS[i],
         "distance": float(2 - 2 * scores[i])}   # same scale as Chroma's distances before
        for i in top if np.isfinite(scores[i])
    ]