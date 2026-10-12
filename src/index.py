import json
import os

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
COLLECTION = os.environ.get("COLLECTION", "scenes_base")
META_KEYS = ("title", "year", "genres", "character", "gender", "generation", "scene_index", "heading")


def main():
    model = SentenceTransformer(MODEL)
    client = chromadb.PersistentClient(path="chroma_db")
    col = client.get_or_create_collection(COLLECTION)
    with open("data/processed/scenes.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    batch = 256
    for i in tqdm(range(0, len(records), batch)):
        b = records[i:i + batch]
        vectors = model.encode(["passage: " + r["text"] for r in b],
                               normalize_embeddings=True).tolist()
        col.upsert(
            ids=[r["id"] for r in b],
            embeddings=vectors,
            documents=[r["text"] for r in b],
            metadatas=[{k: r[k] for k in META_KEYS} for r in b],
        )
    print("Indexed", col.count(), "chunks into", COLLECTION, "with", MODEL)


if __name__ == "__main__":
    main()