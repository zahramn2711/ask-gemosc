import json
import re
from collections import defaultdict

THRESHOLD = 0.5   # two chunks count as the same scene if they share at least half their words


def scene_id(chunk_id):
    return chunk_id.rsplit("_p", 1)[0]


def film_of(chunk_id):
    return chunk_id.split("_", 1)[0]


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def load_index(path="data/processed/scenes.jsonl"):
    """Return (all chunks by id, chunks grouped by film)."""
    by_id, by_film = {}, defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_id[r["id"]] = r
            by_film[film_of(r["id"])].append(r)
    return by_id, by_film


def copies_of(index, chunk_id):
    """Scene IDs of all chunks in the same film whose text mostly matches this chunk."""
    by_id, by_film = index
    w = words(by_id[chunk_id]["text"])
    found = set()
    for other in by_film[film_of(chunk_id)]:
        ow = words(other["text"])
        if len(w & ow) / max(len(w | ow), 1) >= THRESHOLD:
            found.add(scene_id(other["id"]))
    return sorted(found)