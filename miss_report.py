from src.evaluate import QS, scene_of
from src.retrieve import parse_filters, retrieve

for item in QS:
    if not item["relevant"]:
        continue
    filters = parse_filters(item["q"])
    found = retrieve(item["q"], k=300, **filters)
    scenes = [scene_of(c["id"]) for c in found]
    ranks = [i for i, s in enumerate(scenes, 1) if s in item["relevant"]]
    rank = ranks[0] if ranks else ">300"
    top = found[0]["meta"]
    print(f"{str(rank):>5} | {item['q'][:50]:50} | 1st: {top['title']} ({top['heading']})")