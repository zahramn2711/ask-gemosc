import json

from scene_copies import copies_of, load_index

# The chunk each generated question was written from (first generator run)
SOURCES = {
    "4191702_Norman Oppenheimer_s35", "2401878_Michael Stone_s10",
    "4622512_Billie Jean King_s80", "1473832_Jack_s118", "4209788_Molly Bloom_s12",
    "0968264_Female Guest 2_s33", "0499549_Jake Sully_s14", "1954843_Anton_s51",
    "4270516_Carla_s13", "0881891_Mary Horowitz_s135", "1065073_Mason_s154",
}

index = load_index()
path = "eval/questions.jsonl"
with open(path, encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

for item in items:
    source = next((s for s in item["relevant"] if s in SOURCES), None)
    if source:
        new = copies_of(index, source + "_p0")
        print(f"{len(item['relevant'])} -> {len(new)} | {item['q'][:60]}")
        item["relevant"] = new

with open(path, "w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")