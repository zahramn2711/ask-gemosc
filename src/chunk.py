import json
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
TEXTS = RAW / "movie_character_texts"
OUT = Path("data/processed/scenes.jsonl")
LINE = re.compile(r"^\s*(\d+)\)\s*(\d+)\)\s*(text|dialog):\s*(.*)$")
MAX_WORDS = 350      # long scenes are split into parts of this size
OVERLAP = 50         # words shared between consecutive parts
MIN_WORDS = 15       # drop tiny fragments
LIMIT_FILMS = None     # quick test run; set to None for the full corpus


def clean(value, default="unknown"):
    return default if pd.isna(value) else str(value).strip()
def norm_gender(value):
    g = clean(value).lower()
    return {"m": "male", "f": "female"}.get(g, g)   # 'M' / 'm' -> 'male'

def load_metadata():
    df = pd.read_csv(RAW / "thesis_data_-_Sheet1.csv")
    df.columns = df.columns.str.strip()          # ' IMDb ID' -> 'IMDb ID'
    df = df.dropna(subset=["IMDb ID"])
    df["film_id"] = df["IMDb ID"].astype(int).astype(str).str.zfill(7)
    df["char_key"] = df["character"].astype(str).str.strip().str.lower()
    films = df.drop_duplicates("film_id").set_index("film_id")
    chars = df.drop_duplicates(["film_id", "char_key"]).set_index(["film_id", "char_key"])
    return films, chars


def parse_scenes(text, speaker):
    """Group lines by scene number into readable text."""
    scenes, last = {}, None
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            scene, _, kind, content = m.groups()
            prefix = f"{speaker.upper()}: " if kind == "dialog" else "[Action] "
            last = int(scene)
            scenes.setdefault(last, []).append(prefix + content.strip())
        elif line.strip() and last is not None:
            scenes[last][-1] += " " + line.strip()   # continuation of previous line
    return {s: "\n".join(lines) for s, lines in scenes.items()}


def split_long(scene):
    words = scene.split()
    if len(words) <= MAX_WORDS:
        return [scene]
    parts, step = [], MAX_WORDS - OVERLAP
    for i in range(0, len(words), step):
        parts.append(" ".join(words[i:i + MAX_WORDS]))
        if i + MAX_WORDS >= len(words):
            break
    return parts


def main():
    films, chars = load_metadata()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in TEXTS.iterdir() if p.is_dir())
    if LIMIT_FILMS:
        folders = folders[:LIMIT_FILMS]

    n_chunks, n_films, skipped = 0, 0, []
    with OUT.open("w", encoding="utf-8") as f:
        for folder in folders:
            film_id = folder.name.rsplit("_", 1)[-1]
            if film_id not in films.index:
                skipped.append(folder.name)
                continue
            film = films.loc[film_id]
            year = int(film["year"]) if not pd.isna(film["year"]) else 0
            n_films += 1
            for txt in sorted(folder.glob("*_text.txt")):
                name = txt.stem.removesuffix("_text")
                key = (film_id, name.strip().lower())
                person = chars.loc[key] if key in chars.index else None
                text = txt.read_text(encoding="utf-8", errors="ignore")
                for s_idx, scene in parse_scenes(text, name).items():
                    for p_idx, part in enumerate(split_long(scene)):
                        if len(part.split()) < MIN_WORDS:
                            continue
                        record = {
                            "id": f"{film_id}_{name}_s{s_idx}_p{p_idx}",
                            "text": part,
                            "title": clean(film["title"]),
                            "year": year,
                            "genres": clean(film["genres"]),
                            "character": name,
                            "gender": norm_gender(person["gender"]) if person is not None else "unknown",
                            "generation": clean(person["generation"]) if person is not None else "unknown",
                            "scene_index": s_idx,
                            "heading": f"{name}, scene {s_idx}",
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        n_chunks += 1

    print(f"Wrote {n_chunks} chunks from {n_films} films to {OUT}")
    if skipped:
        print(f"{len(skipped)} folders had no match in the table, e.g. {skipped[:3]}")


if __name__ == "__main__":
    main()