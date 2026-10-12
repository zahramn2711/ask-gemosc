import json
import time

from src.generate import NOT_FOUND, answer, call_llm, format_context
from src.retrieve import parse_filters, retrieve
from src.rerank import retrieve_reranked

with open("eval/questions.jsonl", encoding="utf-8") as f:
    QS = [json.loads(line) for line in f if line.strip()]

JUDGE = """You grade a question-answering system.
Given CONTEXT and ANSWER, reply with one word:
SUPPORTED if every claim in the answer is supported by the context,
UNSUPPORTED otherwise."""


def scene_of(chunk_id):
    return chunk_id.rsplit("_p", 1)[0]


def retrieval_metrics(retriever=retrieve, k=5):
    hits, rr, n = 0, 0.0, 0
    for item in QS:
        if not item["relevant"]:
            continue
        n += 1
        found = retriever(item["q"], k=k, **parse_filters(item["q"]))
        scenes = [scene_of(c["id"]) for c in found]
        ranks = [i for i, s in enumerate(scenes, 1) if s in item["relevant"]]
        if ranks:
            hits += 1
            rr += 1 / ranks[0]
        else:
            print(f"  MISS: {item['q'][:70]}")
    return hits / n, rr / n


def generation_metrics():
    supported = answered = wrongly_refused = refused_ok = unanswerable = 0
    log = []
    for item in QS:
        text, chunks = answer(item["q"])
        refused = NOT_FOUND in text
        if not item["relevant"]:
            unanswerable += 1
            refused_ok += refused
        elif refused:
            wrongly_refused += 1
            print(f"  WRONGLY REFUSED: {item['q'][:70]}")
        else:
            verdict = call_llm(f"CONTEXT:\n{format_context(chunks)}\n\nANSWER:\n{text}",
                               JUDGE, temperature=0).strip().upper()
            answered += 1
            supported += verdict.startswith("SUPPORTED")
            log.append({"q": item["q"], "answer": text, "verdict": verdict})
        time.sleep(8)   # stay under free-tier rate limits
    with open("eval/judge_log.jsonl", "w", encoding="utf-8") as f:
        for row in log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_answerable = answered + wrongly_refused
    return {
        "faithfulness": supported / max(answered, 1),
        "false_refusals": wrongly_refused / max(n_answerable, 1),
        "refusal_accuracy": refused_ok / max(unanswerable, 1),
    }


if __name__ == "__main__":
    print("Retrieval:")
    hit, mrr = retrieval_metrics(retriever=retrieve_reranked)
    print(f"Hit@5 {hit:.2f}  MRR {mrr:.2f}")
    print("\nGeneration (takes several minutes):")
    for name, value in generation_metrics().items():
        print(f"{name} {value:.2f}")