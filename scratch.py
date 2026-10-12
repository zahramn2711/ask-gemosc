import time

from src.generate import answer

for q in [
    "How do parents talk to teenagers about technology?",
    "How do Gen Z characters talk about money?",
    "Come parlano del lavoro le donne della generazione X?",
    "What does Darth Vader say to Luke on Cloud City?",
]:
    text, chunks = answer(q)
    print("\nQ:", q, "\nA:", text)
    for n, c in enumerate(chunks, 1):
        m = c["meta"]
        print(f"  [{n}] {m['title']} ({m['year']}) - {m['heading']}")
    time.sleep(5)   # stay under free-tier rate limits