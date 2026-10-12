import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src.rerank import retrieve_reranked
from src.retrieve import parse_filters

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
LLM = "gemini-3.5-flash-lite"
FALLBACK_LLM = "gemini-3.8-flash"   # used for the last attempt if the main model stays busy
MAX_DISTANCE = 0.6                  # good matches were 0.27-0.47 on Day 2
NOT_FOUND = "The corpus excerpts I found don't answer this question."

SYSTEM = f"""You are a research assistant for GeMoSC, a corpus of movie scripts from 2008-2017.
The context contains numbered excerpts. Each excerpt is one character's part of one scene:
their dialogue (NAME: ...) and stage directions ([Action] ...).

How to answer:
- Questions are usually about how characters talk or behave (themes, attitudes, patterns).
  Answer by describing what the excerpts show, even if they only partly answer the question.
- Use ONLY the numbered excerpts. Never use outside knowledge about the films or characters.
- After every claim, cite the supporting excerpt as [n]. Quote short phrases exactly as written in the excerpt (never translate a quote).- Ignore excerpts that are not relevant. If NONE of the excerpts is relevant,
  reply exactly: "{NOT_FOUND}" (in English, even for Italian questions)
- Answer in the language of the question (Italian or English).
- At most 150 words."""


def call_llm(contents, system, temperature=0.2, attempts=4):
    """Call Gemini, retrying when it is busy (503) or rate-limited (429)."""
    for attempt in range(attempts):
        model = LLM if attempt < attempts - 1 else FALLBACK_LLM
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system,
                                                   temperature=temperature),
            )
            return response.text or ""
        except (errors.ServerError, errors.ClientError) as e:
            busy = isinstance(e, errors.ServerError) or getattr(e, "code", None) == 429
            if not busy or attempt == attempts - 1:
                raise
            wait = 5 * 2 ** attempt          # 5, 10, 20 seconds
            print(f"  Gemini busy ({getattr(e, 'code', '?')}), retrying in {wait}s...")
            time.sleep(wait)


def format_context(chunks):
    blocks = []
    for n, c in enumerate(chunks, 1):
        m = c["meta"]
        blocks.append(f"[{n}] {m['title']} ({m['year']}) - {m['heading']}\n{c['text']}")
    return "\n\n".join(blocks)


def answer(question, k=5):
    chunks = retrieve_reranked(question, k=k, **parse_filters(question))
    if not chunks or min(c["distance"] for c in chunks) > MAX_DISTANCE:
        return NOT_FOUND, []
    prompt = (f"Context scenes:\n\n{format_context(chunks)}\n\nQuestion: {question}\n\n"
              "Write your answer in the same language as the question.")
    return call_llm(prompt, SYSTEM, temperature=0), chunks