import re

import gradio as gr

from src.generate import NOT_FOUND, answer


def chat(message, history):
    text, chunks = answer(message)
    if not chunks or NOT_FOUND in text:
        return text                      # refusal: no sources, since nothing supports an answer
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    shown = [(n, c) for n, c in enumerate(chunks, 1) if n in cited] or list(enumerate(chunks, 1))
    sources = "\n".join(
        f"**[{n}]** {c['meta']['title']} ({c['meta']['year']}) - {c['meta']['heading']}"
        for n, c in shown
    )
    return f"{text}\n\n---\n**Sources**\n{sources}"


demo = gr.ChatInterface(
    chat,
    title="Ask GeMoSC",
    description="Ask about the GeMoSC movie script corpus (2008-2017) in English or Italian. "
                "Answers cite the scenes they are based on, and the assistant says so "
                "when the corpus has no evidence.",
    examples=[
        "How do parents talk to teenagers about technology?",
        "Come parlano del lavoro le donne della generazione X?",
        "What announcement does a company representative make to the staff regarding their jobs and the upcoming company transition?",
    ],
)

if __name__ == "__main__":
    demo.launch()