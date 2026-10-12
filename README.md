# Ask GeMoSC

A bilingual (English/Italian) question-answering assistant over **GeMoSC**, the Generational Movie Script Corpus I built for my MSc thesis at the University of Turin. You ask how characters talk or behave, and it answers from the film scripts, cites every claim to a specific scene, and refuses when the corpus has no evidence.

The corpus covers films released between 2008 and 2017, with each character annotated for gender and generation (derived from the actor's year of birth). The assistant indexes **110,889 scene excerpts from 674 films**.

> **Demo video:** https://drive.google.com/file/d/1AFrVQLNsnpcrZ7VQTEIajTP18RlAhsbm/view?usp=sharing
## What it does

- **Cited answers.** Every claim ends with a citation like `[3]`, and the interface lists only the scenes the answer actually uses.
- **Filters from the question.** "Gen Z", "Boomer", "women", "uomini" or a year between 2008 and 2017 restricts the search to matching characters or films.
- **English or Italian.** Questions can be asked in either language; answers come back in the same language, while quotes stay in the original English of the script.
- **Refusal.** If none of the retrieved scenes is relevant, it says so instead of answering from general knowledge. Asked how Darth Vader reveals he is Luke's father, it refuses, even though _Star Wars: The Force Awakens_ (2015) is in the corpus and its scenes are retrieved.

Example questions that work well:

- How do parents talk to teenagers about technology?
- What announcement does a company representative make to the staff regarding their jobs and the upcoming company transition?
- Come parlano del lavoro le donne della generazione X?
- What do Boomer characters say about marriage?

## How it works

```
Question
  │
  ├─ parse filters (generation, gender, year) with regular expressions
  ├─ embed with multilingual-e5-base ("query: " prefix)
  ├─ exact cosine search over all 110,889 excerpts → top 50
  ├─ rerank with a multilingual cross-encoder → top 5
  └─ Gemini 3.5 Flash-Lite, grounded prompt → answer with [n] citations or refusal
```

**Chunking.** The source files store each character's lines separately, with a scene number on every line. One chunk is one character's part of one scene: their dialogue (`NAME: ...`) plus the stage directions in that file (`[Action] ...`). Long scenes are split into overlapping 350-word parts, and fragments under 15 words are dropped. Each chunk carries film, year, genres, character, gender and generation.

**Retrieval.** Embeddings come from `intfloat/multilingual-e5-base` and are stored in ChromaDB. Search is done exactly in memory (one matrix multiplication over 110,889 normalised vectors) rather than through Chroma's approximate index; the evaluation section explains why. A cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) re-scores the top 50 candidates, reading question and excerpt together.

**Generation.** The prompt tells the model to use only the numbered excerpts, cite after every claim, quote without translating, answer in the question's language, and reply with a fixed English refusal sentence when nothing is relevant. Temperature is 0. Calls retry on rate limits and server overload, with a fallback model on the last attempt.

**Interface.** Gradio chat app (`app.py`).

## Evaluation

### Test set

33 questions, each linked to the scene(s) that answer it:

| Type                   | Count | How it was built                                                                                                                           |
| ---------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Single-scene questions | 23    | Drafted by Gemini from 30 randomly sampled scenes, then reviewed by me: 7 rejected as too vague or not answered by the excerpt, 7 reworded |
| Written by hand        | 3     | Scenes found with a keyword search over the corpus                                                                                         |
| Italian                | 4     | Translations of English questions, to measure the cost of asking in another language                                                       |
| Unanswerable           | 3     | Famous scenes from films outside the corpus, plus Darth Vader                                                                              |

The same scene is stored once per character, so a correct answer can come from several files. While building the test set I found that scene numbers count scenes _within each character's file_, not within the film, so matching scene numbers across characters was wrong. Copies of the same scene are now identified by text overlap (at least half of their words in common).

### Metrics

- **Hit@k:** share of answerable questions with a correct scene among the top k results
- **MRR:** mean reciprocal rank of the first correct scene
- **Faithfulness:** share of answers an LLM judge rates as fully supported by the retrieved excerpts
- **False refusals:** share of answerable questions the assistant refused
- **Refusal accuracy:** share of unanswerable questions it refused

### Results

| Setup                                          | Hit@5    | Hit@20   | Hit@100 | MRR      | Faithfulness | False refusals | Refusal acc. |
| ---------------------------------------------- | -------- | -------- | ------- | -------- | ------------ | -------------- | ------------ |
| e5-small, Chroma approximate search (baseline) | 0.30     | 0.33     | 0.40    | 0.17     | 1.00         | 0.57           | 1.00         |
| + HyDE query rewriting                         | 0.13     | 0.20     | 0.37    | 0.11     | –            | –              | –            |
| e5-base, approximate search                    | 0.20     | 0.20     | 0.30    | –        | –            | –              | –            |
| e5-small, exact search                         | 0.30     | 0.43     | 0.47    | –        | –            | –              | –            |
| e5-base, exact search                          | 0.33     | 0.47     | 0.77    | –        | –            | –              | –            |
| **e5-base, exact search + reranker (final)**   | **0.40** | **0.53** | –       | **0.33** | **0.94**     | **0.40**       | **1.00**     |

With 30 answerable questions, one question is worth 0.033, so differences of one or two questions are within noise.

### What I learned from it

**The vector database was losing the best matches.** For one test question, the correct scene was the true closest match, yet it was missing from the top 5 and even the top 100. Asking Chroma for 300 results found it at rank 1. Chroma's approximate (HNSW) search explores the index more deeply when more results are requested, and at normal depths it was getting stuck. This corpus has many near-identical chunks, since every scene is stored once per character, which plausibly makes that worse. Switching to exact search fixed it, and it reversed an earlier conclusion: measured through the approximate search, e5-base looked _worse_ than e5-small. Measured exactly, it finds the right scene in the top 50 for 73% of questions against 43%.

**HyDE made retrieval worse.** Searching with a Gemini-written fake script excerpt instead of the question dropped Hit@5 from 0.30 to 0.13. Every chunk is already script-like text, so matching the style adds nothing, while the invented details pull the search toward the wrong scenes. (This was measured before the switch to exact search and was not re-run.)

**The reranker improved ranking.** Once the right scene was usually among the top 50 candidates, re-scoring them raised Hit@5 from 0.33 to 0.40 and roughly doubled MRR.

**Every false refusal is a retrieval miss.** When the correct scene is not retrieved, the model refuses because the excerpts it was given do not answer the question, which is what the prompt asks for. The remaining errors are retrieval errors, not generation errors.

**The judge made a mistake.** The LLM judge rated 17 of 18 answers as supported. I checked the one rejection by hand: both quotes appear word for word in the cited excerpts. The answer had said that no excerpt mentioned a magazine, a statement about missing evidence, which the strict judge counted as unsupported.

**Distance alone cannot detect unanswerable questions.** The closest-excerpt distance ranged from 0.287 to 0.442 for answerable questions and from 0.294 to 0.422 for unanswerable ones. The Darth Vader question scored 0.294 because _Star Wars_ scenes are in the corpus. Refusal has to come from the model reading the excerpts; the distance threshold (0.6) only catches unrelated input.

**Asking in Italian has a measurable cost.** The four Italian questions have the highest closest-excerpt distances in the test set (0.409–0.442), and the Italian version of one question ranked the correct scene 6th where the English version ranked it 1st.

## Connection to the thesis

GeMoSC(is an extention of my supervisor's work) was built for my MSc thesis (University of Turin, in progress) , which studies how the dialogue of film characters varies with generation, gender and genre using corpus-wide linguistic features.

The two approaches answer different kinds of questions. Semantic retrieval matches _what_ characters talk about, so this assistant can find and quote scenes, but it cannot measure _how_ a group talks. Questions such as "Do Millennials use more slang than Boomers?" need statistics over the whole corpus, not five excerpts. A planned next step is an agent that chooses between this retrieval tool and a tool that looks up corpus statistics.

## Limitations

- **Small test set.** 30 answerable questions, reviewed by one person.
- **Same model family as judge.** Gemini judges Gemini's answers; faithfulness was checked by hand only for the rejected answer.
- **Rule-based filters.** The generation and gender filters use regular expressions. A question that mentions "a woman" filters to female characters even when the relevant excerpt belongs to a male character in the same scene, and a question naming two generations filters to the first one only.
- **Sensitive to phrasing.** A shortened version of an example question retrieved different scenes and was refused.
- **Animated films are not yet excluded.** The thesis excludes them because generation is derived from the actor's age, which does not hold for voice roles; the retrieval filters do not apply that rule yet.
- **Noisy source text.** Dialogue lines in the source files sometimes contain stage directions.

## Run it locally

Requires Python 3.10+ and a Gemini API key.

```bash
git clone https://github.com/zahramn2711/ask-gemosc.git
cd ask-gemosc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key-here" > .env
```

Place the corpus in `data/raw/` (`thesis_data_-_Sheet1.csv` and the `movie_character_texts/` folder), then:

```bash
python -m src.chunk       # 110,889 chunks → data/processed/scenes.jsonl
python -m src.index       # embeddings into ChromaDB (about 80 minutes on a laptop CPU)
python app.py             # chat interface at http://127.0.0.1:7860
python -m src.evaluate    # full evaluation (about 8 minutes)
```

## Project structure

```
src/
  chunk.py      corpus → scene-level chunks with metadata
  index.py      embeddings → ChromaDB
  retrieve.py   filter parsing and exact search
  rerank.py     cross-encoder reranking
  generate.py   grounded prompt, citations, refusal, retries
  evaluate.py   retrieval and generation metrics
  hyde.py       HyDE experiment (not used in the final setup)
eval/
  questions.jsonl   the 33-question test set
app.py          Gradio interface
```

## Data

The dialogue comes from a public movie script corpus on Kaggle; character metadata (actor, gender, year of birth) was annotated by me from IMDb as part of the thesis. The scripts are copyrighted and are not included in this repository.

## Stack

Python · sentence-transformers · ChromaDB · NumPy · Gemini API · Gradio
