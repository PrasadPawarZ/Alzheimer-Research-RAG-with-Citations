> THESE README WAS RECREATED BY AI.

# Potens AI/ML Internship Take-Home

Document Q&A with citations over Alzheimer disease research papers.

This project is my submission for the Potens AI/ML internship assignment, Q1: Document Q&A with Citations. I chose research papers around Alzheimer's disease detection because the domain is technical enough to make retrieval quality, citation discipline, and hallucination control matter.

The system was tested locally with 10 Alzheimer/MRI/deep-learning research papers. The paper PDFs are intentionally not committed to the repository because they may be third-party copyrighted material. The app expects PDF/TXT files to be placed in `papers/` before ingestion.

## Private Review Data Note

This repository is prepared for private recruiter review, but `.env`, raw research PDFs, generated ChromaDB files, and virtual environment files are still intentionally excluded from Git.

- `.env` can contain real API keys, so only `.env.example` is committed.
- Research papers may have third-party copyright restrictions, so only the `papers/` folder placeholder is committed.
- ChromaDB files are generated after ingestion and should be recreated locally.
- If private test papers or keys need to be shared, they should be shared separately and only when redistribution is permitted.
- If the repository is ever made public, it remains safe because secrets and third-party documents are not in Git history.

## Research Papers Used Locally

The following paper titles were used during local testing. I did not commit the PDF files, but these titles can be searched and downloaded from their original publisher or source pages where available:

- A novel CNN architecture for accurate early detection and classification of Alzheimer's disease using MRI data
- Accurate Detection of Alzheimer's Disease Using Lightweight Deep Learning Model on MRI Data
- Advanced interpretable diagnosis of Alzheimer's disease using SECNN-RF framework with explainable AI
- Advancements in deep learning for early diagnosis of Alzheimer's disease using multimodal neuroimaging challenges and future directions
- Alzheimer's Disease Detection Through Whole-Brain 3D-CNN MRI
- Classifying and diagnosing Alzheimer's disease with deep learning using 6735 brain MRI Images
- Deep Multi-Branch CNN Architecture for Early Alzheimer's Detection from Brain MRIs
- Deep learning techniques for Alzheimer's disease detection in 3D imaging A systematic review
- Intelligent Diagnosis of Alzheimer's Disease Based on Machine Learning
- MRI-Driven Alzheimer's Disease Diagnosis Using Deep Network Fusion and Optimal Selection of Feature

## My Role in This Submission

This is an AI-assisted build, but it was not submitted blindly. My work was focused on turning the assignment brief into a working, reviewable product:

- Interpreted the Potens AI/ML problem statement and selected Q1: Document Q&A with Citations.
- Chose Alzheimer's disease research papers as the test domain because they contain dense metrics, model comparisons, and citation-sensitive claims.
- Defined the expected user flow: add papers, ingest, ask questions, verify citations, compare documents, and review confidence.
- Reviewed and iterated on the AI-generated base application instead of leaving it as a simple scaffold.
- Checked that the project supports the required `/ask` and `/contradict` flows.
- Validated the README against the assignment requirements so the reviewer can quickly see what is implemented.
- Kept private keys, local vector databases, virtual environments, and third-party PDFs out of Git.
- Ran local tests and basic safety checks before preparing the submission.
- Documented known limitations honestly so the project can be discussed clearly in an interview.

I used AI tools as engineering assistants for scaffolding, refactoring, and documentation, but I am responsible for the final project direction, testing flow, requirement coverage, and submission quality.

## What It Does

- Ingests PDF/TXT documents from `papers/`.
- Extracts text page by page.
- Chunks documents using sentence-aware token windows.
- Embeds chunks with `sentence-transformers`.
- Stores vectors and metadata in ChromaDB.
- Answers questions through `/ask` with source, page, chunk, snippet, and confidence.
- Supports `/contradict` to compare two documents on a topic.
- Supports multilingual queries through language detection and optional translation at the boundary.
- Refuses or flags weak answers instead of silently hallucinating.
- Includes a Streamlit dashboard so reviewers can try the app without Postman.
- Falls back to extractive answers if no Gemini/Groq key is configured.

## Assignment Coverage

| Potens requirement | Implementation |
| --- | --- |
| Ingest, chunk, embed, and store documents | `ingest.py` extracts PDF/TXT text, applies token-aware chunking, embeds with SentenceTransformers, and stores chunks in ChromaDB. |
| Explain chunking strategy | Documented in the Chunking Strategy section below. |
| `/ask` with citations | `POST /ask` returns answer, coverage status, confidence, source file, page, chunk ID, and snippet citations. |
| `/contradict` between two documents | `POST /contradict` compares two document IDs on a supplied topic using retrieved evidence. |
| Multilingual query flow | `translate.py` detects language and optionally translates at the retrieval boundary when an LLM key is available. |
| Simple UI | `streamlit_app.py` provides an interactive reviewer dashboard. |
| No silent hallucination | Low evidence returns `covered=false` or `needs_human_review=true`; invalid citation output falls back to extractive evidence. |
| Stretch: confidence score | Retrieval-based confidence score plus `CONFIDENCE_THRESHOLD`. |
| Stretch: reranking | Hybrid vector plus keyword reranking. |
| Stretch: eval set | `eval/run_eval.py` and `eval/eval_set.json` provide a small test harness. |

## Quick Start

Double-click:

```bat
run.bat
```

Then choose:

```text
1. Start Streamlit dashboard
2. Start FastAPI server
3. Ingest papers
4. Reset vector DB and re-ingest
5. Run evaluation set
6. Open .env
```

The easiest flow is:

1. Put at least five PDF/TXT documents in `papers/`.
2. Double-click `run.bat`.
3. Choose `3` to ingest papers.
4. Choose `1` to open the Streamlit dashboard.

## Five-Minute Reviewer Demo

After ingestion, try these checks:

1. Ask a covered question, for example: `What accuracy values are reported for CNN models in Alzheimer's MRI detection?`
2. Ask an out-of-scope question, for example: `What is the best treatment plan for a patient?` The app should avoid inventing an answer.
3. Ask a multilingual question, for example in Hindi or Marathi, and confirm the answer stays citation-backed.
4. Open the citations and verify each one includes source, page, chunk, and snippet.
5. Use the Compare Papers tab or `/contradict` endpoint to compare two document IDs on a topic such as `reported accuracy`.

## Manual Setup

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Optional: add either a Gemini or Groq API key in `.env`.

```env
LLM_PROVIDER=auto
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

If no key is configured, the app still runs in `extractive` mode and returns the most relevant retrieved snippets with citations.

## Run Commands

Ingest documents:

```bat
python ingest.py --reset
```

Start Streamlit:

```bat
streamlit run streamlit_app.py
```

Start FastAPI:

```bat
uvicorn app:app --reload --port 8000
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Run tests:

```bat
python -m unittest discover -s tests -p "test_*.py"
```

## API Surface

### Health and Stats

```http
GET /health
GET /stats
GET /documents
```

### Retrieval

```http
POST /retrieve
{
  "query": "CNN accuracy for Alzheimer's MRI classification",
  "top_k": 5
}
```

### Ask

```http
POST /ask
{
  "query": "What accuracy values are reported for CNN models?",
  "top_k": 5,
  "doc_id": null
}
```

Response includes:

- `answer`
- `covered`
- `confidence`
- `needs_human_review`
- `answer_mode`
- `detected_language`
- `citation_check_passed`
- `citations`

### Contradiction Check

```http
POST /contradict
{
  "doc_id_a": "paper_one",
  "doc_id_b": "paper_two",
  "topic": "reported model accuracy"
}
```

### Upload and Ingest

```http
POST /upload
POST /ingest?reset=true
POST /clear
```

More request examples are in `api_examples.http`.

## Architecture

```text
PDF/TXT files
   |
   v
ingest.py
   - extracts page text
   - chunks text
   - creates metadata
   - embeds chunks
   |
   v
ChromaDB vector store
   |
   v
rag_core.py
   - hybrid retrieval
   - confidence scoring
   - citation validation
   - LLM or extractive answer mode
   |
   +--> FastAPI API
   |
   +--> Streamlit dashboard
```

## Chunking Strategy

The chunking code is in `text_utils.py`.

I used sentence-aware token chunking:

- Split text into sentences using punctuation boundaries.
- Pack sentences into chunks of about `220` tokens.
- Carry `40` tokens of overlap into the next chunk.
- Store page number, source file, document ID, chunk index, title, year, and token count as metadata.

Why this choice:

- Research papers often contain dense claims, metrics, and model names inside a sentence.
- Cutting mid-sentence can break the evidence needed for citation-backed answers.
- Token-sized chunks are more stable for LLM context than character-sized chunks.
- Overlap keeps claims readable when important context sits near a chunk boundary.

Config:

```env
CHUNK_SIZE_TOKENS=220
CHUNK_OVERLAP_TOKENS=40
```

## Retrieval Strategy

The retrieval code is in `rag_core.py`.

I used hybrid retrieval:

- Vector similarity from sentence-transformer embeddings.
- Keyword overlap score for exact terms like model names, datasets, accuracy, MRI, CNN, etc.
- Weighted reranking:

```text
hybrid_score = 0.72 * vector_similarity + 0.28 * keyword_score
```

This helps avoid a common RAG problem: vector search can miss exact technical terms, while keyword search alone misses semantic matches.

## Confidence Score

The confidence score is retrieval-based, not the LLM's self-confidence.

It combines:

- Best retrieved chunk score.
- Top-3 average score.
- Overall average score.
- Document diversity.

If confidence is below:

```env
CONFIDENCE_THRESHOLD=0.35
```

then the response is marked:

```text
needs_human_review = true
```

This is meant as a human-in-the-loop gate, not a guarantee of correctness.

## Hallucination Controls

I used multiple guardrails:

1. The LLM prompt says to answer only from retrieved excerpts.
2. The LLM must return `NOT_COVERED_BY_DOCUMENTS` if evidence is insufficient.
3. The app validates citation numbers in the generated answer.
4. If citation validation fails, the app falls back to extractive snippets.
5. If no LLM key is configured, the app does not fake generation. It returns retrieved evidence directly.
6. Low-confidence answers are flagged for human review.

## Multilingual Flow

The multilingual boundary is intentionally simple for a 24-hour build:

1. Detect query language with `langdetect`.
2. If an LLM key exists, translate non-English queries to English before retrieval.
3. Retrieve from the English-indexed document chunks.
4. Translate the final answer back to the original language.
5. Keep citations tied to the original source snippets.

If no LLM key exists, the app still runs but avoids translation and uses extractive retrieval.

## Streamlit UI

The dashboard includes:

- System status sidebar.
- Provider mode and confidence threshold.
- Ask tab with answer confidence, mode, coverage, and citation check.
- Upload and ingest tab.
- Reset and clear vector DB actions.
- Paper comparison tab.
- Document browser with chunk/page stats.
- Retrieval diagnostics.
- API examples.

## What Is Broken or Unfinished

This is the honest list I would improve with more time:

- No OCR yet. Scanned PDFs with image-only text will not ingest well.
- No production authentication on the API.
- No Docker setup yet.
- The eval set is small and mostly retrieval-focused.
- The contradiction endpoint depends on retrieved excerpts and can still miss conflicts if retrieval misses the right chunks.
- The fallback extractive mode is safe but less polished than LLM-generated answers.
- Translation quality depends on the configured LLM.
- Uploaded files are stored locally only; there is no cloud storage.
- There is no advanced reranker model yet, only hybrid vector plus keyword reranking.

## What I Would Build Next

1. Add OCR for scanned papers.
2. Add a cross-encoder reranker after initial retrieval.
3. Expand the eval set with ground-truth answers and citation checks.
4. Add Docker Compose for one-command setup.
5. Add API key auth for FastAPI.
6. Add duplicate document detection.
7. Add structured paper metadata extraction: title, authors, year, dataset, modality, model, reported metrics.
8. Add answer trace export so reviewers can inspect the full retrieval and prompt path.

## Project Structure

```text
app.py              FastAPI backend
streamlit_app.py    Streamlit dashboard
ingest.py           PDF/TXT ingestion and embedding pipeline
rag_core.py         Retrieval, confidence, citation validation, Q&A, contradiction
llm_client.py       Gemini/Groq/auto/extractive provider handling
translate.py        Language detection and translation boundary
text_utils.py       Chunking, keyword scoring, citation validation helpers
config.py           Environment-driven configuration
eval/               Small evaluation harness and sample eval set
tests/              Unit tests for pure utility logic
api_examples.http   Example API requests
run.bat             One-click Windows runner
```

## Submission Notes

- Repository name follows the requested format: `potens-intern-aiml-prasad-pawar`.
- I picked exactly one assignment: AI/ML Q1, Document Q&A with Citations.
- The local test documents were Alzheimer research papers.
- The repository intentionally excludes third-party PDFs and generated ChromaDB files.
- I can explain the retrieval, chunking, confidence score, and hallucination controls in review.

## AI Use Log

The assignment asks for an honest AI use log. Approximate usage:

| Tool | Approx. use | What I used it for |
| --- | ---: | --- |
| Claude AI | ~10-15 prompts | Initial base application scaffold: FastAPI routes, Streamlit starter, ingestion pipeline, and RAG skeleton. |
| ChatGPT | ~10-20 messages | Research support and understanding RAG design choices, chunking tradeoffs, confidence thresholds, and how to present the project clearly. |
| OpenAI Codex | ~40-60 tool-assisted turns | Refactored and extended the project: safe config, extractive fallback, hybrid retrieval, token chunking utilities, metadata, Streamlit dashboard, API additions, the Windows `run.bat` one-click runner, tests, validation, and README enhancement. |

The README was first outlined from my project understanding, assignment requirements, and local testing flow, then recreated and enhanced with Codex into a more comprehensive document with clearer setup steps, project details, requirement mapping, limitations, and AI-use disclosure.

I used AI tools as pair-programming assistants, but I reviewed the assignment requirements, selected the AI/ML Q1 approach, adapted the design, checked the code locally, and can explain the implementation decisions.
