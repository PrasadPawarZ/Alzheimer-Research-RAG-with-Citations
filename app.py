"""FastAPI backend for the Alzheimer research RAG application."""
import os
import shutil
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import config
import ingest as ingest_pipeline
import rag_core

app = FastAPI(title="Alzheimer's Research RAG API", version="1.1")


class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    doc_id: Optional[str] = None


class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    doc_id: Optional[str] = None


class CitationOut(BaseModel):
    source: str
    doc_id: str
    page: int
    chunk_index: int
    snippet: str
    similarity: float
    vector_similarity: float = 0.0
    keyword_score: float = 0.0
    title: str = ""
    year: str = ""
    chunk_tokens: int = 0


class AskResponse(BaseModel):
    answer: str
    covered: bool
    confidence: float
    needs_human_review: bool
    answer_mode: str
    detected_language: str
    citation_check_passed: bool
    citations: List[CitationOut]


class RetrieveResponse(BaseModel):
    citations: List[CitationOut]


class ContradictRequest(BaseModel):
    doc_id_a: str
    doc_id_b: str
    topic: str


class ContradictResponse(BaseModel):
    verdict: str
    reasoning: str
    answer_mode: str
    citations_doc_a: List[CitationOut]
    citations_doc_b: List[CitationOut]


class IngestResponse(BaseModel):
    chunks_ingested: int
    stats: dict


def _citations(items):
    return [CitationOut(**item.__dict__) for item in items]


@app.get("/health")
def health():
    return {"status": "ok", "summary": rag_core.database_summary()}


@app.get("/stats")
def stats():
    return rag_core.database_summary()


@app.get("/documents")
def documents():
    try:
        return {"documents": rag_core.document_stats()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        citations = rag_core.retrieve(req.query, top_k=req.top_k, doc_id_filter=req.doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return RetrieveResponse(citations=_citations(citations))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        result = rag_core.answer_question(req.query, top_k=req.top_k, doc_id_filter=req.doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return AskResponse(
        answer=result.answer,
        covered=result.covered,
        confidence=result.confidence,
        needs_human_review=result.needs_human_review,
        answer_mode=result.answer_mode,
        detected_language=result.detected_language,
        citation_check_passed=result.citation_check_passed,
        citations=_citations(result.citations),
    )


@app.post("/contradict", response_model=ContradictResponse)
def contradict(req: ContradictRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="topic must not be empty")
    try:
        result = rag_core.compare_documents(req.doc_id_a, req.doc_id_b, req.topic)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ContradictResponse(
        verdict=result.verdict,
        reasoning=result.reasoning,
        answer_mode=result.answer_mode,
        citations_doc_a=_citations(result.citations_doc_a),
        citations_doc_b=_citations(result.citations_doc_b),
    )


@app.post("/upload")
def upload_paper(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    os.makedirs(config.PAPERS_DIR, exist_ok=True)
    destination = os.path.join(config.PAPERS_DIR, filename)
    with open(destination, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"saved": filename, "path": destination}


@app.post("/ingest", response_model=IngestResponse)
def ingest(reset: bool = False):
    try:
        chunks = ingest_pipeline.ingest(config.PAPERS_DIR, config.CHROMA_DIR, reset=reset)
        rag_core.reset_runtime_cache()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestResponse(chunks_ingested=chunks, stats=rag_core.database_summary())


@app.post("/clear")
def clear():
    rag_core.clear_vector_store()
    return {"status": "cleared", "stats": rag_core.database_summary()}
