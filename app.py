"""
FastAPI backend.

Run:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /health
    GET  /documents
    POST /ask         {"query": "...", "top_k": 5, "doc_id": null}
    POST /contradict   {"doc_id_a": "...", "doc_id_b": "...", "topic": "..."}
"""
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag_core

app = FastAPI(title="Alzheimer's Research RAG API", version="1.0")


class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    doc_id: Optional[str] = None  # restrict retrieval to one document


class CitationOut(BaseModel):
    source: str
    doc_id: str
    page: int
    chunk_index: int
    snippet: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    covered: bool
    confidence: float
    needs_human_review: bool
    citations: List[CitationOut]


class ContradictRequest(BaseModel):
    doc_id_a: str
    doc_id_b: str
    topic: str


class ContradictResponse(BaseModel):
    verdict: str
    reasoning: str
    citations_doc_a: List[CitationOut]
    citations_doc_b: List[CitationOut]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents():
    try:
        return {"documents": rag_core.list_documents()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        result = rag_core.answer_question(req.query, top_k=req.top_k, doc_id_filter=req.doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(
        answer=result.answer,
        covered=result.covered,
        confidence=result.confidence,
        needs_human_review=result.needs_human_review,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
    )


@app.post("/contradict", response_model=ContradictResponse)
def contradict(req: ContradictRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="topic must not be empty")
    try:
        result = rag_core.compare_documents(req.doc_id_a, req.doc_id_b, req.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ContradictResponse(
        verdict=result.verdict,
        reasoning=result.reasoning,
        citations_doc_a=[CitationOut(**c.__dict__) for c in result.citations_doc_a],
        citations_doc_b=[CitationOut(**c.__dict__) for c in result.citations_doc_b],
    )
