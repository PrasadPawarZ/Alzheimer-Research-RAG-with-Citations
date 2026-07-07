"""
Core retrieval-augmented generation logic.

Design principles enforced here (see README "No hallucination" section):
- The LLM is ALWAYS shown only the retrieved chunks, never asked to use
  outside/general knowledge.
- The LLM is explicitly instructed to emit a fixed sentinel
  (NOT_COVERED_BY_DOCUMENTS) when the retrieved context doesn't answer
  the question, and we treat that sentinel as a hard stop before it
  reaches the user.
- We ALSO compute a numeric confidence score independent of the LLM
  (based on vector-similarity of the retrieved chunks). If that score is
  below CONFIDENCE_THRESHOLD we flag the answer for human review
  regardless of what the LLM said, because a low-similarity retrieval
  means the "grounding" itself is shaky even if the model sounds sure.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

import config
import llm_client

NOT_COVERED_SENTINEL = "NOT_COVERED_BY_DOCUMENTS"

_embedder: Optional[SentenceTransformer] = None
_client = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


@dataclass
class Citation:
    source: str
    doc_id: str
    page: int
    chunk_index: int
    snippet: str
    similarity: float


@dataclass
class AnswerResult:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    needs_human_review: bool = False
    covered: bool = True


def list_documents() -> List[str]:
    col = _get_collection()
    data = col.get(include=["metadatas"])
    doc_ids = sorted({m["doc_id"] for m in data["metadatas"]})
    return doc_ids


def retrieve(query_en: str, top_k: int = None, doc_id_filter: Optional[str] = None):
    top_k = top_k or config.TOP_K
    col = _get_collection()
    embedder = _get_embedder()
    query_emb = embedder.encode([query_en], normalize_embeddings=True).tolist()

    where = {"doc_id": doc_id_filter} if doc_id_filter else None
    results = col.query(
        query_embeddings=query_emb,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    citations = []
    if results["ids"] and results["ids"][0]:
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # chroma cosine "distance" -> similarity
            similarity = 1 - dist
            snippet = doc[:280] + ("..." if len(doc) > 280 else "")
            citations.append(Citation(
                source=meta["source"],
                doc_id=meta["doc_id"],
                page=meta["page"],
                chunk_index=meta["chunk_index"],
                snippet=snippet,
                similarity=round(similarity, 4),
            ))
    return citations


def _build_context_block(citations: List[Citation]) -> str:
    blocks = []
    for i, c in enumerate(citations, start=1):
        blocks.append(f"[{i}] Source: {c.source} (page {c.page})\n{c.snippet}")
    return "\n\n".join(blocks)


ANSWER_SYSTEM_PROMPT = f"""You are a research assistant answering questions ONLY from the numbered
document excerpts the user provides. Rules:
1. Use ONLY the provided excerpts. Never use outside knowledge, never guess.
2. Every factual sentence in your answer must be traceable to at least one excerpt.
   Cite excerpts inline using their bracket number, e.g. "CNNs reached 99% accuracy [2]."
3. If the excerpts do not contain enough information to answer the question,
   respond with EXACTLY this token and nothing else: {NOT_COVERED_SENTINEL}
4. Do not apologize, do not hedge with disclaimers beyond what rule 3 already covers.
5. Be concise and technical; this is for a researcher, not a general audience.
"""


def answer_question(query: str, top_k: int = None, doc_id_filter: Optional[str] = None) -> AnswerResult:
    from translate import detect_language, to_english, translate as translate_to

    orig_lang = detect_language(query)
    query_en = to_english(query, orig_lang)

    citations = retrieve(query_en, top_k=top_k, doc_id_filter=doc_id_filter)

    if not citations:
        msg = "The provided documents do not cover this question."
        return AnswerResult(
            answer=translate_to(msg, orig_lang),
            citations=[],
            confidence=0.0,
            needs_human_review=True,
            covered=False,
        )

    avg_similarity = sum(c.similarity for c in citations) / len(citations)
    top_similarity = citations[0].similarity

    context_block = _build_context_block(citations)
    user_prompt = f"Document excerpts:\n\n{context_block}\n\nQuestion: {query_en}\n\nAnswer:"

    raw_answer = llm_client.chat(ANSWER_SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=700)

    covered = NOT_COVERED_SENTINEL not in raw_answer
    if not covered:
        final_en = "The provided documents do not cover this question."
        citations_out = []  # don't show misleading citations for an uncovered question
        confidence = 0.0
    else:
        final_en = raw_answer
        citations_out = citations
        confidence = round((0.5 * top_similarity + 0.5 * avg_similarity), 4)

    needs_review = (not covered) or (confidence < config.CONFIDENCE_THRESHOLD)

    final_answer = translate_to(final_en, orig_lang)

    return AnswerResult(
        answer=final_answer,
        citations=citations_out,
        confidence=confidence,
        needs_human_review=needs_review,
        covered=covered,
    )


CONTRADICT_SYSTEM_PROMPT = f"""You are a scientific fact-checking assistant. You are given excerpts from
TWO different research papers on the same general topic. Decide whether the two papers
CONTRADICT each other, AGREE with each other, or the excerpts are NOT_ENOUGH_INFO to tell,
specifically regarding the topic given.

Rules:
1. Use ONLY the provided excerpts, nothing else.
2. Respond in this exact format:
VERDICT: <CONTRADICT | AGREE | NOT_ENOUGH_INFO>
REASONING: <2-5 sentences explaining your verdict, citing excerpts by bracket number and
which document they came from>
3. If the excerpts don't actually discuss the given topic, use NOT_ENOUGH_INFO.
"""


@dataclass
class ContradictionResult:
    verdict: str
    reasoning: str
    citations_doc_a: List[Citation]
    citations_doc_b: List[Citation]


def compare_documents(doc_id_a: str, doc_id_b: str, topic: str, top_k: int = 4) -> ContradictionResult:
    citations_a = retrieve(topic, top_k=top_k, doc_id_filter=doc_id_a)
    citations_b = retrieve(topic, top_k=top_k, doc_id_filter=doc_id_b)

    if not citations_a or not citations_b:
        missing = doc_id_a if not citations_a else doc_id_b
        return ContradictionResult(
            verdict="NOT_ENOUGH_INFO",
            reasoning=f"No relevant chunks were found for '{topic}' in document '{missing}'.",
            citations_doc_a=citations_a,
            citations_doc_b=citations_b,
        )

    block_a = _build_context_block(citations_a)
    n_a = len(citations_a)
    block_b_raw = citations_b
    blocks_b = []
    for i, c in enumerate(block_b_raw, start=n_a + 1):
        blocks_b.append(f"[{i}] Source: {c.source} (page {c.page})\n{c.snippet}")
    block_b = "\n\n".join(blocks_b)

    user_prompt = (
        f"Topic to compare: {topic}\n\n"
        f"--- Document A ({doc_id_a}) excerpts ---\n{block_a}\n\n"
        f"--- Document B ({doc_id_b}) excerpts ---\n{block_b}\n\n"
        f"Do these two documents CONTRADICT, AGREE, or is there NOT_ENOUGH_INFO on this topic?"
    )

    raw = llm_client.chat(CONTRADICT_SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=500)

    verdict = "NOT_ENOUGH_INFO"
    reasoning = raw
    for v in ("CONTRADICT", "AGREE", "NOT_ENOUGH_INFO"):
        if f"VERDICT: {v}" in raw.upper().replace(" ", " "):
            verdict = v
            break
    if "REASONING:" in raw:
        reasoning = raw.split("REASONING:", 1)[1].strip()

    return ContradictionResult(
        verdict=verdict,
        reasoning=reasoning,
        citations_doc_a=citations_a,
        citations_doc_b=citations_b,
    )
