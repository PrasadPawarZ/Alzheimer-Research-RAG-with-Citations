"""Core retrieval-augmented generation logic."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

import config
import llm_client
from text_utils import keyword_overlap_score, validate_citation_numbers

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


def reset_runtime_cache() -> None:
    global _client, _collection
    _client = None
    _collection = None


@dataclass
class Citation:
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


@dataclass
class AnswerResult:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    needs_human_review: bool = False
    covered: bool = True
    answer_mode: str = "llm"
    detected_language: str = "en"
    citation_check_passed: bool = True


@dataclass
class ContradictionResult:
    verdict: str
    reasoning: str
    citations_doc_a: List[Citation]
    citations_doc_b: List[Citation]
    answer_mode: str = "llm"


def _collection_rows(doc_id_filter: Optional[str] = None) -> List[Dict]:
    col = _get_collection()
    data = col.get(include=["documents", "metadatas"])
    rows = []
    for row_id, document, metadata in zip(
        data.get("ids") or [],
        data.get("documents") or [],
        data.get("metadatas") or [],
    ):
        if doc_id_filter and metadata.get("doc_id") != doc_id_filter:
            continue
        rows.append({"id": row_id, "document": document, "metadata": metadata})
    return rows


def list_documents() -> List[str]:
    return [item["doc_id"] for item in document_stats()]


def document_stats() -> List[Dict]:
    stats: Dict[str, Dict] = {}
    for row in _collection_rows():
        meta = row["metadata"]
        doc_id = meta.get("doc_id", "")
        item = stats.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "source": meta.get("source", ""),
                "title": meta.get("title", doc_id),
                "year": meta.get("year", ""),
                "chunks": 0,
                "pages": set(),
                "tokens": 0,
            },
        )
        item["chunks"] += 1
        item["pages"].add(meta.get("page", 0))
        item["tokens"] += int(meta.get("chunk_tokens", 0) or 0)

    result = []
    for item in stats.values():
        pages = sorted(page for page in item.pop("pages") if page)
        item["page_count"] = len(pages)
        item["page_range"] = f"{pages[0]}-{pages[-1]}" if pages else ""
        item["avg_chunk_tokens"] = round(item["tokens"] / item["chunks"], 1) if item["chunks"] else 0
        result.append(item)
    return sorted(result, key=lambda value: value["source"].lower())


def database_summary() -> Dict:
    docs = document_stats()
    return {
        "documents": len(docs),
        "chunks": sum(doc["chunks"] for doc in docs),
        "pages": sum(doc["page_count"] for doc in docs),
        "provider": llm_client.provider_status().__dict__,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
    }


def clear_vector_store() -> None:
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    reset_runtime_cache()


def retrieve(query_en: str, top_k: int = None, doc_id_filter: Optional[str] = None) -> List[Citation]:
    top_k = top_k or config.TOP_K
    all_rows = _collection_rows(doc_id_filter)
    if not all_rows:
        return []

    candidate_k = min(max(top_k * config.VECTOR_POOL_MULTIPLIER, top_k), len(all_rows))
    by_id: Dict[str, Dict] = {
        row["id"]: {
            **row,
            "vector_similarity": 0.0,
            "keyword_score": keyword_overlap_score(query_en, row["document"]),
        }
        for row in all_rows
    }

    col = _get_collection()
    embedder = _get_embedder()
    query_emb = embedder.encode([query_en], normalize_embeddings=True).tolist()
    where = {"doc_id": doc_id_filter} if doc_id_filter else None
    vector_results = col.query(
        query_embeddings=query_emb,
        n_results=candidate_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    for row_id, document, metadata, distance in zip(
        vector_results.get("ids", [[]])[0],
        vector_results.get("documents", [[]])[0],
        vector_results.get("metadatas", [[]])[0],
        vector_results.get("distances", [[]])[0],
    ):
        item = by_id.setdefault(
            row_id,
            {
                "id": row_id,
                "document": document,
                "metadata": metadata,
                "keyword_score": keyword_overlap_score(query_en, document),
            },
        )
        item["vector_similarity"] = max(0.0, min(1.0, 1 - distance))

    ranked = []
    for item in by_id.values():
        vector_score = item.get("vector_similarity", 0.0)
        keyword_score = item.get("keyword_score", 0.0)
        if vector_score <= 0 and keyword_score <= 0:
            continue
        rerank_score = (
            config.VECTOR_WEIGHT * vector_score
            + config.KEYWORD_WEIGHT * keyword_score
        )
        ranked.append((rerank_score, vector_score, keyword_score, item))

    ranked.sort(key=lambda value: (value[0], value[1], value[2]), reverse=True)

    citations = []
    for rerank_score, vector_score, keyword_score, item in ranked[:top_k]:
        meta = item["metadata"]
        doc = item["document"]
        citations.append(Citation(
            source=meta.get("source", ""),
            doc_id=meta.get("doc_id", ""),
            page=int(meta.get("page", 0) or 0),
            chunk_index=int(meta.get("chunk_index", 0) or 0),
            snippet=doc[:520] + ("..." if len(doc) > 520 else ""),
            similarity=round(rerank_score, 4),
            vector_similarity=round(vector_score, 4),
            keyword_score=round(keyword_score, 4),
            title=meta.get("title", ""),
            year=meta.get("year", ""),
            chunk_tokens=int(meta.get("chunk_tokens", 0) or 0),
        ))
    return citations


def confidence_from_citations(citations: List[Citation]) -> float:
    if not citations:
        return 0.0
    scores = [citation.similarity for citation in citations]
    top = scores[0]
    top3 = sum(scores[:3]) / min(len(scores), 3)
    avg = sum(scores) / len(scores)
    diversity = len({citation.doc_id for citation in citations}) / max(len(citations), 1)
    confidence = 0.55 * top + 0.30 * top3 + 0.10 * avg + 0.05 * diversity
    return round(min(confidence, 1.0), 4)


def _build_context_block(citations: List[Citation]) -> str:
    blocks = []
    for index, citation in enumerate(citations, start=1):
        blocks.append(
            f"[{index}] Source: {citation.source} | page {citation.page} | "
            f"score {citation.similarity:.2f}\n{citation.snippet}"
        )
    return "\n\n".join(blocks)


def _extractive_answer(citations: List[Citation], confidence: float) -> str:
    if confidence < config.CONFIDENCE_THRESHOLD:
        lead = (
            "The indexed documents do not confidently cover this question. "
            "Closest retrieved excerpts:"
        )
    else:
        lead = "Extractive answer from the most relevant retrieved excerpts:"
    lines = [lead]
    for index, citation in enumerate(citations[:3], start=1):
        lines.append(f"- [{index}] {citation.source}, page {citation.page}: {citation.snippet}")
    return "\n".join(lines)


ANSWER_SYSTEM_PROMPT = f"""You are a research assistant answering questions ONLY from the numbered
document excerpts the user provides. Rules:
1. Use ONLY the provided excerpts. Never use outside knowledge, never guess.
2. Every factual sentence in your answer must cite at least one excerpt using [1], [2], etc.
3. If the excerpts do not contain enough information to answer the question,
   respond with EXACTLY this token and nothing else: {NOT_COVERED_SENTINEL}
4. Do not apologize. Be concise and technical.
"""


def answer_question(query: str, top_k: int = None, doc_id_filter: Optional[str] = None) -> AnswerResult:
    from translate import detect_language, to_english, translate as translate_to

    orig_lang = detect_language(query)
    query_en = to_english(query, orig_lang)
    citations = retrieve(query_en, top_k=top_k, doc_id_filter=doc_id_filter)
    confidence = confidence_from_citations(citations)

    if not citations:
        msg = "The provided documents do not cover this question."
        return AnswerResult(
            answer=translate_to(msg, orig_lang),
            citations=[],
            confidence=0.0,
            needs_human_review=True,
            covered=False,
            answer_mode="none",
            detected_language=orig_lang,
            citation_check_passed=True,
        )

    if not llm_client.has_generation_provider():
        final = _extractive_answer(citations, confidence)
        return AnswerResult(
            answer=final,
            citations=citations,
            confidence=confidence,
            needs_human_review=confidence < config.CONFIDENCE_THRESHOLD,
            covered=confidence >= config.CONFIDENCE_THRESHOLD,
            answer_mode="extractive",
            detected_language=orig_lang,
            citation_check_passed=True,
        )

    context_block = _build_context_block(citations)
    user_prompt = f"Document excerpts:\n\n{context_block}\n\nQuestion: {query_en}\n\nAnswer:"

    try:
        raw_answer = llm_client.chat(ANSWER_SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=700)
    except Exception as exc:
        final = _extractive_answer(citations, confidence)
        final += f"\n\nLLM fallback reason: {exc}"
        return AnswerResult(
            answer=final,
            citations=citations,
            confidence=confidence,
            needs_human_review=True,
            covered=confidence >= config.CONFIDENCE_THRESHOLD,
            answer_mode="extractive_fallback",
            detected_language=orig_lang,
            citation_check_passed=True,
        )

    covered = NOT_COVERED_SENTINEL not in raw_answer
    citation_check_passed = validate_citation_numbers(raw_answer, len(citations))

    if not covered:
        final_en = "The provided documents do not cover this question."
        citations_out: List[Citation] = []
        confidence = 0.0
    elif not citation_check_passed:
        final_en = _extractive_answer(citations, confidence)
        citations_out = citations
    else:
        final_en = raw_answer
        citations_out = citations

    needs_review = (
        (not covered)
        or (not citation_check_passed)
        or (confidence < config.CONFIDENCE_THRESHOLD)
    )

    return AnswerResult(
        answer=translate_to(final_en, orig_lang),
        citations=citations_out,
        confidence=confidence,
        needs_human_review=needs_review,
        covered=covered and confidence >= config.CONFIDENCE_THRESHOLD,
        answer_mode="llm" if citation_check_passed else "extractive_fallback",
        detected_language=orig_lang,
        citation_check_passed=citation_check_passed,
    )


CONTRADICT_SYSTEM_PROMPT = """You are a scientific fact-checking assistant. You are given excerpts
from TWO different research papers. Decide whether the papers CONTRADICT each other, AGREE, or
the excerpts are NOT_ENOUGH_INFO for the requested topic.

Rules:
1. Use ONLY the provided excerpts.
2. Respond in this exact format:
VERDICT: <CONTRADICT | AGREE | NOT_ENOUGH_INFO>
REASONING: <2-5 sentences, citing excerpts by bracket number>
3. If the excerpts do not discuss the topic, use NOT_ENOUGH_INFO.
"""


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
            answer_mode="none",
        )

    if not llm_client.has_generation_provider():
        return ContradictionResult(
            verdict="NOT_ENOUGH_INFO",
            reasoning="No LLM key is configured. Review the side-by-side retrieved excerpts below.",
            citations_doc_a=citations_a,
            citations_doc_b=citations_b,
            answer_mode="extractive",
        )

    block_a = _build_context_block(citations_a)
    offset = len(citations_a)
    block_b = "\n\n".join(
        f"[{index}] Source: {citation.source} | page {citation.page}\n{citation.snippet}"
        for index, citation in enumerate(citations_b, start=offset + 1)
    )

    user_prompt = (
        f"Topic to compare: {topic}\n\n"
        f"--- Document A ({doc_id_a}) excerpts ---\n{block_a}\n\n"
        f"--- Document B ({doc_id_b}) excerpts ---\n{block_b}\n\n"
        "Do these two documents CONTRADICT, AGREE, or is there NOT_ENOUGH_INFO?"
    )

    raw = llm_client.chat(CONTRADICT_SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=500)
    verdict = "NOT_ENOUGH_INFO"
    for value in ("CONTRADICT", "AGREE", "NOT_ENOUGH_INFO"):
        if f"VERDICT: {value}" in raw.upper():
            verdict = value
            break

    reasoning = raw.split("REASONING:", 1)[1].strip() if "REASONING:" in raw else raw
    return ContradictionResult(
        verdict=verdict,
        reasoning=reasoning,
        citations_doc_a=citations_a,
        citations_doc_b=citations_b,
        answer_mode="llm",
    )
