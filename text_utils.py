"""Pure text utilities used by ingestion, retrieval, and tests."""
import os
import re
from collections import Counter
from typing import Iterable, List, Sequence, Set

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CITATION_RE = re.compile(r"\[(\d+)\]")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "what", "which", "when", "where", "how", "does", "did", "into", "using",
    "used", "use", "has", "have", "had", "not", "but", "about", "over",
    "their", "its", "can", "could", "would", "should", "than", "then", "also",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text or "")


def keyword_terms(text: str) -> List[str]:
    return [
        term.lower()
        for term in WORD_RE.findall(text or "")
        if len(term) > 2 and term.lower() not in STOPWORDS
    ]


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def split_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def _tail_token_text(text: str, token_count: int) -> str:
    tokens = tokenize(text)
    if token_count <= 0 or len(tokens) <= token_count:
        return text
    return " ".join(tokens[-token_count:])


def chunk_text(text: str, chunk_size_tokens: int, overlap_tokens: int) -> List[str]:
    """Sentence-aware, token-budgeted sliding window chunker."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if not current:
            current.append(sentence)
            current_tokens = sentence_tokens
            continue

        if current_tokens + sentence_tokens <= chunk_size_tokens:
            current.append(sentence)
            current_tokens += sentence_tokens
            continue

        chunk = normalize_whitespace(" ".join(current))
        chunks.append(chunk)
        overlap = _tail_token_text(chunk, overlap_tokens)
        current = [overlap, sentence] if overlap else [sentence]
        current_tokens = count_tokens(" ".join(current))

    if current:
        chunks.append(normalize_whitespace(" ".join(current)))

    return chunks


def keyword_overlap_score(query: str, document: str) -> float:
    query_terms = keyword_terms(query)
    if not query_terms:
        return 0.0

    doc_counts = Counter(keyword_terms(document))
    if not doc_counts:
        return 0.0

    query_counts = Counter(query_terms)
    matched = sum(min(count, doc_counts.get(term, 0)) for term, count in query_counts.items())
    return round(matched / max(sum(query_counts.values()), 1), 4)


def validate_citation_numbers(answer: str, max_citation: int) -> bool:
    refs = [int(value) for value in CITATION_RE.findall(answer or "")]
    if not refs:
        return False
    return all(1 <= ref <= max_citation for ref in refs)


def clean_title_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"[_-]+", " ", stem)
    return normalize_whitespace(stem).title()


def extract_year(*texts: Sequence[str]) -> str:
    for text in texts:
        match = YEAR_RE.search(text or "")
        if match:
            return match.group(0)
    return ""


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
