"""Ingestion pipeline for local PDF/TXT research documents."""
import argparse
import hashlib
import logging
import os
import re
from typing import Dict, Iterable, List, Tuple

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config
from text_utils import (
    chunk_text,
    clean_title_from_filename,
    count_tokens,
    extract_year,
    normalize_whitespace,
)

CHROMA_SETTINGS = Settings(anonymized_telemetry=False)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def supported_files(papers_dir: str) -> List[str]:
    if not os.path.isdir(papers_dir):
        return []
    return sorted(
        fname
        for fname in os.listdir(papers_dir)
        if fname.lower().endswith((".pdf", ".txt"))
    )


def extract_pages(filepath: str) -> List[Tuple[int, str]]:
    """Return a list of (page_number, text) tuples, 1-indexed."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(filepath)
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = normalize_whitespace(page.extract_text() or "")
            if text:
                pages.append((index, text))
        return pages

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
            text = normalize_whitespace(handle.read())
        return [(1, text)] if text else []

    return []


def build_chunk_id(source: str, page: int, idx: int) -> str:
    raw = f"{source}::p{page}::c{idx}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def document_id(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return stem or hashlib.md5(filename.encode("utf-8")).hexdigest()


def infer_document_metadata(filename: str, pages: Iterable[Tuple[int, str]]) -> Dict[str, str]:
    page_list = list(pages)
    first_page = page_list[0][1] if page_list else ""
    return {
        "source": filename,
        "doc_id": document_id(filename),
        "title": clean_title_from_filename(filename),
        "year": extract_year(filename, first_page),
        "file_type": os.path.splitext(filename)[1].lower().lstrip("."),
    }


def reset_collection(chroma_dir: str = config.CHROMA_DIR) -> None:
    client = chromadb.PersistentClient(path=chroma_dir, settings=CHROMA_SETTINGS)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass


def ingest(papers_dir: str = config.PAPERS_DIR, chroma_dir: str = config.CHROMA_DIR, reset: bool = False) -> int:
    os.makedirs(papers_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)

    files = supported_files(papers_dir)
    if not files:
        print(f"No PDF/TXT files found in {papers_dir}.")
        return 0

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=chroma_dir, settings=CHROMA_SETTINGS)

    if reset:
        reset_collection(chroma_dir)
        print("Existing collection wiped.")

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for filename in files:
        filepath = os.path.join(papers_dir, filename)
        pages = extract_pages(filepath)
        if not pages:
            print(f"  [skip] {filename}: no extractable text")
            continue

        base_meta = infer_document_metadata(filename, pages)
        doc_chunks, doc_metas, doc_ids = [], [], []

        for page_num, page_text in pages:
            chunks = chunk_text(
                page_text,
                config.CHUNK_SIZE_TOKENS,
                config.CHUNK_OVERLAP_TOKENS,
            )
            for chunk_index, chunk in enumerate(chunks):
                doc_chunks.append(chunk)
                doc_metas.append({
                    **base_meta,
                    "page": page_num,
                    "chunk_index": chunk_index,
                    "chunk_tokens": count_tokens(chunk),
                })
                doc_ids.append(build_chunk_id(filename, page_num, chunk_index))

        if not doc_chunks:
            print(f"  [skip] {filename}: text existed but no chunks were produced")
            continue

        embeddings = embedder.encode(
            doc_chunks,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        collection.upsert(
            ids=doc_ids,
            embeddings=embeddings,
            documents=doc_chunks,
            metadatas=doc_metas,
        )
        print(f"  [ok] {filename}: {len(doc_chunks)} chunks across {len(pages)} pages")
        total_chunks += len(doc_chunks)

    print(
        f"\nDone. {total_chunks} chunks stored in Chroma collection "
        f"'{config.COLLECTION_NAME}' at '{chroma_dir}'."
    )
    return total_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-dir", default=config.PAPERS_DIR)
    parser.add_argument("--chroma-dir", default=config.CHROMA_DIR)
    parser.add_argument("--reset", action="store_true", help="Wipe the collection before ingesting")
    args = parser.parse_args()
    ingest(args.papers_dir, args.chroma_dir, args.reset)
