"""
Ingestion pipeline
===================
Reads every .pdf / .txt file from PAPERS_DIR, extracts text page-by-page,
splits it into overlapping chunks, embeds the chunks with a local
sentence-transformers model, and upserts everything into a persistent
Chroma collection.

Run:
    python ingest.py
    python ingest.py --reset      # wipe and rebuild the collection
"""
import argparse
import hashlib
import os
import re
import sys

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config


def extract_pages(filepath: str):
    """Return a list of (page_number, text) tuples, 1-indexed."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(filepath)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append((i, text))
        return pages
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = re.sub(r"\s+", " ", text).strip()
        return [(1, text)] if text else []
    else:
        return []


def chunk_text(text: str, chunk_size: int, overlap: int):
    """
    Sliding-window chunker over sentences.

    Why this approach (see README for full rationale):
    - Research papers pack a lot of meaning per sentence (methods, results,
      numbers). Splitting mid-sentence risks cutting a claim in half and
      handing the LLM a mutilated fact.
    - So we split on sentence boundaries first, then greedily pack sentences
      into ~chunk_size-character windows, carrying the last `overlap`
      characters of context into the next chunk so a fact that spans a
      chunk boundary in the source text still appears whole somewhere.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= chunk_size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying overlap from the tail of the previous one
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sent}".strip()
    if current:
        chunks.append(current)
    return chunks


def build_chunk_id(source: str, page: int, idx: int) -> str:
    raw = f"{source}::p{page}::c{idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest(papers_dir: str, chroma_dir: str, reset: bool = False):
    if not os.path.isdir(papers_dir):
        print(f"Papers directory '{papers_dir}' does not exist.")
        sys.exit(1)

    files = [
        f for f in sorted(os.listdir(papers_dir))
        if f.lower().endswith((".pdf", ".txt"))
    ]
    if not files:
        print(f"No .pdf or .txt files found in '{papers_dir}'.")
        sys.exit(1)

    print(f"Found {len(files)} document(s): {files}")
    print(f"Loading embedding model '{config.EMBEDDING_MODEL}' ...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=chroma_dir)

    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            print("Existing collection wiped.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for fname in files:
        fpath = os.path.join(papers_dir, fname)
        pages = extract_pages(fpath)
        if not pages:
            print(f"  [skip] {fname}: no extractable text")
            continue

        doc_chunks, doc_metas, doc_ids = [], [], []
        for page_num, page_text in pages:
            for idx, chunk in enumerate(
                chunk_text(page_text, config.CHUNK_SIZE_CHARS, config.CHUNK_OVERLAP_CHARS)
            ):
                cid = build_chunk_id(fname, page_num, idx)
                doc_chunks.append(chunk)
                doc_metas.append({
                    "source": fname,
                    "doc_id": os.path.splitext(fname)[0],
                    "page": page_num,
                    "chunk_index": idx,
                })
                doc_ids.append(cid)

        if not doc_chunks:
            continue

        embeddings = embedder.encode(doc_chunks, show_progress_bar=False, normalize_embeddings=True).tolist()
        collection.upsert(
            ids=doc_ids,
            embeddings=embeddings,
            documents=doc_chunks,
            metadatas=doc_metas,
        )
        print(f"  [ok] {fname}: {len(doc_chunks)} chunks across {len(pages)} pages")
        total_chunks += len(doc_chunks)

    print(f"\nDone. {total_chunks} chunks stored in Chroma collection '{config.COLLECTION_NAME}' at '{chroma_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-dir", default=config.PAPERS_DIR)
    parser.add_argument("--chroma-dir", default=config.CHROMA_DIR)
    parser.add_argument("--reset", action="store_true", help="Wipe the collection before ingesting")
    args = parser.parse_args()
    ingest(args.papers_dir, args.chroma_dir, args.reset)
