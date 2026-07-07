"""Streamlit dashboard for the local Alzheimer research RAG app."""
import os
from typing import Iterable

import streamlit as st

import config
import ingest
import llm_client
import rag_core
from translate import detect_language, language_name

st.set_page_config(page_title="Alzheimer Research RAG", layout="wide")


def _ensure_dirs() -> None:
    os.makedirs(config.PAPERS_DIR, exist_ok=True)
    os.makedirs(config.CHROMA_DIR, exist_ok=True)


def _safe_stats() -> dict:
    try:
        return rag_core.database_summary()
    except Exception as exc:
        return {
            "documents": 0,
            "chunks": 0,
            "pages": 0,
            "error": str(exc),
            "provider": llm_client.provider_status().__dict__,
        }


def _doc_options() -> list:
    try:
        return rag_core.list_documents()
    except Exception:
        return []


def _citation_rows(citations: Iterable[rag_core.Citation]) -> list:
    return [
        {
            "#": index,
            "source": citation.source,
            "page": citation.page,
            "hybrid": citation.similarity,
            "vector": citation.vector_similarity,
            "keyword": citation.keyword_score,
        }
        for index, citation in enumerate(citations, start=1)
    ]


def render_citations(citations: list[rag_core.Citation]) -> None:
    if not citations:
        return
    st.markdown("#### Citations")
    st.dataframe(_citation_rows(citations), use_container_width=True, hide_index=True)
    for index, citation in enumerate(citations, start=1):
        label = (
            f"[{index}] {citation.source} | page {citation.page} | "
            f"hybrid {citation.similarity:.2f}"
        )
        with st.expander(label):
            st.write(citation.snippet)


_ensure_dirs()
stats = _safe_stats()
provider = llm_client.provider_status()

st.title("Alzheimer Research RAG")
st.caption("Grounded question answering over local research papers with hybrid retrieval and citation checks.")

with st.sidebar:
    st.subheader("System Status")
    st.metric("Documents", stats.get("documents", 0))
    st.metric("Chunks", stats.get("chunks", 0))
    st.metric("Pages", stats.get("pages", 0))
    st.divider()
    st.write(f"Provider: `{provider.active}`")
    st.caption(provider.message)
    st.write(f"Confidence threshold: `{config.CONFIDENCE_THRESHOLD}`")
    st.write(f"Chunking: `{config.CHUNK_SIZE_TOKENS}` tokens + `{config.CHUNK_OVERLAP_TOKENS}` overlap")
    if stats.get("error"):
        st.warning(stats["error"])

tab_ask, tab_ingest, tab_compare, tab_docs, tab_api = st.tabs([
    "Ask",
    "Upload & Ingest",
    "Compare Papers",
    "Document Browser",
    "API Examples",
])

with tab_ask:
    st.subheader("Ask a grounded question")
    docs = _doc_options()
    col_query, col_settings = st.columns([3, 1])
    with col_query:
        query = st.text_input(
            "Question",
            placeholder="Example: What accuracy did 3D-CNN models achieve for Alzheimer's classification?",
        )
    with col_settings:
        doc_filter = st.selectbox("Restrict to document", ["(all documents)"] + docs)
        top_k = st.slider("Top K chunks", 1, 12, config.TOP_K)

    if query.strip():
        detected = detect_language(query)
        st.caption(f"Detected language: `{language_name(detected)}`")

    if st.button("Ask question", type="primary", disabled=not query.strip()):
        with st.spinner("Retrieving evidence and preparing answer..."):
            selected_doc = None if doc_filter == "(all documents)" else doc_filter
            result = rag_core.answer_question(query, top_k=top_k, doc_id_filter=selected_doc)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Confidence", f"{result.confidence:.2f}")
        metric_cols[1].metric("Mode", result.answer_mode)
        metric_cols[2].metric("Covered", "yes" if result.covered else "no")
        metric_cols[3].metric("Citation check", "pass" if result.citation_check_passed else "review")

        if result.needs_human_review:
            st.warning("This answer needs human review because coverage, confidence, or citation validation is weak.")
        elif result.answer_mode == "extractive":
            st.info("No LLM key is configured, so the app returned extractive grounded snippets.")

        st.markdown("### Answer")
        st.write(result.answer)
        render_citations(result.citations)

with tab_ingest:
    st.subheader("Upload and ingest papers")
    st.write("Upload PDF or TXT research documents, then ingest them into the local Chroma vector store.")

    uploads = st.file_uploader(
        "Upload papers",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Save uploaded files"):
        saved = []
        for upload in uploads:
            destination = os.path.join(config.PAPERS_DIR, os.path.basename(upload.name))
            with open(destination, "wb") as handle:
                handle.write(upload.getbuffer())
            saved.append(upload.name)
        st.success(f"Saved {len(saved)} file(s): {', '.join(saved)}")

    local_files = ingest.supported_files(config.PAPERS_DIR)
    st.write(f"Files waiting in `{config.PAPERS_DIR}`: `{len(local_files)}`")
    if local_files:
        with st.expander("Show local files"):
            for filename in local_files:
                st.write(f"- {filename}")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Ingest new/changed files", type="primary"):
            with st.spinner("Embedding and indexing papers..."):
                chunks = ingest.ingest(config.PAPERS_DIR, config.CHROMA_DIR, reset=False)
                rag_core.reset_runtime_cache()
            st.success(f"Ingestion complete. {chunks} chunks processed.")
            st.rerun()
    with col_b:
        if st.button("Reset and re-ingest"):
            with st.spinner("Clearing and rebuilding vector store..."):
                chunks = ingest.ingest(config.PAPERS_DIR, config.CHROMA_DIR, reset=True)
                rag_core.reset_runtime_cache()
            st.success(f"Re-ingestion complete. {chunks} chunks indexed.")
            st.rerun()
    with col_c:
        if st.button("Clear vector database"):
            rag_core.clear_vector_store()
            st.warning("Vector database cleared. Papers were not deleted.")
            st.rerun()

with tab_compare:
    st.subheader("Compare two papers")
    docs = _doc_options()
    if len(docs) < 2:
        st.info("Ingest at least two documents before using comparison.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            doc_a = st.selectbox("Document A", docs, key="doc_a")
        with col_b:
            doc_b = st.selectbox("Document B", docs, index=1, key="doc_b")

        topic = st.text_input("Topic to compare", placeholder="Example: model accuracy, dataset size, interpretability")
        if st.button("Compare papers", type="primary", disabled=not topic.strip() or doc_a == doc_b):
            with st.spinner("Retrieving comparison evidence..."):
                result = rag_core.compare_documents(doc_a, doc_b, topic)
            st.markdown(f"### Verdict: `{result.verdict}`")
            st.caption(f"Mode: `{result.answer_mode}`")
            st.write(result.reasoning)

            left, right = st.columns(2)
            with left:
                st.markdown(f"#### Evidence from {doc_a}")
                render_citations(result.citations_doc_a)
            with right:
                st.markdown(f"#### Evidence from {doc_b}")
                render_citations(result.citations_doc_b)

with tab_docs:
    st.subheader("Document browser")
    docs = rag_core.document_stats()
    if not docs:
        st.info("No documents are indexed yet. Upload papers and run ingestion first.")
    else:
        st.dataframe(docs, use_container_width=True, hide_index=True)

    st.markdown("#### Retrieval diagnostics")
    diagnostic_query = st.text_input("Test retrieval query", key="diagnostic_query")
    if st.button("Run retrieval test", disabled=not diagnostic_query.strip()):
        citations = rag_core.retrieve(diagnostic_query, top_k=8)
        render_citations(citations)

with tab_api:
    st.subheader("API examples")
    st.write("Run the API with `run.bat api`, then use these examples.")
    st.code(
        """curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What dataset sizes are reported?\",\"top_k\":5}"

curl -X POST http://127.0.0.1:8000/retrieve ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"CNN accuracy for Alzheimer's detection\",\"top_k\":5}"

curl -X POST http://127.0.0.1:8000/ingest?reset=true
""",
        language="bash",
    )
