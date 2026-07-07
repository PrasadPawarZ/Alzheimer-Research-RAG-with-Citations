"""
Streamlit UI.

Run:
    streamlit run streamlit_app.py

This talks directly to rag_core (no need to have the FastAPI server
running too) so it's the fastest way to try the system out.
"""
import streamlit as st

import rag_core

st.set_page_config(page_title="Alzheimer's Research RAG", layout="wide")

st.title("🧠 Alzheimer's Detection Research — RAG Q&A")
st.caption(
    "Answers are generated only from the ingested papers. "
    "If the papers don't cover a question, the system says so — it will not guess."
)

try:
    all_docs = rag_core.list_documents()
except Exception as e:
    all_docs = []
    st.error(f"Could not connect to the vector store. Have you run `python ingest.py`? ({e})")

tab_ask, tab_contradict, tab_docs = st.tabs(["🔍 Ask", "⚖️ Contradiction Check", "📚 Documents"])

# ---------------------------------------------------------------- Ask tab
with tab_ask:
    st.subheader("Ask a question")
    st.write("Try it in any language — English, Hindi, Spanish, etc.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Your question", placeholder="e.g. What accuracy did 3D-CNN models achieve for Alzheimer's classification?")
    with col2:
        doc_filter = st.selectbox("Restrict to document (optional)", ["(all documents)"] + all_docs)

    top_k = st.slider("Number of chunks to retrieve (top_k)", 1, 10, 5)

    if st.button("Ask", type="primary") and query.strip():
        with st.spinner("Retrieving and generating grounded answer..."):
            doc_id_filter = None if doc_filter == "(all documents)" else doc_filter
            result = rag_core.answer_question(query, top_k=top_k, doc_id_filter=doc_id_filter)

        if result.needs_human_review:
            st.warning(
                f"⚠️ Low confidence ({result.confidence:.2f}) or the documents may not "
                "cover this question. Treat this answer as provisional — human review recommended."
            )

        st.markdown("### Answer")
        st.write(result.answer)

        if not result.covered:
            st.info("The system explicitly determined the documents do not cover this question. "
                     "No answer was fabricated.")

        if result.citations:
            st.markdown("### Citations")
            for i, c in enumerate(result.citations, start=1):
                with st.expander(f"[{i}] {c.source} — page {c.page} (similarity {c.similarity:.2f})"):
                    st.write(c.snippet)

            st.markdown(f"**Confidence score:** `{result.confidence:.2f}` "
                        f"(threshold for auto-approval: `{rag_core.config.CONFIDENCE_THRESHOLD}`)")

# ---------------------------------------------------------- Contradict tab
with tab_contradict:
    st.subheader("Check whether two papers contradict each other")

    c1, c2 = st.columns(2)
    with c1:
        doc_a = st.selectbox("Document A", all_docs, key="doc_a")
    with c2:
        doc_b = st.selectbox("Document B", all_docs, index=min(1, len(all_docs) - 1) if len(all_docs) > 1 else 0, key="doc_b")

    topic = st.text_input("Topic to compare", placeholder="e.g. classification accuracy, dataset size, model interpretability")

    if st.button("Compare", type="primary") and topic.strip():
        if doc_a == doc_b:
            st.error("Please choose two different documents.")
        else:
            with st.spinner("Comparing documents..."):
                result = rag_core.compare_documents(doc_a, doc_b, topic)

            verdict_color = {"CONTRADICT": "🔴", "AGREE": "🟢", "NOT_ENOUGH_INFO": "🟡"}
            st.markdown(f"### Verdict: {verdict_color.get(result.verdict, '')} {result.verdict}")
            st.write(result.reasoning)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Excerpts from {doc_a}**")
                for c in result.citations_doc_a:
                    with st.expander(f"page {c.page} (sim {c.similarity:.2f})"):
                        st.write(c.snippet)
            with col_b:
                st.markdown(f"**Excerpts from {doc_b}**")
                for c in result.citations_doc_b:
                    with st.expander(f"page {c.page} (sim {c.similarity:.2f})"):
                        st.write(c.snippet)

# --------------------------------------------------------------- Docs tab
with tab_docs:
    st.subheader("Ingested documents")
    if all_docs:
        for d in all_docs:
            st.write(f"- {d}")
    else:
        st.write("No documents ingested yet. Drop PDFs into `papers/` and run `python ingest.py`.")
