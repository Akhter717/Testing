import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG + RAGAS (Groq)", layout="wide")
st.title("RAG + RAGAS evaluator (Groq free models)")

# Streamlit Cloud: values come from st.secrets. Local: from .env or .streamlit/secrets.toml.
try:
    for _k in ("GROQ_API_KEY", "RAG_MODEL", "JUDGE_MODEL", "EMBED_MODEL"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets file locally; fall back to .env

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. On Streamlit Cloud add it under App settings -> Secrets; "
             "locally put it in .env or .streamlit/secrets.toml.")
    st.stop()

import rag
import evaluator

# ---- Sidebar: upload + index ----
with st.sidebar:
    st.header("1. Upload PDF(s)")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Build index", disabled=not files):
        with st.spinner("Reading PDFs and building index..."):
            chunks = []
            for f in files:
                chunks += rag.chunk_pdf(f.getvalue(), f.name)
            try:
                st.session_state.index = rag.RagIndex(chunks)
                st.session_state.pop("last", None)
                st.success(f"Indexed {len(chunks)} chunks from {len(files)} file(s).")
            except ValueError as e:
                st.error(str(e))
    st.caption(f"RAG model: {os.getenv('RAG_MODEL', 'llama-3.1-8b-instant')}")
    st.caption(f"Judge model: {os.getenv('JUDGE_MODEL', 'llama-3.3-70b-versatile')}")

if "index" not in st.session_state:
    st.info("Upload a PDF in the sidebar and click **Build index**.")
    st.stop()

# ---- Ask ----
st.header("2. Ask a question")
question = st.text_input("Question")
if st.button("Get answer", disabled=not question):
    with st.spinner("Retrieving and answering..."):
        ctx = st.session_state.index.retrieve(question)
        ans = rag.answer(question, ctx)
    st.session_state.last = {"question": question, "answer": ans, "contexts": ctx}

last = st.session_state.get("last")
if last:
    st.subheader("Answer")
    st.write(last["answer"])
    with st.expander("Retrieved chunks (check the answer against these)", expanded=True):
        for i, c in enumerate(last["contexts"], 1):
            st.markdown(f"**{i}. {c.source} - page {c.page}**")
            st.write(c.text)

    # ---- Evaluate ----
    st.header("3. Evaluate with RAGAS")
    reference = st.text_area(
        "Correct answer (optional)",
        help="Leave blank if you don't know it. You'll get 3 reference-free metrics. "
             "Fill it to also get context recall, reference-based precision and factual correctness.",
    )
    if st.button("Run RAGAS"):
        with st.spinner("Judge model is scoring (can take a minute on free tier)..."):
            scores = evaluator.evaluate_one(
                last["question"], last["answer"],
                [c.text for c in last["contexts"]], reference,
            )
        rows = []
        for name, val in scores.items():
            rows.append({"Metric": name,
                         "Score": round(val, 3) if isinstance(val, float) else val})
        st.table(rows)
        st.caption("Scores are 0-1, higher is better. They come from an LLM judge: treat as signals, not proof.")
