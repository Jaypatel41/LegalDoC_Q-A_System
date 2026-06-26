"""Streamlit demo for the Legal QA RAG pipeline.

  streamlit run app/streamlit_app.py

Shows the answer, citations, the routed store, MultiQuery variants, retrieved
context, and the live confidence / faithfulness / hallucination-risk numbers —
so an interviewer can watch every accuracy layer act in real time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.pipeline import AblationFlags, LegalRAG
from src.evaluation import ragas_eval

st.set_page_config(page_title=" Legal Document QA System", page_icon="⚖️", layout="wide")

st.title("⚖️ Legal Document QA System")
st.caption("MultiQuery · Hybrid retrieval (BM25+dense) · RRF · Cross-encoder rerank "
           "· 3-gate guardrails · trained ML router + hallucination detector")

with st.sidebar:
    st.header("Accuracy layers")
    mq = st.checkbox("MultiQuery expansion", value=True)
    hy = st.checkbox("Hybrid retrieval + rerank", value=True)
    gr = st.checkbox("Guardrail stack", value=True)
    st.divider()
    st.caption(f"Metrics mode: **{ragas_eval.metrics_mode()}**")
    st.caption("Set keys in `.env` and unset USE_STUB_LLM for real generation.")
    examples = [
        "What is the punishment for murder under the IPC?",
        "What did the Supreme Court hold in Kesavananda Bharati?",
        "What is the termination notice period in the service agreement?",
        "How long do confidentiality obligations survive termination?",
        "Recommend a good biryani recipe",   # out-of-scope -> blocked
    ]
    st.write("**Try:**")
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["q"] = ex

query = st.text_input("Ask a legal question",
                      value=st.session_state.get("q", ""),
                      placeholder="e.g. What is the punishment for cheating under Section 420?")

if st.button("Answer", type="primary") or st.session_state.get("q"):
    if query.strip():
        flags = AblationFlags(multiquery=mq, hybrid=hy, rerank=hy, guardrails=gr)
        with st.spinner("Routing → expanding → retrieving → reranking → generating…"):
            res = LegalRAG(flags).answer(query)
        st.session_state.pop("q", None)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", res.route)
        c2.metric("Confidence", f"{res.confidence:.2f}")
        c3.metric("Faithfulness",
                  f"{res.faithfulness:.2f}" if res.faithfulness is not None else "—")
        c4.metric("Halluc. risk",
                  f"{res.hallucination_risk:.2f}" if res.hallucination_risk is not None else "—")

        if res.blocked:
            st.warning(f"🛡️ Guardrail blocked this query: {res.block_reason}")
        if res.regenerated:
            st.info("♻️ Output failed the faithfulness gate — answer was "
                    "automatically regenerated with a stricter grounded prompt.")

        st.subheader("Answer")
        st.write(res.answer)

        if res.citations:
            st.subheader("Citations")
            for c in res.citations:
                st.markdown(f"- {c}")

        with st.expander("🔍 MultiQuery variants"):
            for q in res.sub_queries:
                st.markdown(f"- {q}")

        with st.expander("📄 Retrieved context"):
            for i, ctx in enumerate(res.contexts, 1):
                st.markdown(f"**[{i}]** {ctx}")

        with st.expander("⏱️ Stage timings (ms)"):
            st.json({k: round(v, 1) for k, v in res.timings_ms.items()})
