import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Docs Assistant", page_icon="📄")
st.title("📄 Docs Assistant")
st.caption("Answers come only from the indexed PDFs, with document + page citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['ref']}] {s['document']} — page {s['page']}** (score {s['score']})")
            st.caption(s["excerpt"] + "…")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources"))

if question := st.chat_input("Ask a question about the documentation"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the docs..."):
            try:
                r = requests.post(f"{API_URL}/chat", json={"question": question}, timeout=60)
                r.raise_for_status()
                data = r.json()
            except requests.RequestException as e:
                data = {"answer": f"API error: {e}", "sources": []}
        st.markdown(data["answer"])
        render_sources(data["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
    )
