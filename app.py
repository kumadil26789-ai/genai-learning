from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag import chunk_text, embed_texts, extract_pages, generate_answer, retrieve_chunks


load_dotenv()

st.set_page_config(page_title="Document Q&A", page_icon="📚")
st.title("Ask your documents")
st.write("Upload PDF or TXT files, then ask questions grounded in their contents.")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.embeddings = []
    st.session_state.history = []

uploads = st.file_uploader(
    "Choose documents",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if st.button("Index documents", disabled=not uploads):
    try:
        with st.spinner("Extracting text and creating embeddings..."):
            chunks = []
            for upload in uploads:
                for page in extract_pages(upload.name, upload.getvalue()):
                    for text in chunk_text(page["text"]):
                        chunks.append({**page, "text": text})

            if not chunks:
                raise ValueError("No readable text was found in the selected documents.")

            client = OpenAI()
            embeddings = embed_texts(client, [chunk["text"] for chunk in chunks])
            if len(embeddings) != len(chunks):
                raise RuntimeError("The embedding service returned an incomplete result.")

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.history = []
        st.success(f"Indexed {len(chunks)} passages. Ask a question below.")
    except Exception as error:
        st.error(f"Could not index documents: {error}")

if st.session_state.chunks:
    with st.form("question_form"):
        question = st.text_input("Your question")
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not question.strip():
            st.warning("Enter a question before submitting.")
        else:
            try:
                with st.spinner("Finding relevant passages and generating an answer..."):
                    client = OpenAI()
                    question_embedding = embed_texts(client, [question.strip()])[0]
                    matches = retrieve_chunks(
                        question_embedding,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                    )
                    answer = generate_answer(client, question.strip(), matches)
                st.session_state.history.append(
                    {"question": question.strip(), "answer": answer, "matches": matches}
                )
            except Exception as error:
                st.error(f"Could not answer the question: {error}")

    for item in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            with st.expander("Retrieved passages"):
                for match in item["matches"]:
                    page_label = (
                        f", page {match['page']}" if match.get("page") is not None else ""
                    )
                    st.caption(f"{match['source']}{page_label}")
                    st.write(match["text"])
