from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_PATH = BASE_DIR / "Attention.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma:2b"

st.set_page_config(page_title="RAG Chatbot - Gemma 2B", page_icon="📚")
st.title("📚 RAG Chatbot with Gemma 2B")
st.caption("Ask questions about the information in Attention.txt.")

if not DOCUMENT_PATH.exists():
    st.error(
        "Attention.txt was not found. Place your Attention.txt file in the same "
        "folder as app.py, then refresh this page."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading document, embeddings, and FAISS index...")
def build_vector_store():
    loader = TextLoader(str(DOCUMENT_PATH), autodetect_encoding=True)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Attention.txt is empty or contains no usable text.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


@st.cache_resource
def get_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)


try:
    vector_store = build_vector_store()
except Exception as exc:
    st.error(f"Could not prepare the document search index: {exc}")
    st.info(
        "Check that the required Python packages are installed and that "
        "Attention.txt contains readable text."
    )
    st.stop()

with st.sidebar:
    st.subheader("About this app")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM:** `{OLLAMA_MODEL}`")
    st.write("The first launch may take longer while the embedding model downloads.")
    if st.button("Rebuild document index"):
        build_vector_store.clear()
        st.rerun()

question = st.text_input(
    "Enter your question:",
    placeholder="Ask something based on Attention.txt...",
)

if question.strip():
    with st.spinner("Searching the document and generating an answer..."):
        try:
            relevant_docs = vector_store.similarity_search(question, k=4)
            context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful question-answering assistant. "
                        "Answer using the supplied context. If the context does "
                        "not contain enough information, say that you cannot find "
                        "the answer in Attention.txt. Do not invent facts.\n\n"
                        "CONTEXT:\n{context}",
                    ),
                    ("human", "{question}"),
                ]
            )

            chain = prompt | get_llm() | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})

            st.subheader("Answer")
            st.write(answer)

            with st.expander("View retrieved context"):
                for index, doc in enumerate(relevant_docs, start=1):
                    st.markdown(f"**Retrieved chunk {index}**")
                    st.write(doc.page_content)
                    st.divider()

        except Exception as exc:
            st.error(f"Could not generate an answer: {exc}")
            st.info(
                "Make sure Ollama is installed and running, and that "
                "`gemma:2b` is available. In a terminal, run `ollama list`."
            )
