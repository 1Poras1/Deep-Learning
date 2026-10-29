
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Streamlit page
st.set_page_config(page_title="Attention Paper RAG", page_icon="📚")
st.title("Chat with Attention Is All You Need")
st.caption("RAG powered by LangChain, FAISS, Hugging Face and Gemma")

# Load and process document
@st.cache_resource
def create_vector_database():
    loader = TextLoader("Attention.txt", autodetect_encoding=True)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# Create vector database
with st.spinner("Loading document and creating searchable index..."):
    db = create_vector_database()

st.success("Document is ready! Ask a question below.")

# User question
question = st.text_input(
    "Ask a question about the Attention paper:",
    placeholder="What is the Transformer architecture?"
)

# Ollama model
llm = ChatOllama(model="gemma:2b", temperature=0)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant answering questions about a research paper.

Use only the provided context to answer the question.
If the answer is not present in the context, say you could not find it in the document.
Do not invent information.

Context:
{context}"""
    ),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

# Retrieve and generate answer
if question:
    with st.spinner("Searching the document and generating an answer..."):
        docs = db.similarity_search(question, k=3)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        answer = chain.invoke({
            "context": context,
            "question": question
        })

    st.subheader("Answer")
    st.write(answer)

    with st.expander("View retrieved document passages"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Passage {i}**")
            st.write(doc.page_content)
