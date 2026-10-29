# RAG Chatbot with Streamlit, Ollama, and Gemma 2B

## Overview

This project turns the retrieval pipeline from `ayna.ipynb` into a Streamlit web app.
It loads `Attention.txt`, splits the text into chunks, embeds the chunks with
`sentence-transformers/all-MiniLM-L6-v2`, stores them in FAISS, retrieves relevant
chunks for a question, and asks Ollama's `gemma:2b` model to answer using that context.

## Project files

- `app.py` — Streamlit application.
- `ayna.ipynb` — original notebook supplied for Practical 2.
- `requirements.txt` — Python dependencies.
- `Attention.txt` — **add your source text file here**. It was not included with the uploaded notebook.

## Prerequisites

- Python installed.
- Ollama installed and running.
- The `gemma:2b` model downloaded locally.
- Your original `Attention.txt` document.

## Setup (Windows PowerShell)
Open PowerShell in this project folder.

1. Create and activate a virtual environment (skip if you are using your existing training environment):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Copy your original `Attention.txt` into this folder, next to `app.py`.

4. In a terminal, download Gemma 2B with Ollama:
   ```powershell
   ollama pull gemma:2b
   ```

   You can check the model is available with:
   ```powershell
   ollama list
   ```

5. Start the Streamlit app:
   ```powershell
   python -m streamlit run app.py
   ```

6. Open the local URL shown by Streamlit, normally `http://localhost:8501`.

## How it works

1. `TextLoader` reads `Attention.txt`.
2. `RecursiveCharacterTextSplitter` splits the document into chunks of 1,000 characters with 20 characters of overlap.
3. `HuggingFaceEmbeddings` converts each chunk into a vector using `sentence-transformers/all-MiniLM-L6-v2`.
4. `FAISS` indexes those vectors.
5. `similarity_search(question, k=4)` retrieves up to four relevant chunks.
6. `ChatPromptTemplate` combines the retrieved context and user question.
7. `ChatOllama(model="gemma:2b")` generates a response.
8. Streamlit displays the answer and offers an expander to inspect the retrieved context.

The first run may download the Hugging Face embedding model. Internet access may be required for that initial download.

The notebook and app expect `Attention.txt`, but that text file was not part of the uploaded notebook. Add your actual file before running the app. Do not upload private or restricted source material to a public repository.
