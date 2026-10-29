C++ RAG Chatbot

## Objective
Build a Streamlit application that retrieves relevant information from a C++ introduction document using Retrieval-Augmented Generation (RAG) techniques.

## Features
- Loads C++ reference text from `CPP.txt`
- Splits the document into overlapping text chunks
- Generates embeddings using Hugging Face
- Uses FAISS for similarity search
- Displays the top 3 relevant text results for a user's question

## Technologies Used
- Python
- Streamlit
- LangChain
- Hugging Face Sentence Transformers
- FAISS

## How to Run

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

Enter a question about C++ in the app to retrieve relevant context from `CPP.txt`.

**Note:** This version retrieves relevant text passages; it does not generate a final answer using a language model.
