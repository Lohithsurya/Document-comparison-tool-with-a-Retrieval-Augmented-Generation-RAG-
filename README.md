# RAG Chatbot for Document Comparison


## Overview

The RAG model for Document Comparison is a web application built using Streamlit (for web interface) Langchain (for handling documents and AI), ChromaDB (for storing data vectors) and Python, designed to facilitate comparison and analysis of textual content extracted from PDF documents. It leverages natural language processing techniques and embedding models to provide insights based on user queries.

## Features

- **Document Upload**: Users can upload multiple PDF files for comparison.
- **Text Extraction**: PDF files are processed to extract textual content for comparison.
- **Semantic Search**: Uses embedding models to perform similarity searches across documents.
- **Interactive Interface**: Web-based interface powered by Streamlit for user interaction.

## Requirements

- Python 3.9+
- pip package manager
- Ollama (for running Mistral AI model locally)

## Setup

1. **Set up Python environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python3 -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Set up Ollama:**
   ```powershell
   ollama pull llama3.2:3b
   ```

   Use `http://localhost:11434/` to check if Ollama is running.

3. **Every time you use the app** (run these in separate terminals):

   Terminal 1 - Add your PDFs to the `data/` folder, then run (first time or when adding new PDFs):
   ```powershell
   python database.py
   ```

   Terminal 2 - Start the app:
   ```powershell
   streamlit run query.py
   ```

5. Open the app in your browser at `http://localhost:8501`.


## RAG Understanding

RAG (Retrieval-Augmented Generation) is an architecture pattern whose pipeline looks like this:

1. Embedding model (being used here, `sentence-transformers/all-MiniLM-L12-v2`) converts your documents into vectors and stores them in a vector database
2. When a query comes in, the same embedding model converts the query into a vector
3. A similarity search (being used here, `cosine similarity`) finds the most relevant document chunks
4. Those chunks are injected into the LLM's context as extra information
5. The LLM (being used here, `llama3.2:3b`) generates a response grounded in that retrieved content