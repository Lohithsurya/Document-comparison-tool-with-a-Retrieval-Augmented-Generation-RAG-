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
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Set up Ollama:**
   ```powershell
   ollama pull mistral
   ```

3. **Every time you use the app** (run these in separate terminals):

   Terminal 1 - Start Ollama:
   ```powershell
   ollama serve
   ```

   Terminal 2 - Add your PDFs to the `data/` folder, then run (first time or when adding new PDFs):
   ```powershell
   python database.py
   ```

   Terminal 3 - Start the app:
   ```powershell
   streamlit run query.py
   ```

5. Open the app in your browser at `http://localhost:8501`.
