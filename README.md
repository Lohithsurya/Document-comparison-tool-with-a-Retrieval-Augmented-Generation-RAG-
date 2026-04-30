# RAG Chatbot for Document Comparison


## Overview

The RAG model for Document Comparison is a web application built using Streamlit and Python, designed to facilitate comparison and analysis of textual content extracted from PDF documents. It leverages natural language processing techniques and embedding models to provide insights based on user queries.

## Features

- **Document Upload**: Users can upload multiple PDF files for comparison.
- **Text Extraction**: PDF files are processed to extract textual content for comparison.
- **Semantic Search**: Uses embedding models to perform similarity searches across documents.
- **Interactive Interface**: Web-based interface powered by Streamlit for user interaction.

## Requirements

- Python 3.9+ (Python 3.12 is fine)
- pip package manager
- Ollama CLI installed separately for `Ollama(model="mistral")`
   ```bash
   ollama --help
   ```

## Setup

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. Install project dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Prepare the document database:
   - Put PDF files into the `data/` folder
   - Run:
     ```powershell
     python database.py
     ```

4. Start the app:
   ```powershell
   streamlit run query.py
   ```

5. Open the app in your browser at `http://localhost:8501`.

