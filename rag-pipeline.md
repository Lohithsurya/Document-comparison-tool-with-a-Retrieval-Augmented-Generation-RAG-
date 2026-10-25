# The RAG pipeline in this project
The project has two phases. Indexing runs once, in database.py. Querying runs on every question, in query.py.

## Indexing steps (database.py)

1. Load: PyPDFDirectoryLoader
Library: langchain-community, which uses pypdf underneath.
It reads every PDF in data/ and returns a list of Document objects, one per page.

2. Split: RecursiveCharacterTextSplitter
Library: langchain-text-splitters.
A page is too big to search precisely, and an LLM has a limited context window. So pages are cut into chunks of about 800 characters.

3. Embed: HuggingFaceEmbeddings
Libraries: langchain-huggingface, sentence-transformers, torch.
An embedding model turns text into a list of numbers (a vector).

4. Store: Chroma
Libraries: chromadb (the database) and langchain-chroma (the LangChain wrapper).
Chroma is a vector database. It stores each vector, its text and its metadata on disk in the chroma/ folder.

## Query (query.py)

1. Embed the question and search
It embeds the question with the same model, then returns the 15 closest chunks.

2. Built the augmented new prompt
Library: langchain-core.
The 15 results are grouped by source file, keeping up to 3 chunks per file. The first 2 files are put into a comparison template.

3. Generate: Ollama
Libraries: langchain-community and ollama
Sends the prompt and returns the answer text.
The answer is based on the retrieved text, not on what the model memorised.

