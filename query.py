import streamlit as st
import tempfile
import os
from PyPDF2 import PdfReader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embedding import get_embedding_function
import logging
from database import split_documents, add_to_chroma, load_documents
from collections import defaultdict


logging.basicConfig(
    filename="rag.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


CHROMA_PATH = "chroma"

COMPARISON_PROMPT_TEMPLATE = """
Compare the two sources below and answer the question directly with specific facts and figures from each source.

Source 1 {source1}:
{source1_context}

Source 2 {source2}:
{source2_context}

Question: {question}

Using the contexts above, answer with 1-2 relevant findings per source."""

@st.cache_resource
def load_embedding_function():
    return get_embedding_function()

def process_uploaded_pdfs(pdf_files):
    with st.status("📄 Processing uploaded PDFs...", expanded=True) as status:
            for pdf_file in pdf_files:
                # save to /data
                save_path = os.path.join("data", pdf_file.name)
                with open(save_path, "wb") as f:
                    f.write(pdf_file.read())
                status.write(f"Saved {pdf_file.name} to /data")
            
            # reindex the entire /data folder including new files
            documents = load_documents()
            chunks = split_documents(documents)
            add_to_chroma(chunks)
            status.update(label="✅ PDFs indexed into DB")

def extract_text_from_pdf(file):
    with open(file.name, "rb") as f:
        pdf = PdfReader(f)
        text = ""
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text += page.extract_text()
        return text

def query_rag(query_text: str, pdf_files=None):
    # Prepare the DB.
    with st.status("🔍 Loading embedding model...", expanded=True) as status:
        embedding_function = load_embedding_function()
        status.update(label="✅ Embedding model loaded")
    
    with st.status("📚 Connecting to vector database...", expanded=True) as status:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        status.update(label="✅ Database connected")

    # this is a dictionary to collect the content from at least 2 different documents since we are comparing
    source_chunks = defaultdict(list)

    # If PDF files are uploaded, process each one.
    if pdf_files:
        process_uploaded_pdfs(pdf_files)

    # Search the DB for relevant documents.
    results = db.similarity_search_with_score(query_text, k=15)

    # The loop filter results to ensure we get relevant content from two different documents.
    # Improved retrieval logic by collecting top three chunks per document to increase context for model; the previous logic retrieved only one best chunk per document 
    
    for doc, _score in results:
        source = doc.metadata.get("source")

        if len(source_chunks[source]) < 3:
            source_chunks[source].append(doc.page_content)
    
    logging.info(f"size of source_chunks: {len(source_chunks)}")

    # Extract contexts.
    doc_ids = list(source_chunks.keys())[:2]
    context_text_doc1 = "\n\n".join(source_chunks[doc_ids[0]])
    context_text_doc2 = "\n\n".join(source_chunks[doc_ids[1]])

    # Create the comparison prompt.
    prompt_template = ChatPromptTemplate.from_template(COMPARISON_PROMPT_TEMPLATE)
    prompt = prompt_template.format(source1={doc_ids[0]}, source1_context=context_text_doc1, source2={doc_ids[1]}, source2_context=context_text_doc2, question=query_text)

    # Invoke the model.
    with st.status("🤖 Generating response with AI...", expanded=True) as status:
        model = Ollama(model="llama3.2:3b")
        response_text = model.invoke(prompt)
        status.update(label="✅ Response generated")

    formatted_response = f"Response: {response_text}\nSources: {doc_ids}"
    return formatted_response

def main():
    st.title("RAG Chatbot for Document Comparison")

    with st.form("query_form"):
        query_text = st.text_input("Enter your query:", placeholder="Type your question here...")
        pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        submitted = st.form_submit_button("Submit Query", use_container_width=True)

    if submitted:
        try:
            if query_text.strip() == "":
                st.error("Please enter a query.")
                return

            response = query_rag(query_text, pdf_files=pdf_files if pdf_files else None)
            st.text(response)
        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
