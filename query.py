import streamlit as st
import tempfile
import os
from PyPDF2 import PdfReader
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embedding import get_embedding_function

CHROMA_PATH = "chroma"

COMPARISON_PROMPT_TEMPLATE = """
Answer the following question by comparing the content from the two different sources provided.

Source 1 context:
{source1_context}

Source 2 context:
{source2_context}

---

Question: {question}

Provide a detailed comparative answer based on the contexts above.
"""

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
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Initialize document contexts.
    doc_contexts = {}

    # If PDF files are uploaded, process each one.
    if pdf_files:
        for pdf_file in pdf_files:
            pdf_text = extract_text_from_pdf(pdf_file)
            doc_contexts[pdf_file.name] = pdf_text

    # Search the DB for relevant documents.
    results = db.similarity_search_with_score(query_text, k=50)  # Increase k to get more documents

    # Filter results to ensure we get relevant content from two different documents.
    for doc, _score in results:
        if len(doc_contexts) >= 2:
            break
        if doc.metadata.get("source") not in doc_contexts:
            doc_contexts[doc.metadata.get("source")] = doc.page_content

    if len(doc_contexts) < 2:
        raise ValueError("Not enough distinct documents found for comparison.")

    # Extract contexts.
    doc_ids = list(doc_contexts.keys())
    context_text_doc1 = doc_contexts[doc_ids[0]]
    context_text_doc2 = doc_contexts[doc_ids[1]]

    # Create the comparison prompt.
    prompt_template = ChatPromptTemplate.from_template(COMPARISON_PROMPT_TEMPLATE)
    prompt = prompt_template.format(source1_context=context_text_doc1, source2_context=context_text_doc2, question=query_text)

    # Invoke the model.
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    formatted_response = f"Response: {response_text}\nSources: {doc_ids}"
    return formatted_response

def main():
    st.title("RAG Chatbot for Document Comparison")

    query_text = st.text_input("Enter your query:")
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Submit"):
        try:
            if pdf_files:
                # Save the uploaded PDF files to temporary locations.
                temp_pdf_files = []
                for pdf_file in pdf_files:
                    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                        temp_pdf.write(pdf_file.read())
                    temp_pdf_files.append(temp_pdf)

                # Perform query with uploaded PDF files.
                response = query_rag(query_text, pdf_files=temp_pdf_files)

                # Remove temporary files after use.
                for temp_pdf in temp_pdf_files:
                    os.remove(temp_pdf.name)
            else:
                # Perform query without uploaded PDF files.
                response = query_rag(query_text)

            st.text(response)
        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
