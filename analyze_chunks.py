import logging
from collections import defaultdict
from langchain_chroma import Chroma
from embedding import get_embedding_function

# ---------- CONFIG ----------
CHROMA_PATH = "chroma"
LOG_FILE = "chunk_analysis.log"
# ----------------------------

# ---------- LOGGING SETUP ----------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
# ----------------------------------


def analyze_chunks():
    logging.info("===== CHUNK ANALYSIS START =====\n")

    # Connect to existing DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    data = db.get(include=["documents", "metadatas"])

    documents = data["documents"]
    metadatas = data["metadatas"]

    # Group by source document
    docs_grouped = defaultdict(list)

    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "unknown")
        docs_grouped[source].append((doc, meta))

    # Analyze each document
    for source, chunks in docs_grouped.items():
        logging.info(f"--- Document: {source} ---")
        logging.info(f"Total chunks: {len(chunks)}")

        total_chars = 0
        page_distribution = defaultdict(int)

        for i, (chunk_text, meta) in enumerate(chunks):
            chunk_len = len(chunk_text)
            total_chars += chunk_len

            page = meta.get("page", "unknown")
            chunk_id = meta.get("id", "no-id")

            page_distribution[page] += 1

            logging.info(
                f"Chunk {i+1} | ID: {chunk_id} | Page: {page} | Length: {chunk_len}"
            )

        avg_chunk_size = total_chars / len(chunks) if chunks else 0

        logging.info(f"Average chunk size: {avg_chunk_size:.2f} characters")

        logging.info("Chunks per page:")
        for page, count in page_distribution.items():
            logging.info(f"  Page {page}: {count} chunks")

        logging.info("")

    logging.info("===== CHUNK ANALYSIS END =====")


if __name__ == "__main__":
    analyze_chunks()
    