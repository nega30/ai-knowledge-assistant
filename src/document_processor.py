import os
import pickle

import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "data/documents"

INDEX_PATH = "data/index/faiss.index"

CHUNKS_PATH = "data/index/chunks.pkl"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model for document processing...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model ready.")


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

def extract_pdf_pages(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(
    pages,
    chunk_size=500,
    overlap=100
):

    chunks = []


    for page_data in pages:

        text = page_data["text"]

        page_number = page_data["page"]


        start = 0


        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]


            if chunk_text.strip():

                chunks.append(
                    {
                        "text": chunk_text.strip(),

                        "page": page_number
                    }
                )


            start += (
                chunk_size - overlap
            )


    return chunks


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(
    chunks,
    document_name
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    # Create embeddings

    embeddings = embedding_model.encode(

        texts,

        normalize_embeddings=True,

        show_progress_bar=True

    )


    embeddings = np.array(
        embeddings
    ).astype("float32")


    # Create FAISS index

    dimension = embeddings.shape[1]


    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        embeddings
    )


    # Add document information

    for chunk in chunks:

        chunk["document"] = document_name


    # Save index

    os.makedirs(
        "data/index",
        exist_ok=True
    )


    faiss.write_index(
        index,
        INDEX_PATH
    )


    # Save metadata

    with open(
        CHUNKS_PATH,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )


    return len(chunks)