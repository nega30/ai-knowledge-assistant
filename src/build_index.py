import os
import pickle
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PDF_PATH = "data/documents/sample.pdf"

INDEX_PATH = "data/index/faiss.index"
CHUNKS_PATH = "data/index/chunks.pkl"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


# ============================================================
# CREATE INDEX DIRECTORY
# ============================================================

os.makedirs("data/index", exist_ok=True)


# ============================================================
# EXTRACT PDF TEXT PAGE BY PAGE
# ============================================================

print("Loading PDF...")

reader = PdfReader(PDF_PATH)

print("Pages:", len(reader.pages))


# ============================================================
# SMART CHUNKING FUNCTION
# ============================================================

def create_chunks(text, page_number, document_name):

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        # If adding the paragraph stays within the limit
        if len(current_chunk) + len(paragraph) <= CHUNK_SIZE:

            current_chunk += paragraph + "\n\n"

        else:

            if current_chunk.strip():

                chunks.append({
                    "text": current_chunk.strip(),
                    "page": page_number,
                    "document": document_name
                })

            # Keep a small overlap
            overlap_text = current_chunk[
                -CHUNK_OVERLAP:
            ]

            current_chunk = (
                overlap_text
                + "\n\n"
                + paragraph
                + "\n\n"
            )


    # Add final chunk
    if current_chunk.strip():

        chunks.append({
            "text": current_chunk.strip(),
            "page": page_number,
            "document": document_name
        })


    return chunks


# ============================================================
# PROCESS ALL PAGES
# ============================================================

all_chunks = []

document_name = os.path.basename(
    PDF_PATH
)


for page_number, page in enumerate(
    reader.pages,
    start=1
):

    page_text = page.extract_text()

    if not page_text:
        continue


    page_chunks = create_chunks(
        page_text,
        page_number,
        document_name
    )


    all_chunks.extend(
        page_chunks
    )


print(
    "Chunks created:",
    len(all_chunks)
)


# ============================================================
# DISPLAY SAMPLE CHUNKS
# ============================================================

print("\n--- SAMPLE CHUNKS ---")


for i, chunk in enumerate(
    all_chunks[:3]
):

    print(
        f"\nChunk {i + 1}"
    )

    print(
        "Page:",
        chunk["page"]
    )

    print(
        "Text:",
        chunk["text"][:300]
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

texts = [
    chunk["text"]
    for chunk in all_chunks
]


print(
    "Creating embeddings..."
)


embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

print("\nCreating FAISS index...")


dimension = embeddings.shape[1]


index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embeddings.astype("float32")
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

faiss.write_index(
    index,
    INDEX_PATH
)


# ============================================================
# SAVE CHUNK METADATA
# ============================================================

with open(
    CHUNKS_PATH,
    "wb"
) as file:

    pickle.dump(
        all_chunks,
        file
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================")
print("INDEX BUILD COMPLETE")
print("==============================")

print(
    "Total chunks:",
    len(all_chunks)
)

print(
    "Embedding dimension:",
    dimension
)

print(
    "FAISS index:",
    INDEX_PATH
)

print(
    "Metadata:",
    CHUNKS_PATH
)