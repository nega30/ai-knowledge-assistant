from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# ==========================================
# 1. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================================
# 2. SPLIT TEXT INTO CHUNKS
# ==========================================

def split_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# 3. LOAD PDF
# ==========================================

pdf_path = "data/documents/sample.pdf"

text = extract_text_from_pdf(pdf_path)

print("Total characters:", len(text))


# ==========================================
# 4. CREATE CHUNKS
# ==========================================

chunks = split_text(text)

print("Number of chunks:", len(chunks))


# ==========================================
# 5. LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 6. CREATE EMBEDDINGS
# ==========================================

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)


# ==========================================
# 7. CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# ==========================================
# 8. ASK QUESTION
# ==========================================

query = input("\nAsk a question about the PDF: ")


# ==========================================
# 9. EMBED THE QUESTION
# ==========================================

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")


# ==========================================
# 10. SEARCH FAISS
# ==========================================

k = 3

distances, indices = index.search(
    query_embedding,
    k
)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n===== RETRIEVED CONTEXT =====")

for i in range(len(indices[0])):

    chunk_index = indices[0][i]

    print(f"\n--- RESULT {i + 1} ---")

    print("Distance:", distances[0][i])

    print(chunks[chunk_index])