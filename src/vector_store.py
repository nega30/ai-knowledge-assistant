import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Our document chunks
chunks = [
    "Artificial Intelligence allows computers to perform intelligent tasks.",
    "Machine Learning allows computers to learn patterns from data.",
    "Deep Learning uses neural networks with multiple layers.",
    "RAG combines information retrieval with large language models."
]


# Convert chunks into embeddings
embeddings = model.encode(chunks)

# FAISS expects float32
embeddings = np.array(embeddings).astype("float32")


print("Embedding shape:", embeddings.shape)


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
index.add(embeddings)


print("Number of vectors in FAISS:", index.ntotal)


# -----------------------------
# Search
# -----------------------------

query = "What is machine learning?"

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")


# Find 2 most similar chunks
distances, indices = index.search(query_embedding, k=2)


print("\nQuery:", query)

print("\nRetrieved chunks:")

for i in range(len(indices[0])):
    chunk_index = indices[0][i]

    print(f"\nResult {i + 1}")
    print("Distance:", distances[0][i])
    print("Chunk:", chunks[chunk_index])