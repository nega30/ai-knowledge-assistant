from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


texts = [
    "Artificial Intelligence allows computers to perform intelligent tasks.",
    "Machine Learning allows computers to learn patterns from data.",
    "Deep Learning uses neural networks with multiple layers.",
    "RAG combines information retrieval with large language models."
]


# Convert text into embeddings
embeddings = model.encode(texts)


print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))

print("\nFirst embedding:")
print(embeddings[0])