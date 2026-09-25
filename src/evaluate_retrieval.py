import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
import pickle


# -----------------------------
# Load evaluation questions
# -----------------------------

with open(
    "data/evaluation_questions.json",
    "r"
) as file:

    questions = json.load(file)


# -----------------------------
# Load FAISS index
# -----------------------------

index = faiss.read_index(
    "data/index/faiss.index"
)


# -----------------------------
# Load chunks
# -----------------------------

with open(
    "data/index/chunks.pkl",
    "rb"
) as file:

    chunks = pickle.load(file)


# -----------------------------
# Load embedding model
# -----------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# Evaluate
# -----------------------------

total = len(questions)

passed = 0


for item in questions:

    question = item["question"]

    expected_keywords = item["expected_keywords"]

    query_embedding = model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        3
    )

    retrieved_text = ""

    for index_id in indices[0]:

        retrieved_text += (
            chunks[index_id]["text"]
            + " "
        )

    retrieved_text = retrieved_text.lower()

    matched_keywords = []

    for keyword in expected_keywords:

        if keyword.lower() in retrieved_text:

            matched_keywords.append(keyword)

    score = (
        len(matched_keywords)
        / len(expected_keywords)
    )

    print("\n==============================")
    print("Question:", question)
    print("Expected:", expected_keywords)
    print("Matched:", matched_keywords)
    print("Score:", round(score, 2))

    if score >= 0.5:

        passed += 1
        print("Status: PASS")

    else:

        print("Status: FAIL")


# -----------------------------
# Overall result
# -----------------------------

accuracy = passed / total


print("\n==============================")
print("RETRIEVAL EVALUATION")
print("==============================")

print(
    "Questions:",
    total
)

print(
    "Passed:",
    passed
)

print(
    "Retrieval score:",
    round(accuracy * 100, 2),
    "%"
)