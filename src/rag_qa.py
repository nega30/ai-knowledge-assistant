import faiss
import pickle
import numpy as np

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from sklearn.feature_extraction.text import TfidfVectorizer

from generator import generate_answer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_PATH = "data/index/faiss.index"
CHUNKS_PATH = "data/index/chunks.pkl"

TOP_K = 3
CANDIDATE_K = 6

SIMILARITY_THRESHOLD = 0.15

MAX_CONTEXT_CHARS_PER_CHUNK = 800


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(
    INDEX_PATH
)

print(
    "Vectors loaded:",
    index.ntotal
)


# ============================================================
# LOAD CHUNK METADATA
# ============================================================

print("Loading chunk metadata...")

with open(
    CHUNKS_PATH,
    "rb"
) as file:

    chunks = pickle.load(file)


print(
    "Chunks loaded:",
    len(chunks)
)


# ============================================================
# EXTRACT CHUNK TEXT
# ============================================================

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Embedding model loaded."
)


# ============================================================
# LOAD RERANKER
# ============================================================

print("Loading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print(
    "Reranker loaded."
)


# ============================================================
# CREATE TF-IDF INDEX
# ============================================================

print("Creating TF-IDF index...")

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

tfidf_matrix = tfidf_vectorizer.fit_transform(
    chunk_texts
)

print(
    "TF-IDF index ready."
)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_history = []


# ============================================================
# MAIN QUESTION LOOP
# ============================================================

while True:

    question = input(
        "\nAsk a question about the PDF "
        "(type 'exit' to quit): "
    )


    # ========================================================
    # EXIT
    # ========================================================

    if question.lower() == "exit":

        print("\nGoodbye!")

        break


    # ========================================================
    # SHOW CONVERSATION HISTORY
    # ========================================================

    if question.lower() == "history":

        print(
            "\n=============================="
        )

        print(
            "CONVERSATION HISTORY"
        )

        print(
            "=============================="
        )


        if not conversation_history:

            print(
                "No conversation history yet."
            )

        else:

            for i, item in enumerate(
                conversation_history,
                start=1
            ):

                print(
                    f"\nQ{i}: {item['question']}"
                )

                print(
                    f"A{i}: {item['answer']}"
                )


        continue


    # ========================================================
    # STEP 1 — CREATE QUERY EMBEDDING
    # ========================================================

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )


    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    # ========================================================
    # STEP 2 — SEMANTIC SEARCH
    # ========================================================

    semantic_scores, semantic_indices = index.search(
        query_embedding,
        min(
            CANDIDATE_K,
            index.ntotal
        )
    )


    # ========================================================
    # STEP 3 — KEYWORD SEARCH
    # ========================================================

    query_tfidf = tfidf_vectorizer.transform(
        [question]
    )


    keyword_scores = (
        tfidf_matrix @ query_tfidf.T
    ).toarray().flatten()


    # ========================================================
    # STEP 4 — HYBRID RETRIEVAL
    # ========================================================

    combined_scores = []


    for chunk_id in range(
        len(chunks)
    ):

        semantic_score = 0.0

        keyword_score = keyword_scores[
            chunk_id
        ]


        # Find semantic score
        for i in range(
            len(semantic_indices[0])
        ):

            if (
                semantic_indices[0][i]
                == chunk_id
            ):

                semantic_score = float(
                    semantic_scores[0][i]
                )

                break


        # Combine scores
        hybrid_score = (
            0.7 * semantic_score
            + 0.3 * keyword_score
        )


        combined_scores.append(
            hybrid_score
        )


    # ========================================================
    # STEP 5 — SORT HYBRID RESULTS
    # ========================================================

    ranked_indices = np.argsort(
        combined_scores
    )[::-1]


    best_score = combined_scores[
        ranked_indices[0]
    ]


    print(
        "\n=============================="
    )

    print(
        "HYBRID RETRIEVAL"
    )

    print(
        "=============================="
    )


    print(
        "Best hybrid score:",
        round(
            float(best_score),
            4
        )
    )


    # ========================================================
    # STEP 6 — RELEVANCE CHECK
    # ========================================================

    if (
        best_score
        < SIMILARITY_THRESHOLD
    ):

        print(
            "\n=============================="
        )

        print(
            "RESULT"
        )

        print(
            "=============================="
        )


        print(
            "I could not find relevant "
            "information in the provided document."
        )


        continue


    # ========================================================
    # STEP 7 — SELECT CANDIDATES
    # ========================================================

    candidate_indices = ranked_indices[
        :CANDIDATE_K
    ]


    # ========================================================
    # STEP 8 — CROSS-ENCODER RERANKING
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "RERANKING"
    )

    print(
        "=============================="
    )


    pairs = [
        (
            question,
            chunks[index]["text"]
        )

        for index in candidate_indices
    ]


    reranker_scores = reranker.predict(
        pairs
    )


    # ========================================================
    # STEP 9 — SORT RERANKER RESULTS
    # ========================================================

    reranked_results = sorted(
        zip(
            candidate_indices,
            reranker_scores
        ),
        key=lambda x: x[1],
        reverse=True
    )


    # ========================================================
    # STEP 10 — SELECT TOP 3
    # ========================================================

    top_results = reranked_results[
        :TOP_K
    ]


    # ========================================================
    # STEP 11 — BUILD CONTEXT
    # ========================================================

    context = ""

    source_pages = set()

    source_number = 1


    print(
        "\n=============================="
    )

    print(
        "FINAL RETRIEVED SOURCES"
    )

    print(
        "=============================="
    )


    for rank, (
        chunk_index,
        reranker_score
    ) in enumerate(
        top_results,
        start=1
    ):

        chunk = chunks[
            chunk_index
        ]


        hybrid_score = combined_scores[
            chunk_index
        ]


        print(
            f"\nSource {rank}"
        )


        print(
            "Document:",
            chunk["document"]
        )


        print(
            "Page:",
            chunk["page"]
        )


        print(
            "Hybrid Score:",
            round(
                float(hybrid_score),
                4
            )
        )


        print(
            "Reranker Score:",
            round(
                float(reranker_score),
                4
            )
        )


        # ----------------------------------------------------
        # LIMIT CONTEXT SIZE
        # ----------------------------------------------------

        chunk_text = chunk["text"][
            :MAX_CONTEXT_CHARS_PER_CHUNK
        ]


        context += f"""
[Source {source_number}]
Document: {chunk["document"]}
Page: {chunk["page"]}

{chunk_text}

"""


        source_number += 1


        source_pages.add(
            (
                chunk["document"],
                chunk["page"]
            )
        )


    # ========================================================
    # STEP 12 — CONVERSATION HISTORY
    # ========================================================

    history_text = ""


    recent_history = conversation_history[
        -5:
    ]


    for item in recent_history:

        history_text += f"""
Previous Question:
{item["question"]}

Previous Answer:
{item["answer"]}

"""


    # ========================================================
    # STEP 13 — GENERATE ANSWER
    # ========================================================

    answer = generate_answer(
        context=context,
        question=question,
        conversation_history=history_text
    )


    # ========================================================
    # STEP 14 — DISPLAY ANSWER
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "AI ANSWER"
    )

    print(
        "=============================="
    )


    print(
        answer
    )


    # ========================================================
    # STEP 15 — DISPLAY SOURCES
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "SOURCES"
    )

    print(
        "=============================="
    )


    for document, page in sorted(
        source_pages
    ):

        print(
            f"📄 {document} — Page {page}"
        )


    # ========================================================
    # STEP 16 — SAVE CONVERSATION
    # ========================================================

    conversation_history.append(
        {
            "question": question,
            "answer": answer
        }
    )