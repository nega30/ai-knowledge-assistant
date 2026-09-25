import os
import faiss
import pickle
import numpy as np

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from sklearn.feature_extraction.text import TfidfVectorizer

from .generator import generate_answer
from .query_rewriter import rewrite_query
from .document_processor import (
    extract_pdf_pages,
    create_chunks,
    build_faiss_index
)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Knowledge Assistant",
    description="RAG-based PDF Question Answering API",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):

    question: str


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
# LOAD CHUNKS
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

print("Embedding model loaded.")


# ============================================================
# LOAD RERANKER
# ============================================================

print("Loading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded.")


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

print("TF-IDF index ready.")


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_history = []


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Knowledge Assistant API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "vectors": index.ntotal,
        "chunks": len(chunks)
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    question = request.question.strip()


    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question:

        return {
            "answer": "Please provide a question.",
            "sources": []
        }


    # ========================================================
    # STEP 1 — BUILD CONVERSATION HISTORY
    # ========================================================

    history_text = ""

    recent_history = conversation_history[-5:]


    for item in recent_history:

        history_text += f"""
Previous Question:
{item["question"]}

Previous Answer:
{item["answer"]}

"""


    # ========================================================
    # STEP 2 — QUERY REWRITING
    # ========================================================

    rewritten_question = rewrite_query(
        question,
        history_text
    )


    # ========================================================
    # STEP 3 — CREATE QUERY EMBEDDING
    # ========================================================

    query_embedding = embedding_model.encode(
        [rewritten_question],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    # ========================================================
    # STEP 4 — SEMANTIC SEARCH
    # ========================================================

    semantic_scores, semantic_indices = index.search(
        query_embedding,
        min(CANDIDATE_K, index.ntotal)
    )


    # ========================================================
    # STEP 5 — KEYWORD SEARCH
    # ========================================================

    query_tfidf = tfidf_vectorizer.transform(
        [rewritten_question]
    )


    keyword_scores = (
        tfidf_matrix @ query_tfidf.T
    ).toarray().flatten()


    # ========================================================
    # STEP 6 — HYBRID RETRIEVAL
    # ========================================================

    combined_scores = []


    for chunk_id in range(
        len(chunks)
    ):

        semantic_score = 0.0

        keyword_score = keyword_scores[
            chunk_id
        ]


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


        hybrid_score = (
            0.7 * semantic_score
            + 0.3 * keyword_score
        )


        combined_scores.append(
            hybrid_score
        )


    # ========================================================
    # STEP 7 — RANK RESULTS
    # ========================================================

    ranked_indices = np.argsort(
        combined_scores
    )[::-1]


    best_score = combined_scores[
        ranked_indices[0]
    ]


    # ========================================================
    # STEP 8 — RELEVANCE CHECK
    # ========================================================

    if best_score < SIMILARITY_THRESHOLD:

        answer = (
            "I could not find relevant "
            "information in the provided document."
        )

        return {
            "question": question,
            "rewritten_question": rewritten_question,
            "answer": answer,
            "sources": []
        }


    # ========================================================
    # STEP 9 — SELECT CANDIDATES
    # ========================================================

    candidate_indices = ranked_indices[
        :CANDIDATE_K
    ]


    # ========================================================
    # STEP 10 — RERANK
    # ========================================================

    pairs = [

        (
            rewritten_question,
            chunks[index]["text"]
        )

        for index in candidate_indices

    ]


    reranker_scores = reranker.predict(
        pairs
    )


    reranked_results = sorted(

        zip(
            candidate_indices,
            reranker_scores
        ),

        key=lambda x: x[1],

        reverse=True

    )


    top_results = reranked_results[
        :TOP_K
    ]


    # ========================================================
    # STEP 11 — BUILD CONTEXT
    # ========================================================

    context = ""

    sources = []


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


        chunk_text = chunk[
            "text"
        ][
            :MAX_CONTEXT_CHARS_PER_CHUNK
        ]


        context += f"""

[Source {rank}]

Document:
{chunk["document"]}

Page:
{chunk["page"]}

Content:
{chunk_text}

"""


        sources.append({

            "document": chunk[
                "document"
            ],

            "page": chunk[
                "page"
            ],

            "hybrid_score": round(
                float(
                    combined_scores[
                        chunk_index
                    ]
                ),
                4
            ),

            "reranker_score": round(
                float(
                    reranker_score
                ),
                4
            )

        })


    # ========================================================
    # STEP 12 — GENERATE ANSWER
    # ========================================================

    answer = generate_answer(

        context=context,

        question=question,

        conversation_history=history_text

    )


    # ========================================================
    # STEP 13 — SAVE CONVERSATION
    # ========================================================

    conversation_history.append({

        "question": question,

        "answer": answer

    })


    # ========================================================
    # STEP 14 — RETURN RESPONSE
    # ========================================================

    return {

        "question": question,

        "rewritten_question": rewritten_question,

        "answer": answer,

        "sources": sources

    }
# ============================================================
# UPLOAD PDF
# ============================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )


    # --------------------------------------------------------
    # Save uploaded PDF
    # --------------------------------------------------------

    os.makedirs(
        "data/documents",
        exist_ok=True
    )


    file_path = os.path.join(
        "data/documents",
        file.filename
    )


    file_content = await file.read()


    with open(
        file_path,
        "wb"
    ) as output_file:

        output_file.write(
            file_content
        )


    # --------------------------------------------------------
    # Extract PDF pages
    # --------------------------------------------------------

    pages = extract_pdf_pages(
        file_path
    )


    if not pages:

        raise HTTPException(
            status_code=400,
            detail="Could not extract text from this PDF."
        )


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        pages
    )


    if not chunks:

        raise HTTPException(
            status_code=400,
            detail="No text chunks were created."
        )


    # --------------------------------------------------------
    # Build FAISS index
    # --------------------------------------------------------

    chunk_count = build_faiss_index(
        chunks,
        file.filename
    )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "message": "PDF uploaded and indexed successfully.",

        "filename": file.filename,

        "pages": len(pages),

        "chunks": chunk_count

    }