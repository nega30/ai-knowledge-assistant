#  AI Knowledge Assistant

An AI-powered document question-answering system built using
Retrieval-Augmented Generation (RAG).

The system allows users to ask questions about a PDF and generates
grounded answers using information retrieved from the document.

---

##  Features

- PDF document processing
- Smart document chunking
- Semantic embeddings
- FAISS vector search
- TF-IDF keyword search
- Hybrid retrieval
- Cross-Encoder reranking
- Local LLM using FLAN-T5
- Source/page references
- Hallucination reduction
- Conversation history
- Retrieval evaluation

---

##  Architecture

```text
                    PDF DOCUMENT
                         │
                         ▼
                  Text Extraction
                         │
                         ▼
                  Smart Chunking
                         │
                         ▼
                    Embeddings
                         │
                         ▼
                  FAISS Vector DB
                         │
                         ▼
               ┌───────────────────┐
               │ Hybrid Retrieval  │
               │                   │
               │ Semantic + TF-IDF │
               └─────────┬─────────┘
                         │
                         ▼
                  Candidate Chunks
                         │
                         ▼
                Cross-Encoder
                   Reranking
                         │
                         ▼
                    Top Results
                         │
                         ▼
                  Local FLAN-T5
                         │
                         ▼
                 Grounded Answer
                         │
                         ▼
                  Source Citation
