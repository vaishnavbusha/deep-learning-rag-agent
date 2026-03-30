# System Architecture
## Team: ___________________
## Date: 2026-03-24
## Members and Roles:
- Corpus Architect: ___________________
- Pipeline Engineer: ___________________
- UX Lead: ___________________
- Prompt Engineer: ___________________
- QA Lead: ___________________

---

## Architecture Diagram

```text
User uploads .md/.pdf files
        |
        v
DocumentChunker
  - detects file type
  - splits markdown/PDF into chunks
  - infers metadata from filename
  - generates deterministic chunk_id
        |
        v
VectorStoreManager.ingest()
  - check_duplicate(chunk_id)
  - embed chunk text
  - upsert into ChromaDB PersistentClient
        |
        v
ChromaDB collection (cosine similarity)

User query
   |
   v
LangGraph
  START
    -> query_rewrite_node
    -> retrieval_node
    -> if no chunks above threshold -> END with hallucination guard
    -> else generation_node
    -> END
   |
   v
Structured AgentResponse
  - answer
  - sources
  - confidence
  - no_context_found
  - rewritten_query

Conversation memory is maintained by LangGraph MemorySaver checkpointer
using thread_id from Streamlit session_state.
```

---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:** `.md` now, `.pdf` supported by the pipeline

- **Landmark papers ingested:**
  - Add your downloaded PDFs here before final submission
  - Rumelhart, Hinton & Williams (1986)
  - LeCun et al. (1998)
  - Hochreiter & Schmidhuber (1997)

- **Chunking strategy:**
  Markdown uses header-aware splitting first, then word-based chunking with target size `220` words and overlap `30`. This keeps authored sections semantically coherent while staying close to the rubric's 100-300 word expectation.

- **Metadata schema:**

  | Field | Type | Purpose |
  |---|---|---|
  | topic | string | Lets the retriever filter by major deep learning topic |
  | difficulty | string | Supports interview-level filtering such as beginner or intermediate |
  | type | string | Describes the educational role of the chunk, such as concept explanation |
  | source | string | Preserves attribution and powers source citations in the UI |
  | related_topics | list | Helps explain conceptual adjacency and future retrieval expansion |
  | is_bonus | bool | Marks bonus topics like GAN or SOM for demo and coverage tracking |

- **Duplicate detection approach:**
  Each chunk ID is generated from `sha256(source + "::" + chunk_text)` and truncated to 16 hex characters. This is more reliable than a filename-based check because identical content will still be detected even if a file is renamed and re-uploaded.

- **Corpus coverage:**
  - [x] ANN
  - [x] CNN
  - [x] RNN
  - [x] LSTM
  - [x] Seq2Seq
  - [x] Autoencoder
  - [ ] SOM *(bonus)*
  - [ ] Boltzmann Machine *(bonus)*
  - [x] GAN *(bonus)*

---

### Vector Store Layer

- **Database:** ChromaDB — PersistentClient
- **Local persistence path:** `./data/chroma_db`

- **Embedding model:**
  `all-MiniLM-L6-v2` via sentence-transformers

- **Why this embedding model:**
  It is local, inexpensive, and lightweight enough for a class assignment without requiring paid API embeddings. The tradeoff is lower quality than premium hosted embeddings, but the setup is simpler and more reproducible.

- **Similarity metric:**
  Cosine similarity via ChromaDB collection metadata. This is a natural fit for embedding-space semantic search because it emphasizes vector direction over magnitude.

- **Retrieval k:**
  `4` chunks by default. This is enough to capture multiple relevant contexts without overloading the prompt with too much low-value text.

- **Similarity threshold:**
  `0.3` by default. Any chunk below that score is excluded so the system can return an honest no-context response instead of hallucinating.

- **Metadata filtering:**
  Users can filter by topic and difficulty from the Streamlit UI. Those filters are converted into ChromaDB `where` conditions during retrieval.

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**

  | Node | Responsibility |
  |---|---|
  | query_rewrite_node | Rewrites the raw user query into a retrieval-friendly keyword query |
  | retrieval_node | Queries ChromaDB with optional topic and difficulty filters |
  | generation_node | Builds grounded context, applies the hallucination guard, and produces the final answer |

- **Conditional edges:**
  After retrieval, the graph checks `no_context_found`. If no chunks meet threshold, it routes directly to `END`; otherwise it continues to `generation_node`.

- **Hallucination guard:**
  The system returns a no-context message explaining that relevant information was not found in the corpus and suggests trying a more specific deep learning topic.

- **Query rewriting:**
  - Raw query: `How do LSTMs remember information for a long time?`
  - Rewritten query: `LSTM long-term memory cell state forget gate mechanism`

- **Conversation memory:**
  LangGraph uses `MemorySaver` with a `thread_id` from Streamlit `session_state`. Message history is trimmed when it approaches `MAX_CONTEXT_TOKENS`.

- **LLM provider:**
  Configurable through `.env`: Groq, Ollama, or LM Studio

- **Why this provider:**
  The codebase uses a provider factory so the same application can run with a cloud API or fully local inference. That makes the architecture easier to defend in an interview because the backend choice is configuration-driven rather than hardcoded.

---

### Prompt Layer

- **System prompt summary:**
  The agent acts like a rigorous but supportive senior machine learning interviewer. It is explicitly constrained to answer only from retrieved context and always cite sources.

- **Question generation prompt:**
  It takes retrieved context plus a difficulty level and returns structured JSON containing a question, model answer, follow-up question, and source citations.

- **Answer evaluation prompt:**
  It takes a question, a candidate answer, and source context, then returns JSON with a score, strengths, gaps, an ideal answer, verdict, and coaching tip.

- **JSON reliability:**
  The prompts explicitly instruct the model to respond with JSON only and to avoid preambles or markdown fences.

- **Failure modes identified:**
  - Query rewriting can fail when no LLM credentials are configured, so the node falls back to the original query.
  - Question generation can drift into trivial recall questions if the context is too narrow.
  - Answer evaluation can score too generously unless the rubric is strict about missing details.

---

### Interface Layer

- **Framework:** Streamlit
- **Deployment platform:** Streamlit Community Cloud or HuggingFace Spaces
- **Public URL:** ___________________

- **Ingestion panel features:**
  Multi-file uploader for `.md` and `.pdf`, ingestion status, duplicate counts, error display, document list, and per-document delete action.

- **Document viewer features:**
  Users can select an ingested source file and view all stored chunks with topic, difficulty, and type metadata.

- **Chat panel features:**
  The chat shows conversation history, grounded source citations, a visible no-context warning, optional topic and difficulty filters, plus dedicated tabs for interview question generation and answer evaluation.

- **Session state keys:**

  | Key | Stores |
  |---|---|
  | chat_history | The rendered chat transcript shown in the UI |
  | ingested_documents | Current document list from ChromaDB |
  | selected_document | The source currently open in the viewer |
  | thread_id | The LangGraph conversation thread identifier |
  | topic_filter | Optional retrieval topic filter |
  | difficulty_filter | Optional retrieval difficulty filter |

- **Stretch features implemented:**
  - Query rewriting before retrieval
  - Corpus statistics panel
  - Duplicate-safe ingestion

---

## Design Decisions

1. **Decision:** Use local sentence-transformer embeddings by default.
   **Rationale:** This avoids paid embedding APIs and keeps ingestion reproducible for a class environment.
   **Interview answer:** We chose local embeddings because they eliminate API cost and reduce setup friction. The tradeoff is lower embedding quality than premium hosted models, but for a course demo the simplicity and privacy benefits were worth it.

2. **Decision:** Use deterministic chunk IDs based on content hash.
   **Rationale:** Duplicate detection should be tied to actual content, not filenames or upload order.
   **Interview answer:** A content hash gives us content-addressed deduplication, which is more robust than checking filenames. If the same chunk is uploaded twice under different names, the system still skips it.

3. **Decision:** Apply a similarity threshold before generation.
   **Rationale:** Low-similarity retrieval results are more dangerous than no results because they invite grounded-sounding hallucinations.
   **Interview answer:** We deliberately prefer a no-context answer over a weakly grounded answer. In production RAG, knowing when not to answer is often more important than answering every query.

4. **Decision:** Use a separate query rewrite step before retrieval.
   **Rationale:** Users ask questions conversationally, but embeddings often perform better with concise technical phrasing.
   **Interview answer:** Query rewriting improves retrieval recall by converting vague natural language into a denser retrieval query. We also added a safe fallback so the graph still works if rewriting fails.

---

## QA Test Results

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks returned with citations | Retrieval path implemented and verified by unit tests plus local app import | Pass |
| Off-topic query | Hallucination guard fires | Similarity threshold tested locally with empty retrieval result | Pass |
| Duplicate ingestion | Second upload skipped | Duplicate detection covered by unit tests | Pass |
| Empty query | Graceful handling without crash | Chat input blocks empty submission; retrieval returns empty list for blank queries | Pass |
| Cross-topic query | Multi-concept answer from corpus | Supported by multi-topic corpus and metadata-aware retrieval; demo still recommended | Ready for demo |
