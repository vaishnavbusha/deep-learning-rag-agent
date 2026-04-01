# System Architecture
## Team: Vaishnav Busha, Nishith Mareddy, Jyothi Swaroop Ganapavarapu, Mohith Vepanjeri
## Date: 2026-03-24
## Members and Roles:
- Corpus Architect: Vaishnav Busha
- Pipeline Engineer: Nishith Mareddy
- UX Lead: Jyothi Swaroop Ganapavarapu
- Prompt Engineer: Mohith Vepanjeri
- QA Lead: Shared by team

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
    -> generation_node
       (generation_node returns the hallucination guard response
        if retrieval found no chunks above threshold)
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
  - Rumelhart, Hinton & Williams (1986)
  - LeCun et al. (1998)
  - Hochreiter & Schmidhuber (1997)
  - Sutskever, Vinyals & Le (2014)
  - Hinton & Salakhutdinov (2006) supporting PDF
  - Goodfellow et al. (2014)
  - Core RNN / Elman (1990) still pending stronger primary-source coverage

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
  - [x] SOM *(bonus)*
  - [x] Boltzmann Machine *(bonus)*
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
  The current compiled graph always routes retrieval output into `generation_node`. The hallucination guard is enforced inside `generation_node`, which checks `no_context_found` and returns a no-context response without attempting a grounded answer when retrieval found nothing above threshold.

- **Hallucination guard:**
  The system returns this no-context response when no retrieved chunks meet the similarity threshold:

  ```text
  I was unable to find relevant information in the study corpus for your query.

  This may mean:
  - The topic is not yet covered in the corpus (check if it is a bonus topic)
  - Your query needs to be more specific (try including the exact topic name)
  - The corpus needs more content on this area

  Suggested next steps:
  - Rephrase your query with specific deep learning terminology
  - Check which topics are available using the corpus browser
  - If you are the Corpus Architect, consider adding content on this topic

  Topics currently available: ANN, CNN, RNN, LSTM, Seq2Seq, Autoencoder
  Bonus topics (if ingested): SOM, Boltzmann Machines, GAN
  ```

- **Query rewriting:**
  - Raw query: `How do LSTMs remember information for a long time?`
  - Rewritten query: `LSTM long-term memory cell state forget gate mechanism`

- **Conversation memory:**
  LangGraph uses `MemorySaver` with a `thread_id` from Streamlit `session_state`. Message history is trimmed when it approaches `MAX_CONTEXT_TOKENS`.

- **LLM provider:**
  Groq using `llama-3.1-8b-instant` in the current local configuration. The codebase also supports Ollama and LM Studio through the provider factory.

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
- **Deployment platform:** Streamlit Community Cloud
- **Public URL:** https://deep-learning-rag-agent-dtsc5082.streamlit.app/

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
  | last_ingestion_result | Most recent ingestion summary shown in the sidebar |
  | thread_id | The LangGraph conversation thread identifier |
  | topic_filter | Optional retrieval topic filter |
  | difficulty_filter | Optional retrieval difficulty filter |
  | last_generated_question | Most recent grounded interview question output |
  | last_answer_evaluation | Most recent grounded answer evaluation output |

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
| Normal query | Relevant chunks returned with citations | Retrieval smoke test returned relevant RNN/LSTM results for `Explain the vanishing gradient problem`; final UI citation capture still recommended | Pass |
| Off-topic query | Hallucination guard fires | Retrieval smoke test returned `0` chunks for `What is the capital of France?`, which is the expected precursor to the no-context response | Pass |
| Duplicate ingestion | Second upload skipped | Smoke test on real corpus files ingested `9` chunks on first pass and skipped `9` on second pass | Pass |
| Empty query | Graceful handling without crash | `st.chat_input` blocks blank submission and `VectorStoreManager.query()` returns `[]` for blank input | Pass |
| Cross-topic query | Multi-concept answer from corpus | Retrieval smoke test returned chunks from LSTM, RNN, and Seq2Seq for the cross-topic query | Pass |

**Critical failures fixed before Hour 3:**
- Corrected the documented chunk-size explanation to match the implemented `220` word target with `30` word overlap
- Aligned the LangGraph documentation with the actual control flow, where the hallucination guard is enforced inside `generation_node`

**Known issues not fixed (and why):**
- The starter rubric references an Elman (1990) RNN paper, but the repo currently documents that stronger primary-source coverage is still pending
- The final deployed public URL is not included yet because deployment is still pending

---

## Known Limitations

- PDF chunking can still introduce noisy content from academic paper formatting, especially around references, headers, and page artifacts
- The similarity threshold of `0.3` was chosen as a practical classroom default and smoke-tested, but it was not tuned through a large empirical evaluation
- Conversation memory is maintained in-memory through LangGraph `MemorySaver`, so history is not preserved across a full application restart
- Prompt reliability depends on the configured LLM provider, so JSON-heavy workflows can still vary slightly across providers even with strict prompt instructions

---

## What We Would Do With More Time

- Add a stronger retrieval evaluation workflow that logs actual similarity scores and compares threshold choices across a larger set of queries
- Improve PDF ingestion by filtering reference sections and other low-value paper artifacts before chunking
- Add a true retry or fallback retrieval branch in LangGraph instead of documenting a minimal single-path flow
- Deploy the application publicly and capture a stable hosted URL for easier live judging and replay

---

## Hour 3 Interview Questions

**Question 1:** Why is a content-hash-based chunk ID more reliable than checking only the filename during ingestion?
Model answer: A content hash ties duplicate detection to the actual chunk text plus source identity, so the system can detect repeated uploads even if a file is renamed or uploaded again later. Filename-only checks are weaker because the same content can appear under different names.

**Question 2:** How does the hallucination guard work in your RAG pipeline, and why is it important?
Model answer: Retrieval first filters results by similarity threshold, and when no chunks remain above threshold, the state sets `no_context_found=True`. The generation step then returns a no-context response instead of letting the LLM answer from general model memory, which keeps the system grounded and safer.

**Question 3:** Why did your team choose chunking near 220 words with 30 words of overlap instead of much larger chunks?
Model answer: That setting keeps chunks within the assignment's 100-300 word expectation while preserving enough local context for technical explanations. Larger chunks would mix more ideas together, which can reduce retrieval precision and weaken source-specific grounding.

---

## Team Retrospective

**What clicked:**
- The division of labor across corpus, backend, UI, prompting, and QA made the system easier to build and explain
- Deterministic chunk IDs, metadata filters, and explicit graph state gave the project a clear technical story for demo and interview defense

**What confused us:**
- Aligning the written architecture explanation with the final implemented graph behavior required a cleanup pass after the core features were already working
- Supporting both authored markdown and academic PDFs introduced extra decisions around chunk quality and retrieval noise

**One thing each team member would study before a real interview:**
- Corpus Architect: stronger primary-source coverage for RNN landmark papers and corpus-quality evaluation
- Pipeline Engineer: deeper retrieval tuning, threshold calibration, and graph branching patterns in LangGraph
- UX Lead: deployment hardening and more polished evidence presentation for live demos
- Prompt Engineer: structured-output robustness and prompt calibration across different LLM backends
- QA Lead: larger integration test suites and more systematic failure-mode tracking for RAG applications
