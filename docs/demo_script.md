# 5-Minute Demo Script

## 1. Introduce the System

“This is our Deep Learning RAG Interview Prep Agent built with LangChain, LangGraph, and ChromaDB. The system ingests authored study material, stores it in a vector database, retrieves grounded context, and answers interview-style questions with source citations.”

## 2. Show the Corpus

- Open the document viewer
- Mention the available topics: ANN, CNN, RNN, LSTM, Seq2Seq, Autoencoder, GAN
- Point out that each document is chunked and tagged with metadata such as topic, difficulty, type, source, related topics, and bonus flag

## 3. Demonstrate Ingestion

- Upload one or more `.md` files from `data/corpus/`
- Show the ingestion result summary
- Mention that chunk IDs are deterministic hashes of source and content

## 4. Demonstrate Duplicate Detection

- Upload the same file again
- Show that the second ingest skips duplicates instead of storing them twice
- Explain that this avoids duplicate vectors and keeps retrieval clean

## 5. Demonstrate a Successful Query

- Ask: `Explain the vanishing gradient problem in RNNs`
- Show the answer
- Expand the source citations
- Mention that the answer is grounded in retrieved chunks rather than general model memory

## 6. Demonstrate an Off-Topic Query

- Ask: `What is the capital of France?`
- Show the hallucination guard response
- Explain that the system prefers returning no-context over hallucinating

## 7. Close with Design Decisions

- Mention local sentence-transformer embeddings for low-cost development
- Mention Chroma persistent storage with cosine similarity
- Mention LangGraph query rewrite -> retrieval -> generation flow
- Mention Groq as the LLM provider for fast inference
