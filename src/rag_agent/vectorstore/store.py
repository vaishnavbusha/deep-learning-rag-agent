"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        db_path = Path(self._settings.chroma_db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        expected_metadata = {
            "hnsw:space": "cosine",
            "embedding_model": self._settings.embedding_model,
            "embedding_provider": self._settings.embedding_provider.value,
        }

        try:
            self._client = chromadb.PersistentClient(path=str(db_path))

            existing_collection = None
            try:
                existing_collection = self._client.get_collection(
                    name=self._settings.chroma_collection_name
                )
            except Exception:
                existing_collection = None

            if existing_collection is not None:
                existing_metadata = existing_collection.metadata or {}
                if (
                    existing_collection.count() > 0
                    and (
                        existing_metadata.get("embedding_model")
                        != expected_metadata["embedding_model"]
                        or existing_metadata.get("embedding_provider")
                        != expected_metadata["embedding_provider"]
                    )
                ):
                    raise RuntimeError(
                        "Existing Chroma collection was created with a different "
                        "embedding configuration. Delete data/chroma_db or change "
                        "CHROMA_COLLECTION_NAME before running with this model."
                    )

            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata=expected_metadata,
            )
            logger.info(
                "ChromaDB ready: collection='{}' count={}",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB at {db_path}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        result = self._collection.get(ids=[chunk_id], include=[])
        return chunk_id in result.get("ids", [])

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()
        seen_sources: set[str] = set()

        for chunk in chunks:
            try:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    continue

                embedding = self._embeddings.embed_documents([chunk.chunk_text])[0]
                self._collection.upsert(
                    ids=[chunk.chunk_id],
                    embeddings=[embedding],
                    documents=[chunk.chunk_text],
                    metadatas=[chunk.metadata.to_dict()],
                )
                result.ingested += 1
                seen_sources.add(chunk.metadata.source)
            except Exception as exc:
                logger.exception("Failed to ingest chunk {}: {}", chunk.chunk_id, exc)
                result.errors.append(f"{chunk.metadata.source}: {exc}")

        result.document_ids.extend(sorted(seen_sources))
        logger.info(
            "Ingestion complete: ingested={} skipped={} errors={}",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        if not query_text.strip():
            return []

        k = k or self._settings.retrieval_k

        filter_clauses = []
        if topic_filter:
            filter_clauses.append({"topic": topic_filter})
        if difficulty_filter:
            filter_clauses.append({"difficulty": difficulty_filter})

        where_filter = None
        if len(filter_clauses) == 1:
            where_filter = filter_clauses[0]
        elif len(filter_clauses) > 1:
            where_filter = {"$and": filter_clauses}

        query_embedding = self._embeddings.embed_query(query_text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks: list[RetrievedChunk] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            score = max(0.0, 1 - float(distance))
            if score < self._settings.similarity_threshold:
                continue
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=score,
                )
            )

        return sorted(retrieved_chunks, key=lambda chunk: chunk.score, reverse=True)

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        results = self._collection.get(include=["metadatas"])
        documents: dict[str, dict] = {}

        for metadata in results.get("metadatas", []):
            source = metadata["source"]
            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": metadata["topic"],
                    "chunk_count": 0,
                }
            documents[source]["chunk_count"] += 1

        return sorted(documents.values(), key=lambda item: item["source"])

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        chunks: list[DocumentChunk] = []
        for chunk_id, document, metadata in zip(
            results.get("ids", []),
            results.get("documents", []),
            results.get("metadatas", []),
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=ChunkMetadata.from_dict(metadata),
                )
            )

        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        return {
            "total_chunks": len(results.get("ids", [])),
            "topics": sorted({metadata["topic"] for metadata in metadatas}),
            "sources": sorted({metadata["source"] for metadata in metadatas}),
            "bonus_topics_present": any(
                metadata.get("is_bonus", "false").lower() == "true"
                for metadata in metadatas
            ),
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        results = self._collection.get(where={"source": source}, include=[])
        ids = results.get("ids", [])
        if not ids:
            return 0

        self._collection.delete(ids=ids)
        return len(ids)
