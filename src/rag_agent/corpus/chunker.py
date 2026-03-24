"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader
from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            raw_chunks = self._chunk_pdf(file_path, chunk_size, chunk_overlap)
        elif suffix == ".md":
            raw_chunks = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        metadata = self._infer_metadata(file_path, metadata_overrides)
        prepared_chunks: list[DocumentChunk] = []

        for raw_chunk in raw_chunks:
            chunk_text = raw_chunk["text"].strip()
            if not chunk_text:
                continue

            prepared_chunks.append(
                DocumentChunk(
                    chunk_id=VectorStoreManager.generate_chunk_id(
                        file_path.name, chunk_text
                    ),
                    chunk_text=chunk_text,
                    metadata=metadata,
                )
            )

        logger.info("Chunked {} into {} chunks", file_path.name, len(prepared_chunks))
        return prepared_chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        chunks: list[DocumentChunk] = []
        for file_path in file_paths:
            try:
                chunks.extend(self.chunk_file(file_path, metadata_overrides))
            except Exception as exc:
                logger.exception("Failed to chunk {}: {}", file_path, exc)
        return chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        raw_chunks: list[dict] = []
        for page in pages:
            text = page.page_content.strip()
            if len(text) < 50:
                continue

            for split in splitter.split_text(text):
                cleaned = " ".join(split.split())
                if cleaned:
                    raw_chunks.append(
                        {
                            "text": cleaned,
                            "page": page.metadata.get("page"),
                        }
                    )
        return raw_chunks

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Interview talking point: header-aware splitting preserves
        semantic coherence better than naive character splitting —
        a concept within one section stays within one chunk.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        markdown_text = file_path.read_text(encoding="utf-8")
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        header_docs = header_splitter.split_text(markdown_text)
        raw_chunks: list[dict] = []

        for doc in header_docs:
            header_parts = [value for value in doc.metadata.values() if value]
            prefix = " > ".join(header_parts)
            combined = (
                f"{prefix}\n\n{doc.page_content.strip()}".strip()
                if prefix
                else doc.page_content.strip()
            )

            for split in recursive_splitter.split_text(combined):
                cleaned = " ".join(split.split())
                if cleaned:
                    raw_chunks.append({"text": cleaned, "header": prefix})

        return raw_chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        overrides = overrides or {}
        stem_parts = file_path.stem.split("_")

        topic_map = {
            "ann": "ANN",
            "cnn": "CNN",
            "rnn": "RNN",
            "lstm": "LSTM",
            "seq2seq": "Seq2Seq",
            "autoencoder": "Autoencoder",
            "som": "SOM",
            "boltzmann": "BoltzmannMachine",
            "boltzmannmachine": "BoltzmannMachine",
            "gan": "GAN",
        }
        related_topics_map = {
            "ANN": ["backpropagation", "activation_functions"],
            "CNN": ["convolution", "pooling", "feature_maps"],
            "RNN": ["sequence_modeling", "hidden_state", "vanishing_gradient"],
            "LSTM": ["RNN", "cell_state", "gating"],
            "Seq2Seq": ["encoder_decoder", "attention", "RNN"],
            "Autoencoder": ["latent_space", "dimensionality_reduction"],
            "SOM": ["clustering", "unsupervised_learning"],
            "BoltzmannMachine": ["energy_based_models", "unsupervised_learning"],
            "GAN": ["generator", "discriminator", "adversarial_training"],
        }

        topic_key = stem_parts[0].lower() if stem_parts else "unknown"
        inferred_topic = topic_map.get(topic_key, topic_key.upper())
        inferred_difficulty = (
            stem_parts[1].lower() if len(stem_parts) > 1 else "intermediate"
        )

        metadata_values = {
            "topic": inferred_topic,
            "difficulty": inferred_difficulty,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": related_topics_map.get(inferred_topic, []),
            "is_bonus": inferred_topic in {"SOM", "BoltzmannMachine", "GAN"},
        }
        metadata_values.update(overrides)

        return ChunkMetadata(**metadata_values)
