"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.nodes import (
    evaluate_candidate_answer,
    generate_interview_question,
)
from rag_agent.agent.state import AgentResponse, AnswerEvaluation, InterviewQuestion
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
        "last_generated_question": None,
        "last_answer_evaluation": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")
    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    if not st.session_state.ingested_documents:
        st.session_state.ingested_documents = store.list_documents()

    if st.sidebar.button(
        "Ingest Documents",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        with st.sidebar:
            with st.spinner("Chunking and ingesting documents..."):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    file_paths: list[Path] = []
                    for uploaded_file in uploaded_files:
                        file_path = Path(tmp_dir) / uploaded_file.name
                        file_path.write_bytes(uploaded_file.getbuffer())
                        file_paths.append(file_path)

                    chunks = chunker.chunk_files(file_paths)
                    result = store.ingest(chunks)

                st.session_state.last_ingestion_result = result
                st.session_state.ingested_documents = store.list_documents()
                if st.session_state.ingested_documents:
                    st.session_state.selected_document = st.session_state.ingested_documents[
                        0
                    ]["source"]

        result = st.session_state.last_ingestion_result
        if result.errors:
            st.sidebar.error(
                f"{result.ingested} chunks added, {result.skipped} duplicates skipped, "
                f"{len(result.errors)} errors."
            )
        elif result.ingested > 0:
            st.sidebar.success(
                f"{result.ingested} chunks added, {result.skipped} duplicates skipped."
            )
        else:
            st.sidebar.warning("No new chunks were added.")

    if st.session_state.last_ingestion_result and st.session_state.last_ingestion_result.errors:
        with st.sidebar.expander("Ingestion Errors"):
            for error in st.session_state.last_ingestion_result.errors:
                st.caption(error)

    st.sidebar.info("Upload .pdf or .md files to populate the corpus.")
    st.sidebar.divider()
    st.sidebar.subheader("Ingested Documents")

    if not st.session_state.ingested_documents:
        st.sidebar.caption("No documents ingested yet.")
        return

    for document in st.session_state.ingested_documents:
        info_col, action_col = st.sidebar.columns([4, 1])
        with info_col:
            st.caption(
                f"{document['source']} | {document['topic']} | {document['chunk_count']} chunks"
            )
        with action_col:
            if st.button("🗑", key=f"delete-{document['source']}"):
                store.delete_document(document["source"])
                st.session_state.ingested_documents = store.list_documents()
                if st.session_state.selected_document == document["source"]:
                    st.session_state.selected_document = None
                st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    stats = store.get_collection_stats()
    st.sidebar.divider()
    st.sidebar.subheader("Corpus Health")
    st.sidebar.metric("Total Chunks", stats["total_chunks"])
    st.sidebar.caption(
        "Topics: " + (", ".join(stats["topics"]) if stats["topics"] else "None yet")
    )
    if stats["bonus_topics_present"]:
        st.sidebar.success("Bonus topics present")
    else:
        st.sidebar.warning("No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    documents = st.session_state.ingested_documents or store.list_documents()
    if not documents:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [document["source"] for document in documents]
    default_index = 0
    if st.session_state.selected_document in sources:
        default_index = sources.index(st.session_state.selected_document)

    selected_source = st.selectbox("Select document", options=sources, index=default_index)
    st.session_state.selected_document = selected_source

    chunks = store.get_document_chunks(selected_source)
    st.caption(f"{len(chunks)} chunks in {selected_source}")

    viewer_container = st.container(height=500)
    with viewer_container:
        for index, chunk in enumerate(chunks, start=1):
            st.markdown(
                f"**Chunk {index}**  \n"
                f"`{chunk.metadata.topic}` | `{chunk.metadata.difficulty}` | "
                f"`{chunk.metadata.type}`"
            )
            st.write(chunk.chunk_text)
            st.divider()


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    col_topic, col_diff = st.columns(2)
    with col_topic:
        topic_options = [None] + sorted(
            {document["topic"] for document in st.session_state.ingested_documents}
        )
        st.session_state.topic_filter = st.selectbox(
            "Topic Filter",
            options=topic_options,
            format_func=lambda value: "All topics" if value is None else value,
            index=topic_options.index(st.session_state.topic_filter)
            if st.session_state.topic_filter in topic_options
            else 0,
        )
    with col_diff:
        difficulty_options = [None, "beginner", "intermediate", "advanced"]
        st.session_state.difficulty_filter = st.selectbox(
            "Difficulty Filter",
            options=difficulty_options,
            format_func=lambda value: "All levels" if value is None else value,
            index=difficulty_options.index(st.session_state.difficulty_filter)
            if st.session_state.difficulty_filter in difficulty_options
            else 0,
        )

    ask_tab, generate_tab, evaluate_tab = st.tabs(
        ["Ask the Corpus", "Generate Question", "Evaluate Answer"]
    )

    with ask_tab:
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("sources"):
                        with st.expander("📎 Sources"):
                            for source in message["sources"]:
                                st.caption(source)
                    if message.get("no_context_found"):
                        st.warning("⚠️ No relevant content found in corpus.")

        query = st.chat_input(
            "Ask about a deep learning topic...",
            disabled=not st.session_state.ingested_documents,
        )

        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.spinner("Thinking..."):
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=query)],
                        "topic_filter": st.session_state.topic_filter,
                        "difficulty_filter": st.session_state.difficulty_filter,
                    },
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                response = result["final_response"]

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "no_context_found": response.no_context_found,
                }
            )
            st.rerun()

    with generate_tab:
        topic_hint = st.text_input(
            "Topic or concept",
            placeholder="Example: LSTM gates or CNN pooling",
            key="question_generation_topic",
        )
        generation_difficulty = st.selectbox(
            "Question difficulty",
            options=["beginner", "intermediate", "advanced"],
            index=1,
            key="question_generation_difficulty",
        )
        if st.button(
            "Generate Interview Question",
            use_container_width=True,
            disabled=not st.session_state.ingested_documents or not topic_hint.strip(),
        ):
            with st.spinner("Generating grounded interview question..."):
                st.session_state.last_generated_question = generate_interview_question(
                    query_text=topic_hint,
                    difficulty=generation_difficulty,
                    topic_filter=st.session_state.topic_filter,
                )

        generated = st.session_state.last_generated_question
        if isinstance(generated, AgentResponse):
            st.warning(generated.answer)
        elif isinstance(generated, InterviewQuestion):
            st.markdown(f"**Question**  \n{generated.question}")
            st.caption(
                f"Topic: {generated.topic} | Difficulty: {generated.difficulty}"
            )
            st.markdown(f"**Model Answer**  \n{generated.model_answer}")
            st.markdown(f"**Follow-up**  \n{generated.follow_up}")
            with st.expander("📎 Sources"):
                for source in generated.source_citations:
                    st.caption(source)

    with evaluate_tab:
        evaluation_question = st.text_area(
            "Interview question",
            placeholder="Paste the interview question here",
            key="evaluation_question",
        )
        candidate_answer = st.text_area(
            "Candidate answer",
            placeholder="Paste the student's answer here",
            key="candidate_answer",
            height=180,
        )
        if st.button(
            "Evaluate Answer",
            use_container_width=True,
            disabled=(
                not st.session_state.ingested_documents
                or not evaluation_question.strip()
                or not candidate_answer.strip()
            ),
        ):
            with st.spinner("Evaluating answer against the corpus..."):
                st.session_state.last_answer_evaluation = evaluate_candidate_answer(
                    question=evaluation_question,
                    candidate_answer=candidate_answer,
                    topic_filter=st.session_state.topic_filter,
                    difficulty_filter=st.session_state.difficulty_filter,
                )

        evaluation = st.session_state.last_answer_evaluation
        if isinstance(evaluation, AgentResponse):
            st.warning(evaluation.answer)
        elif isinstance(evaluation, AnswerEvaluation):
            st.metric("Score", f"{evaluation.score}/10")
            st.markdown(f"**What Was Correct**  \n{evaluation.what_was_correct}")
            st.markdown(f"**What Was Missing**  \n{evaluation.what_was_missing}")
            st.markdown(f"**Ideal Answer**  \n{evaluation.ideal_answer}")
            st.markdown(
                f"**Interview Verdict**  \n{evaluation.interview_verdict}"
            )
            st.markdown(f"**Coaching Tip**  \n{evaluation.coaching_tip}")
            with st.expander("📎 Sources"):
                for source in evaluation.source_citations:
                    st.caption(source)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
