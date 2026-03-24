"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from loguru import logger

from rag_agent.agent.prompts import (
    QUERY_REWRITE_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


@lru_cache(maxsize=1)
def get_vector_store_manager() -> VectorStoreManager:
    """Reuse a single vector store manager across graph invocations."""
    return VectorStoreManager()


def _state_get(state: AgentState | dict, key: str, default=None):
    """Read from LangGraph state whether it arrives as a dict or typed object."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


DOMAIN_KEYWORDS = {
    "deep learning",
    "neural",
    "network",
    "ann",
    "cnn",
    "rnn",
    "lstm",
    "seq2seq",
    "autoencoder",
    "gan",
    "gradient",
    "backprop",
    "backpropagation",
    "convolution",
    "pooling",
    "activation",
    "recurrent",
    "encoder",
    "decoder",
    "cell state",
    "gate",
}


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    messages = _state_get(state, "messages", [])
    human_messages = [
        message for message in messages if isinstance(message, HumanMessage)
    ]
    original_query = human_messages[-1].content if human_messages else ""
    if not original_query:
        return {"original_query": "", "rewritten_query": ""}

    lowered_query = original_query.lower()
    if not any(keyword in lowered_query for keyword in DOMAIN_KEYWORDS):
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
        }

    try:
        llm = LLMFactory(get_settings()).create()
        rewritten = llm.invoke(
            QUERY_REWRITE_PROMPT.format(original_query=original_query)
        ).content.strip()
    except Exception as exc:
        logger.warning("Query rewrite failed, using original query: {}", exc)
        rewritten = original_query

    return {
        "original_query": original_query,
        "rewritten_query": rewritten or original_query,
    }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    manager = get_vector_store_manager()
    chunks = manager.query(
        query_text=_state_get(state, "rewritten_query") or _state_get(state, "original_query", ""),
        topic_filter=_state_get(state, "topic_filter"),
        difficulty_filter=_state_get(state, "difficulty_filter"),
    )
    return {
        "retrieved_chunks": chunks,
        "no_context_found": len(chunks) == 0,
    }


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # ---- Hallucination Guard -----------------------------------------------
    if _state_get(state, "no_context_found", False):
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=_state_get(state, "rewritten_query", ""),
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    # ---- Build Context from Retrieved Chunks --------------------------------
    context_blocks = []
    citations = []
    retrieved_chunks = _state_get(state, "retrieved_chunks", [])
    for chunk in retrieved_chunks:
        citation = f"[SOURCE: {chunk.metadata.topic} | {chunk.metadata.source}]"
        citations.append(citation)
        context_blocks.append(f"{citation}\n{chunk.chunk_text}")

    context_message = (
        "Use only the following retrieved context when answering.\n\n"
        + "\n\n".join(context_blocks)
    )
    average_confidence = sum(
        chunk.score for chunk in retrieved_chunks
    ) / len(retrieved_chunks)

    history = trim_messages(
        [
            message
            for message in _state_get(state, "messages", [])
            if not isinstance(message, SystemMessage)
        ],
        strategy="last",
        token_counter=len,
        max_tokens=settings.max_context_tokens,
        start_on="human",
    )
    grounded_system_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "RETRIEVED CONTEXT:\n"
        f"{context_message}\n\n"
        "GROUNDING REQUIREMENTS:\n"
        "- Treat the retrieved context as the only valid knowledge source.\n"
        "- If the answer is present in the context, answer directly and cite sources.\n"
        "- If the answer is not present in the context, say that clearly and do not use outside knowledge.\n"
        "- Do not claim the context is missing if the answer is explicitly present.\n"
    )
    prior_history = history[:-1] if history else []
    prompt_messages = [SystemMessage(content=grounded_system_prompt)]
    prompt_messages.extend(prior_history)
    prompt_messages.append(
        HumanMessage(
            content=(
                "Answer this question using only the retrieved context.\n\n"
                f"Question: {_state_get(state, 'original_query', '')}"
            )
        )
    )

    llm_response = llm.invoke(prompt_messages)
    answer = llm_response.content.strip()
    response = AgentResponse(
        answer=answer,
        sources=sorted(set(citations)),
        confidence=average_confidence,
        no_context_found=False,
        rewritten_query=_state_get(state, "rewritten_query", ""),
    )

    return {
        "final_response": response,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    return "generate"
