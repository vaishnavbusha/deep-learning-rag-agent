# QA Test Plan

## Integration Tests

| Test | Input | Expected Behavior | Status |
|---|---|---|---|
| Normal query | `Explain the vanishing gradient problem` | Relevant RNN or LSTM chunks retrieved, answer grounded in corpus, at least one source citation shown | Ready to run |
| Off-topic query | `What is the capital of France?` | Hallucination guard fires, answer says relevant context was not found, no fabricated deep learning response | Ready to run |
| Duplicate ingestion | Upload `rnn_intermediate.md` twice | First ingest adds chunks, second ingest reports duplicates skipped | Ready to run |
| Empty query | Submit blank chat input | UI should not crash and should avoid sending an invalid query | Ready to run |
| Cross-topic query | `How do LSTMs improve on RNNs for Seq2Seq tasks?` | Retrieval spans more than one concept area and answer synthesizes grounded context with citations | Ready to run |

## Risk Assessment

| Risk | Why It Matters | Mitigation |
|---|---|---|
| Thin corpus coverage | Retrieval quality falls quickly when the corpus is too small or too generic | Add LSTM, Seq2Seq, Autoencoder, and at least one bonus topic |
| Environment drift | Chroma collections or local caches can break when embedding config changes | Reset `data/chroma_db` when changing embedding models or collection settings |
| Prompt drift | The model may answer from prior knowledge instead of retrieved context | Keep strict grounding instructions and verify citations during the demo |

## Manual Verification Notes

- Unit tests pass with `uv run pytest tests/ -v`
- Duplicate skipping has been verified in smoke testing
- Real retrieval and Groq-backed generation have been verified in smoke testing
- Final UI walkthrough should still be recorded for the submission video
