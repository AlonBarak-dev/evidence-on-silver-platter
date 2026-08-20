"""Rebuilds the frozen original trajectory from a no-injection RunRecord.

The summary JSONL (outputs/initial_gold_injection_confirmatory_no_injection/
runs.jsonl) deliberately does not store full search-result previews or
get_document text (CLAUDE-style convention: don't duplicate content FigBrowse
already stores elsewhere). Instead we re-derive the exact original content
via FigBrowse's own retrieval cache (``cached_search`` hits the same key —
retriever_id/query/top_k/preview_tokens — so it returns the identical
previously-computed result, no re-embedding) and the corpus store (for
``get_document`` text, which is a pure deterministic lookup).

This keeps "the frozen trajectory" as a *reconstruction*, not a stored
duplicate — consistent with how the original experiment already avoided
persisting raw document content in its own JSONL.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from figbrowse.agent import cached_search
from figbrowse.cache import SimpleCache
from figbrowse.retrieval import _truncate_to_tokens
from figbrowse.schemas import RetrievedDocument, TurnRecord


class ReconstructionError(Exception):
    """Raised when the live retrieval cache no longer matches the doc IDs
    recorded in the summary JSONL — the trajectory can't be faithfully
    frozen/replayed."""


class ReconstructedTrajectory(BaseModel):
    turns: list[TurnRecord] = Field(default_factory=list)
    original_search_turns: int = 0
    original_result_slots: int = 0
    trajectory_result_ids: list[str] = Field(default_factory=list)
    get_document_ids: list[str] = Field(default_factory=list)
    original_final_answer: str | None = None
    original_cited_doc_ids: list[str] = Field(default_factory=list)


def reconstruct_original_trajectory(
    run_record: dict[str, Any],
    *,
    retriever: Any,
    corpus_store: Any,
    top_k: int,
    preview_tokens: int,
    document_view_tokens: int,
    retrieval_cache: SimpleCache | None,
) -> ReconstructedTrajectory:
    query_id = run_record["question_id"]
    turns: list[TurnRecord] = []
    result_ids: list[str] = []
    get_document_ids: list[str] = []
    final_answer: str | None = None
    cited: list[str] = []
    turn_index = 0
    search_turns = 0

    for action in run_record.get("actions") or []:
        atype = action["action_type"]

        if atype == "search":
            query = action["arguments"]["query"]
            expected_ids = list(action.get("document_ids_returned") or [])
            hits: list[RetrievedDocument] = cached_search(
                retriever, query, corpus_store, top_k, preview_tokens,
                retrieval_cache=retrieval_cache,
            )
            actual_ids = [h.doc_id for h in hits]
            if actual_ids != expected_ids:
                raise ReconstructionError(
                    f"query_id={query_id}: retrieval cache mismatch for query "
                    f"{query!r} — expected {expected_ids}, got {actual_ids}. "
                    "The trajectory can no longer be faithfully reconstructed."
                )
            turns.append(TurnRecord(
                run_id="distributed_evidence_replay", episode_id=f"replay_{query_id}",
                query_id=query_id, variant_id=f"{query_id}_replay", variant_type="replay",
                agent_id="", retriever_id=getattr(retriever, "retriever_id", "unknown"),
                turn_index=turn_index, remaining_search_calls_before=0,
                action="search", search_query=query,
                retrieved_doc_ids=[h.doc_id for h in hits],
                retrieved_scores=[h.score for h in hits],
                retrieved_previews=[h.preview for h in hits],
            ))
            result_ids.extend(actual_ids)
            search_turns += 1
            turn_index += 1

        elif atype == "get_document":
            doc_id = action["arguments"]["doc_id"]
            record = corpus_store.get(doc_id) if corpus_store is not None else None
            text = _truncate_to_tokens(record["text"], document_view_tokens) if record else None
            turns.append(TurnRecord(
                run_id="distributed_evidence_replay", episode_id=f"replay_{query_id}",
                query_id=query_id, variant_id=f"{query_id}_replay", variant_type="replay",
                agent_id="", retriever_id=getattr(retriever, "retriever_id", "unknown"),
                turn_index=turn_index, remaining_search_calls_before=0,
                action="get_document", document_id=doc_id, document_text=text,
            ))
            get_document_ids.append(doc_id)
            turn_index += 1

        elif atype == "answer":
            final_answer = action["arguments"].get("answer")
            cited = list(action["arguments"].get("cited_doc_ids") or [])
            # answer is excluded from the replayed trajectory — not appended

        else:
            raise ReconstructionError(f"query_id={query_id}: unknown action_type {atype!r}")

    return ReconstructedTrajectory(
        turns=turns,
        original_search_turns=search_turns,
        original_result_slots=len(result_ids),
        trajectory_result_ids=result_ids,
        get_document_ids=get_document_ids,
        original_final_answer=final_answer,
        original_cited_doc_ids=cited,
    )
