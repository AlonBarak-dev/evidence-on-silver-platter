"""Builds the synthetic bootstrap search result injected at trajectory start.

Reuses FigBrowse's ``RetrievedDocument`` schema and ``CorpusStore`` so the
injected turn is byte-for-byte the same shape as an ordinary FigBrowse
search result (CorpusStore documents only ever carry ``text``/``url`` — no
title field — so leaving ``title=None`` matches real search output exactly,
see figbrowse/retrieval.py Qwen3ShardedRetriever.search()).
"""

from __future__ import annotations

import random

from figbrowse.data import CorpusStore
from figbrowse.retrieval import _truncate_to_tokens
from figbrowse.schemas import BenchmarkExample, RetrievedDocument


def select_evidence_doc_ids(
    example: BenchmarkExample,
    *,
    prefer_evidence_documents: bool = True,
    fallback_to_gold_documents: bool = True,
) -> tuple[list[str], str]:
    """Return (doc_ids, source) where source is 'evidence' or 'gold'.

    Evidence documents are the preferred injection source (they are
    scaffolding rather than answer-bearing) since they best represent
    "potentially sufficient" search results an ordinary search might have
    surfaced. Falls back to gold (answer-bearing) documents only if no
    evidence qrels exist for the query.
    """
    if prefer_evidence_documents and example.evidence_qrels:
        return sorted(example.evidence_qrels.keys()), "evidence"
    if fallback_to_gold_documents and example.gold_qrels:
        return sorted(example.gold_qrels.keys()), "gold"
    if example.evidence_qrels:
        return sorted(example.evidence_qrels.keys()), "evidence"
    if example.gold_qrels:
        return sorted(example.gold_qrels.keys()), "gold"
    return [], "none"


def check_accessible(doc_ids: list[str], corpus_store: CorpusStore) -> tuple[list[str], list[str]]:
    """Split doc_ids into (accessible, missing) via corpus_store.get()."""
    accessible: list[str] = []
    missing: list[str] = []
    for doc_id in doc_ids:
        if corpus_store.get(doc_id) is not None:
            accessible.append(doc_id)
        else:
            missing.append(doc_id)
    return accessible, missing


def build_injected_documents(
    doc_ids: list[str],
    corpus_store: CorpusStore,
    *,
    preview_tokens: int,
    document_view_tokens: int,
    expose_full_text_immediately: bool,
    shuffle: bool,
    seed: int,
    query_id: str,
) -> list[RetrievedDocument]:
    """Build RetrievedDocument previews for the accessible injected doc_ids.

    Order is randomized with a query-specific but fully deterministic seed
    (derived from the run seed + query_id, not from call order) so re-runs
    reproduce the exact same injected ordering.
    """
    ordered = list(doc_ids)
    if shuffle:
        rng = random.Random(f"{seed}:{query_id}")
        rng.shuffle(ordered)

    max_tokens = document_view_tokens if expose_full_text_immediately else preview_tokens
    docs: list[RetrievedDocument] = []
    for rank, doc_id in enumerate(ordered, start=1):
        record = corpus_store.get(doc_id)
        text = _truncate_to_tokens(record["text"], max_tokens) if record else ""
        # Synthetic score: strictly descending so the result list looks like
        # an ordinary ranked search response. It carries no meaning beyond
        # presentation ordering — these documents were not actually scored
        # against a query.
        score = round(1.0 - 0.01 * (rank - 1), 4)
        docs.append(
            RetrievedDocument(doc_id=doc_id, rank=rank, score=score, preview=text)
        )
    return docs
