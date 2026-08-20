from __future__ import annotations

from experiments.initial_gold_injection.injection import (
    build_injected_documents,
    check_accessible,
    select_evidence_doc_ids,
)
from tests.fixtures import FakeCorpusStore, fake_benchmark_example


def test_prefers_evidence_over_gold():
    ex = fake_benchmark_example()
    doc_ids, source = select_evidence_doc_ids(ex)
    assert source == "evidence"
    assert set(doc_ids) == {"doc_001", "doc_002"}


def test_falls_back_to_gold_when_no_evidence():
    ex = fake_benchmark_example()
    ex.evidence_qrels = None
    doc_ids, source = select_evidence_doc_ids(ex)
    assert source == "gold"
    assert doc_ids == ["doc_002"]


def test_check_accessible_splits_missing_docs():
    store = FakeCorpusStore()
    accessible, missing = check_accessible(["doc_000", "doc_999"], store)
    assert accessible == ["doc_000"]
    assert missing == ["doc_999"]


def test_build_injected_documents_shuffles_deterministically():
    store = FakeCorpusStore()
    doc_ids = ["doc_000", "doc_001", "doc_002", "doc_003"]
    docs_a = build_injected_documents(
        doc_ids, store, preview_tokens=100, document_view_tokens=1000,
        expose_full_text_immediately=False, shuffle=True, seed=42, query_id="qX",
    )
    docs_b = build_injected_documents(
        doc_ids, store, preview_tokens=100, document_view_tokens=1000,
        expose_full_text_immediately=False, shuffle=True, seed=42, query_id="qX",
    )
    assert [d.doc_id for d in docs_a] == [d.doc_id for d in docs_b]
    assert {d.doc_id for d in docs_a} == set(doc_ids)


def test_build_injected_documents_different_query_id_different_order():
    store = FakeCorpusStore()
    doc_ids = [f"doc_{i:03d}" for i in range(6)]
    docs_a = build_injected_documents(
        doc_ids, store, preview_tokens=100, document_view_tokens=1000,
        expose_full_text_immediately=False, shuffle=True, seed=42, query_id="qA",
    )
    docs_b = build_injected_documents(
        doc_ids, store, preview_tokens=100, document_view_tokens=1000,
        expose_full_text_immediately=False, shuffle=True, seed=42, query_id="qB",
    )
    assert [d.doc_id for d in docs_a] != [d.doc_id for d in docs_b]


def test_injected_documents_carry_no_leakage_labels():
    store = FakeCorpusStore()
    docs = build_injected_documents(
        ["doc_000", "doc_001"], store, preview_tokens=100, document_view_tokens=1000,
        expose_full_text_immediately=False, shuffle=False, seed=42, query_id="qX",
    )
    for d in docs:
        assert d.title is None  # matches ordinary FigBrowse search() output
        assert d.metadata == {}  # no gold/evidence/relevant/sufficient labels
    # scores strictly descending, like an ordinary ranked result list
    scores = [d.score for d in docs]
    assert scores == sorted(scores, reverse=True)


def test_scores_descending_regardless_of_shuffle():
    store = FakeCorpusStore()
    docs = build_injected_documents(
        [f"doc_{i:03d}" for i in range(5)], store, preview_tokens=100, document_view_tokens=1000,
        expose_full_text_immediately=False, shuffle=True, seed=7, query_id="qZ",
    )
    scores = [d.score for d in docs]
    assert scores == sorted(scores, reverse=True)
