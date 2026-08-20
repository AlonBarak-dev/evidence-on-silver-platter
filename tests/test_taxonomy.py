from __future__ import annotations

from experiments.initial_gold_injection.taxonomy import assign_control_label, assign_labels


def _base(**overrides):
    d = dict(
        answer_correct=False,
        immediate_answer=False,
        termination_reason="final",
        error=None,
        search_calls_after_injection=0,
        get_document_calls=0,
        injected_documents_opened=[],
        injected_document_coverage=0.0,
    )
    d.update(overrides)
    return d


def test_correct_immediate():
    label, _ = assign_labels(**_base(answer_correct=True, immediate_answer=True))
    assert label == "CORRECT_IMMEDIATE"


def test_correct_after_document_access():
    label, _ = assign_labels(**_base(
        answer_correct=True, get_document_calls=1, injected_documents_opened=["d1"],
        injected_document_coverage=1.0,
    ))
    assert label == "CORRECT_AFTER_DOCUMENT_ACCESS"


def test_correct_after_extra_search():
    label, _ = assign_labels(**_base(
        answer_correct=True, search_calls_after_injection=2, get_document_calls=1,
        injected_documents_opened=["d1"], injected_document_coverage=1.0,
    ))
    assert label == "CORRECT_AFTER_EXTRA_SEARCH"


def test_wrong_without_opening_evidence():
    label, secondary = assign_labels(**_base(answer_correct=False, search_calls_after_injection=1))
    assert label == "WRONG_WITHOUT_OPENING_EVIDENCE"
    assert "SEARCHED_WITHOUT_OPENING_INJECTED_DOCS" in secondary


def test_wrong_after_partial_evidence():
    label, _ = assign_labels(**_base(
        answer_correct=False, injected_documents_opened=["d1"], injected_document_coverage=0.5,
    ))
    assert label == "WRONG_AFTER_PARTIAL_EVIDENCE"


def test_wrong_after_full_evidence():
    label, _ = assign_labels(**_base(
        answer_correct=False, injected_documents_opened=["d1", "d2"], injected_document_coverage=1.0,
    ))
    assert label == "WRONG_AFTER_FULL_EVIDENCE"


def test_budget_exhausted_overrides_correctness():
    label, _ = assign_labels(**_base(answer_correct=False, termination_reason="budget_exhausted"))
    assert label == "BUDGET_EXHAUSTED"


def test_tool_or_data_error():
    label, _ = assign_labels(**_base(error="boom"))
    assert label == "TOOL_OR_DATA_ERROR"


def test_control_label_correct():
    assert assign_control_label(answer_correct=True, termination_reason="final", error=None) == "CORRECT"


def test_control_label_wrong():
    assert assign_control_label(answer_correct=False, termination_reason="final", error=None) == "WRONG"


def test_control_label_budget_exhausted():
    label = assign_control_label(answer_correct=False, termination_reason="budget_exhausted", error=None)
    assert label == "BUDGET_EXHAUSTED"


def test_control_label_error():
    label = assign_control_label(answer_correct=False, termination_reason="final", error="boom")
    assert label == "TOOL_OR_DATA_ERROR"


def test_secondary_flag_not_set_when_docs_opened():
    _, secondary = assign_labels(**_base(
        search_calls_after_injection=1, injected_documents_opened=["d1"],
        injected_document_coverage=0.5,
    ))
    assert "SEARCHED_WITHOUT_OPENING_INJECTED_DOCS" not in secondary
