"""Deterministic failure-taxonomy labeling (spec: "Lightweight failure taxonomy").

One primary label per run plus zero or more secondary flags. Pure function
of already-computed per-run fields — no LLM calls, no extra gold access.
"""

from __future__ import annotations

PRIMARY_LABELS = [
    "CORRECT_IMMEDIATE",
    "CORRECT_AFTER_DOCUMENT_ACCESS",
    "CORRECT_AFTER_EXTRA_SEARCH",
    "WRONG_WITHOUT_OPENING_EVIDENCE",
    "WRONG_AFTER_PARTIAL_EVIDENCE",
    "WRONG_AFTER_FULL_EVIDENCE",
    "BUDGET_EXHAUSTED",
    "ANSWERED_BEFORE_INJECTION",
    "TOOL_OR_DATA_ERROR",
]

SECONDARY_LABELS = [
    "SEARCHED_WITHOUT_OPENING_INJECTED_DOCS",
]


def assign_labels(
    *,
    answer_correct: bool,
    immediate_answer: bool,
    termination_reason: str,
    error: str | None,
    search_calls_after_injection: int,
    get_document_calls: int,
    injected_documents_opened: list[str],
    injected_document_coverage: float,
) -> tuple[str, list[str]]:
    secondary: list[str] = []
    if search_calls_after_injection > 0 and not injected_documents_opened:
        secondary.append("SEARCHED_WITHOUT_OPENING_INJECTED_DOCS")

    if error or termination_reason == "error":
        return "TOOL_OR_DATA_ERROR", secondary

    if termination_reason == "answered_before_injection":
        return "ANSWERED_BEFORE_INJECTION", secondary

    if termination_reason == "budget_exhausted":
        return "BUDGET_EXHAUSTED", secondary

    if answer_correct:
        if immediate_answer:
            return "CORRECT_IMMEDIATE", secondary
        if search_calls_after_injection > 0:
            return "CORRECT_AFTER_EXTRA_SEARCH", secondary
        if get_document_calls > 0:
            return "CORRECT_AFTER_DOCUMENT_ACCESS", secondary
        return "CORRECT_IMMEDIATE", secondary

    if not injected_documents_opened:
        return "WRONG_WITHOUT_OPENING_EVIDENCE", secondary
    if injected_document_coverage < 1.0:
        return "WRONG_AFTER_PARTIAL_EVIDENCE", secondary
    return "WRONG_AFTER_FULL_EVIDENCE", secondary


CONTROL_LABELS = ["CORRECT", "WRONG", "BUDGET_EXHAUSTED", "TOOL_OR_DATA_ERROR"]


def assign_control_label(
    *, answer_correct: bool, termination_reason: str, error: str | None,
) -> str:
    """Label for the no-injection control condition.

    Deliberately does not reuse the injection-evidence vocabulary above
    (WRONG_WITHOUT_OPENING_EVIDENCE etc.) — those labels make claims about
    injected documents that don't exist in this condition.
    """
    if error or termination_reason == "error":
        return "TOOL_OR_DATA_ERROR"
    if termination_reason == "budget_exhausted":
        return "BUDGET_EXHAUSTED"
    return "CORRECT" if answer_correct else "WRONG"
