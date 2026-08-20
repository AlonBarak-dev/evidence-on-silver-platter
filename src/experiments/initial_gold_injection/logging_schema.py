"""Per-run JSONL record schema for the initial_gold_injection experiment.

Mirrors the field contract in the experiment spec. Full document contents
are never placed here — FigBrowse's own ``turns.jsonl``-style records (not
written by this experiment; see ``episode.py``) already hold previews/text,
and this record only carries doc-id references, matching FigBrowse's
gold-isolation convention of keeping heavy/raw content out of summary logs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionRecord(BaseModel):
    turn: int
    action_type: Literal["search", "get_document", "answer", "other"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation_summary: str | None = None
    document_ids_returned: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_seconds: float | None = None


class RunRecord(BaseModel):
    question_id: str
    question: str
    gold_answer: str | None = None
    injected_document_ids: list[str] = Field(default_factory=list)
    num_injected_documents: int = 0
    all_injected_documents_accessible: bool = True

    model: str = ""
    seed: int = 42
    max_post_injection_actions: int = 8

    injection_position: str = "beginning"
    injection_delivered: bool = True
    injection_delivered_after_actions: int = 0
    pre_injection_actions: list[ActionRecord] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)

    search_calls: int = 0
    get_document_calls: int = 0
    unique_documents_opened: int = 0
    injected_documents_opened: list[str] = Field(default_factory=list)
    injected_document_coverage: float = 0.0
    all_injected_documents_opened_turn: int | None = None
    first_answer_support_turn: int | None = None
    final_answer_turn: int | None = None
    post_access_delay: int | None = None

    final_answer: str | None = None
    answer_correct: bool = False
    termination_reason: str = "unknown"

    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost: float | None = None
    latency_seconds: float | None = None
    error: str | None = None

    # Not part of the required contract, but needed for the failure taxonomy
    # and analysis without re-deriving from actions every time.
    immediate_answer: bool = False
    primary_label: str | None = None
    secondary_labels: list[str] = Field(default_factory=list)
