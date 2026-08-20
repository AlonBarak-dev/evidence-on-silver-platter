"""Per-run JSONL record schema for distributed_evidence_replay, matching the
exact field contract from the experiment spec."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .placement import Replacement


class RunRecord(BaseModel):
    question_id: str
    condition: Literal["original_replay", "distributed_evidence_replay"]
    placement_seed: int | None = None
    model: str = ""
    original_trajectory_path: str = ""
    original_search_turns: int = 0
    original_result_slots: int = 0

    total_evidence_documents: int = 0
    naturally_present_evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence_ids: list[str] = Field(default_factory=list)
    injected_evidence_ids: list[str] = Field(default_factory=list)
    final_present_evidence_ids: list[str] = Field(default_factory=list)
    all_evidence_present: bool = True

    replacements: list[Replacement] = Field(default_factory=list)

    mean_normalized_evidence_position: float | None = None
    first_evidence_turn: int | None = None
    last_evidence_turn: int | None = None

    final_answer: str | None = None
    cited_document_ids: list[str] = Field(default_factory=list)
    answer_correct: bool = False
    termination_reason: str = "final"
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    # Not in the required contract but useful for analysis without
    # re-deriving from gold data every time.
    gold_answer: str | None = None
