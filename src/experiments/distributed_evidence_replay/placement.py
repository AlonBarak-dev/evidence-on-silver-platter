"""Evidence-placement algorithm: distribute missing evidence documents across
eligible search-result slots in a frozen trajectory.

Genuinely new logic (not present anywhere in FigBrowse or the earlier
initial_gold_injection experiment) — everything it touches (RetrievedDocument
construction, preview truncation) still reuses FigBrowse's own conventions.
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel, Field

from figbrowse.retrieval import _truncate_to_tokens
from figbrowse.schemas import TurnRecord


class Replacement(BaseModel):
    turn: int
    rank: int
    removed_document_id: str
    inserted_document_id: str


class PlacementResult(BaseModel):
    ok: bool
    turns: list[TurnRecord] = Field(default_factory=list)
    replacements: list[Replacement] = Field(default_factory=list)
    required_slots: int = 0
    available_slots: int = 0
    error: str | None = None


def eligible_slots_by_turn(
    turns: list[TurnRecord],
    *,
    evidence_ids: set[str],
    protected_doc_ids: set[str],
) -> dict[int, list[int]]:
    """turn_index -> list of 0-indexed slot positions eligible for replacement.

    A slot is eligible iff its current doc_id is not an evidence document and
    not protected (opened via get_document, or cited by the removed answer).
    """
    slots: dict[int, list[int]] = {}
    for t in turns:
        if t.action != "search":
            continue
        eligible = [
            rank for rank, doc_id in enumerate(t.retrieved_doc_ids)
            if doc_id not in evidence_ids and doc_id not in protected_doc_ids
        ]
        if eligible:
            slots[t.turn_index] = eligible
    return slots


def assign_placements(
    missing_evidence_ids: list[str],
    slots_by_turn: dict[int, list[int]],
    *,
    seed: int,
) -> list[tuple[str, int, int]]:
    """Deterministic round-robin placement: (evidence_id, turn_index, rank).

    Shuffles evidence order and turn order with the given seed, then cycles
    through turns assigning one evidence document per turn per pass — so
    with more evidence documents than turns, later passes reuse turns with
    remaining capacity ("cycle through the turns again using other eligible
    ranks").
    """
    rng = random.Random(seed)
    evidence_order = list(missing_evidence_ids)
    rng.shuffle(evidence_order)

    turn_order = list(slots_by_turn.keys())
    rng.shuffle(turn_order)

    pools: dict[int, list[int]] = {t: list(slots_by_turn[t]) for t in turn_order}
    for t in turn_order:
        rng.shuffle(pools[t])

    assignments: list[tuple[str, int, int]] = []
    remaining = list(evidence_order)
    while remaining:
        progressed = False
        for t in turn_order:
            if not remaining:
                break
            if pools[t]:
                rank = pools[t].pop()
                doc_id = remaining.pop(0)
                assignments.append((doc_id, t, rank))
                progressed = True
        if not progressed:
            break
    return assignments


def apply_placements(
    turns: list[TurnRecord],
    assignments: list[tuple[str, int, int]],
    *,
    corpus_store: Any,
    preview_tokens: int,
) -> tuple[list[TurnRecord], list[Replacement]]:
    """Returns a new turns list with the assigned slots swapped in, using the
    same doc-id/score/preview representation as an ordinary search result
    (score is preserved from the removed slot so the swap is undetectable —
    no invented titles/snippets, preview built the same way real search
    results are built)."""
    by_turn: dict[int, list[tuple[str, int]]] = {}
    for doc_id, turn_idx, rank in assignments:
        by_turn.setdefault(turn_idx, []).append((doc_id, rank))

    new_turns: list[TurnRecord] = []
    replacements: list[Replacement] = []
    for t in turns:
        if t.action != "search" or t.turn_index not in by_turn:
            new_turns.append(t)
            continue

        doc_ids = list(t.retrieved_doc_ids)
        scores = list(t.retrieved_scores)
        previews = list(t.retrieved_previews)

        for new_doc_id, rank in by_turn[t.turn_index]:
            removed = doc_ids[rank]
            record = corpus_store.get(new_doc_id) if corpus_store is not None else None
            preview = _truncate_to_tokens(record["text"], preview_tokens) if record else ""
            doc_ids[rank] = new_doc_id
            previews[rank] = preview
            # scores[rank] left unchanged — the swap keeps the original
            # ranking score so it's indistinguishable from a real result.
            replacements.append(Replacement(
                turn=t.turn_index, rank=rank,
                removed_document_id=removed, inserted_document_id=new_doc_id,
            ))

        new_turns.append(t.model_copy(update={
            "retrieved_doc_ids": doc_ids, "retrieved_scores": scores, "retrieved_previews": previews,
        }))

    return new_turns, replacements


def place_evidence(
    turns: list[TurnRecord],
    *,
    missing_evidence_ids: set[str],
    evidence_ids: set[str],
    protected_doc_ids: set[str],
    corpus_store: Any,
    preview_tokens: int,
    seed: int,
) -> PlacementResult:
    missing_list = sorted(missing_evidence_ids)
    if not missing_list:
        return PlacementResult(ok=True, turns=turns, replacements=[], required_slots=0, available_slots=0)

    slots = eligible_slots_by_turn(turns, evidence_ids=evidence_ids, protected_doc_ids=protected_doc_ids)
    available = sum(len(v) for v in slots.values())
    required = len(missing_list)
    if available < required:
        return PlacementResult(
            ok=False, required_slots=required, available_slots=available,
            error="INSUFFICIENT_REPLACEMENT_SLOTS",
        )

    assignments = assign_placements(missing_list, slots, seed=seed)
    if len(assignments) != required:
        return PlacementResult(
            ok=False, required_slots=required, available_slots=available,
            error="INSUFFICIENT_REPLACEMENT_SLOTS",
        )

    new_turns, replacements = apply_placements(
        turns, assignments, corpus_store=corpus_store, preview_tokens=preview_tokens,
    )
    return PlacementResult(
        ok=True, turns=new_turns, replacements=replacements,
        required_slots=required, available_slots=available,
    )
