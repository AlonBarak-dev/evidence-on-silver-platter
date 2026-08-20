from __future__ import annotations

import json

from figbrowse.schemas import RetrievedDocument, TurnRecord

from experiments.distributed_evidence_replay.placement import (
    apply_placements,
    assign_placements,
    eligible_slots_by_turn,
    place_evidence,
)
from experiments.distributed_evidence_replay.reconstruct import (
    ReconstructionError,
    reconstruct_original_trajectory,
)
from experiments.distributed_evidence_replay.replay import build_replay_prompt, render_full_trajectory
from tests.fixtures import FakeCorpusStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _search_turn(turn_index, query, doc_ids, scores=None):
    scores = scores or [1.0 - 0.1 * i for i in range(len(doc_ids))]
    return TurnRecord(
        run_id="r", episode_id="e", query_id="q001", variant_id="v", variant_type="replay",
        agent_id="a", retriever_id="r", turn_index=turn_index, remaining_search_calls_before=0,
        action="search", search_query=query, retrieved_doc_ids=doc_ids, retrieved_scores=scores,
        retrieved_previews=[f"preview of {d}" for d in doc_ids],
    )


def _get_document_turn(turn_index, doc_id, text="some text"):
    return TurnRecord(
        run_id="r", episode_id="e", query_id="q001", variant_id="v", variant_type="replay",
        agent_id="a", retriever_id="r", turn_index=turn_index, remaining_search_calls_before=0,
        action="get_document", document_id=doc_id, document_text=text,
    )


class FakeReplayRetriever:
    retriever_id = "fake-retriever"

    def __init__(self, results_by_query: dict[str, list[str]]):
        self._results = results_by_query

    def search(self, query, corpus_store=None, top_k=None):
        ids = self._results[query]
        return [
            RetrievedDocument(doc_id=d, rank=i + 1, score=1.0 - 0.1 * i, preview=f"preview of {d}")
            for i, d in enumerate(ids)
        ]


def _fake_source_record(actions):
    return {"question_id": "q001", "actions": actions}


# ---------------------------------------------------------------------------
# Requirement 11: original final answer excluded from reconstructed turns
# ---------------------------------------------------------------------------


def test_reconstruction_excludes_final_answer_action():
    actions = [
        {"turn": 1, "action_type": "search", "arguments": {"query": "q1"},
         "document_ids_returned": ["a", "b"]},
        {"turn": 2, "action_type": "answer",
         "arguments": {"answer": "Titanic", "cited_doc_ids": ["a"]}},
    ]
    retriever = FakeReplayRetriever({"q1": ["a", "b"]})
    store = FakeCorpusStore({"a": "text a", "b": "text b"})
    traj = reconstruct_original_trajectory(
        _fake_source_record(actions), retriever=retriever, corpus_store=store,
        top_k=5, preview_tokens=100, document_view_tokens=1000, retrieval_cache=None,
    )
    assert len(traj.turns) == 1  # only the search turn remains
    assert all(t.action != "answer" for t in traj.turns)
    assert traj.original_final_answer == "Titanic"
    assert traj.original_cited_doc_ids == ["a"]


# ---------------------------------------------------------------------------
# Requirement 4 (partial): reconstruction preserves queries/action order
# ---------------------------------------------------------------------------


def test_reconstruction_preserves_query_text_and_order():
    actions = [
        {"turn": 1, "action_type": "search", "arguments": {"query": "first query"},
         "document_ids_returned": ["a", "b"]},
        {"turn": 2, "action_type": "get_document", "arguments": {"doc_id": "a"}},
        {"turn": 3, "action_type": "search", "arguments": {"query": "second query"},
         "document_ids_returned": ["c", "d"]},
    ]
    retriever = FakeReplayRetriever({"first query": ["a", "b"], "second query": ["c", "d"]})
    store = FakeCorpusStore({"a": "A", "b": "B", "c": "C", "d": "D"})
    traj = reconstruct_original_trajectory(
        _fake_source_record(actions), retriever=retriever, corpus_store=store,
        top_k=5, preview_tokens=100, document_view_tokens=1000, retrieval_cache=None,
    )
    assert [t.action for t in traj.turns] == ["search", "get_document", "search"]
    assert traj.turns[0].search_query == "first query"
    assert traj.turns[2].search_query == "second query"
    assert traj.turns[1].document_id == "a"


def test_reconstruction_raises_on_cache_mismatch():
    actions = [
        {"turn": 1, "action_type": "search", "arguments": {"query": "q1"},
         "document_ids_returned": ["a", "b"]},  # doesn't match retriever below
    ]
    retriever = FakeReplayRetriever({"q1": ["x", "y"]})
    store = FakeCorpusStore({"x": "X", "y": "Y"})
    try:
        reconstruct_original_trajectory(
            _fake_source_record(actions), retriever=retriever, corpus_store=store,
            top_k=5, preview_tokens=100, document_view_tokens=1000, retrieval_cache=None,
        )
        assert False, "expected ReconstructionError"
    except ReconstructionError:
        pass


# ---------------------------------------------------------------------------
# Requirement 1: every annotated evidence document present after modification
# ---------------------------------------------------------------------------


def test_all_evidence_present_after_placement():
    turns = [
        _search_turn(0, "q1", ["a", "b", "c"]),
        _search_turn(1, "q2", ["d", "e", "f"]),
    ]
    store = FakeCorpusStore({d: f"text {d}" for d in "abcdefgh"})
    result = place_evidence(
        turns, missing_evidence_ids={"g", "h"}, evidence_ids={"g", "h"},
        protected_doc_ids=set(), corpus_store=store, preview_tokens=100, seed=42,
    )
    assert result.ok
    all_ids = {doc_id for t in result.turns if t.action == "search" for doc_id in t.retrieved_doc_ids}
    assert {"g", "h"} <= all_ids


# ---------------------------------------------------------------------------
# Requirement 2: naturally present evidence is not duplicated / not touched
# ---------------------------------------------------------------------------


def test_naturally_present_evidence_slot_is_not_eligible_and_not_replaced():
    turns = [_search_turn(0, "q1", ["a", "evidence1", "c"])]
    slots = eligible_slots_by_turn(turns, evidence_ids={"evidence1"}, protected_doc_ids=set())
    assert 1 not in slots.get(0, [])  # rank 1 holds the evidence doc — not eligible

    store = FakeCorpusStore({"a": "A", "evidence1": "E1", "c": "C", "new_ev": "NEW"})
    result = place_evidence(
        turns, missing_evidence_ids={"new_ev"}, evidence_ids={"evidence1", "new_ev"},
        protected_doc_ids=set(), corpus_store=store, preview_tokens=100, seed=1,
    )
    assert result.ok
    assert result.turns[0].retrieved_doc_ids[1] == "evidence1"  # untouched
    assert "new_ev" in result.turns[0].retrieved_doc_ids


# ---------------------------------------------------------------------------
# Requirement 3: search-result counts (slot count per turn) unchanged
# ---------------------------------------------------------------------------


def test_result_slot_count_per_turn_unchanged():
    turns = [_search_turn(0, "q1", ["a", "b", "c", "d", "e"])]
    store = FakeCorpusStore({d: f"t{d}" for d in "abcdefgh"})
    result = place_evidence(
        turns, missing_evidence_ids={"g", "h"}, evidence_ids={"g", "h"},
        protected_doc_ids=set(), corpus_store=store, preview_tokens=100, seed=3,
    )
    assert result.ok
    assert len(result.turns[0].retrieved_doc_ids) == 5
    assert len(result.turns[0].retrieved_scores) == 5
    assert len(result.turns[0].retrieved_previews) == 5


# ---------------------------------------------------------------------------
# Requirement 5/6: get_document'd and cited docs are protected, never replaced
# ---------------------------------------------------------------------------


def test_get_document_referenced_doc_is_protected():
    turns = [_search_turn(0, "q1", ["a", "b", "c"])]
    slots = eligible_slots_by_turn(turns, evidence_ids=set(), protected_doc_ids={"b"})
    assert slots[0] == [0, 2]  # rank 1 (doc "b") excluded


def test_cited_doc_is_protected():
    turns = [_search_turn(0, "q1", ["a", "b", "c"])]
    slots = eligible_slots_by_turn(turns, evidence_ids=set(), protected_doc_ids={"c"})
    assert slots[0] == [0, 1]


# ---------------------------------------------------------------------------
# Requirement 7/8: seed determinism and seed-dependent variation
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_placement():
    slots = {0: [0, 1, 2], 1: [0, 1, 2]}
    a = assign_placements(["e1", "e2", "e3"], slots, seed=42)
    b = assign_placements(["e1", "e2", "e3"], slots, seed=42)
    assert a == b


def test_different_seeds_can_give_different_placements():
    slots = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2]}
    results = {
        seed: assign_placements(["e1", "e2", "e3", "e4", "e5"], slots, seed=seed)
        for seed in (42, 43, 44)
    }
    # not all three identical
    assert len({tuple(sorted(v)) for v in results.values()}) > 1


# ---------------------------------------------------------------------------
# Insufficient slots
# ---------------------------------------------------------------------------


def test_insufficient_slots_reported_not_silently_dropped():
    turns = [_search_turn(0, "q1", ["a"])]  # only 1 eligible slot
    store = FakeCorpusStore({"a": "A", "e1": "E1", "e2": "E2"})
    result = place_evidence(
        turns, missing_evidence_ids={"e1", "e2"}, evidence_ids={"e1", "e2"},
        protected_doc_ids=set(), corpus_store=store, preview_tokens=100, seed=1,
    )
    assert not result.ok
    assert result.error == "INSUFFICIENT_REPLACEMENT_SLOTS"
    assert result.required_slots == 2
    assert result.available_slots == 1


# ---------------------------------------------------------------------------
# Requirement 9: no gold/evidence leakage in the replay prompt
# ---------------------------------------------------------------------------


def test_replay_prompt_has_no_leakage_words():
    turns = [_search_turn(0, "q1", ["a", "b"]), _get_document_turn(1, "a", "full text of a")]
    prompt = build_replay_prompt("What is X?", turns)
    forbidden = ["gold", "injected", "sufficient", "evidence label", "relevance annotation"]
    lowered = prompt.lower()
    for word in forbidden:
        assert word not in lowered, f"leaked forbidden term: {word}"


def test_replay_prompt_does_not_contain_gold_answer_text():
    turns = [_search_turn(0, "q1", ["a"])]
    prompt = build_replay_prompt("What sank in 1912?", turns)
    assert "Titanic" not in prompt  # gold answer never passed to build_replay_prompt at all


def test_render_full_trajectory_shows_all_turns_not_just_recent_three():
    # Unlike figbrowse.agent._render_trajectory, nothing should collapse —
    # this is a one-shot full dump, not incremental multi-turn context.
    turns = [_search_turn(i, f"q{i}", [f"doc{i}"]) for i in range(6)]
    rendered = render_full_trajectory(turns)
    for i in range(6):
        assert f"preview: preview of doc{i}" in rendered
    assert "collapsed" not in rendered


# ---------------------------------------------------------------------------
# Requirement 10: R0 and R-all share exactly the same prompt template
# ---------------------------------------------------------------------------


def test_r0_and_rall_use_same_prompt_template():
    turns_r0 = [_search_turn(0, "q1", ["a", "b"])]
    turns_rall = [_search_turn(0, "q1", ["a", "evidence1"])]  # evidence swapped in
    prompt_r0 = build_replay_prompt("Q?", turns_r0)
    prompt_rall = build_replay_prompt("Q?", turns_rall)
    # same header/instructions (everything before "## Session record")
    header_r0 = prompt_r0.split("## Session record")[0]
    header_rall = prompt_rall.split("## Session record")[0]
    assert header_r0 == header_rall


# ---------------------------------------------------------------------------
# Requirement 12: resume skips completed (question, condition, seed) tuples
# ---------------------------------------------------------------------------


def test_load_completed_keys_by_question_condition_seed(tmp_path):
    from experiments.distributed_evidence_replay.run import _load_completed

    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps({"question_id": "q1", "condition": "original_replay", "placement_seed": None}) + "\n"
        + json.dumps({"question_id": "q1", "condition": "distributed_evidence_replay", "placement_seed": 42}) + "\n"
    )
    done = _load_completed(path)
    assert done == {("q1", "original_replay", None), ("q1", "distributed_evidence_replay", 42)}
