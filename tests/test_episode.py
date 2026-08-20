from __future__ import annotations

import json

from experiments.initial_gold_injection.episode import (
    run_episode_no_injection,
    run_episode_with_injection,
    run_full_evidence_reader,
)
from figbrowse.schemas import RetrievedDocument
from tests.fixtures import FakeCorpusStore, FakeLLMClient, FakeRetriever

INJECTED_DOCS = [
    RetrievedDocument(doc_id="doc_001", rank=1, score=1.0, preview="preview of doc_001"),
    RetrievedDocument(doc_id="doc_002", rank=2, score=0.99, preview="preview of doc_002"),
]
INJECTED_IDS = {"doc_001", "doc_002"}


def _final(answer="Titanic", cited=None):
    return json.dumps({
        "thought": "I know it", "action": "final", "answer": answer,
        "confidence": 0.9, "cited_doc_ids": cited or [], "brief_support": "x",
    })


def _get_document(doc_id):
    return json.dumps({
        "thought": "reading", "action": "get_document", "doc_id": doc_id,
        "cited_doc_ids": [], "brief_support": None,
    })


def _search(query="more info"):
    return json.dumps({
        "thought": "searching", "action": "search", "search_query": query,
        "cited_doc_ids": [], "brief_support": None,
    })


def _run(responses, max_actions=8, injection_delay=0, injection_position="beginning"):
    client = FakeLLMClient(responses=responses)
    return run_episode_with_injection(
        run_id="r1", episode_id="e1", query_id="q001", variant_id="v1",
        question="What sank in 1912?", agent_id="fake-model", retriever_id="fake-retriever",
        client=client, retriever=FakeRetriever(), corpus_store=FakeCorpusStore(),
        injected_docs=INJECTED_DOCS, injected_doc_ids=INJECTED_IDS,
        max_post_injection_actions=max_actions,
        injection_delay=injection_delay, injection_position=injection_position,
    )


class TestBootstrapNotCountedAsAction:
    def test_injected_turn_is_turn_zero_not_in_actions(self):
        result = _run([_final()])
        assert result.turns[0].turn_index == 0
        assert result.turns[0].action == "search"
        assert set(result.turns[0].retrieved_doc_ids) == INJECTED_IDS
        assert len(result.actions) == 1  # only the real final action
        assert result.actions[0].action_type == "answer"


class TestImmediateAnswer:
    def test_immediate_answer_flag_and_turn_numbers(self):
        result = _run([_final(answer="Titanic")])
        assert result.immediate_answer is True
        assert result.final_answer_turn == 1
        assert result.final_answer == "Titanic"
        assert result.search_calls_after_injection == 0
        assert result.get_document_calls == 0
        assert result.termination_reason == "final"
        assert result.completed is True


class TestDocumentAccess:
    def test_get_document_then_final_tracks_coverage(self):
        result = _run([_get_document("doc_001"), _final(answer="Titanic", cited=["doc_001"])])
        assert result.immediate_answer is False
        assert result.get_document_calls == 1
        assert result.documents_opened == ["doc_001"]
        assert result.injected_documents_opened == ["doc_001"]
        assert result.injected_document_coverage == 0.5
        assert result.all_injected_documents_opened_turn is None  # only 1 of 2 opened
        assert result.first_answer_support_turn == 1
        assert result.final_answer_turn == 2

    def test_opening_all_injected_docs_sets_complete_access_turn(self):
        result = _run([
            _get_document("doc_001"), _get_document("doc_002"), _final(answer="Titanic"),
        ])
        assert result.injected_document_coverage == 1.0
        assert result.all_injected_documents_opened_turn == 2
        assert result.first_answer_support_turn == 1
        assert result.final_answer_turn == 3


class TestSearchAfterInjection:
    def test_real_search_counts_separately_from_bootstrap(self):
        result = _run([_search(), _final(answer="Titanic")])
        assert result.search_calls_after_injection == 1
        assert result.final_answer_turn == 2
        # the real search turn should be turn_index 1 (bootstrap occupies 0)
        assert result.turns[1].turn_index == 1
        assert result.turns[1].action == "search"


class TestMidTrajectoryInjection:
    def test_injection_delivered_after_delay(self):
        # 2 real actions before injection (delay=2), then evidence appears,
        # then the agent opens an injected doc and answers.
        result = _run(
            [_search("who sank in 1912"), _search("more"), _get_document("doc_001"),
             _final(answer="Titanic", cited=["doc_001"])],
            injection_delay=2, injection_position="mid_trajectory",
        )
        assert result.injection_delivered is True
        assert result.injection_delivered_after_actions == 2
        assert len(result.pre_injection_actions) == 2
        assert result.pre_injection_actions[0].action_type == "search"
        assert result.pre_injection_actions[1].action_type == "search"
        # post-injection actions/turn numbering restarts at 1
        assert len(result.actions) == 2
        assert result.actions[0].action_type == "get_document"
        assert result.actions[1].action_type == "answer"
        assert result.final_answer_turn == 2
        assert result.injected_documents_opened == ["doc_001"]
        assert result.termination_reason == "final"
        # the injected turn is somewhere in the middle of the trajectory,
        # after the 2 pre-injection search turns (turn_index 0, 1) and
        # before the post-injection get_document/final turns
        injected_turns = [t for t in result.turns if t.model_parameters.get("bootstrap_injection")]
        assert len(injected_turns) == 1
        assert injected_turns[0].turn_index == 2

    def test_answered_before_injection_point(self):
        # agent answers on its very first action, before delay=2 is reached
        result = _run([_final(answer="Titanic")], injection_delay=2, injection_position="mid_trajectory")
        assert result.injection_delivered is False
        assert result.termination_reason == "answered_before_injection"
        assert len(result.actions) == 0
        assert len(result.pre_injection_actions) == 1
        assert result.pre_injection_actions[0].action_type == "answer"
        assert result.immediate_answer is True
        assert result.injected_documents_opened == []


class TestBudgetEnforcement:
    def test_budget_exhaustion_forces_final(self):
        # budget=1: first search consumes it, then the agent tries to search
        # again but must be forced to FINAL.
        result = _run([_search(), _search(), _final(answer="Titanic")], max_actions=1)
        assert result.search_calls_after_injection == 1
        assert result.completed is True
        assert result.final_answer is not None


class TestNoInjectionControl:
    def test_delegates_to_figbrowse_run_episode_with_no_synthetic_turn(self):
        client = FakeLLMClient(responses=[_search("who sank in 1912"), _final(answer="Titanic")])
        result = run_episode_no_injection(
            run_id="r1", episode_id="e1", query_id="q001", variant_id="v1",
            question="What sank in 1912?", agent_id="fake-model", retriever_id="fake-retriever",
            client=client, retriever=FakeRetriever(), corpus_store=FakeCorpusStore(),
            max_actions=10,
        )
        assert result.injection_position == "no_injection"
        assert result.injection_delivered is False
        # no bootstrap turn — every turn is a real agent decision
        assert len(result.turns) == 2
        assert len(result.actions) == 2
        assert result.actions[0].action_type == "search"
        assert result.actions[1].action_type == "answer"
        assert result.search_calls_after_injection == 1
        assert result.final_answer == "Titanic"
        assert result.termination_reason == "final"
        assert result.completed is True

    def test_respects_its_own_action_ceiling(self):
        # budget=1: only one search/get_document action allowed before a
        # forced final, exactly like FigBrowse's own max_search_calls.
        client = FakeLLMClient(responses=[_search(), _search(), _final(answer="Titanic")])
        result = run_episode_no_injection(
            run_id="r1", episode_id="e1", query_id="q001", variant_id="v1",
            question="What sank in 1912?", agent_id="fake-model", retriever_id="fake-retriever",
            client=client, retriever=FakeRetriever(), corpus_store=FakeCorpusStore(),
            max_actions=1,
        )
        assert result.search_calls_after_injection == 1
        assert result.completed is True


class TestSearchDisabled:
    def test_search_repaired_once_then_agent_complies(self):
        # first search attempt gets one repair note (no budget cost); the
        # agent then complies with a final answer on its next turn.
        client = FakeLLMClient(responses=[_search(), _final(answer="Titanic")])
        result = run_episode_with_injection(
            run_id="r1", episode_id="e1", query_id="q001", variant_id="v1",
            question="What sank in 1912?", agent_id="fake-model", retriever_id="fake-retriever",
            client=client, retriever=FakeRetriever(), corpus_store=FakeCorpusStore(),
            injected_docs=INJECTED_DOCS, injected_doc_ids=INJECTED_IDS,
            max_post_injection_actions=8, injection_delay=0, injection_position="beginning_no_search",
            enable_search=False,
        )
        assert result.search_calls_after_injection == 0
        assert result.completed is True
        assert result.final_answer == "Titanic"

    def test_search_still_disabled_after_repair_forces_synthetic_final(self):
        # agent tries to search twice despite it being disabled; after the
        # one repair attempt is exhausted it is forced to a final answer
        # without asking the model again.
        client = FakeLLMClient(responses=[_search(), _search()])
        result = run_episode_with_injection(
            run_id="r1", episode_id="e1", query_id="q001", variant_id="v1",
            question="What sank in 1912?", agent_id="fake-model", retriever_id="fake-retriever",
            client=client, retriever=FakeRetriever(), corpus_store=FakeCorpusStore(),
            injected_docs=INJECTED_DOCS, injected_doc_ids=INJECTED_IDS,
            max_post_injection_actions=8, injection_delay=0, injection_position="beginning_no_search",
            enable_search=False,
        )
        assert result.search_calls_after_injection == 0
        assert result.completed is True
        assert "search disabled" in result.final_answer

    def test_get_document_still_works_when_search_disabled(self):
        client = FakeLLMClient(responses=[_get_document("doc_001"), _final(answer="Titanic", cited=["doc_001"])])
        result = run_episode_with_injection(
            run_id="r1", episode_id="e1", query_id="q001", variant_id="v1",
            question="What sank in 1912?", agent_id="fake-model", retriever_id="fake-retriever",
            client=client, retriever=FakeRetriever(), corpus_store=FakeCorpusStore(),
            injected_docs=INJECTED_DOCS, injected_doc_ids=INJECTED_IDS,
            max_post_injection_actions=8, injection_delay=0, injection_position="beginning_no_search",
            enable_search=False,
        )
        assert result.get_document_calls == 1
        assert result.injected_documents_opened == ["doc_001"]
        assert result.final_answer == "Titanic"


class TestFullEvidenceReader:
    def test_one_call_no_trajectory_immediate_answer(self):
        client = FakeLLMClient(responses=[json.dumps({
            "answer": "Titanic", "confidence": 0.9,
            "cited_doc_ids": ["doc_001"], "brief_support": "the ship that sank",
        })])
        result = run_full_evidence_reader(
            question="What sank in 1912?", docs=INJECTED_DOCS, client=client,
        )
        assert result.injection_position == "full_evidence_reader"
        assert result.injection_delivered is True
        assert result.final_answer == "Titanic"
        assert result.turns == []
        assert len(result.actions) == 1
        assert result.actions[0].action_type == "answer"
        assert result.immediate_answer is True
        assert result.final_answer_turn == 1
        assert set(result.injected_documents_opened) == INJECTED_IDS
        assert result.injected_document_coverage == 1.0
        assert result.termination_reason == "final"
        assert result.completed is True

    def test_prompt_contains_full_document_text_not_just_preview(self):
        from experiments.initial_gold_injection.episode import build_reader_prompt
        prompt = build_reader_prompt("What sank in 1912?", INJECTED_DOCS)
        assert "doc_001" in prompt
        assert "preview of doc_001" in prompt
        assert "What sank in 1912?" in prompt

    def test_client_error_is_captured_not_raised(self):
        class BrokenClient:
            def _model(self):
                return "fake"

            def generate_structured(self, **kwargs):
                raise RuntimeError("boom")

        result = run_full_evidence_reader(question="Q?", docs=INJECTED_DOCS, client=BrokenClient())
        assert result.completed is False
        assert result.termination_reason == "error"
        assert "boom" in result.error
