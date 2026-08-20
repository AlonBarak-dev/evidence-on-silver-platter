"""Runs one initial_gold_injection episode.

Thin adapter around FigBrowse's ReAct loop (figbrowse.agent): reuses
``get_agent_action`` (LLM decision + cache), ``cached_search`` (retrieval +
cache), ``build_agent_prompt``/its trajectory renderer, and the
``TurnRecord``/``RetrievedDocument`` schemas verbatim. The only new behavior
is seeding the trajectory with one synthetic search turn containing the
injected evidence documents, and tracking the extra bookkeeping (post-
injection action counts, coverage, timing) this experiment needs.

Two injection positions share this one loop, controlled by
``injection_delay`` (number of *real* agent actions to run before the
synthetic evidence is shown):

    injection_delay=0 ("beginning", CLAUDE.md-equivalent default):
        turn 0            : SEARCH(<question>) -> injected results   (bootstrap, uncounted)
        turn 1 (action 1) : first real agent decision
        ...

    injection_delay=N>0 ("mid_trajectory"):
        turn 0..N-1        : N real agent actions, no evidence shown yet
        turn N              : SEARCH(<question>) -> injected results   (bootstrap, uncounted)
        turn N+1 (action 1) : first real agent decision post-injection
        ...

If the agent answers FINAL before N real actions are taken, injection never
happens for that episode (termination_reason="answered_before_injection");
it is still graded for accuracy but excluded from post-injection efficiency
metrics, since there is no post-injection trajectory to measure.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from figbrowse.agent import cached_search, get_agent_action
from figbrowse.agent import run_episode as figbrowse_run_episode
from figbrowse.cache import SimpleCache
from figbrowse.llm import AzureOpenAIClient
from figbrowse.retrieval import _truncate_to_tokens
from figbrowse.run_dir import prompt_hash
from figbrowse.schemas import RetrievedDocument, TurnRecord

from .logging_schema import ActionRecord


class InjectionEpisodeResult(BaseModel):
    turns: list[TurnRecord] = Field(default_factory=list)
    pre_injection_actions: list[ActionRecord] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)

    injection_position: str = "beginning"
    injection_delivered: bool = True
    injection_delivered_after_actions: int = 0

    final_answer: str | None = None
    cited_doc_ids: list[str] = Field(default_factory=list)

    search_calls_after_injection: int = 0
    get_document_calls: int = 0
    unique_documents_opened: int = 0
    documents_opened: list[str] = Field(default_factory=list)
    injected_documents_opened: list[str] = Field(default_factory=list)
    injected_document_coverage: float = 0.0
    all_injected_documents_opened_turn: int | None = None
    first_answer_support_turn: int | None = None
    final_answer_turn: int | None = None
    immediate_answer: bool = False

    termination_reason: str = "unknown"
    completed: bool = False
    error: str | None = None
    latency_seconds: float = 0.0


def _build_injected_turn(
    *,
    run_id: str,
    episode_id: str,
    query_id: str,
    variant_id: str,
    agent_id: str,
    retriever_id: str,
    synthetic_query: str,
    injected_docs: list[RetrievedDocument],
    turn_index: int,
    remaining: int,
) -> TurnRecord:
    """One ordinary-shaped SEARCH turn carrying the injected documents.

    Rendered by figbrowse.agent._render_trajectory exactly like any other
    search turn — the agent cannot distinguish it from a real search result.
    """
    return TurnRecord(
        run_id=run_id,
        episode_id=episode_id,
        query_id=query_id,
        variant_id=variant_id,
        variant_type="initial_gold_injection",
        agent_id=agent_id,
        retriever_id=retriever_id,
        turn_index=turn_index,
        remaining_search_calls_before=remaining,
        action="search",
        search_query=synthetic_query,
        rationale=None,
        retrieved_doc_ids=[d.doc_id for d in injected_docs],
        retrieved_scores=[d.score for d in injected_docs],
        retrieved_previews=[d.preview for d in injected_docs],
        result_preview_hashes=[prompt_hash(d.preview) for d in injected_docs],
        prompt_hash=prompt_hash(synthetic_query),
        model_parameters={"bootstrap_injection": True},
    )


def run_episode_with_injection(
    *,
    run_id: str,
    episode_id: str,
    query_id: str,
    variant_id: str,
    question: str,
    agent_id: str,
    retriever_id: str,
    client: AzureOpenAIClient,
    retriever: Any,
    corpus_store: Any,
    injected_docs: list[RetrievedDocument],
    injected_doc_ids: set[str],
    max_post_injection_actions: int = 8,
    injection_delay: int = 0,
    injection_position: str = "beginning",
    top_k: int = 5,
    preview_tokens: int = 512,
    document_view_tokens: int = 4096,
    temperature: float = 0.0,
    enable_search: bool = True,
    enable_get_document: bool = True,
    llm_cache: SimpleCache | None = None,
    retrieval_cache: SimpleCache | None = None,
) -> InjectionEpisodeResult:
    """Run one episode, injecting evidence either immediately (injection_delay=0)
    or after ``injection_delay`` real agent actions ("mid_trajectory").

    Budget is shared across pre- and post-injection phases: the agent always
    has exactly ``max_post_injection_actions`` real actions total, regardless
    of where the injection lands, so conditions stay comparable.
    """
    t0 = time.perf_counter()

    turns: list[TurnRecord] = []
    pre_injection_actions: list[ActionRecord] = []
    actions: list[ActionRecord] = []

    seen_doc_ids: set[str] = set()
    documents_opened: set[str] = set()
    injected_documents_opened: set[str] = set()
    all_injected_opened_turn: int | None = None
    first_support_turn: int | None = None

    remaining = max_post_injection_actions
    turn_index = 0
    pre_action_no = 0
    action_no = 0
    search_calls = 0
    get_document_calls = 0
    injected = False

    def _inject_now() -> None:
        nonlocal turns, turn_index, injected, seen_doc_ids
        injected_turn = _build_injected_turn(
            run_id=run_id, episode_id=episode_id, query_id=query_id, variant_id=variant_id,
            agent_id=agent_id, retriever_id=retriever_id, synthetic_query=question,
            injected_docs=injected_docs, turn_index=turn_index, remaining=remaining,
        )
        turns.append(injected_turn)
        turn_index += 1
        seen_doc_ids.update(d.doc_id for d in injected_docs)
        injected = True

    if injection_delay <= 0:
        _inject_now()

    invalid_citation_note: str | None = None
    repaired_once = False
    budget_violation_repaired = False
    disabled_action_repaired = False

    def _mark_evidence_access(turn_no: int) -> None:
        nonlocal all_injected_opened_turn, first_support_turn
        if injected_documents_opened and first_support_turn is None:
            first_support_turn = turn_no
        if injected_doc_ids and injected_documents_opened >= injected_doc_ids:
            if all_injected_opened_turn is None:
                all_injected_opened_turn = turn_no

    try:
        while True:
            call_t0 = time.perf_counter()
            action = get_agent_action(
                question, turns, remaining, max_post_injection_actions, client,
                temperature=temperature, llm_cache=llm_cache,
                invalid_citation_note=invalid_citation_note,
            )
            call_latency = time.perf_counter() - call_t0
            invalid_citation_note = None

            if action.action == "search" and not enable_search:
                if not disabled_action_repaired:
                    disabled_action_repaired = True
                    invalid_citation_note = (
                        "The search action is disabled for this episode. Use "
                        "get_document on a previously seen doc_id, or respond FINAL."
                    )
                    continue
                action.action = "final"
                action.answer = action.answer or "(no answer — search disabled, agent did not comply)"
                action.confidence = 0.0

            if action.action == "get_document" and not enable_get_document:
                if not disabled_action_repaired:
                    disabled_action_repaired = True
                    invalid_citation_note = (
                        "The get_document action is disabled for this episode. "
                        "Respond FINAL with your best answer."
                    )
                    continue
                action.action = "final"
                action.answer = action.answer or "(no answer — get_document disabled, agent did not comply)"
                action.confidence = 0.0

            if action.action in ("search", "get_document") and remaining == 0:
                if not budget_violation_repaired:
                    budget_violation_repaired = True
                    invalid_citation_note = (
                        "Your search/get_document budget is exhausted (0 remaining). You "
                        "must respond with a FINAL answer now — no more searches or "
                        "document reads are possible."
                    )
                    continue
                action.action = "final"
                action.answer = action.answer or "(no answer — budget exhausted, agent did not comply)"
                action.confidence = 0.0

            if action.action == "search" and remaining > 0:
                hits = cached_search(
                    retriever, action.search_query, corpus_store, top_k, preview_tokens,
                    retrieval_cache=retrieval_cache,
                )
                seen_doc_ids.update(h.doc_id for h in hits)
                remaining -= 1
                turns.append(TurnRecord(
                    run_id=run_id, episode_id=episode_id, query_id=query_id,
                    variant_id=variant_id, variant_type="initial_gold_injection",
                    agent_id=agent_id, retriever_id=retriever_id,
                    turn_index=turn_index,
                    remaining_search_calls_before=remaining + 1,
                    action="search",
                    search_query=action.search_query,
                    rationale=action.thought,
                    retrieved_doc_ids=[h.doc_id for h in hits],
                    retrieved_scores=[h.score for h in hits],
                    retrieved_previews=[h.preview for h in hits],
                    result_preview_hashes=[prompt_hash(h.preview) for h in hits],
                    prompt_hash=prompt_hash(action.search_query),
                    model_parameters={"temperature": temperature},
                ))
                turn_index += 1
                if injected:
                    action_no += 1
                    search_calls += 1
                    actions.append(ActionRecord(
                        turn=action_no, action_type="search",
                        arguments={"query": action.search_query},
                        observation_summary=f"{len(hits)} result(s) returned",
                        document_ids_returned=[h.doc_id for h in hits],
                        latency_seconds=call_latency,
                    ))
                else:
                    pre_action_no += 1
                    pre_injection_actions.append(ActionRecord(
                        turn=pre_action_no, action_type="search",
                        arguments={"query": action.search_query},
                        observation_summary=f"{len(hits)} result(s) returned",
                        document_ids_returned=[h.doc_id for h in hits],
                        latency_seconds=call_latency,
                    ))
                    if pre_action_no >= injection_delay:
                        _inject_now()
                continue

            if action.action == "get_document" and remaining > 0:
                if action.doc_id not in seen_doc_ids:
                    if not repaired_once:
                        repaired_once = True
                        invalid_citation_note = (
                            f"doc_id {action.doc_id!r} was never returned by a search in this "
                            f"episode. Only request one of: {sorted(seen_doc_ids)}. Try again."
                        )
                        continue
                    text = None
                else:
                    record = corpus_store.get(action.doc_id) if corpus_store is not None else None
                    text = _truncate_to_tokens(record["text"], document_view_tokens) if record else None

                remaining -= 1
                turns.append(TurnRecord(
                    run_id=run_id, episode_id=episode_id, query_id=query_id,
                    variant_id=variant_id, variant_type="initial_gold_injection",
                    agent_id=agent_id, retriever_id=retriever_id,
                    turn_index=turn_index,
                    remaining_search_calls_before=remaining + 1,
                    action="get_document",
                    rationale=action.thought,
                    document_id=action.doc_id,
                    document_text=text,
                    prompt_hash=prompt_hash(action.doc_id or ""),
                    model_parameters={"temperature": temperature},
                ))
                turn_index += 1

                opened = text is not None and bool(action.doc_id)
                if opened:
                    documents_opened.add(action.doc_id)

                if injected:
                    action_no += 1
                    get_document_calls += 1
                    if opened and action.doc_id in injected_doc_ids:
                        injected_documents_opened.add(action.doc_id)
                    _mark_evidence_access(action_no)
                    actions.append(ActionRecord(
                        turn=action_no, action_type="get_document",
                        arguments={"doc_id": action.doc_id},
                        observation_summary="found" if opened else "not_found",
                        document_ids_returned=[action.doc_id] if opened else [],
                        latency_seconds=call_latency,
                    ))
                else:
                    pre_action_no += 1
                    pre_injection_actions.append(ActionRecord(
                        turn=pre_action_no, action_type="get_document",
                        arguments={"doc_id": action.doc_id},
                        observation_summary="found" if opened else "not_found",
                        document_ids_returned=[action.doc_id] if opened else [],
                        latency_seconds=call_latency,
                    ))
                    if pre_action_no >= injection_delay:
                        _inject_now()
                continue

            # FINAL
            cited = list(action.cited_doc_ids or [])
            invalid = [c for c in cited if c not in seen_doc_ids]
            if invalid and not repaired_once:
                repaired_once = True
                invalid_citation_note = (
                    f"You cited document ID(s) {invalid} that were never retrieved in this "
                    f"episode. Only cite from: {sorted(seen_doc_ids)}. Provide a corrected "
                    "FINAL answer now."
                )
                continue
            if invalid and repaired_once:
                cited = [c for c in cited if c in seen_doc_ids]

            turns.append(TurnRecord(
                run_id=run_id, episode_id=episode_id, query_id=query_id,
                variant_id=variant_id, variant_type="initial_gold_injection",
                agent_id=agent_id, retriever_id=retriever_id,
                turn_index=turn_index,
                remaining_search_calls_before=remaining,
                action="final",
                rationale=action.thought,
                answer=action.answer,
                confidence=action.confidence,
                cited_doc_ids=cited,
                stop_reason="final" if action.action == "final" else "budget_exhausted",
                prompt_hash=prompt_hash(action.answer or ""),
                model_parameters={"temperature": temperature},
            ))

            if not injected:
                # Agent answered before the mid-trajectory injection point —
                # evidence was never shown for this episode.
                pre_action_no += 1
                pre_injection_actions.append(ActionRecord(
                    turn=pre_action_no, action_type="answer",
                    arguments={"answer": action.answer, "cited_doc_ids": cited},
                    observation_summary=None, document_ids_returned=[],
                    latency_seconds=call_latency,
                ))
                return InjectionEpisodeResult(
                    turns=turns,
                    pre_injection_actions=pre_injection_actions,
                    actions=actions,
                    injection_position=injection_position,
                    injection_delivered=False,
                    final_answer=action.answer or "",
                    cited_doc_ids=cited,
                    immediate_answer=(pre_action_no == 1),
                    termination_reason="answered_before_injection",
                    completed=True,
                    latency_seconds=time.perf_counter() - t0,
                )

            action_no += 1
            actions.append(ActionRecord(
                turn=action_no, action_type="answer",
                arguments={"answer": action.answer, "cited_doc_ids": cited},
                observation_summary=None, document_ids_returned=[],
                latency_seconds=call_latency,
            ))

            coverage = (
                len(injected_documents_opened) / len(injected_doc_ids)
                if injected_doc_ids else 0.0
            )
            return InjectionEpisodeResult(
                turns=turns,
                pre_injection_actions=pre_injection_actions,
                actions=actions,
                injection_position=injection_position,
                injection_delivered=True,
                injection_delivered_after_actions=pre_action_no,
                final_answer=action.answer or "",
                cited_doc_ids=cited,
                search_calls_after_injection=search_calls,
                get_document_calls=get_document_calls,
                unique_documents_opened=len(documents_opened),
                documents_opened=sorted(documents_opened),
                injected_documents_opened=sorted(injected_documents_opened),
                injected_document_coverage=coverage,
                all_injected_documents_opened_turn=all_injected_opened_turn,
                first_answer_support_turn=first_support_turn,
                final_answer_turn=action_no,
                immediate_answer=(action_no == 1),
                termination_reason="final" if action.action == "final" else "budget_exhausted",
                completed=True,
                latency_seconds=time.perf_counter() - t0,
            )
    except Exception as e:
        coverage = (
            len(injected_documents_opened) / len(injected_doc_ids)
            if injected_doc_ids else 0.0
        )
        return InjectionEpisodeResult(
            turns=turns,
            pre_injection_actions=pre_injection_actions,
            actions=actions,
            injection_position=injection_position,
            injection_delivered=injected,
            injection_delivered_after_actions=pre_action_no,
            search_calls_after_injection=search_calls,
            get_document_calls=get_document_calls,
            unique_documents_opened=len(documents_opened),
            documents_opened=sorted(documents_opened),
            injected_documents_opened=sorted(injected_documents_opened),
            injected_document_coverage=coverage,
            all_injected_documents_opened_turn=all_injected_opened_turn,
            first_answer_support_turn=first_support_turn,
            termination_reason="error",
            completed=False,
            error=f"{type(e).__name__}: {e}",
            latency_seconds=time.perf_counter() - t0,
        )


def _turn_to_action(turn: TurnRecord, turn_no: int) -> ActionRecord:
    if turn.action == "search":
        return ActionRecord(
            turn=turn_no, action_type="search",
            arguments={"query": turn.search_query},
            observation_summary=f"{len(turn.retrieved_doc_ids)} result(s) returned",
            document_ids_returned=turn.retrieved_doc_ids,
        )
    if turn.action == "get_document":
        found = turn.document_text is not None
        return ActionRecord(
            turn=turn_no, action_type="get_document",
            arguments={"doc_id": turn.document_id},
            observation_summary="found" if found else "not_found",
            document_ids_returned=[turn.document_id] if found and turn.document_id else [],
        )
    return ActionRecord(
        turn=turn_no, action_type="answer",
        arguments={"answer": turn.answer, "cited_doc_ids": turn.cited_doc_ids},
    )


def run_episode_no_injection(
    *,
    run_id: str,
    episode_id: str,
    query_id: str,
    variant_id: str,
    question: str,
    agent_id: str,
    retriever_id: str,
    client: AzureOpenAIClient,
    retriever: Any,
    corpus_store: Any,
    max_actions: int = 10,
    top_k: int = 5,
    preview_tokens: int = 512,
    document_view_tokens: int = 4096,
    temperature: float = 0.0,
    llm_cache: SimpleCache | None = None,
    retrieval_cache: SimpleCache | None = None,
) -> InjectionEpisodeResult:
    """No-injection control: a direct, unmodified call to FigBrowse's own
    ``figbrowse.agent.run_episode`` — the ordinary FigBrowse search loop with
    no synthetic evidence turn at all, just a different action ceiling.
    """
    t0 = time.perf_counter()
    result = figbrowse_run_episode(
        run_id=run_id, episode_id=episode_id, query_id=query_id,
        variant_id=variant_id, variant_type="no_injection_control",
        question=question, agent_id=agent_id, retriever_id=retriever_id,
        client=client, retriever=retriever, corpus_store=corpus_store,
        max_search_calls=max_actions, top_k=top_k, preview_tokens=preview_tokens,
        document_view_tokens=document_view_tokens, temperature=temperature,
        llm_cache=llm_cache, retrieval_cache=retrieval_cache,
    )

    actions = [_turn_to_action(t, i + 1) for i, t in enumerate(result.turns)]
    search_calls = sum(1 for t in result.turns if t.action == "search")
    get_document_calls = sum(1 for t in result.turns if t.action == "get_document")
    documents_opened = sorted({
        t.document_id for t in result.turns
        if t.action == "get_document" and t.document_text is not None and t.document_id
    })

    return InjectionEpisodeResult(
        turns=result.turns,
        pre_injection_actions=[],
        actions=actions,
        injection_position="no_injection",
        injection_delivered=False,
        injection_delivered_after_actions=0,
        final_answer=result.final_answer,
        cited_doc_ids=result.cited_doc_ids,
        search_calls_after_injection=search_calls,
        get_document_calls=get_document_calls,
        unique_documents_opened=len(documents_opened),
        documents_opened=documents_opened,
        injected_documents_opened=[],
        injected_document_coverage=0.0,
        all_injected_documents_opened_turn=None,
        first_answer_support_turn=None,
        final_answer_turn=len(actions) if result.completed else None,
        immediate_answer=(len(actions) == 1),
        termination_reason=result.termination,
        completed=result.completed,
        error=result.error,
        latency_seconds=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Condition D: full-evidence reader — one call, no tools, no trajectory
# ---------------------------------------------------------------------------


class ReaderAnswer(BaseModel):
    answer: str
    confidence: float = 0.0
    cited_doc_ids: list[str] = Field(default_factory=list)
    brief_support: str | None = None


def _load_local_prompt(name: str) -> str:
    # stopBrowse's own prompts/ dir — this capability (single-shot full-
    # context reading) doesn't exist in FigBrowse, so it isn't reusing an
    # existing prompt file the way the agent loop reuses search_agent.md.
    return (Path(__file__).parent.parent.parent.parent / "prompts" / name).read_text(encoding="utf-8")


def build_reader_prompt(question: str, docs: list[RetrievedDocument]) -> str:
    template = _load_local_prompt("full_evidence_reader.md")
    doc_blocks = []
    for d in docs:
        doc_blocks.append(f"### Document [{d.doc_id}]\n\n{d.preview}")
    return template.replace("{question}", question).replace("{documents}", "\n\n".join(doc_blocks))


def run_full_evidence_reader(
    *,
    question: str,
    docs: list[RetrievedDocument],
    client: AzureOpenAIClient,
    temperature: float = 0.0,
    llm_cache: SimpleCache | None = None,
) -> InjectionEpisodeResult:
    """Condition D: give the model the full text of every injected document
    up front, no tools, no trajectory, one immediate answer. Answers the
    question "can this model solve it when evidence access and stopping
    are removed?" — the evidence-answerability ceiling that condition B
    (agentic access to the same documents) is compared against.
    """
    t0 = time.perf_counter()
    prompt_text = build_reader_prompt(question, docs)
    messages = [{"role": "user", "content": prompt_text}]

    if llm_cache is not None:
        from figbrowse.cache import llm_cache_key
        key = llm_cache_key(
            model=client._model(), messages=messages, schema_name="ReaderAnswer",
            temperature=temperature, prompt_template_id="full_evidence_reader_v1",
        )
        cached = llm_cache.get(key)
        if cached is not None:
            answer = ReaderAnswer.model_validate(cached)
            return _reader_result(answer, docs, time.perf_counter() - t0)

    try:
        answer, _ = client.generate_structured(
            messages=messages, schema=ReaderAnswer,
            prompt_template_id="full_evidence_reader_v1",
            temperature=temperature, max_schema_retries=2,
        )
    except Exception as e:
        return InjectionEpisodeResult(
            injection_position="full_evidence_reader",
            injection_delivered=True,
            termination_reason="error",
            completed=False,
            error=f"{type(e).__name__}: {e}",
            latency_seconds=time.perf_counter() - t0,
        )

    if llm_cache is not None:
        from figbrowse.cache import llm_cache_key
        key = llm_cache_key(
            model=client._model(), messages=messages, schema_name="ReaderAnswer",
            temperature=temperature, prompt_template_id="full_evidence_reader_v1",
        )
        llm_cache.set(
            key, answer.model_dump(mode="json"),
            provenance={"prompt_template_id": "full_evidence_reader_v1", "model": client._model()},
        )

    return _reader_result(answer, docs, time.perf_counter() - t0)


def _reader_result(answer: ReaderAnswer, docs: list[RetrievedDocument], latency: float) -> InjectionEpisodeResult:
    doc_ids = [d.doc_id for d in docs]
    action = ActionRecord(
        turn=1, action_type="answer",
        arguments={"answer": answer.answer, "cited_doc_ids": answer.cited_doc_ids},
    )
    return InjectionEpisodeResult(
        actions=[action],
        injection_position="full_evidence_reader",
        injection_delivered=True,
        final_answer=answer.answer,
        cited_doc_ids=answer.cited_doc_ids,
        injected_documents_opened=doc_ids,
        injected_document_coverage=1.0 if doc_ids else 0.0,
        final_answer_turn=1,
        immediate_answer=True,
        termination_reason="final",
        completed=True,
        latency_seconds=latency,
    )
