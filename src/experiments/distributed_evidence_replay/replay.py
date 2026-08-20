"""One-shot replay call: frozen trajectory in, single final answer out.

Deliberately does NOT reuse figbrowse.agent._render_trajectory — that
renderer collapses all but the 3 most recent search turns' previews (an
agent-loop context-management optimization for incremental multi-turn
calls). Here the model sees the entire trajectory exactly once, so every
search turn must show its full preview or injected evidence placed in an
early turn would be invisible to the model, silently defeating the
"distributed throughout the trajectory" design. Turn/action formatting
otherwise mirrors FigBrowse's own rendering conventions for continuity.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field

from figbrowse.cache import SimpleCache, llm_cache_key
from figbrowse.llm import AzureOpenAIClient
from figbrowse.schemas import TurnRecord


class ReplayAnswer(BaseModel):
    answer: str
    cited_document_ids: list[str] = Field(default_factory=list)


def _load_local_prompt(name: str) -> str:
    return (Path(__file__).parent.parent.parent.parent / "prompts" / name).read_text(encoding="utf-8")


def render_full_trajectory(turns: list[TurnRecord]) -> str:
    if not turns:
        return "(no turns)"
    lines = []
    for t in turns:
        if t.action == "search":
            lines.append(f"Turn {t.turn_index}: SEARCH({t.search_query!r})")
            previews = t.retrieved_previews or [""] * len(t.retrieved_doc_ids)
            for doc_id, score, preview in zip(t.retrieved_doc_ids, t.retrieved_scores, previews):
                lines.append(f"  - [{doc_id}] score={score:.3f}")
                if preview:
                    lines.append(f"    preview: {preview}")
        elif t.action == "get_document":
            lines.append(f"Turn {t.turn_index}: GET_DOCUMENT({t.document_id!r})")
            lines.append(f"    text: {t.document_text or '(document not found)'}")
    return "\n".join(lines)


def build_replay_prompt(question: str, turns: list[TurnRecord]) -> str:
    template = _load_local_prompt("replay_answer.md")
    return (
        template
        .replace("{question}", question)
        .replace("{trajectory}", render_full_trajectory(turns))
    )


class ReplayResult(BaseModel):
    final_answer: str | None = None
    cited_document_ids: list[str] = Field(default_factory=list)
    termination_reason: str = "final"
    completed: bool = False
    error: str | None = None
    latency_seconds: float = 0.0


def run_replay_answer(
    *,
    question: str,
    turns: list[TurnRecord],
    client: AzureOpenAIClient,
    temperature: float = 0.0,
    llm_cache: SimpleCache | None = None,
) -> ReplayResult:
    t0 = time.perf_counter()
    prompt_text = build_replay_prompt(question, turns)
    messages = [{"role": "user", "content": prompt_text}]
    prompt_template_id = "distributed_evidence_replay_v1"

    if llm_cache is not None:
        key = llm_cache_key(
            model=client._model(), messages=messages, schema_name="ReplayAnswer",
            temperature=temperature, prompt_template_id=prompt_template_id,
        )
        cached = llm_cache.get(key)
        if cached is not None:
            answer = ReplayAnswer.model_validate(cached)
            return ReplayResult(
                final_answer=answer.answer, cited_document_ids=answer.cited_document_ids,
                termination_reason="final", completed=True,
                latency_seconds=time.perf_counter() - t0,
            )

    try:
        answer, _ = client.generate_structured(
            messages=messages, schema=ReplayAnswer,
            prompt_template_id=prompt_template_id,
            temperature=temperature, max_schema_retries=2,
        )
    except Exception as e:
        return ReplayResult(
            termination_reason="error", completed=False,
            error=f"{type(e).__name__}: {e}", latency_seconds=time.perf_counter() - t0,
        )

    if llm_cache is not None:
        key = llm_cache_key(
            model=client._model(), messages=messages, schema_name="ReplayAnswer",
            temperature=temperature, prompt_template_id=prompt_template_id,
        )
        llm_cache.set(
            key, answer.model_dump(mode="json"),
            provenance={"prompt_template_id": prompt_template_id, "model": client._model()},
        )

    return ReplayResult(
        final_answer=answer.answer, cited_document_ids=answer.cited_document_ids,
        termination_reason="final", completed=True,
        latency_seconds=time.perf_counter() - t0,
    )
