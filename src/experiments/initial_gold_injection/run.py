"""CLI entrypoint: run the initial_gold_injection experiment.

    python -m experiments.initial_gold_injection.run \\
        --config configs/initial_gold_injection.yaml --limit 3

Thin orchestration layer only: all real work (data loading, retrieval,
agent decisions, caching, grading) is delegated to FigBrowse modules; this
file wires them together around the injected-evidence bootstrap turn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from figbrowse.data import BrowseCompPlusLoader, CorpusStore
from figbrowse.evaluation import judge_answer
from figbrowse.llm import AzureOpenAIClient
from figbrowse.cache import SimpleCache
from figbrowse.retrieval import Qwen3ShardedRetriever

from experiments.common.llm_clients import build_llm_client

from .config import ExperimentConfig, load_config
from .episode import run_episode_no_injection, run_episode_with_injection, run_full_evidence_reader
from .injection import build_injected_documents, check_accessible, select_evidence_doc_ids
from .logging_schema import RunRecord
from .taxonomy import assign_control_label, assign_labels

# Positions whose whole point is disabling search (subject to the same
# injected-evidence accessibility check as "beginning"/"mid_trajectory").
_NO_SEARCH_POSITIONS = {"beginning_no_search"}

app = typer.Typer(add_completion=False)


def _load_manifest(path: Path, limit: int | None) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit is not None:
        records = records[:limit]
    return records


def _load_completed_ids(output_path: Path) -> set[tuple[str, str]]:
    """(question_id, injection_position) pairs already present in runs.jsonl.

    Keyed by the pair — not just question_id — so multiple injection
    positions for the same question can coexist in one output file and
    resume independently.
    """
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("question_id")
            if qid:
                done.add((qid, rec.get("injection_position", "beginning")))
    return done


def run_experiment(config: ExperimentConfig, *, limit: int | None, resume: bool) -> Path:
    # Reuses FigBrowse's own credential file (same env-var names/conventions
    # as figbrowse.cli's load_dotenv()) rather than duplicating secrets.
    load_dotenv(config.figbrowse_root() / ".env")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runs.jsonl"

    manifest = _load_manifest(config.manifest_path(), limit or config.sample_limit)

    already_done = _load_completed_ids(output_path) if resume else set()
    if already_done:
        print(f"[resume] {len(already_done)} question(s) already completed; skipping them.")

    loader = BrowseCompPlusLoader(
        questions_path=config.resolve_path(config.benchmark.questions_path),
        evidence_qrels_path=config.resolve_path(config.benchmark.evidence_qrels_path),
        gold_qrels_path=config.resolve_path(config.benchmark.gold_qrels_path),
        answers_path=config.resolve_path(config.benchmark.answers_path),
    )
    corpus_store = CorpusStore(config.resolve_path(config.benchmark.corpus_path))
    retriever = Qwen3ShardedRetriever(
        index_path=config.resolve_path(config.retriever.index_path),
        query_encoder=config.retriever.query_encoder,
        task_prefix=config.retriever.task_prefix,
        normalize=config.retriever.normalize,
        top_k=config.retriever.top_k,
        preview_tokens=config.retriever.preview_tokens,
        retriever_id=config.retriever.id,
    )

    agent_model = config.model.resolved_name()
    agent_client = build_llm_client(
        provider=config.model.provider, model_name=agent_model,
        temperature=config.model.temperature, max_output_tokens=config.agent.max_output_tokens,
    )
    judge_client = AzureOpenAIClient(
        model_env="FIGBROWSE_GENERATOR_MODEL",
        default_model="gpt-4o",
        temperature=0.0,
    )

    cache_root = config.figbrowse_root() / ".cache" / "figbrowse"
    llm_cache = SimpleCache(cache_root / "llm")
    retrieval_cache = SimpleCache(cache_root / "retrieval")

    positions = config.injection.positions or ["beginning"]

    n_run = 0
    with output_path.open("a", encoding="utf-8") as out_f:
        for rec in manifest:
            query_id = str(rec["query_id"])

            example = loader.get(query_id)
            question = example.inference_view().question  # gold-free view

            doc_ids, source = select_evidence_doc_ids(
                example,
                prefer_evidence_documents=config.injection.prefer_evidence_documents,
                fallback_to_gold_documents=config.injection.fallback_to_gold_documents,
            )
            accessible, missing = check_accessible(doc_ids, corpus_store)
            all_accessible = len(missing) == 0
            gold_answer = (example.reference_answers or [None])[0]

            for position in positions:
                if (query_id, position) in already_done:
                    continue

                if position == "no_injection":
                    episode_id = f"igi_{query_id}_no_injection"
                    result = run_episode_no_injection(
                        run_id=config.experiment_name,
                        episode_id=episode_id,
                        query_id=query_id,
                        variant_id=f"{query_id}_no_injection",
                        question=question,
                        agent_id=agent_model,
                        retriever_id=config.retriever.id,
                        client=agent_client,
                        retriever=retriever,
                        corpus_store=corpus_store,
                        max_actions=config.injection.no_injection_max_actions,
                        top_k=config.retriever.top_k,
                        preview_tokens=config.retriever.preview_tokens,
                        document_view_tokens=config.agent.document_view_tokens,
                        temperature=config.model.temperature,
                        llm_cache=llm_cache,
                        retrieval_cache=retrieval_cache,
                    )

                    answer_correct = False
                    judge_error = None
                    if result.completed and gold_answer and result.final_answer:
                        try:
                            judged = judge_answer(question, result.final_answer, gold_answer, judge_client)
                            answer_correct = judged.correct
                        except Exception as e:  # noqa: BLE001
                            judge_error = f"{type(e).__name__}: {e}"

                    primary_label = assign_control_label(
                        answer_correct=answer_correct,
                        termination_reason=result.termination_reason,
                        error=result.error or judge_error,
                    )

                    run_record = RunRecord(
                        question_id=query_id,
                        question=question,
                        gold_answer=gold_answer,
                        injected_document_ids=[],
                        num_injected_documents=0,
                        all_injected_documents_accessible=True,
                        model=agent_model,
                        seed=config.seed,
                        max_post_injection_actions=config.injection.no_injection_max_actions,
                        injection_position="no_injection",
                        injection_delivered=False,
                        actions=result.actions,
                        search_calls=result.search_calls_after_injection,
                        get_document_calls=result.get_document_calls,
                        unique_documents_opened=result.unique_documents_opened,
                        injected_documents_opened=[],
                        injected_document_coverage=0.0,
                        final_answer_turn=result.final_answer_turn,
                        final_answer=result.final_answer,
                        answer_correct=answer_correct,
                        termination_reason=result.termination_reason,
                        latency_seconds=result.latency_seconds,
                        error=result.error or judge_error,
                        immediate_answer=result.immediate_answer,
                        primary_label=primary_label,
                    )
                    out_f.write(run_record.model_dump_json() + "\n")
                    out_f.flush()
                    n_run += 1
                    print(
                        f"[{query_id}/no_injection] {primary_label} "
                        f"correct={answer_correct} actions={len(result.actions)}"
                    )
                    continue

                if position == "full_evidence_reader":
                    if not accessible:
                        run_record = RunRecord(
                            question_id=query_id, question=question, gold_answer=gold_answer,
                            injected_document_ids=[], num_injected_documents=0,
                            all_injected_documents_accessible=False,
                            model=agent_model, seed=config.seed,
                            injection_position=position,
                            termination_reason="tool_or_data_error",
                            error=f"No accessible {source} documents for query_id={query_id}",
                            primary_label="TOOL_OR_DATA_ERROR",
                        )
                        out_f.write(run_record.model_dump_json() + "\n")
                        out_f.flush()
                        n_run += 1
                        print(f"[{query_id}/{position}] SKIPPED — no accessible evidence/gold documents")
                        continue

                    full_docs = build_injected_documents(
                        accessible, corpus_store,
                        preview_tokens=config.retriever.preview_tokens,
                        document_view_tokens=config.agent.document_view_tokens,
                        expose_full_text_immediately=True,  # condition D always sees full text
                        shuffle=config.injection.shuffle_documents,
                        seed=config.seed,
                        query_id=query_id,
                    )
                    result = run_full_evidence_reader(
                        question=question, docs=full_docs, client=agent_client,
                        temperature=config.model.temperature, llm_cache=llm_cache,
                    )

                    answer_correct = False
                    judge_error = None
                    if result.completed and gold_answer and result.final_answer:
                        try:
                            judged = judge_answer(question, result.final_answer, gold_answer, judge_client)
                            answer_correct = judged.correct
                        except Exception as e:  # noqa: BLE001
                            judge_error = f"{type(e).__name__}: {e}"

                    primary_label = assign_control_label(
                        answer_correct=answer_correct,
                        termination_reason=result.termination_reason,
                        error=result.error or judge_error,
                    )

                    run_record = RunRecord(
                        question_id=query_id, question=question, gold_answer=gold_answer,
                        injected_document_ids=sorted(d.doc_id for d in full_docs),
                        num_injected_documents=len(full_docs),
                        all_injected_documents_accessible=all_accessible,
                        model=agent_model, seed=config.seed,
                        max_post_injection_actions=1,
                        injection_position="full_evidence_reader",
                        injection_delivered=True,
                        actions=result.actions,
                        search_calls=0,
                        get_document_calls=0,
                        unique_documents_opened=len(full_docs),
                        injected_documents_opened=result.injected_documents_opened,
                        injected_document_coverage=result.injected_document_coverage,
                        final_answer_turn=1,
                        final_answer=result.final_answer,
                        answer_correct=answer_correct,
                        termination_reason=result.termination_reason,
                        latency_seconds=result.latency_seconds,
                        error=result.error or judge_error,
                        immediate_answer=True,
                        primary_label=primary_label,
                    )
                    out_f.write(run_record.model_dump_json() + "\n")
                    out_f.flush()
                    n_run += 1
                    print(f"[{query_id}/full_evidence_reader] {primary_label} correct={answer_correct}")
                    continue

                if not accessible:
                    run_record = RunRecord(
                        question_id=query_id,
                        question=question,
                        gold_answer=gold_answer,
                        injected_document_ids=[],
                        num_injected_documents=0,
                        all_injected_documents_accessible=False,
                        model=agent_model,
                        seed=config.seed,
                        max_post_injection_actions=config.agent.max_post_injection_actions,
                        injection_position=position,
                        termination_reason="tool_or_data_error",
                        error=f"No accessible {source} documents for query_id={query_id}",
                        primary_label="TOOL_OR_DATA_ERROR",
                    )
                    out_f.write(run_record.model_dump_json() + "\n")
                    out_f.flush()
                    n_run += 1
                    print(f"[{query_id}/{position}] SKIPPED — no accessible evidence/gold documents")
                    continue

                injected_docs = build_injected_documents(
                    accessible, corpus_store,
                    preview_tokens=config.retriever.preview_tokens,
                    document_view_tokens=config.agent.document_view_tokens,
                    expose_full_text_immediately=config.injection.expose_full_text_immediately,
                    shuffle=config.injection.shuffle_documents,
                    seed=config.seed,
                    query_id=query_id,
                )
                injected_doc_ids = set(accessible)
                # "beginning_no_search" is condition E: same beginning-of-
                # trajectory injection as "beginning", but with the search
                # tool removed — an oracle diagnostic, not a new timing.
                is_no_search = position in _NO_SEARCH_POSITIONS
                injection_delay = 0 if position in ("beginning", *_NO_SEARCH_POSITIONS) else config.injection.mid_trajectory_delay
                enable_search = False if is_no_search else config.agent.enable_search

                episode_id = f"igi_{query_id}_{position}"
                result = run_episode_with_injection(
                    run_id=config.experiment_name,
                    episode_id=episode_id,
                    query_id=query_id,
                    variant_id=f"{query_id}_igi_{position}",
                    question=question,
                    agent_id=agent_model,
                    retriever_id=config.retriever.id,
                    client=agent_client,
                    retriever=retriever,
                    corpus_store=corpus_store,
                    injected_docs=injected_docs,
                    injected_doc_ids=injected_doc_ids,
                    max_post_injection_actions=config.agent.max_post_injection_actions,
                    injection_delay=injection_delay,
                    injection_position=position,
                    top_k=config.retriever.top_k,
                    preview_tokens=config.retriever.preview_tokens,
                    document_view_tokens=config.agent.document_view_tokens,
                    temperature=config.model.temperature,
                    enable_search=enable_search,
                    enable_get_document=config.agent.enable_get_document,
                    llm_cache=llm_cache,
                    retrieval_cache=retrieval_cache,
                )

                answer_correct = False
                judge_error = None
                if result.completed and gold_answer and result.final_answer:
                    try:
                        judged = judge_answer(question, result.final_answer, gold_answer, judge_client)
                        answer_correct = judged.correct
                    except Exception as e:  # noqa: BLE001 - grading failure must not crash the run
                        judge_error = f"{type(e).__name__}: {e}"

                post_access_delay = None
                if (
                    result.all_injected_documents_opened_turn is not None
                    and result.final_answer_turn is not None
                ):
                    post_access_delay = (
                        result.final_answer_turn - result.all_injected_documents_opened_turn
                    )

                primary_label, secondary_labels = assign_labels(
                    answer_correct=answer_correct,
                    immediate_answer=result.immediate_answer,
                    termination_reason=result.termination_reason,
                    error=result.error or judge_error,
                    search_calls_after_injection=result.search_calls_after_injection,
                    get_document_calls=result.get_document_calls,
                    injected_documents_opened=result.injected_documents_opened,
                    injected_document_coverage=result.injected_document_coverage,
                )

                run_record = RunRecord(
                    question_id=query_id,
                    question=question,
                    gold_answer=gold_answer,
                    injected_document_ids=sorted(injected_doc_ids),
                    num_injected_documents=len(injected_doc_ids),
                    all_injected_documents_accessible=all_accessible,
                    model=agent_model,
                    seed=config.seed,
                    max_post_injection_actions=config.agent.max_post_injection_actions,
                    injection_position=position,
                    injection_delivered=result.injection_delivered,
                    injection_delivered_after_actions=result.injection_delivered_after_actions,
                    pre_injection_actions=result.pre_injection_actions,
                    actions=result.actions,
                    search_calls=result.search_calls_after_injection,
                    get_document_calls=result.get_document_calls,
                    unique_documents_opened=result.unique_documents_opened,
                    injected_documents_opened=result.injected_documents_opened,
                    injected_document_coverage=result.injected_document_coverage,
                    all_injected_documents_opened_turn=result.all_injected_documents_opened_turn,
                    first_answer_support_turn=result.first_answer_support_turn,
                    final_answer_turn=result.final_answer_turn,
                    post_access_delay=post_access_delay,
                    final_answer=result.final_answer,
                    answer_correct=answer_correct,
                    termination_reason=result.termination_reason,
                    latency_seconds=result.latency_seconds,
                    error=result.error or judge_error,
                    immediate_answer=result.immediate_answer,
                    primary_label=primary_label,
                    secondary_labels=secondary_labels,
                )
                out_f.write(run_record.model_dump_json() + "\n")
                out_f.flush()
                n_run += 1
                print(
                    f"[{query_id}/{position}] {primary_label} "
                    f"correct={answer_correct} actions={len(result.actions)} "
                    f"coverage={result.injected_document_coverage:.2f}"
                )

    print(f"Done. {n_run} sample(s) processed this invocation -> {output_path}")
    return output_path


@app.command()
def main(
    config: str = typer.Option(..., "--config", help="Path to experiment YAML config"),
    limit: int = typer.Option(None, "--limit", help="Override sample_limit for this invocation"),
    resume: bool = typer.Option(None, "--resume/--no-resume", help="Skip already-completed question_ids"),
) -> None:
    cfg = load_config(config)
    resume_flag = cfg.resume if resume is None else resume
    run_experiment(cfg, limit=limit, resume=resume_flag)


if __name__ == "__main__":
    app()
