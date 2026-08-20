"""CLI entrypoint: run the distributed_evidence_replay experiment.

    python -m experiments.distributed_evidence_replay.run \\
        --config configs/distributed_evidence_replay.yaml --limit 3 --placement-seeds 42 43

Thin orchestration only: trajectory reconstruction, placement, and the
one-shot replay call are all delegated to reconstruct.py/placement.py/
replay.py; this file just wires them together and handles JSONL I/O/resume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv

from figbrowse.data import BrowseCompPlusLoader, CorpusStore
from figbrowse.evaluation import judge_answer
from figbrowse.llm import AzureOpenAIClient
from figbrowse.cache import SimpleCache
from figbrowse.retrieval import Qwen3ShardedRetriever

from experiments.common.llm_clients import build_llm_client
from experiments.initial_gold_injection.injection import check_accessible, select_evidence_doc_ids

from .config import DistributedEvidenceReplayConfig, load_config
from .logging_schema import RunRecord
from .placement import place_evidence
from .reconstruct import ReconstructionError, reconstruct_original_trajectory
from .replay import run_replay_answer

app = typer.Typer(add_completion=False)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_completed(output_path: Path) -> set[tuple[str, str, int | None]]:
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
                done.add((qid, rec.get("condition"), rec.get("placement_seed")))
    return done


def _evidence_position_stats(
    evidence_ids: set[str], turns, original_search_turns: int,
) -> tuple[float | None, int | None, int | None]:
    """mean_normalized_evidence_position, first_evidence_turn, last_evidence_turn
    over the (possibly modified) trajectory's search turns."""
    positions = []
    for t in turns:
        if t.action != "search":
            continue
        if any(doc_id in evidence_ids for doc_id in t.retrieved_doc_ids):
            positions.append(t.turn_index)
    if not positions:
        return None, None, None
    denom = max(1, original_search_turns - 1)
    normalized = [p / denom for p in positions]
    return sum(normalized) / len(normalized), min(positions), max(positions)


def run_experiment(
    config: DistributedEvidenceReplayConfig, *, limit: int | None, placement_seeds: list[int] | None,
    resume: bool, include_r0: bool = True,
) -> Path:
    load_dotenv(config.figbrowse_root() / ".env")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runs.jsonl"

    source_path = config.resolve_own_path(config.source.runs_path)
    source_records_all = _load_jsonl(source_path)
    source_records = [
        r for r in source_records_all
        if r.get("injection_position") == config.source.injection_position
        and r.get("primary_label") != "TOOL_OR_DATA_ERROR"
    ]
    limit_n = limit or config.sample_limit
    source_records = source_records[:limit_n]

    seeds = placement_seeds or config.replay.placement_seeds

    already_done = _load_completed(output_path) if resume else set()
    if already_done:
        print(f"[resume] {len(already_done)} (question, condition, seed) record(s) already completed.")

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
        provider=config.model.provider, model_name=agent_model, temperature=config.model.temperature,
    )
    judge_client = AzureOpenAIClient(
        model_env="FIGBROWSE_GENERATOR_MODEL", default_model="gpt-4o", temperature=0.0,
    )

    cache_root = config.figbrowse_root() / ".cache" / "figbrowse"
    llm_cache = SimpleCache(cache_root / "llm")
    retrieval_cache = SimpleCache(cache_root / "retrieval")

    conditions: list[tuple[str, int | None]] = [("original_replay", None)] if include_r0 else []
    conditions += [("distributed_evidence_replay", s) for s in seeds]

    n_run = 0
    with output_path.open("a", encoding="utf-8") as out_f:
        for source_record in source_records:
            query_id = source_record["question_id"]

            try:
                trajectory = reconstruct_original_trajectory(
                    source_record, retriever=retriever, corpus_store=corpus_store,
                    top_k=config.retriever.top_k, preview_tokens=config.retriever.preview_tokens,
                    document_view_tokens=config.source.document_view_tokens,
                    retrieval_cache=retrieval_cache,
                )
            except ReconstructionError as e:
                for condition, seed in conditions:
                    if (query_id, condition, seed) in already_done:
                        continue
                    rec = RunRecord(
                        question_id=query_id, condition=condition, placement_seed=seed,
                        model=agent_model, original_trajectory_path=str(source_path),
                        termination_reason="reconstruction_error", error=str(e),
                    )
                    out_f.write(rec.model_dump_json() + "\n")
                    out_f.flush()
                    n_run += 1
                print(f"[{query_id}] RECONSTRUCTION_ERROR — {e}")
                continue

            example = loader.get(query_id)
            gold_answer = (example.reference_answers or [None])[0]
            question = example.inference_view().question

            evidence_ids_list, source = select_evidence_doc_ids(
                example,
                prefer_evidence_documents=config.replay.prefer_evidence_documents,
                fallback_to_gold_documents=config.replay.fallback_to_gold_documents,
            )
            accessible, _missing_from_corpus = check_accessible(evidence_ids_list, corpus_store)
            evidence_ids = set(accessible)

            trajectory_result_ids = set(trajectory.trajectory_result_ids)
            naturally_present = evidence_ids & trajectory_result_ids
            missing = evidence_ids - naturally_present
            protected = set(trajectory.get_document_ids) | set(trajectory.original_cited_doc_ids)

            for condition, seed in conditions:
                if (query_id, condition, seed) in already_done:
                    continue

                if condition == "original_replay":
                    replay_turns = trajectory.turns
                    replacements = []
                    injected = []
                    final_present = naturally_present
                    all_present = len(missing) == 0
                else:
                    placement = place_evidence(
                        trajectory.turns, missing_evidence_ids=missing, evidence_ids=evidence_ids,
                        protected_doc_ids=protected, corpus_store=corpus_store,
                        preview_tokens=config.retriever.preview_tokens, seed=seed,
                    )
                    if not placement.ok:
                        rec = RunRecord(
                            question_id=query_id, condition=condition, placement_seed=seed,
                            model=agent_model, original_trajectory_path=str(source_path),
                            original_search_turns=trajectory.original_search_turns,
                            original_result_slots=trajectory.original_result_slots,
                            total_evidence_documents=len(evidence_ids),
                            naturally_present_evidence_ids=sorted(naturally_present),
                            missing_evidence_ids=sorted(missing),
                            all_evidence_present=False,
                            termination_reason="insufficient_replacement_slots",
                            error=(
                                f"INSUFFICIENT_REPLACEMENT_SLOTS: required={placement.required_slots} "
                                f"available={placement.available_slots}"
                            ),
                            gold_answer=gold_answer,
                        )
                        out_f.write(rec.model_dump_json() + "\n")
                        out_f.flush()
                        n_run += 1
                        print(f"[{query_id}/{condition}/seed={seed}] INSUFFICIENT_REPLACEMENT_SLOTS "
                              f"required={placement.required_slots} available={placement.available_slots}")
                        continue

                    replay_turns = placement.turns
                    replacements = placement.replacements
                    injected = sorted(missing)
                    final_present = evidence_ids

                    modified_result_ids = {
                        doc_id for t in replay_turns if t.action == "search" for doc_id in t.retrieved_doc_ids
                    }
                    if not evidence_ids <= modified_result_ids:
                        rec = RunRecord(
                            question_id=query_id, condition=condition, placement_seed=seed,
                            model=agent_model, original_trajectory_path=str(source_path),
                            total_evidence_documents=len(evidence_ids),
                            naturally_present_evidence_ids=sorted(naturally_present),
                            missing_evidence_ids=sorted(missing),
                            all_evidence_present=False,
                            termination_reason="assertion_failed",
                            error="evidence_ids <= modified_trajectory_result_ids assertion failed",
                            gold_answer=gold_answer,
                        )
                        out_f.write(rec.model_dump_json() + "\n")
                        out_f.flush()
                        n_run += 1
                        print(f"[{query_id}/{condition}/seed={seed}] ASSERTION_FAILED")
                        continue
                    all_present = True

                mean_pos, first_turn, last_turn = _evidence_position_stats(
                    final_present, replay_turns, trajectory.original_search_turns,
                )

                result = run_replay_answer(
                    question=question, turns=replay_turns, client=agent_client,
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

                rec = RunRecord(
                    question_id=query_id, condition=condition, placement_seed=seed,
                    model=agent_model, original_trajectory_path=str(source_path),
                    original_search_turns=trajectory.original_search_turns,
                    original_result_slots=trajectory.original_result_slots,
                    total_evidence_documents=len(evidence_ids),
                    naturally_present_evidence_ids=sorted(naturally_present),
                    missing_evidence_ids=sorted(missing),
                    injected_evidence_ids=injected,
                    final_present_evidence_ids=sorted(final_present),
                    all_evidence_present=all_present,
                    replacements=replacements,
                    mean_normalized_evidence_position=mean_pos,
                    first_evidence_turn=first_turn,
                    last_evidence_turn=last_turn,
                    final_answer=result.final_answer,
                    cited_document_ids=result.cited_document_ids,
                    answer_correct=answer_correct,
                    termination_reason=result.termination_reason,
                    latency_seconds=result.latency_seconds,
                    error=result.error or judge_error,
                    gold_answer=gold_answer,
                )
                out_f.write(rec.model_dump_json() + "\n")
                out_f.flush()
                n_run += 1
                print(f"[{query_id}/{condition}/seed={seed}] correct={answer_correct}")

    print(f"Done. {n_run} record(s) processed this invocation -> {output_path}")
    return output_path


@app.command()
def main(
    config: str = typer.Option(..., "--config", help="Path to experiment YAML config"),
    limit: int = typer.Option(None, "--limit", help="Override sample_limit for this invocation"),
    placement_seeds: list[int] = typer.Option(None, "--placement-seeds", help="Override replay.placement_seeds"),
    resume: bool = typer.Option(None, "--resume/--no-resume"),
    include_r0: bool = typer.Option(
        True, "--include-r0/--no-r0",
        help="Whether to (re)generate original_replay (R0) records this invocation",
    ),
) -> None:
    cfg = load_config(config)
    resume_flag = cfg.resume if resume is None else resume
    run_experiment(
        cfg, limit=limit, placement_seeds=placement_seeds or None,
        resume=resume_flag, include_r0=include_r0,
    )


if __name__ == "__main__":
    app()
