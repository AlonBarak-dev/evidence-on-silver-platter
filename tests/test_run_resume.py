from __future__ import annotations

import json

import experiments.initial_gold_injection.run as run_mod
from experiments.initial_gold_injection.config import ExperimentConfig
from experiments.initial_gold_injection.run import _load_completed_ids, _load_manifest, run_experiment
from tests.fixtures import FakeCorpusStore, FakeLLMClient, FakeLoader, FakeRetriever


def _write_manifest(path, query_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in query_ids:
            f.write(json.dumps({"query_id": qid, "question": "irrelevant"}) + "\n")


def _immediate_final_response():
    return json.dumps({
        "thought": "known", "action": "final", "answer": "Titanic",
        "confidence": 0.9, "cited_doc_ids": [], "brief_support": "x",
    })


def _judge_response(correct=True):
    return json.dumps({
        "extracted_final_answer": "Titanic", "reasoning": "matches", "correct": correct,
    })


def _patch_run_deps(monkeypatch):
    monkeypatch.setattr(run_mod, "BrowseCompPlusLoader", lambda **kw: FakeLoader())
    monkeypatch.setattr(run_mod, "CorpusStore", lambda *a, **kw: FakeCorpusStore())
    monkeypatch.setattr(run_mod, "Qwen3ShardedRetriever", lambda **kw: FakeRetriever())

    def _fake_client_factory(**kw):
        # judge client is still built via AzureOpenAIClient directly in
        # run.py, keyed off model_env.
        if kw.get("model_env") == "FIGBROWSE_GENERATOR_MODEL":
            return FakeLLMClient(responses=[_judge_response(True)])
        return FakeLLMClient(responses=[_immediate_final_response()])

    monkeypatch.setattr(run_mod, "AzureOpenAIClient", _fake_client_factory)
    # the agent client goes through build_llm_client (provider-aware
    # factory), which internally imports its own AzureOpenAIClient — patch
    # it directly rather than relying on the run_mod-level patch above.
    monkeypatch.setattr(
        run_mod, "build_llm_client",
        lambda **kw: FakeLLMClient(responses=[_immediate_final_response()]),
    )


def test_load_manifest_respects_limit(tmp_path):
    path = tmp_path / "manifest.jsonl"
    _write_manifest(path, ["q1", "q2", "q3"])
    assert [r["query_id"] for r in _load_manifest(path, None)] == ["q1", "q2", "q3"]
    assert [r["query_id"] for r in _load_manifest(path, 2)] == ["q1", "q2"]


def test_load_completed_ids_empty_for_missing_file(tmp_path):
    assert _load_completed_ids(tmp_path / "nope.jsonl") == set()


def test_load_completed_ids_reads_question_id_position_pairs(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps({"question_id": "q1", "injection_position": "beginning"}) + "\n"
        + json.dumps({"question_id": "q2", "injection_position": "mid_trajectory"}) + "\n"
    )
    assert _load_completed_ids(path) == {("q1", "beginning"), ("q2", "mid_trajectory")}


def test_load_completed_ids_defaults_missing_position_to_beginning(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps({"question_id": "q1"}) + "\n")
    assert _load_completed_ids(path) == {("q1", "beginning")}


def test_full_run_writes_jsonl_and_resume_skips_completed(tmp_path, monkeypatch):
    _patch_run_deps(monkeypatch)

    figbrowse_root = tmp_path / "fig"
    manifest_path = figbrowse_root / "data" / "manifests" / "development_15.jsonl"
    _write_manifest(manifest_path, ["q001", "q002"])

    cfg = ExperimentConfig(
        figbrowse_path=str(figbrowse_root),
        split="validation",
        sample_limit=10,
        output_dir=str(tmp_path / "out"),
    )
    cfg.agent.max_post_injection_actions = 4

    output_path = run_experiment(cfg, limit=None, resume=True)
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(l) for l in lines]
    assert {r["question_id"] for r in records} == {"q001", "q002"}
    for r in records:
        assert r["answer_correct"] is True
        assert r["gold_answer"] == "Titanic"
        # gold answer must never leak into the question the agent saw
        assert "Titanic" not in r["question"]
        assert r["num_injected_documents"] == 2
        assert r["all_injected_documents_accessible"] is True

    # Re-running with resume must not duplicate or re-invoke the agent.
    _patch_run_deps(monkeypatch)  # fresh scripted clients
    output_path2 = run_experiment(cfg, limit=None, resume=True)
    lines2 = output_path2.read_text().strip().splitlines()
    assert len(lines2) == 2  # unchanged — nothing new appended


def test_multi_position_run_produces_one_record_per_question_per_position(tmp_path, monkeypatch):
    _patch_run_deps(monkeypatch)

    figbrowse_root = tmp_path / "fig"
    manifest_path = figbrowse_root / "data" / "manifests" / "development_15.jsonl"
    _write_manifest(manifest_path, ["q001"])

    cfg = ExperimentConfig(
        figbrowse_path=str(figbrowse_root),
        split="validation",
        sample_limit=10,
        output_dir=str(tmp_path / "out"),
    )
    cfg.agent.max_post_injection_actions = 4
    cfg.injection.positions = ["beginning", "mid_trajectory"]
    cfg.injection.mid_trajectory_delay = 2

    output_path = run_experiment(cfg, limit=None, resume=True)
    records = [json.loads(l) for l in output_path.read_text().strip().splitlines()]
    assert len(records) == 2
    by_position = {r["injection_position"]: r for r in records}
    assert set(by_position) == {"beginning", "mid_trajectory"}

    # The fake agent always answers immediately, so "beginning" delivers the
    # injection (turn 0) before that answer, while "mid_trajectory" (delay=2)
    # never reaches the injection point — evidence is never shown.
    assert by_position["beginning"]["injection_delivered"] is True
    assert by_position["mid_trajectory"]["injection_delivered"] is False
    assert by_position["mid_trajectory"]["termination_reason"] == "answered_before_injection"
    assert by_position["mid_trajectory"]["primary_label"] == "ANSWERED_BEFORE_INJECTION"


def test_no_injection_control_run(tmp_path, monkeypatch):
    _patch_run_deps(monkeypatch)

    figbrowse_root = tmp_path / "fig"
    manifest_path = figbrowse_root / "data" / "manifests" / "development_15.jsonl"
    _write_manifest(manifest_path, ["q001"])

    cfg = ExperimentConfig(
        figbrowse_path=str(figbrowse_root),
        split="validation",
        sample_limit=10,
        output_dir=str(tmp_path / "out"),
    )
    cfg.injection.positions = ["no_injection"]
    cfg.injection.no_injection_max_actions = 10

    output_path = run_experiment(cfg, limit=None, resume=True)
    records = [json.loads(l) for l in output_path.read_text().strip().splitlines()]
    assert len(records) == 1
    r = records[0]
    assert r["injection_position"] == "no_injection"
    assert r["num_injected_documents"] == 0
    assert r["injected_document_ids"] == []
    assert r["answer_correct"] is True
    assert r["primary_label"] == "CORRECT"
    assert r["max_post_injection_actions"] == 10
