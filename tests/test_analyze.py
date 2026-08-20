from __future__ import annotations

import json

from experiments.initial_gold_injection.analyze import compute_summary, group_by_position, main


def _rec(**overrides):
    d = dict(
        question_id="q1", answer_correct=True, immediate_answer=True,
        actions=[{"turn": 1, "action_type": "answer"}],
        search_calls=0, get_document_calls=0, num_injected_documents=2,
        injected_document_coverage=1.0, primary_label="CORRECT_IMMEDIATE",
        termination_reason="final",
    )
    d.update(overrides)
    return d


def test_compute_summary_basic_rates():
    runs = [
        _rec(question_id="q1", answer_correct=True, immediate_answer=True),
        _rec(question_id="q2", answer_correct=False, immediate_answer=False,
             search_calls=1, primary_label="WRONG_WITHOUT_OPENING_EVIDENCE"),
    ]
    summary = compute_summary(runs)
    assert summary["n_scored"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["immediate_answer_rate"] == 0.5
    assert summary["redundant_search_rate"] == 0.5
    assert summary["failure_counts"]["CORRECT_IMMEDIATE"] == 1
    assert summary["failure_counts"]["WRONG_WITHOUT_OPENING_EVIDENCE"] == 1


def test_compute_summary_excludes_data_errors_from_accuracy():
    runs = [
        _rec(question_id="q1", answer_correct=True),
        _rec(question_id="q2", primary_label="TOOL_OR_DATA_ERROR", answer_correct=False,
             termination_reason="tool_or_data_error"),
    ]
    summary = compute_summary(runs)
    assert summary["n_attempted"] == 2
    assert summary["n_scored"] == 1
    assert summary["accuracy"] == 1.0


def test_analyze_cli_writes_all_outputs(tmp_path):
    runs_path = tmp_path / "runs.jsonl"
    with runs_path.open("w") as f:
        for r in [_rec(question_id="q1"), _rec(question_id="q2", answer_correct=False,
                                                 immediate_answer=False, search_calls=1,
                                                 primary_label="WRONG_WITHOUT_OPENING_EVIDENCE")]:
            f.write(json.dumps(r) + "\n")

    main(input=str(runs_path), output_dir=str(tmp_path))

    for name in [
        "summary.json", "per_question.csv", "metrics.md", "failures.jsonl",
        "hist_actions_to_answer.png", "hist_actions_to_answer.pdf",
        "hist_search_calls_after_injection.png", "accuracy_by_num_injected.png",
        "coverage_vs_correctness.png", "search_vs_get_document.png",
    ]:
        assert (tmp_path / name).exists(), f"missing {name}"

    failures = (tmp_path / "failures.jsonl").read_text().strip().splitlines()
    assert len(failures) == 1
    assert json.loads(failures[0])["question_id"] == "q2"


def test_group_by_position_defaults_missing_to_beginning():
    runs = [_rec(question_id="q1"), _rec(question_id="q2", **{"injection_position": "mid_trajectory"})]
    groups = group_by_position(runs)
    assert set(groups) == {"beginning", "mid_trajectory"}
    assert len(groups["beginning"]) == 1
    assert len(groups["mid_trajectory"]) == 1


def test_analyze_cli_writes_position_comparison_plot_when_multiple_positions(tmp_path):
    runs_path = tmp_path / "runs.jsonl"
    with runs_path.open("w") as f:
        f.write(json.dumps(_rec(question_id="q1", injection_position="beginning")) + "\n")
        f.write(json.dumps(_rec(question_id="q2", injection_position="mid_trajectory",
                                 answer_correct=False)) + "\n")

    main(input=str(runs_path), output_dir=str(tmp_path))

    assert (tmp_path / "accuracy_by_injection_position.png").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary["by_position"]) == {"beginning", "mid_trajectory"}
    assert "overall" in summary
