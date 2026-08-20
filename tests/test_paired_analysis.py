from __future__ import annotations

from experiments.initial_gold_injection.paired_analysis import (
    check_budget_parity,
    contingency_table,
    mcnemar_exact,
    paired_records,
    run_paired_analysis,
)


def _rec(qid, position, correct, search_calls=0, n_actions=1, max_budget=8):
    return {
        "question_id": qid, "injection_position": position, "answer_correct": correct,
        "search_calls": search_calls, "actions": [{}] * n_actions,
        "primary_label": "CORRECT_IMMEDIATE" if correct else "WRONG_WITHOUT_OPENING_EVIDENCE",
        "max_post_injection_actions": max_budget,
    }


def test_paired_records_only_keeps_shared_error_free_questions():
    runs = [
        _rec("q1", "beginning", True), _rec("q1", "mid_trajectory", False),
        _rec("q2", "beginning", True),  # missing from mid_trajectory
        _rec("q3", "beginning", True, max_budget=8), _rec("q3", "mid_trajectory", True),
    ]
    paired = paired_records(runs, "beginning", "mid_trajectory")
    assert set(paired) == {"q1", "q3"}


def test_contingency_table_counts():
    paired = {
        "q1": (_rec("q1", "a", True), _rec("q1", "b", True)),      # both correct
        "q2": (_rec("q2", "a", True), _rec("q2", "b", False)),     # a only
        "q3": (_rec("q3", "a", False), _rec("q3", "b", True)),     # b only
        "q4": (_rec("q4", "a", False), _rec("q4", "b", False)),    # both incorrect
    }
    t = contingency_table(paired)
    assert t == {"both_correct": 1, "a_only_correct": 1, "b_only_correct": 1, "both_incorrect": 1}


def test_mcnemar_exact_no_discordant_pairs_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_exact_symmetric_discordance_is_high_p():
    # 5 vs 5 discordant pairs -> no evidence of a systematic difference
    p = mcnemar_exact(5, 5)
    assert p > 0.5


def test_mcnemar_exact_lopsided_discordance_is_significant():
    # 0 vs 10 discordant pairs -> strong evidence of a systematic difference
    p = mcnemar_exact(0, 10)
    assert p < 0.01


def test_budget_parity_detected_when_equal():
    runs = [_rec("q1", "a", True, max_budget=8), _rec("q1", "b", True, max_budget=8)]
    budgets = check_budget_parity(runs, "a", "b")
    assert budgets == {"a": {8}, "b": {8}}


def test_run_paired_analysis_flags_budget_mismatch():
    runs = [
        _rec("q1", "a", True, max_budget=8), _rec("q1", "b", True, max_budget=12),
    ]
    result = run_paired_analysis(runs, "a", "b")
    assert result["budget_parity_ok"] is False


def test_run_paired_analysis_end_to_end():
    runs = [
        _rec("q1", "a", True, search_calls=2, n_actions=3),
        _rec("q1", "b", True, search_calls=0, n_actions=1),
        _rec("q2", "a", False, search_calls=1, n_actions=2),
        _rec("q2", "b", True, search_calls=0, n_actions=1),
    ]
    result = run_paired_analysis(runs, "a", "b")
    assert result["n_paired_questions"] == 2
    assert result["budget_parity_ok"] is True
    assert result["contingency_table"]["b_only_correct"] == 1
    assert result["search_calls_diff"]["mean_diff"] == 1.5  # (2-0 + 1-0) / 2
