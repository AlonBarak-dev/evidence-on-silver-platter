"""Analysis CLI for the initial_gold_injection experiment.

    python -m experiments.initial_gold_injection.analyze \\
        --input outputs/initial_gold_injection/runs.jsonl

Produces summary.json, per_question.csv, metrics.md, failures.jsonl, and
PNG+PDF plots, all written next to the input file unless --output-dir is
given.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402

app = typer.Typer(add_completion=False)


def _load_runs(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _bootstrap_ci(
    values: list[float], *, n_samples: int = 10000, seed: int = 42, alpha: float = 0.05
) -> tuple[float | None, float | None, float | None]:
    """Paired bootstrap CI by resampling questions (values already one-per-question)."""
    if not values:
        return None, None, None
    rng = random.Random(seed)
    n = len(values)
    point = statistics.mean(values)
    boots = []
    for _ in range(n_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.mean(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_samples)]
    hi = boots[int((1 - alpha / 2) * n_samples) - 1]
    return point, lo, hi


def compute_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = len(runs)
    completed = [r for r in runs if r.get("termination_reason") not in ("tool_or_data_error",) and r.get("error") is None]
    non_error = [r for r in runs if r.get("primary_label") != "TOOL_OR_DATA_ERROR"]

    correctness = [1.0 if r.get("answer_correct") else 0.0 for r in non_error]
    acc_point, acc_lo, acc_hi = _bootstrap_ci(correctness)

    immediate = [r for r in non_error if r.get("immediate_answer")]
    immediate_correct = [r for r in immediate if r.get("answer_correct")]
    immediate_incorrect = [r for r in immediate if not r.get("answer_correct")]

    redundant = [r for r in non_error if (r.get("search_calls") or 0) > 0]

    correct_runs = [r for r in non_error if r.get("answer_correct")]

    def _actions_to_answer(r: dict[str, Any]) -> int:
        return len(r.get("actions") or [])

    summary = {
        "n_attempted": attempted,
        "n_completed": len(completed),
        "n_scored": len(non_error),
        "accuracy": acc_point,
        "accuracy_ci95": [acc_lo, acc_hi],
        "immediate_answer_rate": len(immediate) / len(non_error) if non_error else None,
        "immediate_correct_rate": len(immediate_correct) / len(non_error) if non_error else None,
        "immediate_incorrect_rate": len(immediate_incorrect) / len(non_error) if non_error else None,
        "redundant_search_rate": len(redundant) / len(non_error) if non_error else None,
        "actions_to_answer_mean": _mean([_actions_to_answer(r) for r in non_error]),
        "actions_to_answer_median": _median([_actions_to_answer(r) for r in non_error]),
        "search_calls_mean": _mean([r.get("search_calls") or 0 for r in non_error]),
        "search_calls_median": _median([r.get("search_calls") or 0 for r in non_error]),
        "get_document_calls_mean": _mean([r.get("get_document_calls") or 0 for r in non_error]),
        "get_document_calls_median": _median([r.get("get_document_calls") or 0 for r in non_error]),
        "injected_document_coverage_mean": _mean(
            [r.get("injected_document_coverage") or 0.0 for r in non_error]
        ),
        "correct_efficiency": {
            "n_correct": len(correct_runs),
            "actions_to_answer_mean": _mean([_actions_to_answer(r) for r in correct_runs]),
            "actions_to_answer_median": _median([_actions_to_answer(r) for r in correct_runs]),
            "search_calls_mean": _mean([r.get("search_calls") or 0 for r in correct_runs]),
            "search_calls_median": _median([r.get("search_calls") or 0 for r in correct_runs]),
            "get_document_calls_mean": _mean([r.get("get_document_calls") or 0 for r in correct_runs]),
            "get_document_calls_median": _median([r.get("get_document_calls") or 0 for r in correct_runs]),
            "coverage_mean": _mean([r.get("injected_document_coverage") or 0.0 for r in correct_runs]),
        },
        "failure_counts": {},
    }
    for r in non_error:
        label = r.get("primary_label") or "UNKNOWN"
        summary["failure_counts"][label] = summary["failure_counts"].get(label, 0) + 1
    err_count = attempted - len(non_error)
    if err_count:
        summary["failure_counts"]["TOOL_OR_DATA_ERROR"] = err_count
    return summary


def group_by_position(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        pos = r.get("injection_position") or "beginning"
        groups.setdefault(pos, []).append(r)
    return groups


def write_per_question_csv(runs: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    rows = []
    for r in runs:
        rows.append({
            "question_id": r.get("question_id"),
            "injection_position": r.get("injection_position") or "beginning",
            "injection_delivered": r.get("injection_delivered", True),
            "answer_correct": r.get("answer_correct"),
            "immediate_answer": r.get("immediate_answer"),
            "actions_to_answer": len(r.get("actions") or []),
            "search_calls": r.get("search_calls"),
            "get_document_calls": r.get("get_document_calls"),
            "num_injected_documents": r.get("num_injected_documents"),
            "injected_document_coverage": r.get("injected_document_coverage"),
            "all_injected_documents_opened_turn": r.get("all_injected_documents_opened_turn"),
            "first_answer_support_turn": r.get("first_answer_support_turn"),
            "final_answer_turn": r.get("final_answer_turn"),
            "post_access_delay": r.get("post_access_delay"),
            "termination_reason": r.get("termination_reason"),
            "primary_label": r.get("primary_label"),
            "secondary_labels": ";".join(r.get("secondary_labels") or []),
            "error": r.get("error"),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def write_failures_jsonl(runs: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in runs:
            if not r.get("answer_correct"):
                f.write(json.dumps(r) + "\n")


def _savefig(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(runs: list[dict[str, Any]], out_dir: Path) -> None:
    non_error = [r for r in runs if r.get("primary_label") != "TOOL_OR_DATA_ERROR"]
    if not non_error:
        return

    actions = [len(r.get("actions") or []) for r in non_error]
    fig, ax = plt.subplots()
    ax.hist(actions, bins=range(0, max(actions, default=0) + 2))
    ax.set_xlabel("Actions to answer")
    ax.set_ylabel("Count")
    ax.set_title("Actions to answer (post-injection)")
    _savefig(fig, out_dir, "hist_actions_to_answer")

    searches = [r.get("search_calls") or 0 for r in non_error]
    fig, ax = plt.subplots()
    ax.hist(searches, bins=range(0, max(searches, default=0) + 2))
    ax.set_xlabel("Search calls after injection")
    ax.set_ylabel("Count")
    ax.set_title("Search calls after injection")
    _savefig(fig, out_dir, "hist_search_calls_after_injection")

    by_n: dict[int, list[bool]] = {}
    for r in non_error:
        n = r.get("num_injected_documents") or 0
        by_n.setdefault(n, []).append(bool(r.get("answer_correct")))
    ns = sorted(by_n)
    accs = [sum(by_n[n]) / len(by_n[n]) for n in ns]
    fig, ax = plt.subplots()
    ax.bar([str(n) for n in ns], accs)
    ax.set_xlabel("Number of injected documents")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy grouped by number of injected documents")
    _savefig(fig, out_dir, "accuracy_by_num_injected")

    cov_correct = [r.get("injected_document_coverage") or 0.0 for r in non_error if r.get("answer_correct")]
    cov_incorrect = [r.get("injected_document_coverage") or 0.0 for r in non_error if not r.get("answer_correct")]
    fig, ax = plt.subplots()
    ax.scatter([1] * len(cov_correct), cov_correct, label="correct", alpha=0.6)
    ax.scatter([0] * len(cov_incorrect), cov_incorrect, label="incorrect", alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["incorrect", "correct"])
    ax.set_ylabel("Injected-document coverage")
    ax.set_title("Injected-document coverage vs. correctness")
    ax.legend()
    _savefig(fig, out_dir, "coverage_vs_correctness")

    gdc = [r.get("get_document_calls") or 0 for r in non_error]
    fig, ax = plt.subplots()
    ax.scatter(searches, gdc, alpha=0.6)
    ax.set_xlabel("Search calls after injection")
    ax.set_ylabel("get_document calls")
    ax.set_title("Search calls vs. get_document calls")
    _savefig(fig, out_dir, "search_vs_get_document")


def make_position_comparison_plot(runs: list[dict[str, Any]], out_dir: Path) -> None:
    """Accuracy by injection position — only meaningful with >1 position."""
    non_error = [r for r in runs if r.get("primary_label") != "TOOL_OR_DATA_ERROR"]
    groups = group_by_position(non_error)
    if len(groups) < 2:
        return
    positions = sorted(groups)
    accs = [
        sum(1 for r in groups[p] if r.get("answer_correct")) / len(groups[p])
        for p in positions
    ]
    fig, ax = plt.subplots()
    ax.bar(positions, accs)
    ax.set_xlabel("Injection position")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by injection position")
    ax.set_ylim(0, 1)
    _savefig(fig, out_dir, "accuracy_by_injection_position")


def _summary_section(summary: dict[str, Any], heading: str) -> list[str]:
    fc = summary["failure_counts"]
    lines = [
        heading,
        "",
        f"- Attempted samples: {summary['n_attempted']}",
        f"- Completed samples: {summary['n_completed']}",
        f"- Scored samples (excluding data errors): {summary['n_scored']}",
        "",
        "### Accuracy",
        f"- Accuracy: {summary['accuracy']}",
        f"- 95% bootstrap CI (question resampling): "
        f"[{summary['accuracy_ci95'][0]}, {summary['accuracy_ci95'][1]}]",
        f"- Immediate-answer rate: {summary['immediate_answer_rate']}",
        f"- Immediate correct-answer rate: {summary['immediate_correct_rate']}",
        f"- Immediate incorrect-answer rate: {summary['immediate_incorrect_rate']}",
        f"- Redundant-search rate: {summary['redundant_search_rate']}",
        "",
        "### Efficiency (all scored samples)",
        f"- Actions to answer — mean/median: "
        f"{summary['actions_to_answer_mean']} / {summary['actions_to_answer_median']}",
        f"- Search calls — mean/median: "
        f"{summary['search_calls_mean']} / {summary['search_calls_median']}",
        f"- get_document calls — mean/median: "
        f"{summary['get_document_calls_mean']} / {summary['get_document_calls_median']}",
        f"- Mean injected-document coverage: {summary['injected_document_coverage_mean']}",
        "",
        "### Correct-efficiency (correct samples only)",
        f"- n correct: {summary['correct_efficiency']['n_correct']}",
        f"- Actions to answer — mean/median: "
        f"{summary['correct_efficiency']['actions_to_answer_mean']} / "
        f"{summary['correct_efficiency']['actions_to_answer_median']}",
        f"- Search calls — mean/median: "
        f"{summary['correct_efficiency']['search_calls_mean']} / "
        f"{summary['correct_efficiency']['search_calls_median']}",
        f"- get_document calls — mean/median: "
        f"{summary['correct_efficiency']['get_document_calls_mean']} / "
        f"{summary['correct_efficiency']['get_document_calls_median']}",
        f"- Mean coverage: {summary['correct_efficiency']['coverage_mean']}",
        "",
        "### Failure taxonomy counts",
    ]
    for label, count in sorted(fc.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {label}: {count}")
    lines.append("")
    return lines


def write_metrics_md(
    overall: dict[str, Any], by_position: dict[str, dict[str, Any]], path: Path
) -> None:
    lines = ["# initial_gold_injection — metrics report", ""]
    if len(by_position) > 1:
        lines += ["## Position comparison", ""]
        lines += ["| position | n | accuracy | immediate rate | redundant search rate | actions/answer (mean) |",
                   "|---|---|---|---|---|---|"]
        for pos in sorted(by_position):
            s = by_position[pos]
            lines.append(
                f"| {pos} | {s['n_scored']} | {s['accuracy']} | {s['immediate_answer_rate']} | "
                f"{s['redundant_search_rate']} | {s['actions_to_answer_mean']} |"
            )
        lines.append("")
    lines += _summary_section(overall, "## Overall (all positions combined)")
    for pos in sorted(by_position):
        lines += _summary_section(by_position[pos], f"## Position: {pos}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command()
def main(
    input: str = typer.Option(..., "--input", help="Path to runs.jsonl"),
    output_dir: str = typer.Option(None, "--output-dir", help="Defaults to the input file's directory"),
) -> None:
    input_path = Path(input)
    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(input_path)
    overall = compute_summary(runs)
    by_position = {pos: compute_summary(rs) for pos, rs in group_by_position(runs).items()}

    summary_out = {"overall": overall, "by_position": by_position}
    (out_dir / "summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    write_per_question_csv(runs, out_dir / "per_question.csv")
    write_metrics_md(overall, by_position, out_dir / "metrics.md")
    write_failures_jsonl(runs, out_dir / "failures.jsonl")
    make_plots(runs, out_dir)
    make_position_comparison_plot(runs, out_dir)

    print(f"Wrote summary.json, per_question.csv, metrics.md, failures.jsonl, and plots to {out_dir}")


if __name__ == "__main__":
    app()
