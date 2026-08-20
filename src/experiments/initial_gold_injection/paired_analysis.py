"""Paired comparison between two injection positions run on the same queries.

    python -m experiments.initial_gold_injection.paired_analysis \\
        --input outputs/initial_gold_injection_confirmatory_multi_position/runs.jsonl \\
        --position-a beginning --position-b mid_trajectory

Unpaired per-condition confidence intervals (as in analyze.py) are not
sufficient to compare two conditions run on the *same* base queries — the
question-to-question variance is shared and should be differenced out.
This module implements the paired analysis instead:

- 2x2 correctness contingency table
- exact McNemar's test on the discordant pairs (matches FigBrowse's own
  CLAUDE.md §8.6 statistical protocol: "use an exact McNemar test for paired
  binary accuracy comparisons")
- paired bootstrap CIs (resampling question IDs, not episodes) for the
  difference in search calls, actions-to-answer, and redundant-search rate
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable

import typer

app = typer.Typer(add_completion=False)


def _load_runs(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def paired_records(
    runs: list[dict[str, Any]], position_a: str, position_b: str,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """question_id -> (record_a, record_b), restricted to questions present
    and error-free in both positions."""
    by_pos: dict[str, dict[str, dict[str, Any]]] = {}
    for r in runs:
        pos = r.get("injection_position") or "beginning"
        if pos not in (position_a, position_b):
            continue
        if r.get("primary_label") in ("TOOL_OR_DATA_ERROR",):
            continue
        by_pos.setdefault(pos, {})[r["question_id"]] = r

    a_map = by_pos.get(position_a, {})
    b_map = by_pos.get(position_b, {})
    shared = sorted(set(a_map) & set(b_map))
    return {qid: (a_map[qid], b_map[qid]) for qid in shared}


def contingency_table(paired: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, int]:
    both_correct = a_only = b_only = both_incorrect = 0
    for a, b in paired.values():
        ac, bc = bool(a.get("answer_correct")), bool(b.get("answer_correct"))
        if ac and bc:
            both_correct += 1
        elif ac and not bc:
            a_only += 1
        elif not ac and bc:
            b_only += 1
        else:
            both_incorrect += 1
    return {
        "both_correct": both_correct, "a_only_correct": a_only,
        "b_only_correct": b_only, "both_incorrect": both_incorrect,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar's test p-value on the discordant pairs
    (b = a-only-correct, c = b-only-correct), via the exact binomial test
    against p=0.5 — no scipy dependency, matches FigBrowse's own protocol."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def _bootstrap_paired_diff(
    paired: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    metric_fn: Callable[[dict[str, Any]], float],
    *, n_samples: int = 10000, seed: int = 42, alpha: float = 0.05,
) -> dict[str, float | None]:
    """Bootstrap CI for mean(metric_a - metric_b), resampling question IDs."""
    diffs = [metric_fn(a) - metric_fn(b) for a, b in paired.values()]
    if not diffs:
        return {"mean_diff": None, "ci_lo": None, "ci_hi": None}
    rng = random.Random(seed)
    n = len(diffs)
    point = statistics.mean(diffs)
    boots = []
    for _ in range(n_samples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.mean(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_samples)]
    hi = boots[int((1 - alpha / 2) * n_samples) - 1]
    return {"mean_diff": point, "ci_lo": lo, "ci_hi": hi}


def _search_calls(r: dict[str, Any]) -> float:
    return float(r.get("search_calls") or 0)


def _actions_to_answer(r: dict[str, Any]) -> float:
    return float(len(r.get("actions") or []))


def _redundant_search(r: dict[str, Any]) -> float:
    return 1.0 if (r.get("search_calls") or 0) > 0 else 0.0


def check_budget_parity(
    runs: list[dict[str, Any]], position_a: str, position_b: str,
) -> dict[str, set]:
    budgets: dict[str, set] = {}
    for r in runs:
        pos = r.get("injection_position") or "beginning"
        if pos in (position_a, position_b):
            budgets.setdefault(pos, set()).add(r.get("max_post_injection_actions"))
    return budgets


def run_paired_analysis(
    runs: list[dict[str, Any]], position_a: str, position_b: str,
) -> dict[str, Any]:
    budgets = check_budget_parity(runs, position_a, position_b)
    budget_parity_ok = len(set().union(*budgets.values())) == 1 if budgets else False

    paired = paired_records(runs, position_a, position_b)
    table = contingency_table(paired)
    p_mcnemar = mcnemar_exact(table["a_only_correct"], table["b_only_correct"])

    return {
        "position_a": position_a,
        "position_b": position_b,
        "n_paired_questions": len(paired),
        "budget_by_position": {k: sorted(v) for k, v in budgets.items()},
        "budget_parity_ok": budget_parity_ok,
        "contingency_table": table,
        "mcnemar_p_value": p_mcnemar,
        "search_calls_diff": _bootstrap_paired_diff(paired, _search_calls),
        "actions_to_answer_diff": _bootstrap_paired_diff(paired, _actions_to_answer),
        "redundant_search_rate_diff": _bootstrap_paired_diff(paired, _redundant_search),
    }


def format_report(result: dict[str, Any]) -> str:
    a, b = result["position_a"], result["position_b"]
    t = result["contingency_table"]
    lines = [
        f"# Paired comparison: {a} vs {b}",
        "",
        f"- Paired questions (present & error-free in both): {result['n_paired_questions']}",
        f"- Budget by position: {result['budget_by_position']}",
        f"- Budget parity OK (both positions used the same action allowance): "
        f"{result['budget_parity_ok']}",
        "",
        "## Paired correctness contingency table",
        "",
        f"|                      | {b} correct | {b} incorrect |",
        f"|---|---|---|",
        f"| **{a} correct**   | {t['both_correct']} (both correct) | {t['a_only_correct']} ({a} only) |",
        f"| **{a} incorrect** | {t['b_only_correct']} ({b} only) | {t['both_incorrect']} (both incorrect) |",
        "",
        f"## McNemar's exact test (accuracy)",
        f"- Discordant pairs: {a}-only={t['a_only_correct']}, {b}-only={t['b_only_correct']}",
        f"- p-value: {result['mcnemar_p_value']}",
        "",
        "## Paired bootstrap differences (10,000 resamples of question IDs, 95% CI)",
        "",
        f"- Search calls ({a} − {b}): mean={result['search_calls_diff']['mean_diff']}, "
        f"CI=[{result['search_calls_diff']['ci_lo']}, {result['search_calls_diff']['ci_hi']}]",
        f"- Actions to answer ({a} − {b}): mean={result['actions_to_answer_diff']['mean_diff']}, "
        f"CI=[{result['actions_to_answer_diff']['ci_lo']}, {result['actions_to_answer_diff']['ci_hi']}]",
        f"- Redundant-search rate ({a} − {b}): mean={result['redundant_search_rate_diff']['mean_diff']}, "
        f"CI=[{result['redundant_search_rate_diff']['ci_lo']}, {result['redundant_search_rate_diff']['ci_hi']}]",
    ]
    return "\n".join(lines) + "\n"


@app.command()
def main(
    input: str = typer.Option(..., "--input", help="Path to runs.jsonl"),
    position_a: str = typer.Option("beginning", "--position-a"),
    position_b: str = typer.Option("mid_trajectory", "--position-b"),
    output_dir: str = typer.Option(None, "--output-dir", help="Defaults to the input file's directory"),
) -> None:
    input_path = Path(input)
    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(input_path)
    result = run_paired_analysis(runs, position_a, position_b)
    report = format_report(result)

    fname_base = f"paired_{position_a}_vs_{position_b}"
    (out_dir / f"{fname_base}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / f"{fname_base}.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {fname_base}.json and {fname_base}.md to {out_dir}")


if __name__ == "__main__":
    app()
