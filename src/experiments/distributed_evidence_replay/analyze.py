"""Analysis CLI for distributed_evidence_replay.

    python -m experiments.distributed_evidence_replay.analyze \\
        --input outputs/distributed_evidence_replay/runs.jsonl

Reuses experiments.initial_gold_injection.paired_analysis's exact-McNemar
and paired-bootstrap primitives (pure functions of counts/diffs, no
dependency on that experiment's schema) rather than reimplementing them.
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

from experiments.initial_gold_injection.paired_analysis import mcnemar_exact

app = typer.Typer(add_completion=False)

# Default locations of the earlier experiment's completed outputs, used only
# for the B/C/D comparison section. Never written to.
_B_C_PATH = "outputs/initial_gold_injection_confirmatory_multi_position/runs.jsonl"
_D_PATH = "outputs/initial_gold_injection_confirmatory_full_evidence_reader/runs.jsonl"


def _load_runs(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _valid(r: dict[str, Any]) -> bool:
    return r.get("termination_reason") == "final" and r.get("error") is None


def _bootstrap_ci(values: list[float], *, n_samples=10000, seed=42, alpha=0.05):
    if not values:
        return None, None, None
    rng = random.Random(seed)
    n = len(values)
    point = statistics.mean(values)
    boots = []
    for _ in range(n_samples):
        boots.append(statistics.mean(values[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    return point, boots[int((alpha / 2) * n_samples)], boots[int((1 - alpha / 2) * n_samples) - 1]


def _paired_accuracy_bootstrap(pairs: list[tuple[bool, bool]], *, seed=42):
    diffs = [float(a) - float(b) for a, b in pairs]
    return _bootstrap_ci(diffs, seed=seed)


def _contingency(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    both = a_only = b_only = neither = 0
    for a, b in pairs:
        if a and b:
            both += 1
        elif a and not b:
            a_only += 1
        elif b and not a:
            b_only += 1
        else:
            neither += 1
    return {"both_correct": both, "a_only_correct": a_only, "b_only_correct": b_only, "both_incorrect": neither}


# ---------------------------------------------------------------------------
# Load and organize
# ---------------------------------------------------------------------------


def organize(runs: list[dict[str, Any]]) -> tuple[dict[str, dict], dict[int, dict[str, dict]]]:
    """(r0_by_qid, {seed: {qid: record}})"""
    r0: dict[str, dict] = {}
    rall: dict[int, dict[str, dict]] = {}
    for r in runs:
        if r["condition"] == "original_replay" and _valid(r):
            r0[r["question_id"]] = r
        elif r["condition"] == "distributed_evidence_replay" and _valid(r):
            rall.setdefault(r["placement_seed"], {})[r["question_id"]] = r
    return r0, rall


# ---------------------------------------------------------------------------
# R0 vs R-all, per seed
# ---------------------------------------------------------------------------


def r0_vs_rall_per_seed(r0: dict[str, dict], rall: dict[int, dict[str, dict]]) -> dict[int, dict[str, Any]]:
    out = {}
    for seed, by_qid in rall.items():
        shared = sorted(set(r0) & set(by_qid))
        pairs = [(bool(r0[q]["answer_correct"]), bool(by_qid[q]["answer_correct"])) for q in shared]
        table = _contingency(pairs)
        p = mcnemar_exact(table["a_only_correct"], table["b_only_correct"])
        acc_diff = _paired_accuracy_bootstrap(pairs)
        out[seed] = {
            "n": len(shared),
            "r0_accuracy": sum(p[0] for p in pairs) / len(pairs) if pairs else None,
            "rall_accuracy": sum(p[1] for p in pairs) / len(pairs) if pairs else None,
            "contingency_table": table,
            "mcnemar_p_value": p,
            "accuracy_diff_bootstrap": {
                "mean_diff": acc_diff[0], "ci_lo": acc_diff[1], "ci_hi": acc_diff[2],
            },
            "rescues": table["b_only_correct"],  # r0 wrong, rall correct
            "harms": table["a_only_correct"],    # r0 correct, rall wrong
        }
    return out


# ---------------------------------------------------------------------------
# Cross-seed aggregation (question-level)
# ---------------------------------------------------------------------------


def cross_seed_aggregation(rall: dict[int, dict[str, dict]]) -> dict[str, Any]:
    seeds = sorted(rall)
    shared_qids = sorted(set.intersection(*(set(rall[s]) for s in seeds))) if seeds else []

    per_question = []
    for qid in shared_qids:
        correctness = [bool(rall[s][qid]["answer_correct"]) for s in seeds]
        per_question.append({
            "question_id": qid,
            "seed_correctness": dict(zip(seeds, correctness)),
            "mean_correctness": sum(correctness) / len(correctness),
            "num_seeds_correct": sum(correctness),
            "all_seeds_correct": all(correctness),
            "any_seed_correct": any(correctness),
            "placement_sensitive": len(set(correctness)) > 1,
        })

    per_seed_acc = {
        s: sum(1 for q in rall[s].values() if q["answer_correct"]) / len(rall[s]) if rall[s] else None
        for s in seeds
    }
    seed_acc_values = [v for v in per_seed_acc.values() if v is not None]

    # bootstrap resampling QUESTIONS, keeping each question's full seed vector together
    mean_correctness_values = [pq["mean_correctness"] for pq in per_question]
    mean_acc_point, mean_acc_lo, mean_acc_hi = _bootstrap_ci(mean_correctness_values)

    placement_sensitive_rate = (
        sum(1 for pq in per_question if pq["placement_sensitive"]) / len(per_question)
        if per_question else None
    )

    return {
        "seeds": seeds,
        "n_shared_questions": len(shared_qids),
        "per_seed_accuracy": per_seed_acc,
        "mean_accuracy_across_seeds": statistics.mean(seed_acc_values) if seed_acc_values else None,
        "stdev_accuracy_across_seeds": statistics.pstdev(seed_acc_values) if len(seed_acc_values) > 1 else 0.0,
        "mean_question_level_correctness_bootstrap": {
            "point": mean_acc_point, "ci_lo": mean_acc_lo, "ci_hi": mean_acc_hi,
        },
        "placement_sensitive_rate": placement_sensitive_rate,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Position vs. accuracy relationships
# ---------------------------------------------------------------------------


def _bucket_accuracy(rall: dict[int, dict[str, dict]], field: str, bins: list[tuple[float, float]] | None = None):
    """Mean accuracy grouped by a numeric field, pooled across all seeds."""
    records = [r for by_qid in rall.values() for r in by_qid.values() if r.get(field) is not None]
    if bins is None:
        # group by exact integer value
        by_key: dict[Any, list[bool]] = {}
        for r in records:
            by_key.setdefault(r[field], []).append(bool(r["answer_correct"]))
        return {k: sum(v) / len(v) for k, v in sorted(by_key.items())}
    by_bin: dict[str, list[bool]] = {}
    for r in records:
        v = r[field]
        for lo, hi in bins:
            if lo <= v < hi:
                by_bin.setdefault(f"[{lo},{hi})", []).append(bool(r["answer_correct"]))
                break
    return {k: sum(v) / len(v) for k, v in by_bin.items()}


# ---------------------------------------------------------------------------
# Comparisons with B/C/D (majority-vote R-all per question — documented
# assumption: with 3 seeds, >=2-of-3 correct is R-all's representative
# binary verdict for a fair single-outcome paired comparison)
# ---------------------------------------------------------------------------


def rall_majority_vote(rall: dict[int, dict[str, dict]]) -> dict[str, bool]:
    seeds = sorted(rall)
    shared = set.intersection(*(set(rall[s]) for s in seeds)) if seeds else set()
    out = {}
    for qid in shared:
        correctness = [bool(rall[s][qid]["answer_correct"]) for s in seeds]
        out[qid] = sum(correctness) > len(correctness) / 2
    return out


def compare_with_condition(
    rall_majority: dict[str, bool], other_runs: list[dict[str, Any]], other_position: str,
) -> dict[str, Any] | None:
    other = {
        r["question_id"]: bool(r["answer_correct"])
        for r in other_runs
        if (r.get("injection_position") or "beginning") == other_position
        and r.get("primary_label") not in ("TOOL_OR_DATA_ERROR",)
    }
    shared = sorted(set(rall_majority) & set(other))
    if not shared:
        return None
    pairs = [(rall_majority[q], other[q]) for q in shared]
    table = _contingency(pairs)  # a=R-all, b=other
    p = mcnemar_exact(table["a_only_correct"], table["b_only_correct"])
    diff = _paired_accuracy_bootstrap(pairs)
    return {
        "n": len(shared),
        "rall_accuracy": sum(a for a, _ in pairs) / len(pairs),
        f"{other_position}_accuracy": sum(b for _, b in pairs) / len(pairs),
        "contingency_table": table,
        "mcnemar_p_value": p,
        "accuracy_diff_bootstrap": {"mean_diff": diff[0], "ci_lo": diff[1], "ci_hi": diff[2]},
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_per_question_csv(runs: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd
    rows = []
    for r in runs:
        rows.append({
            "question_id": r["question_id"], "condition": r["condition"],
            "placement_seed": r.get("placement_seed"),
            "answer_correct": r.get("answer_correct"),
            "total_evidence_documents": r.get("total_evidence_documents"),
            "naturally_present": len(r.get("naturally_present_evidence_ids") or []),
            "missing": len(r.get("missing_evidence_ids") or []),
            "injected": len(r.get("injected_evidence_ids") or []),
            "all_evidence_present": r.get("all_evidence_present"),
            "mean_normalized_evidence_position": r.get("mean_normalized_evidence_position"),
            "first_evidence_turn": r.get("first_evidence_turn"),
            "last_evidence_turn": r.get("last_evidence_turn"),
            "original_search_turns": r.get("original_search_turns"),
            "termination_reason": r.get("termination_reason"),
            "error": r.get("error"),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def write_placements_jsonl(runs: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in runs:
            if r["condition"] == "distributed_evidence_replay":
                f.write(json.dumps({
                    "question_id": r["question_id"], "placement_seed": r["placement_seed"],
                    "replacements": r.get("replacements") or [],
                    "injected_evidence_ids": r.get("injected_evidence_ids"),
                    "mean_normalized_evidence_position": r.get("mean_normalized_evidence_position"),
                    "first_evidence_turn": r.get("first_evidence_turn"),
                    "last_evidence_turn": r.get("last_evidence_turn"),
                }) + "\n")


def _savefig(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(rall: dict[int, dict[str, dict]], out_dir: Path) -> None:
    seeds = sorted(rall)
    if not seeds:
        return

    accs = [sum(1 for r in rall[s].values() if r["answer_correct"]) / len(rall[s]) if rall[s] else 0 for s in seeds]
    fig, ax = plt.subplots()
    ax.bar([str(s) for s in seeds], accs)
    ax.set_xlabel("Placement seed")
    ax.set_ylabel("Accuracy")
    ax.set_title("R-all accuracy by placement seed")
    ax.set_ylim(0, 1)
    _savefig(fig, out_dir, "accuracy_by_placement_seed")

    pos_bins = _bucket_accuracy(rall, "mean_normalized_evidence_position",
                                 bins=[(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)])
    if pos_bins:
        fig, ax = plt.subplots()
        ax.bar(list(pos_bins.keys()), list(pos_bins.values()))
        ax.set_xlabel("Mean normalized evidence position")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs. mean normalized evidence position")
        _savefig(fig, out_dir, "accuracy_vs_evidence_position")

    last_turn_acc = _bucket_accuracy(rall, "last_evidence_turn")
    if last_turn_acc:
        fig, ax = plt.subplots()
        ax.bar([str(k) for k in last_turn_acc], list(last_turn_acc.values()))
        ax.set_xlabel("Last evidence turn")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs. last evidence turn")
        _savefig(fig, out_dir, "accuracy_vs_last_evidence_turn")

    traj_len_acc = _bucket_accuracy(rall, "original_search_turns")
    if traj_len_acc:
        fig, ax = plt.subplots()
        ax.bar([str(k) for k in traj_len_acc], list(traj_len_acc.values()))
        ax.set_xlabel("Original trajectory length (search turns)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs. trajectory length")
        _savefig(fig, out_dir, "accuracy_vs_trajectory_length")

    n_ev_acc = _bucket_accuracy(rall, "total_evidence_documents")
    if n_ev_acc:
        fig, ax = plt.subplots()
        ax.bar([str(k) for k in n_ev_acc], list(n_ev_acc.values()))
        ax.set_xlabel("Number of evidence documents")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs. number of evidence documents")
        _savefig(fig, out_dir, "accuracy_vs_num_evidence_documents")


def write_summary_md(summary: dict[str, Any], path: Path) -> None:
    lines = ["# distributed_evidence_replay — summary", ""]
    lines.append(
        "We construct off-policy counterfactual replays by distributing every "
        "annotated evidence document across the search-result observations of "
        "a frozen no-injection trajectory. We then regenerate only the final "
        "answer, isolating evidence utilization from search-policy adaptation."
    )
    lines.append("")
    lines.append(
        "**Limitation:** later search actions were generated under the "
        "original observations and may not be behaviorally consistent with "
        "the injected evidence. This inconsistency is intentional: freezing "
        "the action sequence prevents evidence-induced policy changes from "
        "confounding the utilization analysis."
    )
    lines.append("")

    lines.append("## R0 vs. R-all, per placement seed")
    lines.append("")
    for seed, s in sorted(summary["r0_vs_rall_per_seed"].items()):
        t = s["contingency_table"]
        lines += [
            f"### Seed {seed} (n={s['n']})",
            f"- R0 accuracy: {s['r0_accuracy']}",
            f"- R-all accuracy: {s['rall_accuracy']}",
            f"- Accuracy diff (R-all - R0): {s['accuracy_diff_bootstrap']['mean_diff']} "
            f"CI=[{s['accuracy_diff_bootstrap']['ci_lo']}, {s['accuracy_diff_bootstrap']['ci_hi']}]",
            f"- Contingency: both_correct={t['both_correct']}, R0_only={t['a_only_correct']}, "
            f"R-all_only={t['b_only_correct']}, both_incorrect={t['both_incorrect']}",
            f"- McNemar p-value: {s['mcnemar_p_value']}",
            f"- Rescues (R0 wrong -> R-all correct): {s['rescues']}",
            f"- Harms (R0 correct -> R-all wrong): {s['harms']}",
            "",
        ]

    agg = summary["cross_seed_aggregation"]
    lines += [
        "## Aggregation across seeds",
        "",
        f"- Shared questions across all seeds: {agg['n_shared_questions']}",
        f"- Per-seed accuracy: {agg['per_seed_accuracy']}",
        f"- Mean accuracy across seeds: {agg['mean_accuracy_across_seeds']}",
        f"- Stdev accuracy across seeds: {agg['stdev_accuracy_across_seeds']}",
        f"- Mean question-level correctness (bootstrap, questions resampled with "
        f"full seed vector kept together): {agg['mean_question_level_correctness_bootstrap']}",
        f"- Placement-sensitive rate (result changes across placements): {agg['placement_sensitive_rate']}",
        "",
    ]

    lines.append("## Comparisons with existing conditions (R-all majority vote vs. B/C/D)")
    lines.append("")
    for name, comp in summary.get("comparisons", {}).items():
        if comp is None:
            lines.append(f"- {name}: no shared questions available")
            continue
        t = comp["contingency_table"]
        lines += [
            f"### R-all vs. {name} (n={comp['n']})",
            f"- R-all accuracy: {comp['rall_accuracy']}",
            f"- {name} accuracy: {comp.get(f'{name}_accuracy')}",
            f"- Accuracy diff bootstrap: {comp['accuracy_diff_bootstrap']}",
            f"- Contingency: {t}",
            f"- McNemar p-value: {comp['mcnemar_p_value']}",
            "",
        ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_manual_examples(
    r0: dict[str, dict], rall: dict[int, dict[str, dict]], agg: dict[str, Any],
    b_runs: list[dict], c_runs: list[dict], d_runs: list[dict], out_path: Path,
) -> None:
    b_by_q = {r["question_id"]: r for r in b_runs if (r.get("injection_position") or "beginning") == "beginning"}
    d_by_q = {r["question_id"]: r for r in d_runs}
    per_q = {pq["question_id"]: pq for pq in agg["per_question"]}

    examples: dict[str, list[dict]] = {
        "r0_wrong_all_rall_correct": [],
        "r0_correct_some_rall_wrong": [],
        "rall_wrong_d_correct": [],
        "rall_correct_b_wrong": [],
        "placement_sensitive": [],
    }
    for qid, pq in per_q.items():
        r0_correct = bool(r0.get(qid, {}).get("answer_correct")) if qid in r0 else None
        if r0_correct is False and pq["all_seeds_correct"]:
            examples["r0_wrong_all_rall_correct"].append(qid)
        if r0_correct is True and pq["num_seeds_correct"] < len(pq["seed_correctness"]):
            examples["r0_correct_some_rall_wrong"].append(qid)
        if pq["mean_correctness"] < 1.0 and d_by_q.get(qid, {}).get("answer_correct"):
            examples["rall_wrong_d_correct"].append(qid)
        if pq["mean_correctness"] > 0.0 and b_by_q.get(qid) and not b_by_q[qid]["answer_correct"]:
            examples["rall_correct_b_wrong"].append(qid)
        if pq["placement_sensitive"]:
            examples["placement_sensitive"].append(qid)

    with out_path.open("w", encoding="utf-8") as f:
        for category, qids in examples.items():
            for qid in qids[:5]:
                record = {
                    "category": category, "question_id": qid,
                    "r0": r0.get(qid),
                    "rall_by_seed": {s: rall[s].get(qid) for s in rall},
                    "b": b_by_q.get(qid), "d": d_by_q.get(qid),
                }
                f.write(json.dumps(record) + "\n")


@app.command()
def main(
    input: str = typer.Option(..., "--input", help="Path to distributed_evidence_replay runs.jsonl"),
    output_dir: str = typer.Option(None, "--output-dir"),
    b_c_input: str = typer.Option(_B_C_PATH, "--b-c-input"),
    d_input: str = typer.Option(_D_PATH, "--d-input"),
) -> None:
    input_path = Path(input)
    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(input_path)
    r0, rall = organize(runs)

    r0_vs_rall = r0_vs_rall_per_seed(r0, rall)
    agg = cross_seed_aggregation(rall)
    majority = rall_majority_vote(rall)

    comparisons = {}
    bc_path = Path(b_c_input)
    d_path = Path(d_input)
    if bc_path.exists():
        bc_runs = _load_runs(bc_path)
        comparisons["beginning"] = compare_with_condition(majority, bc_runs, "beginning")
        comparisons["mid_trajectory"] = compare_with_condition(majority, bc_runs, "mid_trajectory")
    else:
        bc_runs = []
    if d_path.exists():
        d_runs = _load_runs(d_path)
        comparisons["full_evidence_reader"] = compare_with_condition(majority, d_runs, "full_evidence_reader")
    else:
        d_runs = []

    summary = {
        "n_r0": len(r0), "n_rall_by_seed": {s: len(v) for s, v in rall.items()},
        "r0_vs_rall_per_seed": r0_vs_rall,
        "cross_seed_aggregation": agg,
        "comparisons": comparisons,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_summary_md(summary, out_dir / "summary.md")
    write_per_question_csv(runs, out_dir / "per_question.csv")
    write_placements_jsonl(runs, out_dir / "placements.jsonl")
    make_plots(rall, out_dir)
    export_manual_examples(r0, rall, agg, bc_runs, bc_runs, d_runs, out_dir / "manual_examples.jsonl")

    print(f"Wrote summary.md, summary.json, per_question.csv, placements.jsonl, plots, "
          f"and manual_examples.jsonl to {out_dir}")


if __name__ == "__main__":
    app()
