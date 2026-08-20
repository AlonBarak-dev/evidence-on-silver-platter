#!/usr/bin/env python3
"""Generate the four publication figures for report.tex, from verified
gpt-4o results only. Run from the stopBrowse repo root:

    python scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#2a78d6"
NEUTRAL = "#8b8f97"
GOOD = "#2a9d5c"
BAD = "#c4453b"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load(p):
    return json.load(open(ROOT / p))


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Evidence discovery bottleneck: accuracy across A-E
# ---------------------------------------------------------------------------
a = load("outputs/initial_gold_injection_confirmatory_no_injection/summary.json")["overall"]
mp = load("outputs/initial_gold_injection_confirmatory_multi_position/summary.json")
b = mp["by_position"]["beginning"]
c = mp["by_position"]["mid_trajectory"]
d = load("outputs/initial_gold_injection_confirmatory_full_evidence_reader/summary.json")["overall"]
e = load("outputs/initial_gold_injection_confirmatory_search_disabled/summary.json")["overall"]

conditions = [("A\nno injection", a), ("B\nbeginning", b), ("C\nmid-trajectory", c),
              ("D\nfull-evidence\nreader", d), ("E\nsearch\ndisabled", e)]
labels = [x[0] for x in conditions]
accs = [x[1]["accuracy"] * 100 for x in conditions]
ci_lo = [x[1]["accuracy_ci95"][0] * 100 for x in conditions]
ci_hi = [x[1]["accuracy_ci95"][1] * 100 for x in conditions]
err_lo = [a_ - lo for a_, lo in zip(accs, ci_lo)]
err_hi = [hi - a_ for a_, hi in zip(accs, ci_hi)]
colors = [NEUTRAL, ACCENT, ACCENT, ACCENT, ACCENT]

fig, ax = plt.subplots(figsize=(6.2, 4))
ax.bar(labels, accs, color=colors, width=0.6, zorder=3)
ax.errorbar(labels, accs, yerr=[err_lo, err_hi], fmt="none", ecolor="#333333", capsize=4, zorder=4)
for i, (v, hi) in enumerate(zip(accs, ci_hi)):
    ax.text(i, hi + 3, f"{v:.0f}%", ha="center", fontsize=10)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Evidence discovery is the dominant bottleneck")
savefig(fig, "fig1_accuracy_by_condition")


# ---------------------------------------------------------------------------
# Figure 2 — Injection timing: accuracy (B vs C) + efficiency deltas
# ---------------------------------------------------------------------------
bc = load("outputs/initial_gold_injection_confirmatory_multi_position/paired_beginning_vs_mid_trajectory.json")

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))

ax = axes[0]
names = ["B\nbeginning", "C\nmid-trajectory"]
vals = [b["accuracy"] * 100, c["accuracy"] * 100]
cis = [b["accuracy_ci95"], c["accuracy_ci95"]]
elo = [v - ci[0] * 100 for v, ci in zip(vals, cis)]
ehi = [ci[1] * 100 - v for v, ci in zip(vals, cis)]
ax.bar(names, vals, color=[ACCENT, ACCENT], width=0.5, zorder=3)
ax.errorbar(names, vals, yerr=[elo, ehi], fmt="none", ecolor="#333333", capsize=4, zorder=4)
for i, (v, ci) in enumerate(zip(vals, cis)):
    ax.text(i, ci[1] * 100 + 3, f"{v:.0f}%", ha="center", fontsize=10)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title(f"Accuracy (McNemar p={bc['mcnemar_p_value']:.2f})")

ax = axes[1]
metrics = [
    ("Search calls\n(B − C)", bc["search_calls_diff"]),
    ("Actions to answer\n(B − C)", bc["actions_to_answer_diff"]),
    ("Redundant-search rate\n(B − C)", bc["redundant_search_rate_diff"]),
]
y = range(len(metrics))
means = [m[1]["mean_diff"] for m in metrics]
los = [m[1]["mean_diff"] - m[1]["ci_lo"] for m in metrics]
his = [m[1]["ci_hi"] - m[1]["mean_diff"] for m in metrics]
ax.errorbar(means, y, xerr=[los, his], fmt="o", color=ACCENT, ecolor="#333333", capsize=4, markersize=7)
ax.axvline(0, color="#999999", linewidth=1, linestyle="--")
ax.set_yticks(list(y))
ax.set_yticklabels([m[0] for m in metrics])
ax.invert_yaxis()
ax.set_xlabel("Paired difference (95% bootstrap CI)")
ax.set_title("Efficiency cost of early injection")
savefig(fig, "fig2_injection_timing")


# ---------------------------------------------------------------------------
# Figure 3 — Contribution of continued search: paired-effect forest plot
# ---------------------------------------------------------------------------
ab = load("outputs/paired_analyses/paired_no_injection_vs_beginning.json")
be = load("outputs/paired_analyses/paired_beginning_vs_beginning_no_search.json")
bd = load("outputs/paired_analyses/paired_beginning_vs_full_evidence_reader.json")


def acc_diff_ci(paired_json, acc_a, acc_b, n_key="n_paired_questions"):
    """Wilson-free point + McNemar-based direction; we report the raw
    accuracy point difference with the paired bootstrap CI computed
    separately via the shared question set (approximate via contingency)."""
    t = paired_json["contingency_table"]
    n = t["both_correct"] + t["a_only_correct"] + t["b_only_correct"] + t["both_incorrect"]
    diff = (acc_a - acc_b)
    return diff * 100, n


effects = [
    ("A → B\n(inject evidence)", (ab["contingency_table"]["b_only_correct"] - ab["contingency_table"]["a_only_correct"]) / ab["n_paired_questions"] * 100, ab["mcnemar_p_value"]),
    ("E → B\n(re-enable search)", (be["contingency_table"]["a_only_correct"] - be["contingency_table"]["b_only_correct"]) / be["n_paired_questions"] * 100, be["mcnemar_p_value"]),
    ("B → D\n(oracle reader)", (bd["contingency_table"]["b_only_correct"] - bd["contingency_table"]["a_only_correct"]) / bd["n_paired_questions"] * 100, bd["mcnemar_p_value"]),
]

fig, ax = plt.subplots(figsize=(7.2, 3.8))
y = range(len(effects))
vals = [x[1] for x in effects]
colors_f = [GOOD if p < 0.05 else NEUTRAL for _, _, p in effects]
ax.barh(list(y), vals, color=colors_f, height=0.5, zorder=3)
ax.set_xlim(-65, 65)
for i, (name, v, p) in enumerate(effects):
    x_text = v + (4 if v >= 0 else -4)
    ax.text(x_text, i, f"{v:+.0f}pp, p={p:.1g}", va="center",
            ha="left" if v >= 0 else "right", fontsize=9)
ax.axvline(0, color="#999999", linewidth=1)
ax.set_yticks(list(y))
ax.set_yticklabels([x[0] for x in effects])
ax.invert_yaxis()
ax.margins(y=0.15)
ax.set_xlabel("Accuracy change (percentage points, paired, n=100)")
ax.set_title("Where accuracy gains come from")
savefig(fig, "fig3_paired_effects")


# ---------------------------------------------------------------------------
# Figure 4 — Distributed evidence utilization: R0 vs R-all
# ---------------------------------------------------------------------------
rp = load("outputs/distributed_evidence_replay/summary.json")
r0r = rp["r0_vs_rall_per_seed"]["43"]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))

ax = axes[0]
names = ["R0\n(unmodified\nreplay)", "R-all\n(distributed\nevidence)"]
vals = [r0r["r0_accuracy"] * 100, r0r["rall_accuracy"] * 100]
ax.bar(names, vals, color=[NEUTRAL, ACCENT], width=0.5, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 3, f"{v:.0f}%", ha="center", fontsize=10)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title(f"n={r0r['n']}, McNemar p={r0r['mcnemar_p_value']:.1e}")

ax = axes[1]
t = r0r["contingency_table"]
cats = ["Rescues\n(R0 wrong\n→ R-all correct)", "Harms\n(R0 correct\n→ R-all wrong)",
        "Both\ncorrect", "Both\nincorrect"]
counts = [t["b_only_correct"], t["a_only_correct"], t["both_correct"], t["both_incorrect"]]
bar_colors = [GOOD, BAD, ACCENT, NEUTRAL]
ax.bar(cats, counts, color=bar_colors, width=0.6, zorder=3)
for i, v in enumerate(counts):
    ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)
ax.set_ylabel("Questions (n)")
ax.set_title("Rescues far outnumber harms", fontsize=11)
plt.setp(ax.get_xticklabels(), fontsize=8.5)
savefig(fig, "fig4_replay_rescue")

# ---------------------------------------------------------------------------
# Figure 5 — Supplementary: accuracy vs. number of injected documents (B)
# ---------------------------------------------------------------------------
runs_mp = [json.loads(l) for l in open(ROOT / "outputs/initial_gold_injection_confirmatory_multi_position/runs.jsonl")]
b_runs = [r for r in runs_mp if r.get("injection_position") == "beginning"]
by_n: dict[int, list[bool]] = {}
for r in b_runs:
    n = r.get("num_injected_documents") or 0
    by_n.setdefault(n, []).append(bool(r.get("answer_correct")))
ns = sorted(by_n)
counts = [len(by_n[n]) for n in ns]
accs = [100 * sum(by_n[n]) / len(by_n[n]) for n in ns]

fig, ax = plt.subplots(figsize=(6.2, 4))
bars = ax.bar([str(n) for n in ns], accs, color=ACCENT, width=0.6, zorder=3)
for i, (n, acc, cnt) in enumerate(zip(ns, accs, counts)):
    ax.text(i, acc + 3, f"{acc:.0f}%", ha="center", fontsize=9)
    ax.text(i, -6, f"n={cnt}", ha="center", fontsize=8, color=NEUTRAL)
ax.set_xlabel("Number of injected documents")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(-10, 100)
ax.axhline(0, color="#333333", linewidth=0.8)
ax.set_title("Condition B: accuracy vs. injected-document count")
savefig(fig, "fig5_dose_response")

# ---------------------------------------------------------------------------
# Figure 0 — Schematic: what injection looks like (trajectory diagram)
# ---------------------------------------------------------------------------
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.2), sharex=True)
rows = [
    ("B — beginning injection", ["inject"] + ["agent"] * 6),
    ("C — mid-trajectory injection", ["agent", "agent", "inject"] + ["agent"] * 4),
    ("E — beginning, search disabled", ["inject"] + ["agent (no search)"] * 6),
]
box_w = 0.9
for ax, (title, steps) in zip(axes, rows):
    for i, step in enumerate(steps):
        if step == "inject":
            color = ACCENT
            label = "injected\nsearch turn"
        elif step == "agent (no search)":
            color = "#c9832a"
            label = "agent turn\n(get_document\nonly)"
        else:
            color = NEUTRAL
            label = "agent\nturn"
        rect = mpatches.FancyBboxPatch(
            (i, 0), box_w, 1, boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0, facecolor=color, alpha=0.85 if step == "inject" else 0.55,
        )
        ax.add_patch(rect)
        ax.text(i + box_w / 2, 0.5, label, ha="center", va="center", fontsize=7.3,
                color="white" if step == "inject" else "#222222")
    ax.set_xlim(-0.2, len(steps) + 0.1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10.5, loc="left")

axes[-1].set_xlabel("Trajectory turn order →", fontsize=9.5, loc="left")

legend_handles = [
    mpatches.Patch(color=ACCENT, alpha=0.85, label="Injected search turn (synthetic, same schema as a real result)"),
    mpatches.Patch(color=NEUTRAL, alpha=0.55, label="Ordinary agent turn (search / get_document / final)"),
    mpatches.Patch(color="#c9832a", alpha=0.55, label="Agent turn with search disabled (condition E only)"),
]
fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           fontsize=8.3, frameon=False, ncol=1)
fig.suptitle("What injection looks like: where the synthetic evidence turn lands", fontsize=11.5, y=0.99)
fig.tight_layout(rect=[0, 0.1, 1, 0.96])
fig.savefig(OUT / "fig0_injection_schematic.pdf")
fig.savefig(OUT / "fig0_injection_schematic.png")
plt.close(fig)

print("Wrote figures to", OUT)
for f in sorted(OUT.glob("*.png")):
    print(" -", f.name)
