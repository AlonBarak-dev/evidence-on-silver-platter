# Evidence on a Silver Platter

**Finding versus using evidence in agentic search.**

A controlled evaluation that disentangles two failure modes in agentic web
search: does the agent fail because it never finds the right evidence, or
because it finds it and still fails to use it? Built on a ReAct-style GPT-4o
search agent evaluated on 100 questions from
[BrowseComp-Plus](https://arxiv.org/abs/2508.06600).

## Headline result

Injecting gold evidence directly into the agent's trajectory raises accuracy
from **18% to 72%** — a 54-point gain (McNemar p = 1.2×10⁻¹⁴). A single-shot
reader given the same evidence with no search access at all reaches 75%,
statistically indistinguishable from the live agent (p = 0.51). **Evidence
discovery, not evidence utilization, is the dominant bottleneck.**

| Condition | Accuracy | 95% CI |
|---|---|---|
| A — No injection (control) | 18.0% | [11.0, 26.0] |
| B — Beginning injection | 72.0% | [63.0, 81.0] |
| C — Mid-trajectory injection | 73.0% | [64.0, 81.0] |
| D — Full-evidence reader | 75.0% | [66.0, 83.0] |
| E — Beginning, search disabled | 64.0% | [54.0, 73.0] |
| R0 — Frozen replay (54-question subset) | 14.8% | — |
| R-all — Distributed evidence (54-question subset) | 50.0% | — |
| R-all — Distributed evidence (full 100) | 53.0% | [43.0, 63.0] |

Two secondary findings still matter: disabling further search after injection
costs 8 points (p = 0.02), and evidence placed into a frozen, off-policy
trajectory recovers most but not all of the live-agent benefit (−19 to −22pp
vs. live conditions, p < 0.002 throughout).

## Research questions

1. How much of the agent's failure is caused by evidence discovery, as
   opposed to failure after evidence was already available?
2. Does the *timing* of evidence injection affect accuracy or efficiency?
3. How do continued search access and the manner of evidence presentation
   (live agentic access vs. a frozen, distributed replay) affect how well the
   agent utilizes evidence it has been given?

## Method

Five live conditions (A–E) plus an offline counterfactual replay (R0/R-all),
all run on the identical 100 questions, holding the agent, retriever, corpus,
and grading procedure fixed:

- **A — No injection.** Unmodified agent, 10-action budget.
- **B — Beginning injection.** Gold evidence injected before the agent's
  first action, 8-action budget, search enabled.
- **C — Mid-trajectory injection.** Identical to B, injection delayed until
  after two initial actions — isolates *timing* from injection itself.
- **D — Full-evidence reader.** Single-shot: full evidence text up front, no
  tools at all.
- **E — Beginning injection, search disabled.** Identical to B, but the
  `search` tool is disabled after injection.
- **R0 / R-all — Distributed-evidence counterfactual replay.** Condition A's
  own frozen trajectories are replayed with either no change (R0) or every
  evidence document distributed into the existing search-result slots
  (R-all), then the final answer is regenerated once, tools disabled. This
  isolates evidence *utilization* from search-policy adaptation.

Injected evidence is inserted as a synthetic `search` turn, formatted
identically to a real search result (same truncation length, ordering
scheme, and synthetic scoring as an ordinary search response — see
`src/experiments/initial_gold_injection/injection.py`).

Cross-condition comparisons use paired statistics — exact McNemar tests and
paired bootstrap confidence intervals over shared question IDs — since every
condition is evaluated on the same 100 questions.

- **Benchmark**: [BrowseComp-Plus](https://arxiv.org/abs/2508.06600),
  validation split, fixed `confirmatory_100` manifest.
- **Agent model**: GPT-4o (Azure OpenAI, temperature 0).
- **Retriever**: Qwen3-Embedding-4B dense retriever, top-k=5, 512-token
  previews.
- **Grading**: LLM judge reusing BrowseComp-Plus's own grading template.

## Repository layout

```
configs/                      one YAML config per condition
prompts/                      all prompt templates used
src/experiments/
  initial_gold_injection/     conditions A–E: agent, injection, taxonomy,
                               paired-analysis code
  distributed_evidence_replay/  R0/R-all: trajectory reconstruction,
                               evidence placement, replay
  common/                     shared LLM client plumbing
tests/                        test suite for the above
scripts/make_paper_figures.py generates the report's figures from outputs/
outputs/                      per-condition results: runs.jsonl, summary.json,
                               metrics.md, plots; outputs/paired_analyses/ and
                               outputs/paper_figures/ hold cross-condition
                               analysis and the final report figures
```

## Reproducing

Each condition is run via its module's CLI, e.g.:

```bash
python -m experiments.initial_gold_injection.run --config configs/initial_gold_injection_confirmatory_no_injection.yaml
python -m experiments.distributed_evidence_replay.run --config configs/distributed_evidence_replay.yaml
```
