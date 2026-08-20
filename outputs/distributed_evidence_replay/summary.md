# distributed_evidence_replay — summary

We construct off-policy counterfactual replays by distributing every annotated evidence document across the search-result observations of a frozen no-injection trajectory. We then regenerate only the final answer, isolating evidence utilization from search-policy adaptation.

**Limitation:** later search actions were generated under the original observations and may not be behaviorally consistent with the injected evidence. This inconsistency is intentional: freezing the action sequence prevents evidence-induced policy changes from confounding the utilization analysis.

## R0 vs. R-all, per placement seed

### Seed 43 (n=54)
- R0 accuracy: 0.14814814814814814
- R-all accuracy: 0.5
- Accuracy diff (R-all - R0): -0.35185185185185186 CI=[-0.48148148148148145, -0.2222222222222222]
- Contingency: both_correct=8, R0_only=0, R-all_only=19, both_incorrect=27
- McNemar p-value: 3.814697265625e-06
- Rescues (R0 wrong -> R-all correct): 19
- Harms (R0 correct -> R-all wrong): 0

## Aggregation across seeds

- Shared questions across all seeds: 100
- Per-seed accuracy: {43: 0.53}
- Mean accuracy across seeds: 0.53
- Stdev accuracy across seeds: 0.0
- Mean question-level correctness (bootstrap, questions resampled with full seed vector kept together): {'point': 0.53, 'ci_lo': 0.43, 'ci_hi': 0.63}
- Placement-sensitive rate (result changes across placements): 0.0

## Comparisons with existing conditions (R-all majority vote vs. B/C/D)

### R-all vs. beginning (n=100)
- R-all accuracy: 0.53
- beginning accuracy: 0.72
- Accuracy diff bootstrap: {'mean_diff': -0.19, 'ci_lo': -0.28, 'ci_hi': -0.1}
- Contingency: {'both_correct': 51, 'a_only_correct': 2, 'b_only_correct': 21, 'both_incorrect': 26}
- McNemar p-value: 6.604194641113281e-05

### R-all vs. mid_trajectory (n=100)
- R-all accuracy: 0.53
- mid_trajectory accuracy: 0.73
- Accuracy diff bootstrap: {'mean_diff': -0.2, 'ci_lo': -0.29, 'ci_hi': -0.11}
- Contingency: {'both_correct': 51, 'a_only_correct': 2, 'b_only_correct': 22, 'both_incorrect': 25}
- McNemar p-value: 3.5881996154785156e-05

### R-all vs. full_evidence_reader (n=100)
- R-all accuracy: 0.53
- full_evidence_reader accuracy: 0.75
- Accuracy diff bootstrap: {'mean_diff': -0.22, 'ci_lo': -0.31, 'ci_hi': -0.13}
- Contingency: {'both_correct': 51, 'a_only_correct': 2, 'b_only_correct': 24, 'both_incorrect': 23}
- McNemar p-value: 1.049041748046875e-05

