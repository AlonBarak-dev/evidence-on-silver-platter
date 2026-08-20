# Paired comparison: no_injection vs beginning

- Paired questions (present & error-free in both): 100
- Budget by position: {'no_injection': [10], 'beginning': [8]}
- Budget parity OK (both positions used the same action allowance): False

## Paired correctness contingency table

|                      | beginning correct | beginning incorrect |
|---|---|---|
| **no_injection correct**   | 16 (both correct) | 2 (no_injection only) |
| **no_injection incorrect** | 56 (beginning only) | 26 (both incorrect) |

## McNemar's exact test (accuracy)
- Discordant pairs: no_injection-only=2, beginning-only=56
- p-value: 1.1879386363489175e-14

## Paired bootstrap differences (10,000 resamples of question IDs, 95% CI)

- Search calls (no_injection − beginning): mean=4.39, CI=[3.96, 4.82]
- Actions to answer (no_injection − beginning): mean=5.0, CI=[4.49, 5.51]
- Redundant-search rate (no_injection − beginning): mean=0.62, CI=[0.52, 0.71]
