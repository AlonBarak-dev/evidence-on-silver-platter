# Paired comparison: beginning vs mid_trajectory

- Paired questions (present & error-free in both): 100
- Budget by position: {'beginning': [8], 'mid_trajectory': [8]}
- Budget parity OK (both positions used the same action allowance): True

## Paired correctness contingency table

|                      | mid_trajectory correct | mid_trajectory incorrect |
|---|---|---|
| **beginning correct**   | 69 (both correct) | 3 (beginning only) |
| **beginning incorrect** | 4 (mid_trajectory only) | 24 (both incorrect) |

## McNemar's exact test (accuracy)
- Discordant pairs: beginning-only=3, mid_trajectory-only=4
- p-value: 1.0

## Paired bootstrap differences (10,000 resamples of question IDs, 95% CI)

- Search calls (beginning − mid_trajectory): mean=0.71, CI=[0.45, 1.0]
- Actions to answer (beginning − mid_trajectory): mean=0.92, CI=[0.61, 1.25]
- Redundant-search rate (beginning − mid_trajectory): mean=0.2, CI=[0.11, 0.3]
