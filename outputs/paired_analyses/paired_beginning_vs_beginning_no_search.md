# Paired comparison: beginning vs beginning_no_search

- Paired questions (present & error-free in both): 100
- Budget by position: {'beginning': [8], 'beginning_no_search': [8]}
- Budget parity OK (both positions used the same action allowance): True

## Paired correctness contingency table

|                      | beginning_no_search correct | beginning_no_search incorrect |
|---|---|---|
| **beginning correct**   | 63 (both correct) | 9 (beginning only) |
| **beginning incorrect** | 1 (beginning_no_search only) | 27 (both incorrect) |

## McNemar's exact test (accuracy)
- Discordant pairs: beginning-only=9, beginning_no_search-only=1
- p-value: 0.021484375

## Paired bootstrap differences (10,000 resamples of question IDs, 95% CI)

- Search calls (beginning − beginning_no_search): mean=1.06, CI=[0.71, 1.43]
- Actions to answer (beginning − beginning_no_search): mean=1.26, CI=[0.86, 1.69]
- Redundant-search rate (beginning − beginning_no_search): mean=0.38, CI=[0.29, 0.48]
