# Paired comparison: beginning vs full_evidence_reader

- Paired questions (present & error-free in both): 100
- Budget by position: {'beginning': [8], 'full_evidence_reader': [1]}
- Budget parity OK (both positions used the same action allowance): False

## Paired correctness contingency table

|                      | full_evidence_reader correct | full_evidence_reader incorrect |
|---|---|---|
| **beginning correct**   | 69 (both correct) | 3 (beginning only) |
| **beginning incorrect** | 6 (full_evidence_reader only) | 22 (both incorrect) |

## McNemar's exact test (accuracy)
- Discordant pairs: beginning-only=3, full_evidence_reader-only=6
- p-value: 0.5078125

## Paired bootstrap differences (10,000 resamples of question IDs, 95% CI)

- Search calls (beginning − full_evidence_reader): mean=1.06, CI=[0.71, 1.43]
- Actions to answer (beginning − full_evidence_reader): mean=2.05, CI=[1.6, 2.53]
- Redundant-search rate (beginning − full_evidence_reader): mean=0.38, CI=[0.29, 0.48]
