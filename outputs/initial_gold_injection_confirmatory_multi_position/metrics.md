# initial_gold_injection — metrics report

## Position comparison

| position | n | accuracy | immediate rate | redundant search rate | actions/answer (mean) |
|---|---|---|---|---|---|
| beginning | 100 | 0.72 | 0.3 | 0.38 | 3.05 |
| mid_trajectory | 100 | 0.73 | 0.37 | 0.18 | 2.13 |

## Overall (all positions combined)

- Attempted samples: 200
- Completed samples: 200
- Scored samples (excluding data errors): 200

### Accuracy
- Accuracy: 0.725
- 95% bootstrap CI (question resampling): [0.665, 0.785]
- Immediate-answer rate: 0.335
- Immediate correct-answer rate: 0.29
- Immediate incorrect-answer rate: 0.045
- Redundant-search rate: 0.28

### Efficiency (all scored samples)
- Actions to answer — mean/median: 2.59 / 2.0
- Search calls — mean/median: 0.705 / 0.0
- get_document calls — mean/median: 0.885 / 1.0
- Mean injected-document coverage: 0.16146554834054833

### Correct-efficiency (correct samples only)
- n correct: 145
- Actions to answer — mean/median: 1.9448275862068964 / 2
- Search calls — mean/median: 0.2620689655172414 / 0
- get_document calls — mean/median: 0.6827586206896552 / 1
- Mean coverage: 0.12628750559785043

### Failure taxonomy counts
- CORRECT_AFTER_DOCUMENT_ACCESS: 61
- CORRECT_IMMEDIATE: 58
- WRONG_AFTER_PARTIAL_EVIDENCE: 40
- CORRECT_AFTER_EXTRA_SEARCH: 26
- WRONG_WITHOUT_OPENING_EVIDENCE: 12
- WRONG_AFTER_FULL_EVIDENCE: 3

## Position: beginning

- Attempted samples: 100
- Completed samples: 100
- Scored samples (excluding data errors): 100

### Accuracy
- Accuracy: 0.72
- 95% bootstrap CI (question resampling): [0.63, 0.81]
- Immediate-answer rate: 0.3
- Immediate correct-answer rate: 0.27
- Immediate incorrect-answer rate: 0.03
- Redundant-search rate: 0.38

### Efficiency (all scored samples)
- Actions to answer — mean/median: 3.05 / 2.0
- Search calls — mean/median: 1.06 / 0.0
- get_document calls — mean/median: 0.99 / 1.0
- Mean injected-document coverage: 0.17567604617604618

### Correct-efficiency (correct samples only)
- n correct: 72
- Actions to answer — mean/median: 2.1805555555555554 / 2.0
- Search calls — mean/median: 0.4027777777777778 / 0.0
- get_document calls — mean/median: 0.7777777777777778 / 1.0
- Mean coverage: 0.13943602693602694

### Failure taxonomy counts
- CORRECT_AFTER_DOCUMENT_ACCESS: 27
- CORRECT_IMMEDIATE: 27
- WRONG_AFTER_PARTIAL_EVIDENCE: 20
- CORRECT_AFTER_EXTRA_SEARCH: 18
- WRONG_WITHOUT_OPENING_EVIDENCE: 6
- WRONG_AFTER_FULL_EVIDENCE: 2

## Position: mid_trajectory

- Attempted samples: 100
- Completed samples: 100
- Scored samples (excluding data errors): 100

### Accuracy
- Accuracy: 0.73
- 95% bootstrap CI (question resampling): [0.64, 0.81]
- Immediate-answer rate: 0.37
- Immediate correct-answer rate: 0.31
- Immediate incorrect-answer rate: 0.06
- Redundant-search rate: 0.18

### Efficiency (all scored samples)
- Actions to answer — mean/median: 2.13 / 2.0
- Search calls — mean/median: 0.35 / 0.0
- get_document calls — mean/median: 0.78 / 1.0
- Mean injected-document coverage: 0.1472550505050505

### Correct-efficiency (correct samples only)
- n correct: 73
- Actions to answer — mean/median: 1.7123287671232876 / 2
- Search calls — mean/median: 0.1232876712328767 / 0
- get_document calls — mean/median: 0.589041095890411 / 1
- Mean coverage: 0.11331910099033386

### Failure taxonomy counts
- CORRECT_AFTER_DOCUMENT_ACCESS: 34
- CORRECT_IMMEDIATE: 31
- WRONG_AFTER_PARTIAL_EVIDENCE: 20
- CORRECT_AFTER_EXTRA_SEARCH: 8
- WRONG_WITHOUT_OPENING_EVIDENCE: 6
- WRONG_AFTER_FULL_EVIDENCE: 1

