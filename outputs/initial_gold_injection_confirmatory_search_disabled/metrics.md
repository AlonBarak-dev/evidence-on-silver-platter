# initial_gold_injection — metrics report

## Overall (all positions combined)

- Attempted samples: 100
- Completed samples: 100
- Scored samples (excluding data errors): 100

### Accuracy
- Accuracy: 0.64
- 95% bootstrap CI (question resampling): [0.54, 0.73]
- Immediate-answer rate: 0.31
- Immediate correct-answer rate: 0.28
- Immediate incorrect-answer rate: 0.03
- Redundant-search rate: 0.0

### Efficiency (all scored samples)
- Actions to answer — mean/median: 1.79 / 2.0
- Search calls — mean/median: 0 / 0.0
- get_document calls — mean/median: 0.79 / 1.0
- Mean injected-document coverage: 0.1589935064935065

### Correct-efficiency (correct samples only)
- n correct: 64
- Actions to answer — mean/median: 1.609375 / 2.0
- Search calls — mean/median: 0 / 0.0
- get_document calls — mean/median: 0.609375 / 1.0
- Mean coverage: 0.10363455988455988

### Failure taxonomy counts
- CORRECT_AFTER_DOCUMENT_ACCESS: 36
- WRONG_AFTER_PARTIAL_EVIDENCE: 32
- CORRECT_IMMEDIATE: 28
- WRONG_WITHOUT_OPENING_EVIDENCE: 3
- WRONG_AFTER_FULL_EVIDENCE: 1

## Position: beginning_no_search

- Attempted samples: 100
- Completed samples: 100
- Scored samples (excluding data errors): 100

### Accuracy
- Accuracy: 0.64
- 95% bootstrap CI (question resampling): [0.54, 0.73]
- Immediate-answer rate: 0.31
- Immediate correct-answer rate: 0.28
- Immediate incorrect-answer rate: 0.03
- Redundant-search rate: 0.0

### Efficiency (all scored samples)
- Actions to answer — mean/median: 1.79 / 2.0
- Search calls — mean/median: 0 / 0.0
- get_document calls — mean/median: 0.79 / 1.0
- Mean injected-document coverage: 0.1589935064935065

### Correct-efficiency (correct samples only)
- n correct: 64
- Actions to answer — mean/median: 1.609375 / 2.0
- Search calls — mean/median: 0 / 0.0
- get_document calls — mean/median: 0.609375 / 1.0
- Mean coverage: 0.10363455988455988

### Failure taxonomy counts
- CORRECT_AFTER_DOCUMENT_ACCESS: 36
- WRONG_AFTER_PARTIAL_EVIDENCE: 32
- CORRECT_IMMEDIATE: 28
- WRONG_WITHOUT_OPENING_EVIDENCE: 3
- WRONG_AFTER_FULL_EVIDENCE: 1

