# initial_gold_injection — metrics report

- Attempted samples: 15
- Completed samples: 15
- Scored samples (excluding data errors): 15

## Accuracy
- Accuracy: 0.6
- 95% bootstrap CI (question resampling): [0.3333333333333333, 0.8666666666666667]
- Immediate-answer rate: 0.26666666666666666
- Immediate correct-answer rate: 0.2
- Immediate incorrect-answer rate: 0.06666666666666667
- Redundant-search rate: 0.26666666666666666

## Efficiency (all scored samples)
- Actions to answer — mean/median: 2.6 / 2
- Search calls — mean/median: 0.4666666666666667 / 0
- get_document calls — mean/median: 1.1333333333333333 / 1
- Mean injected-document coverage: 0.18756613756613758

## Correct-efficiency (correct samples only)
- n correct: 9
- Actions to answer — mean/median: 2.6666666666666665 / 2
- Search calls — mean/median: 0.4444444444444444 / 0
- get_document calls — mean/median: 1.2222222222222223 / 1
- Mean coverage: 0.18271604938271604

## Failure taxonomy counts
- WRONG_AFTER_PARTIAL_EVIDENCE: 5
- CORRECT_AFTER_DOCUMENT_ACCESS: 4
- CORRECT_IMMEDIATE: 3
- CORRECT_AFTER_EXTRA_SEARCH: 2
- WRONG_WITHOUT_OPENING_EVIDENCE: 1
