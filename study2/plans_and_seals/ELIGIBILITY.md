# X-IIoTID eligibility check (XIIOTID_CRITERIA.md). No model has been run.
- Criterion 1: Scr_IP, Des_IP and Timestamp are present; 819,791 of 820,834 rows are kept -> met
- Criterion 2: class3 has the values Normal and Attack -> met
- Criterion 3: training-region attack sequences, total over clients, are 4,485 in every partition (>= 500) -> met
- Criterion 4: the validation region holds 3,394 attack sequences and 270 benign sequences (both >= 30) -> met
Decision: X-IIoTID is ELIGIBLE as the second independent test.
Note: this dataset is attack-majority, with about 93% attack windows in the validation region.
