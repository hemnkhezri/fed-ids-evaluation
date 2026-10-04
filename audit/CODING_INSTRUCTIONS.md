# Coding instructions (apply AUDIT_PROTOCOL.md; do not edit it)

For each paper, open the full text and code items A1-A11 exactly as defined in AUDIT_PROTOCOL.md, using ONLY these
code values:
- A1: random | temporal_or_capture | provided_split | not_reported
- A2: yes | no | unclear
- A3: train_only | all_data | not_reported
- A4: both | one | neither
- A5: same | different_or_unstated | copied
- A6: rerun | copied | none
- A7: yes | no
- A8: yes | no | unclear
- A9: runs_with_dispersion_or_test | runs_no_dispersion | single_or_not_reported
- A10: yes | no
- A11: yes | no
Guidance:
- Code what the text states. If it is not stated, use not_reported / neither / different_or_unstated / no /
  single_or_not_reported as defined. Do not infer from common practice.
- A1 "random" includes stratified random, k-fold cross-validation, and train_test_split over records or samples.
  "provided_split" = the dataset's own released train/test files (e.g. UNSW-NB15 training/testing set, NSL-KDD).
  If several datasets use different splits, code the dominant one and explain in the note.
- A2 = yes only if the model consumes windows/sequences built from consecutive records (e.g. sliding windows,
  time steps, sequences for LSTM/GRU/TCN) AND the split is random over those windows/sequences or over records
  before sequencing without temporal separation. A per-record LSTM with timesteps=1 counts as no windows (no).
- A5 = same only if the paper says the baselines use the same rounds/epochs (or a shared setting explicitly applied
  to all methods). Baselines whose numbers are quoted from other papers = copied.
- A10 = yes only for evaluation on a dataset/capture not used in developing or training that model.
- A11 = yes only if a working link to code is given in the paper.
- Evidence: a short verbatim quote (<= 30 words) or a section/table reference for EVERY item, including
  "not stated in Sections X-Y" for not_reported codes.

Write one JSON file per paper to /home/claude/audit/coding/<id>.json:
{"id": ..., "citation": ..., "url_read": ..., "items": {"A1": {"code": ..., "evidence": ...}, ... "A11": {...}},
 "datasets": [...], "notes": "..."}
