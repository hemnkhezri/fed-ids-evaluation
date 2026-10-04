# Literature audit of evaluation practice in federated IDS for IoT/IIoT (protocol)

Written before any paper is screened or coded. Its SHA-256 is recorded in AUDIT_SEAL.sha256. Changes after that
are dated addenda at the end of this file.

## Question
How often do published federated intrusion detection studies for IoT/IIoT report, or avoid, the evaluation choices
examined in the paper: random splits of time-ordered data, preprocessing fitted on all data, unreported or unequal
training budgets, accuracy/F1-only reporting, single runs, and the absence of an independent test dataset?

## Inclusion criteria
A paper is included if all hold:
1. Journal article or peer-reviewed conference paper published 2021-2026 (a preprint of the same work may be read if
   the published version is not accessible).
2. It proposes or evaluates a federated learning based intrusion or attack detection system for IoT, IIoT or edge
   networks, working on network traffic (flows, packets or windows of them).
3. It reports experiments on at least one public dataset and compares against at least one baseline.
4. Its full text (methods and results) can be read by the auditors.
Excluded: surveys, theses, papers without experiments, papers on host logs, malware binaries or sensor-value anomalies
only, and the authors' own work.

## Search and selection
Candidates come from (a) the federated IDS papers cited in the manuscript's related work and (b) the first results of
these web searches, run in this order:
Q1 "federated learning intrusion detection industrial IoT"; Q2 "federated learning intrusion detection IoT dataset";
Q3 "federated intrusion detection IIoT non-IID"; Q4 "federated learning intrusion detection Edge-IIoTset";
Q5 "federated learning intrusion detection TON_IoT"; Q6 "federated graph neural network intrusion detection IoT".
Candidates are screened in that order and eligible papers are included until 30 are reached (or the candidates run
out). Every screened candidate is listed with its decision and reason.

## Coding items (each coded from the paper's text, with a quotation or section reference as evidence)
- A1 Split: random/stratified | temporal or by capture/file/device | dataset-provided train/test files | not reported.
- A2 Sequence leakage risk: model uses sliding windows or sequences AND the split is random over windows/sequences
  (yes) | split made before windowing, or no windows (no) | unclear.
- A3 Preprocessing fit: scaling/encoding fitted on training data only (stated) | on all data (stated) | not reported.
- A4 Budget reported: rounds and local epochs/steps both reported | only one | neither.
- A5 Budget parity: baselines stated to use the same rounds/epochs | different or unstated for baselines | baseline
  numbers copied from other papers.
- A6 Baselines: all re-run by the authors | some or all copied from other papers | no baselines beyond ablations.
- A7 Imbalance-robust metric: MCC or balanced accuracy (or G-mean) reported | only accuracy/precision/recall/F1/AUC.
- A8 Non-IID: a non-IID client partition evaluated | IID or natural split only | unclear.
- A9 Repetition: several runs with dispersion and/or a statistical test | several runs, no dispersion | single run
  or not reported.
- A10 Independent test: a model evaluated on a dataset or capture never used in its development (cross-dataset
  transfer or held-out dataset) | no.
- A11 Code: public link to code | no.

"Not reported" is coded when the text does not state the item; it is not assumed either way.

## Reliability
A second coder, who does not see the first coding, re-codes a random sample of at least 8 papers (seed 2026).
Agreement per item and Cohen's kappa are reported. Disagreements are resolved by re-reading the paper; the
resolution is recorded.

## Reporting
The paper reports, per item, counts and percentages over included papers, and the full coding table with evidence
in the supplementary material. No paper is named in the main text as an example of poor practice.
