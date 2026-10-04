# X-IIoTID as a second independent test: eligibility criteria (fixed before the file is opened)

Written on 2026-09-27, after the phase-2/3 outcome and before any inspection of X-IIoTID. The only thing known
about the file is its name and its size (338 MB).

Why a second independent test is added: on CICAPT-IIoT2024 (held-out test 1) no method learned. The whole
federation had about 5 labelled attack sequences at p_label = 0.1, so that test did not probe the hypothesis
(label scarcity, not label absence). The CICAPT result stays in the paper unchanged. X-IIoTID is added as a
second test and does not replace it.

## Eligibility (checked only from schema and label counts; no model is run)
X-IIoTID is used only if all of the following hold:
1. It has source-IP, destination-IP and timestamp columns that can be parsed into absolute time.
2. It has a benign/attack label, or a class label that maps to one.
3. After the standard pipeline, the non-IID partitions hold at least 500 attack sequences in the training
   region in total (so that p_label = 0.1 leaves at least about 50 labelled attack sequences). The standard
   pipeline is the CICAPT_PROTOCOL.md rules: 60-s windows, five chronological blocks with blocks 2 and 4 as
   validation, training-region normalisation, a 30-host cap, T = 5 sequences, partition seeds 101-110.
4. The validation region holds at least 30 attack sequences and at least 30 benign sequences.

If any criterion fails, X-IIoTID is reported as ineligible, and no other dataset is substituted in this
study.

## Preprocessing
The CICAPT_PROTOCOL.md rules apply, with one addition: columns that carry absolute time, or that are
identifiers (IP addresses, ports, MAC addresses, date and time fields), are excluded from the features. The
exact column list is fixed after the schema check and before windows are built.

## Test
The phase-2 configurations, seeds, test and decision rule are applied unchanged. The new baseline is added as
described in the next plan, which is sealed before any run.

## Addendum (after the schema check and before any window is built)
The schema check gave: 820,834 rows, 68 columns, class3 = Normal 421,417 / Attack 399,417, and Timestamp
parseable on 820,383 rows (2019-05-09 to 2020-03-07; most days fall between Dec 2019 and Mar 2020).

Fixed column roles:
- **Label:** class3 == "Attack" (class1 and class2 are not used).
- **Excluded:** Date, Timestamp, Scr_IP, Des_IP, Scr_port and Des_port (identifiers / absolute time), and
  class1, class2 and class3.
- **Categorical, integer-encoded** (as for the other datasets, since each has at most 50 distinct values):
  Protocol and Service.
- **Features:** every other column. Booleans become 0/1. Non-numeric tokens such as "-", "?" and "aza"
  become 0.
- **Rows dropped:** rows with an unparseable Timestamp, and rows whose source or destination is not a
  dotted IP address.

Everything else follows CICAPT_PROTOCOL.md.
