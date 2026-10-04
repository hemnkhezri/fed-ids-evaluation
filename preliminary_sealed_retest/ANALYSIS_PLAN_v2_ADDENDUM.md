# Addendum 1 to the pre-registered analysis plan v2

Parent plan: `ANALYSIS_PLAN_v2.md`, sha256 b5d262579fe89ad16699788248775594bf4e2472aee47d1dc5dcfd62fa563ec0,
sealed 2026-09-26T05:43:46Z. This addendum was written and hashed before any confirmatory run of the
re-implemented code. It changes nothing in the parent plan's design, arms, metric, hypotheses, test,
multiplicity correction, decision rules or exploratory list.

## What happened
On 2026-09-26 (about 12:10 UTC) the compute environment was reset. The working directory was lost,
including the fixed pipeline (`22_fixed.py`, `23_run.py`, the corrected partition builders) and the
partial results file. The parent plan survived because it was stored separately.

## Disclosure of outcomes already seen
The analyst had already seen partial outcomes produced by the lost implementation:
- WUSTL-IIoT-2021, all 7 arms × 10 seeds.
- TON_IoT-Network, seeds 1–2 (all arms) and seed 3 (A1–A4).

Those outcomes are discarded. They cannot be reproduced bit-for-bit, and mixing them with runs from new
code would compare two different implementations. No design choice below was made to change those outcomes.

## Re-implementation
Code: `setup_data.py` and `train.py` in this folder. `04_model.py` is the unchanged encoder. The code was
rebuilt from the written specification: the parent plan, manuscript Section 3, and manuscript Section 4.2.

**Fidelity check.** The rebuilt WUSTL-IIoT-2021 non-IID partition reproduces exactly the figures the
manuscript states for the lost partition:
- 243 training sequences and 160 validation sequences;
- per-client training attack counts of 0, 0, 55 and 1.

The Dirichlet fallback was not used on any dataset.

## Implementation choices
Where the lost code's exact behaviour is not recoverable, the following choices were fixed now. Each
follows the manuscript as written.
1. **Local objective.** The local loss is L = L_con + β(e)·L_sup, with β(e) = (e+1)/E (Eq. 6 as printed).
2. **Contrastive denominator.** It is exp(s_pos) + Σ over negatives of γ·exp(s). γ weights negatives only
   ("each negative pair", Section 3.4). The positive pair is not weighted. Plain NT-Xent (A3) sets γ = 1.
3. **Mini-batches.** Every arm, FedGTCL arms included, iterates over every local mini-batch of 16 in each
   of the E = 3 local epochs. FedGTCL arms train on all local sequences, with supervision on the fixed
   labelled mask. Baselines train on labelled sequences only.
4. **A2 zero-gradient batches.** In A2, a batch with no labelled item has an exactly zero gradient. The
   Adam step is skipped for such a batch, so stale momentum cannot move the weights.
5. **Normalisation.** Eq. (1) min/max is fitted on training-region rows and clipped to [0, 1] for
   validation rows.
6. **Server optimiser.** ε = 1e-8, with no bias correction, for both FedAdaptOpt (A1–A3) and FedAdam (B3).
   Client-side top-s (s = 0.3) applies to A1–A4 only.
7. **Seed control.** One seed fixes model initialisation and all training randomness. The labelled mask
   uses `random.Random(100003·seed + round(1000·p_label))` and is identical for every arm.
8. **Partitions** use partition seed 42. The partition files and their SHA-256 hashes are listed in
   `partition_summary_*.json`.

## Engineering checks
Only engineering-check runs were made before sealing: seed 99, R = 2, with results not stored in the
results file. They verified three things:
- the caller's data is not modified;
- the labelled masks are identical across arms;
- final weights are bit-identical when the arm run order is reversed.

## Confirmatory execution
The confirmatory execution is the parent plan's, unchanged:
- non-IID partition, p_label = 0.5;
- seeds 1–10;
- arms A1–A4 and B1–B3;
- all three datasets;
- H1–H3 per dataset, two-sided Wilcoxon (zsplit), Holm correction over 9 tests at α = 0.05.

Results go to `results_v2r.jsonl`.
