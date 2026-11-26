# Plan — Project 1

Design lives in `docs/SPEC.md`. This file is the schedule, the gates and the checklist.
Off days are Mon–Wed (~3 h), work days Thu–Sun (~90 min). Through December the project
gets **one off-day block per week**; GPU runs go in the background while other work happens.

## Gates (hard — do not start the next phase until the gate passes)

| Gate | Date | Pass condition | If it fails |
|---|---|---|---|
| G1 | Tue 9 Dec | B0 = 90.1 ± 0.5 · B-rel = 94.6 ± 0.3 · B1 seed 1 = 94.6 ± 1.0 | Debug the pipeline. No mechanism code until G1 passes |
| G2 | Tue 23 Dec | Meta, Pyra, MoH all train end-to-end on last-4 without NaN; healing loss falls; unit tests pass; timing sweep table exists | Drop the failing mechanism to Tier B, run Tier A with the rest |
| Freeze | Tue 6 Jan | Tier A results + saved embeddings committed | Whatever isn't done is cut, not squeezed into exam weeks |

If the CV written moves from 22 Jan to the 18 Feb fallback, the freeze moves to 20 Jan.

## P0 — Setup and priors · Wed 25 – Sun 29 Nov

- [x] Repo structure, spec, FLOP priors (`scripts/flop_model.py`)
- [ ] Enter Project 1 in Amerini's winter scheduling form
- [ ] Find the March presentation date (form due 7 days before, repo frozen 2 days before)
- [ ] Ask a TA whether written and project each need 18/30 on their own
- [ ] Read, in the short blocks:
  - DreamSim §3–4 and Appendix A (training details)
  - Schiavella et al. 2025, methods + entropy analysis + Pareto section (Amerini's paper — know it best)
  - PoolFormer §3, PVT §3 (spatial-reduction attention), MoH §3

## P1 — Data and baseline · Mon 30 Nov – Tue 9 Dec

- [ ] Colab day 1: locate `data.csv` for the chunked download (SPEC §6, O1)
- [ ] Measure the 224 px size on chunk 000 → full train or subset (O2)
- [ ] Run the resumable prep job; verify 15,941 / 1,958 / 2,120 triplets and that every path exists
- [ ] B0 zero-shot DINO (target 90.1)
- [ ] B-rel: released DreamSim `dino_vitb16` scored with our eval code (target 94.6, O4)
- [ ] B1 seed 1, record minutes per epoch on T4 (O3)
- [ ] Timing harness on the baseline: T4 (fp16/fp32, batch 1/64) and X270 single core
- [ ] Pin `requirements.txt` against the working Colab environment
- [ ] **G1**

## P2 — Mechanisms and healing · Wed 10 – Tue 23 Dec

- [ ] Meta, Pyra (R=2), MoH (2 shared + 4/10 routed), each with its CLS rule
- [ ] Unit tests: Pyra with R=1 and identity W equals the original attention; MoH with all heads at weight 1 equals the original; Meta's CLS receives patch information; shapes at 224 and 448
- [ ] Stage A healing; Stage B reuses the baseline training code
- [ ] Smoke run each mechanism on last-4 on the smoke subset (SPEC §6 Subsets)
- [ ] Tier C timing sweep at 224/336/448/672 px (no training needed)
- [ ] **G2**

## P3 — Experiments · Wed 24 Dec – Tue 6 Jan

- [ ] B1 seeds 2 and 3
- [ ] Tier A: M-L4, M-A12, P-L4, P-A12, H-L4, H-A12
- [ ] Save test embeddings + per-block features for every trained model (SPEC §8.3)
- [ ] Log every run in `results/runs.csv`
- [ ] Tier B, only if Tier A is done by 3 Jan, in this order: C1 → healing ablation (×3) → first-4 (×3) → H75-L4
- [ ] **Freeze**

## P4 — Exams · 7 Jan – 12 Feb

No project work. CV written 22 Jan, Robotics 1 written 12 Feb.

## P5 — Analysis and delivery · 13 Feb – presentation

- [ ] Statistics: bootstrap CIs, McNemar + Holm (laptop)
- [ ] Dispersion: entropy (two estimators), effective rank, CKA from saved embeddings (laptop)
- [ ] Pareto plots: 2AFC vs T4 latency, vs CPU latency, vs FLOPs
- [ ] Hypotheses H1–H5: confirmed / refuted, one line each
- [ ] Notebook runs end-to-end on a fresh Colab with GPU (course requirement)
- [ ] README and slides, written by me
- [ ] Course form submitted 7 days before the presentation; repo final 2 days before
