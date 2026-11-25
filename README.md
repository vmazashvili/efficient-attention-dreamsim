# efficient-attention-dreamsim

Computer Vision course project (Sapienza, Prof. Amerini), Project 1: does replacing
self-attention with efficient variants (Meta, Pyra, MoH) in a DreamSim-style ViT keep
its agreement with human similarity judgments, and what does it buy in speed?

Status: setup. Design in `docs/SPEC.md`, schedule in `PLAN.md`.

```
docs/SPEC.md            design: formulation, method, experiments, risks
PLAN.md                 phases, gates, checklist
configs/                base hyperparameters and the run matrix
scripts/flop_model.py   analytic FLOP model (priors for H1)
src/effsim/             package
notebooks/project.ipynb driver notebook in the course's required section order
results/                run log and result tables
```
