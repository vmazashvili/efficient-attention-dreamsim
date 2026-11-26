# Project 1 — Efficient Attention in DreamSim: Specification

Status: v1, 25 Nov 2026. Single source of truth for design decisions. `PLAN.md` holds
the schedule and gates; this file holds the *what* and *why*.

---

## 1. Context and constraints

| Item | Value |
|---|---|
| Course | Computer Vision, Prof. Irene Amerini, Fall/Winter 2026 list, **Project 1** |
| Presentation | March 2027 session (date to confirm). Both written exam and project must be done by then |
| Team | Solo |
| Framework | PyTorch (mandated) |
| Compute | Colab free (T4, 16 GB) primary · Kaggle (free GPU quota) backup · ThinkPad X270 (CPU only) for analysis and CPU timing |
| Competing load | Elective in AI (Nov–Dec, paper mid-Jan) · CV written 22 Jan (fallback 18 Feb) · Robotics 1 written 12 Feb |
| Time budget | ~5 h/week on the project through early January, then frozen until exams end |

### Course obligations (verbatim from the brief, all mandatory)

- Code in a notebook or IDE project, **GPU-ready, runs without error**, organised as
  `Imports → Globals → Utils → Data → Network → Train → Evaluation`
- GitHub repo containing: code, dataset or link, **presentation**, detailed README
- Plagiarism clause: no copying other students' work or online code verbatim, no content
  "highly retrieved from generative AI models". Consulting external resources is allowed

### Project 1 objectives → where they are covered

| Objective (brief) | Covered by |
|---|---|
| Dataset preparation: manageable NIGHTS subset, preprocessed to model input size | §6 |
| Model implementation: ViT backbone(s) in a DreamSim-like pipeline | §5.1, §7 |
| Efficient attention integration: replace self-attention with Meta / Pyra / MoH | §5.2 |
| Evaluation: 2AFC agreement + inference speed on a GPU-limited environment vs baseline | §8 |
| Ablation: across mechanisms and encoder configurations (all vs some layers), trade-offs | §7.3, §8.4 |

---

## 2. Why this project is framed the way it is

**Reference [2] is from Amerini's own group.** Schiavella, Cirillo, Papa, Russo, Amerini,
*Efficient attention vision transformers for monocular depth estimation on
resource-limited hardware*, Sci. Rep. 15:24001 (2025). The "Meta", "Pyra", "MoH" names in
the brief are that paper's names. Consequences:

1. Use the same names, definitions and analysis tools: full vs partial replacement, a
   **Pareto frontier** of quality vs inference time, and **embedding-entropy analysis**.
2. The brief's phrase "counter-effects on maintaining the informativeness of the
   embeddings … information dispersion" refers to that paper's finding: *"optimised
   encoders tend to generate embeddings that are less effective in guiding the depth map
   reconstruction process"*, leading to the hypothesis that modified encoders *"generate
   degraded latent representations"*. They measured dispersion with Gaussian
   differential entropy and the Kozachenko–Leonenko estimator; any encoder change made
   embeddings less compact. They state it as a hypothesis, not a proven cause.
3. Their speed measurements were on **a single CPU core**. We add that protocol (X270)
   next to the Colab GPU timing the brief asks for.

**The original angle this project contributes.** In depth estimation, the encoder feeds
a decoder that has to reconstruct from the embedding. DreamSim has no decoder: the
embedding *is* the output, compared directly with cosine distance. So their dispersion
hypothesis is genuinely open here — dispersion could hurt, be harmless, or even help
relative distances. Testing whether their finding transfers to a decoder-free
metric-learning setting is the central intellectual contribution, and it is directly
relevant to the professor's own research line.

---

## 3. Problem formulation

**Data.** A 2AFC triplet is $(x, \tilde x_0, \tilde x_1, y)$: reference $x$, two
variations, and the human choice $y \in \{0,1\}$ of which variation looks more similar.

**Metric.** An encoder $f_\theta$ maps an image to an embedding. Perceptual distance is
$$D(a,b) = 1 - \cos\big(f_\theta(a), f_\theta(b)\big).$$
The model predicts $\hat y = \mathbb 1[D(x,\tilde x_1) < D(x,\tilde x_0)]$ and the
**2AFC score** is the fraction of test triplets where $\hat y = y$.

**Training objective (DreamSim).** With $\Delta d = D(x,\tilde x_0) - D(x,\tilde x_1)$ and
$\bar y = 2y - 1$: $\;\mathcal L = \max(0,\; m - \bar y\,\Delta d)$, margin $m = 0.05$.

**Intervention.** For a mechanism $\mathcal M \in \{\text{Meta}, \text{Pyra}, \text{MoH}\}$
and a set of block indices $S \subseteq \{1,\dots,12\}$, $f_\theta^{\mathcal M,S}$ is the
encoder with the token mixer of every block in $S$ replaced by $\mathcal M$.

**Questions, as a bi-objective problem.** Maximise 2AFC, minimise cost
(latency, FLOPs, memory). Characterise the Pareto frontier over $(\mathcal M, S)$, and
explain the representational changes behind it.

### Research questions

- **RQ1 — Alignment.** How much human agreement (2AFC) is lost per mechanism and per
  replaced-block set?
- **RQ2 — Efficiency.** What FLOP, latency and memory gains result on a T4 GPU and on a
  single CPU core, and how do they scale with token count?
- **RQ3 — Dispersion.** Do modified encoders produce more dispersed embeddings, as
  Schiavella et al. found for depth encoders? Does dispersion track alignment loss when
  there is no decoder?
- **RQ4 — Trade-off.** Which configurations are Pareto-optimal?
- **Method ablation.** Does a label-free healing stage (§5.3) matter?

### Hypotheses (stated before any experiment; each is falsifiable)

| ID | Hypothesis | Basis |
|---|---|---|
| H1 | At 224 px, Pyra and MoH give encoder-level FLOP reductions under 10% and latency reductions within timing noise; only Meta gives a large reduction. Gains grow with resolution | §4 FLOP model |
| H2 | Replacing early blocks costs less 2AFC than replacing late blocks, most of all for Meta | Early ViT layers attend more locally (mean attention distance, Dosovitskiy et al. 2021; Raghu et al. 2021) |
| H3 | All-12 replacements increase embedding entropy relative to the baseline | Schiavella et al. 2025 |
| H4 | MoH stays within ~1 pp of the baseline but shows no wall-clock gain with a dense implementation | Attention is retained, only heads are gated |
| H5 | Meta on all 12 blocks drops below the zero-shot baseline (90.1) despite healing | Pooling-only mixing, small adaptation data |

A refuted hypothesis is a result, not a failure. Report all five either way.

---

## 4. Priors: where efficiency can and cannot come from

Computed by `scripts/flop_model.py` (multiply-adds × 2; ViT-B/16, D = 768, 12 heads,
MLP ratio 4). The baseline at 224 px is 35.1 GFLOPs = 17.6 GMACs, matching the
standard ViT-B/16 figure.

| Resolution | Tokens | Quadratic-attention share of a block | Meta last-4 / all-12 | Pyra (R=2) last-4 / all-12 | MoH 50% ideal last-4 / all-12 | MoH dense |
|---|---|---|---|---|---|---|
| 224 | 197 | **4.1%** | −11.9% / −35.8% | −2.3% / −7.0% | −3.3% / −9.9% | ±0% |
| 448 | 785 | 14.6% | −14.3% / −42.8% | −4.8% / −14.4% | −4.7% / −14.2% | ±0% |
| 672 | 1765 | 27.7% | −17.2% / −51.5% | −7.9% / −23.7% | −6.6% / −19.7% | ±0% |

What this means, before any code:

- At 224 px the cost is dominated by linear layers (QKV, output projection, MLP), not the
  $N^2$ term. Mechanisms that only shrink the $N^2$ term (Pyra, MoH) can barely help.
- **Meta** removes the QKV and output projections entirely, which is why it alone gives
  a large reduction. It should also cost the most accuracy.
- **MoH** saves compute only if unselected heads are actually skipped. Per-token routing
  is hard to make fast in PyTorch; a straightforward implementation computes all heads
  and masks them, giving zero saving (the "dense" column). Report FLOPs-ideal and
  measured latency separately and explain the gap.
- Only DINO ViT-B/16 is worth studying: CLIP and OpenCLIP B/32 run on 50 tokens, where
  the quadratic share is about 1% and efficient attention is moot.
- A resolution sweep (timing only, no retraining) shows where the crossover happens.
  This is the cheapest strong result in the project.

---

## 5. Method

### 5.1 Backbone and embedding (matches DreamSim's single-branch model)

| Element | Decision | Source |
|---|---|---|
| Backbone | DINO ViT-B/16, ImageNet-pretrained self-supervised weights (timm `vit_base_patch16_224.dino`) | DreamSim `dino_vitb16` |
| Input | 224×224, bicubic resize, ImageNet mean/std | DreamSim `get_preprocess_fn("DEFAULT")`, `_preprocess` |
| Embedding | **CLS token of the 12th block's output, before the final LayerNorm** | DreamSim paper; code extracts layer 11 (0-indexed) |
| Distance | Cosine distance | DreamSim |
| Fine-tuning | LoRA on `qkv` only, r = 16, α = 32, dropout 0.2; Adam, lr 3e-4, wd 0, batch 32, 8 epochs, margin 0.05 | DreamSim repo `configs/train_single_model_lora.yaml` |

Note the paper reports α = 0.5, dropout 0.3, batch 512 for single models, which differs
from the repo config. We follow the repo config and state the discrepancy.

Reproduction targets: zero-shot DINO **90.1**, LoRA-tuned DINO **94.6** (NIGHTS test,
DreamSim Table 5).

### 5.2 The three mechanisms

All three are drop-in replacements for the token mixer inside an existing block. The
block keeps its pretrained `norm1`, `norm2` and MLP.

**Meta** (MetaFormer / PoolFormer token mixer, Yu et al. 2022)
- Patch tokens reshaped to the 14×14 grid; mixer = `AvgPool3x3(x, stride 1, pad 1,
  count_include_pad=False) − x`, as in PoolFormer. No parameters.
- **CLS rule (required):** PoolFormer has no CLS token, but DreamSim's embedding *is* the
  CLS token. Without a rule, the CLS token receives no patch information in any
  Meta block — fatal when the last blocks are replaced. Rule: CLS mixer =
  `mean(patch tokens) − cls` (global average pooling, the 1×1 analogue).
- Deviation to declare: Schiavella et al. write Meta as `Pooling_k(x)`. We use
  PoolFormer's `pool − identity` form, which is what PoolFormer computes inside the
  residual.

**Pyra** (PVT v1 spatial-reduction attention, Wang et al. 2021)
- Queries from all tokens. Keys and values from a reduced sequence:
  `SR(x) = LN(Reshape(x, R) · W)`, kernel = stride = R on the patch grid.
- **R = 2** at 224 px (14×14 → 7×7). R must divide the grid size: valid values at 224
  are 2, 7, 14.
- **CLS rule:** CLS is kept unreduced and appended to the reduced key/value sequence.
- Init: Q, K, V and output projections from the pretrained `qkv`/`proj`. `W` initialised
  to average each R×R window, so at step 0 the module equals attention over
  window-averaged keys and values.
- Unit test: with R = 1 and identity `W`, output must equal the original attention.

**MoH** (Mixture-of-Head attention, Jin et al., ICML 2025)
- Heads treated as experts: **2 shared heads + top-4 of 10 routed heads = 6 of 12 active
  (50%)** per token. Head outputs combined by router weights instead of a plain sum.
- Init from the pretrained `qkv`/`proj`, router initialised so all heads start with
  weight ≈ 1. MoH's own paper shows pretrained MHA models can be continue-tuned into MoH,
  so this conversion has precedent.
- Load-balancing loss on the router, as in MoH, to prevent collapse onto a few heads.
  Log the head-usage histogram.
- Unit test: with all heads active and all weights = 1, output must equal the original
  attention.
- 75% (9 of 12) variant: Tier B only.

**Kernel fairness.** timm's baseline attention uses fused `scaled_dot_product_attention`.
Pyra and MoH must call the same fused kernel wherever their math allows, otherwise the
baseline gets a kernel advantage that has nothing to do with the mechanism.

### 5.3 Two-stage adaptation

Swapping a pretrained attention layer for a new module disrupts every layer after it.
NIGHTS has only 13,900 training triplets — not enough to retrain new token mixers from
the human labels alone. So:

**Stage A — healing (label-free distillation).**
- Teacher: frozen pretrained DINO. Student: the modified encoder.
- Loss: per-replaced-block normalised MSE between student and teacher hidden states, plus
  `1 − cos` between final CLS embeddings.
- Trainable: replaced blocks only (mixer, norms, MLP). Everything else frozen.
- Data: **NIGHTS train-split images only** (no labels used; val/test images never seen).
- Budget: 2 epochs, AdamW, lr 1e-4.
- Precedent: attention-transfer-then-LoRA conversions of pretrained transformers, e.g.
  LoLCATs (Zhang et al., 2024) — verify the reference before citing it in slides.

**Stage B — alignment (DreamSim objective).**
- Same recipe and budget as the baseline (§5.1): LoRA on all remaining `qkv`, plus
  replaced blocks fully trainable (lr 1e-4 for those parameters).
- Model selection on val 2AFC. **Test evaluated once per model, at the end.**

**Why the capacity control exists.** In DreamSim, LoRA targets `qkv` only. A Meta block
has no `qkv`, so replaced blocks are trained fully while the baseline only gets LoRA.
Any difference could then come from trainable capacity, not the mechanism. Control run
C1: baseline with blocks 9–12 fully trainable plus LoRA elsewhere.

---

## 6. Data

### Verified facts

- NIGHTS `data.csv` (checked 26 Nov 2026): **20,019 rows**, generated with Stable
  Diffusion 2.1, images **768×768**. Unfiltered split: train 15,941 · val 1,958 · test
  2,120. Some rows have fewer than 6 votes (2,575 of them).
- After DreamSim's `votes >= 6` filter: **17,444 triplets → train 13,900 · val 1,720 ·
  test 1,824** (test: 1,041 ImageNet / 783 non-ImageNet via `is_imagenet`; 5,472
  distinct test images). These are the counts this project uses. Whether the published
  90.1 / 94.6 are on 1,824 or 2,120 test triplets is settled by B-rel (O4): DreamSim's
  code filters, so 1,824 is expected.
- Columns: `id, left_vote, right_vote, votes, ref_path, left_path, right_path, split,
  is_imagenet, prompt`. `left_vote`/`right_vote` are 0/1. DreamSim reads its target
  from position 2, which is **`right_vote`**; ref/left/right paths are positions 4–6.
  Our code reads columns by name.
- Category = `prompt` (1,448 categories in filtered train; median 9 triplets each,
  median 1 in val and test).
- Full archive `nights.zip`: **61.5 GB** (58 GB unzipped), served with HTTP range
  support.
- Storage (checked 26 Nov): Colab VM disk **113 GB** free; Drive is the institutional
  account, **800 GB** per-user quota. Storage no longer limits the dataset choice; the
  chunked pipeline stays because training needs the small 224 px copy on local disk.
- A chunked download exists: 100 `ref` + 100 `distort` zips from
  `data.csail.mit.edu/nights/nights_chunked/` (DreamSim `dataset/download_chunked_dataset.sh`).
  It contains no `data.csv` (`nights_chunked/data.csv` and `nights/data.csv` are 404).
  `scripts/fetch_data_csv.py` extracts `nights/data.csv` (1.9 MB) from `nights.zip`
  with range requests and checks its sha256.
- Every chunk directory holds kept triplets (141–218 each), so filtering cannot skip
  downloads; it skips 7,725 of 60,057 images at resize time.
- DreamSim code license: MIT.

### Unverified — Day-1 checks

1. **Size of the 224 px copy and time to copy it from Drive to `/content`** (the
   per-session setup cost; decides whether train stays full).

### Preparation pipeline (one Colab job, resumable per chunk)

```
for each chunk: download → resize to 224 bicubic every image of a votes ≥ 6 triplet
(identical to DreamSim's online preprocessing, so this is a no-op deviation) → save
lossless PNG → delete chunk → record progress
```

- Store on Drive; copy to `/content` at the start of each session (Colab has 2 vCPUs, so
  decoding 768 px PNGs online would bottleneck training).
- "Manageable subset", as the brief puts it: the 224 px lossless copy replaces the 58 GB
  originals. **Test and val are always full**, for comparability with published numbers.
  **Train is used in full** by default. Fallback: subsample train (stratified by
  category) only if copying the 224 px copy to `/content` each session is impractical
  (O2). If so, run the baseline once on full train too, to show subsampling isn't what
  explains any gap.
- Verify after preparation: triplet counts per split equal 13,900 / 1,720 / 1,824; every
  path exists; checksum the manifest.

### Subsets

Generated by `scripts/make_subsets.py` from `data.csv`, committed in `data/splits/`.

| Subset | Split | Size | Unit | File |
|---|---|---|---|---|
| smoke | train / val | 256 / 128 | triplets | `smoke_train.csv`, `smoke_val.csv` (ids) |
| dev | train / val | 4,000 / 500 | triplets | `dev_train.csv`, `dev_val.csv` (ids) |
| full | train / val / test | 13,900 / 1,720 / 1,824 | triplets | `data.csv`, `votes >= 6` |
| probe | test | 1,000 | images | `probe.csv` (paths) |
| timing | test | 64 | images | `timing.csv` (paths) |

- Filter `votes >= 6` first. The category column is found by name; the script stops if
  it isn't there rather than guessing.
- Rank each triplet by `sha1(f"{seed}:{id}")`, seed 1234. Within each category, a
  triplet's key is its rank quantile `(rank + 0.5) / N_category`; sorting by key gives
  one order per split. Every subset is a prefix of that order, so totals are exact,
  category shares match the full split to within one per category, and
  **smoke ⊂ dev ⊂ full**.
- Probe: all distinct test images (ref, left, right), stratified the same way by the
  category of their triplet, keyed by `sha1(f"{seed}:{path}")`. Timing = the first 64
  of the probe order, so **timing ⊂ probe**. Used by §8.3 (per-block features) and §8.2.
- smoke and dev are for development only. **Reported numbers always use full val and
  test.**
- `--available` keeps only triplets whose three images exist on disk, for use while the
  download is running. Its output goes to `data/splits/available/` (not committed) and
  is not the final selection.

---

## 7. Experiments

### 7.1 Baselines

| ID | Model | Purpose |
|---|---|---|
| B0 | Pretrained DINO, zero-shot | Lower bound; pipeline check (target 90.1) |
| B1 | DINO + LoRA (§5.1), **3 seeds** | Reference for every comparison (target 94.6); seed spread = noise floor |
| B-rel | Released DreamSim `dino_vitb16` checkpoint, scored with *our* eval code | Validates the evaluation pipeline only (must give 94.6). Not a comparison baseline |

### 7.2 Run matrix

All variant runs = Stage A + Stage B, 1 seed, unless stated.

| Tier | ID | Mechanism | Blocks replaced | Healing | Answers |
|---|---|---|---|---|---|
| A | M-L4 | Meta | 9–12 | yes | RQ1, RQ3 |
| A | M-A12 | Meta | 1–12 | yes | RQ1, RQ3, H5 |
| A | P-L4 | Pyra R=2 | 9–12 | yes | RQ1 |
| A | P-A12 | Pyra R=2 | 1–12 | yes | RQ1, RQ3 |
| A | H-L4 | MoH 50% | 9–12 | yes | RQ1, H4 |
| A | H-A12 | MoH 50% | 1–12 | yes | RQ1, RQ3 |
| B | M-F4, P-F4, H-F4 | each | 1–4 | yes | H2 (layer position) |
| B | M-L4-nh, P-L4-nh, H-L4-nh | each | 9–12 | **no** | Healing ablation |
| B | C1 | none | 9–12 fully trainable | — | Capacity control |
| B | H75-L4 | MoH 75% | 9–12 | yes | Head-budget ablation |
| C | timing sweep | all mechanisms × configs | — | — | RQ2 at 224/448/672/896 px (patch grids 14/28/42/56; 336 px gives 21×21, which Pyra's R=2 can't halve). **No training needed**: latency doesn't depend on trained weights |

Tier A = 9 trained runs (B1 × 3 + 6 variants). Estimated 1–1.5 GPU-hours each on a T4
(to be measured on Day 1), ≈ 12 GPU-hours total. Tier B runs only after Tier A is
complete.

Machine-readable version: `configs/experiments.yaml`.

### 7.3 What the ablations isolate

| Contrast | Isolates |
|---|---|
| X-L4 vs X-A12 | How much replacement (the brief's "all vs some layers") |
| X-F4 vs X-L4 | Where in the encoder (H2) |
| X-L4 vs X-L4-nh | Value of healing |
| X-L4 vs C1 | Mechanism vs extra trainable capacity |
| H-L4 vs H75-L4 | Head budget |
| Meta vs Pyra vs MoH | Mechanism family |

---

## 8. Evaluation

### 8.1 Alignment

- 2AFC on the full test set: overall, ImageNet subset, non-ImageNet subset.
- **Statistics:** 95% bootstrap CI (10,000 resamples over test triplets); paired
  McNemar test against B1, Holm-corrected across comparisons. With n = 1,824 and accuracy
  near 94%, the standard error is ≈ 0.5 pp, so **differences under ~1.5 pp are not
  interpretable**. B1's 3-seed spread is reported next to every comparison.
- Context: majority labels on the *unfiltered* 100k NIGHTS corpus reproduce only ~70% of
  the time ("Cosine is Human", NeurIPS 2026). The standard benchmark uses the unanimous
  subset, so the ceiling there is high. Mention this when discussing the top of the
  range, not as an excuse.

### 8.2 Efficiency

| Measure | Protocol |
|---|---|
| GPU latency/throughput | Colab T4; `eval()` + `inference_mode`; fp16 autocast **and** fp32; batch 1 and 64; 25 warm-up + 200 timed iterations with CUDA events; median + IQR |
| GPU fairness | All models timed **in one session**, random order, baseline timed at start and end (drift check). Record GPU name, driver, torch/timm versions, SDPA backend |
| CPU latency | X270, `torch.set_num_threads(1)`, batch 1, fp32, 20 warm-up + 100 runs, median; on mains power, cool-down between models (mirrors Schiavella et al.'s single-core protocol) |
| FLOPs | `torch.utils.flop_counter.FlopCounterMode`, cross-checked against `scripts/flop_model.py` |
| Memory, params | `torch.cuda.max_memory_allocated`, parameter count |

Never compare timings across sessions without re-timing the baseline in the same session.

### 8.3 Dispersion (RQ3) — runs on the laptop from saved embeddings

- **Save for every trained model:** test-set final CLS embeddings (5,472 × 768, fp16,
  ≈ 8 MB) and per-block CLS + mean-patch features on a fixed 1,000-image subset. All
  analysis below then needs no GPU.
- Gaussian differential entropy, $\tfrac12\log\det(2\pi e\,\Sigma)$, on L2-normalised
  embeddings, PCA to 64 dimensions, Ledoit–Wolf shrinkage.
- Kozachenko–Leonenko k-NN entropy (k = 5) in the same PCA space.
- Effective rank of the covariance (Roy & Vetterli 2007).
- Linear CKA against B1, per block (Kornblith et al. 2019): shows *where* representations
  diverge.
- Relationship to alignment: Spearman correlation between Δentropy and Δ2AFC across
  variants — descriptive only, n is small.
- Pitfalls handled: entropy depends on scale (hence L2 normalisation); the k-NN
  estimator is biased in high dimensions (hence PCA); n ≈ 6k vs d = 768 (hence
  shrinkage). Report both estimators; treat agreement between them as the signal.

### 8.4 Trade-off (RQ4)

Pareto frontiers, 2AFC (↑) against: T4 fp16 batch-64 latency, CPU single-core latency,
and FLOPs. A model is dominated if another is at least as good on both axes and strictly
better on one (Schiavella et al.'s definition).

---

## 9. Risk register

| # | Risk | Effect if ignored | Mitigation |
|---|---|---|---|
| R1 | 58 GB dataset vs Colab/Drive limits | Can't start | Chunked download + resize-on-the-fly (§6); 800 GB institutional Drive and 113 GB VM disk remove the hard limit |
| R2 | `data.csv` missing from chunked download | No labels | Confirmed missing; extracted from `nights.zip` with `remotezip` (§6) |
| R3 | Colab free volatility: no GPU, timeouts, quotas | Lost runs, slipped schedule | Resumable runs, checkpoint to Drive every epoch, short jobs, run log; Kaggle as backup |
| R4 | 2 vCPUs → dataloader bottleneck | GPU idle, runs 3–5× slower | Pre-resized lossless data copied to local disk |
| R5 | Baseline doesn't reproduce 94.6 | Every comparison is suspect | Gate G1. Score the released checkpoint (B-rel) with our eval code to separate eval bugs from training differences |
| R6 | New modules wreck pretrained features | Big drops that say nothing about the mechanism | Init close to the original (§5.2), healing (§5.3), healing ablation |
| R7 | LoRA's `qkv` target vanishes in replaced blocks | Mechanism confounded with capacity | Capacity control C1 |
| R8 | CLS token mishandled in spatial mixers | Silent collapse of the embedding | Explicit CLS rules + unit tests (§5.2) |
| R9 | FLOPs ≠ latency (MoH dense, pooling memory-bound, fused kernels) | Wrong efficiency claims | Report FLOPs, GPU and CPU latency separately; explain gaps |
| R10 | Timing noise on shared GPUs | Spurious speedups | §8.2 protocol: same session, drift check, medians |
| R11 | Low statistical power (SE ≈ 0.5 pp) | Over-reading noise | CIs, McNemar + Holm, B1 seeds |
| R12 | Entropy estimator artefacts | False dispersion claims | §8.3 pitfalls |
| R13 | MoH router collapse on small data | MoH looks worse than it is | Load-balancing loss, head-usage logging |
| R14 | Library drift (timm, peft) | Broken environment mid-project | Pin versions once the first Colab environment works; don't depend on the `dreamsim` package beyond B-rel |
| R15 | Test leakage | Inflated results | Healing uses train images only; selection on val; test touched once per model |
| R16 | Plagiarism clause | Failing grade | Own code for data, training, mixers; cite every consulted repo in the README; student writes README and slide prose |
| R17 | Over-claiming at high resolution | Indefensible claims | Accuracy only at 224; higher resolutions are timing-only, stated as such |
| R18 | Exam calendar overrun | Project or exam suffers | Freeze dates, Tier A/B/C priorities, gates (`PLAN.md`) |

---

## 10. Out of scope (cut deliberately)

The DreamSim ensemble · CLIP/OpenCLIP backbones · other efficient-attention families
(linear attention, windowed attention, token merging — mention as future work) · training
or accuracy evaluation at resolutions above 224 · other perceptual datasets (BAPPS,
THINGS) · quantisation, pruning, distillation into smaller backbones · a LaTeX paper.

---

## 11. Open items — resolved by checks, not opinion

| # | Item | Resolved by | When |
|---|---|---|---|
| O1 | `data.csv` source in chunked path | **Resolved 26 Nov**: only in `nights.zip`; `scripts/fetch_data_csv.py` (§6) | — |
| O2 | Size of the 224 px copy and Drive → `/content` copy time → full train or subset | Measure on chunk 000, extrapolate ×100 | P1 day 1 |
| O3 | T4 time per epoch → is Tier A feasible as estimated | First B1 run | P1 |
| O4 | Our eval reproduces the released checkpoint (B-rel = 94.6) | B-rel | P1 |
| O5 | March presentation date and form deadline | Course page / Amerini | This week |
| O6 | Do written and project each need 18/30 independently | TA | This week |

---

## 12. Presentation mapping (course's suggested scheme)

| Slide section | Content source |
|---|---|
| Problem statement | §3 |
| State of the art | DreamSim, MetaFormer, PVT, MoH, Schiavella et al. 2025 |
| Proposed method | §5 (mechanisms, CLS rules, healing) |
| Dataset | §6 |
| Experimental setup | §7, §8.2 |
| Model evaluation | 2AFC table with CIs, timing table, resolution sweep, Pareto plots, dispersion and CKA plots |
| Conclusions | Hypotheses H1–H5 confirmed/refuted; does the dispersion finding transfer without a decoder? |

---

## 13. References

1. Fu, Tamir, Sundaram, Chai, Zhang, Dekel, Isola. *DreamSim: Learning New Dimensions of
   Human Visual Similarity using Synthetic Data.* NeurIPS 2023. arXiv:2306.09344.
   Code (MIT): github.com/ssundaram21/dreamsim
2. Schiavella, Cirillo, Papa, Russo, Amerini. *Efficient attention vision transformers for
   monocular depth estimation on resource-limited hardware.* Sci. Rep. 15:24001 (2025).
   doi:10.1038/s41598-025-06112-8
3. Schiavella, Cirillo, Papa, Russo, Amerini. *Optimize Vision Transformer Architecture via
   Efficient Attention Modules: A Study on the Monocular Depth Estimation Task.* ICIAP 2023
   Workshops, LNCS 14365, pp. 383–394 (Meta-METER, Pyra-METER).
4. Yu et al. *MetaFormer Is Actually What You Need for Vision.* CVPR 2022. arXiv:2111.11418
5. Wang et al. *Pyramid Vision Transformer.* ICCV 2021.
6. Jin, Zhu, Yuan, Yan. *MoH: Multi-Head Attention as Mixture-of-Head Attention.* ICML 2025.
   arXiv:2410.11842
7. Caron et al. *Emerging Properties in Self-Supervised Vision Transformers* (DINO). ICCV 2021.
8. Dosovitskiy et al. *An Image is Worth 16x16 Words.* ICLR 2021 (mean attention distance).
9. Raghu et al. *Do Vision Transformers See Like Convolutional Neural Networks?* NeurIPS 2021.
10. Kornblith et al. *Similarity of Neural Network Representations Revisited* (CKA). ICML 2019.
11. Roy & Vetterli. *The effective rank.* EUSIPCO 2007.
12. Kozachenko & Leonenko. *Sample estimate of the entropy of a random vector.* 1987.
13. Ledoit & Wolf. *A well-conditioned estimator for large-dimensional covariance matrices.* 2004.
