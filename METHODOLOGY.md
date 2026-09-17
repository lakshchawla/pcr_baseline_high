# Methodology: Part-Aware, Semantically-Regularized Person Re-Identification

Last reset: 2026-08-29. This document describes the training pipeline as it actually exists in
this repo today, stage by stage. It replaces a previous version that had drifted from the code in
several places (stale loss temperature, a removed upstream filtering step, an incomplete Stage 1
negative-pool description, and a wrong foreground loss term) — everything below was checked
directly against the current code, not carried over from the old draft.

Three techniques are combined: **BPBreID** (WACV 2023) for part-attention pooling and visibility
scoring, **CLIP-ReID** (AAAI 2023) for CLIP-text-anchored prompt learning (generalized here to
per-body-part prompts), and **SPCL** (NeurIPS 2020) for self-paced unsupervised domain adaptation.

## 1. Notation

| Symbol | Meaning |
|---|---|
| $I$ | an input person image |
| $F$ | the backbone's spatial feature map, $\mathbb{R}^{H\times W\times C}$ |
| $K$ | number of body parts (5 in every config in this repo) |
| $M = 1+K$ | number of branches: index 0 is foreground/global, 1..K are the parts |
| $A_m$ | branch $m$'s spatial attention map (softmax over pixel-classifier logits) |
| $f_m$ | branch $m$'s pooled embedding, $\mathbb{R}^D$ |
| $v_m$ | branch $m$'s visibility score, continuous in Stage 0/1/2, in $[0,1]$ |
| $y_i$ | identity label of image $i$ |
| $\tau$ | a softmax temperature (each loss below states its own) |

## 2. Stage 0 — BPA Segmentation Pretraining

`examples/train_bpa_segmentation.py`. Trains BPBreID's pixel-to-part classifier as a plain
supervised segmentation task against PifPaf/MaskRCNN-derived ground-truth part masks
(`BodyPartAttentionLoss`, cross-entropy per pixel), before any CLIP/prompt machinery exists. This
is the only signal in the whole pipeline that ties branch index $m$ to a specific real anatomical
region — Stage 1's prompts use placeholder context tokens (no body-part words), and Stage 2/3's
id/triplet/align gradients have no notion of "this pixel is a knee." Runs only on Market1501
(DukeMTMC-reID has no masks on disk). Output: a plain BPBreID checkpoint, loadable as
`model.checkpoint_path` by every later stage.

## 3. Part-Based Visual Encoder

`pcr/models/bpbreid_encoder.py`, wrapping BPBreID (`third_party/torchreid`).

1. **Backbone** produces $F \in \mathbb{R}^{H\times W\times C}$ (HRNet32 or ResNet50).
2. **Pixel classifier (BPAM)** is a $1\times1$ conv over $F$ producing $M$ logit channels,
   softmaxed into attention maps $A_0 \ldots A_K$ (branch 0 = background).
3. **GWAP** (gated weighted average pooling) pools $F$ once per branch, weighted by that branch's
   own attention map: $f_m = \text{pool}(A_m \odot F)$. This is the only place $F$ and the
   attention maps are combined — the encoder's own input is just the raw image; $F$ and
   $A_0\ldots A_K$ are both produced internally, one from the other, not supplied externally.
4. **Visibility** $v_m$ is each map's peak confidence, $\max_{h,w} A_m[h,w]$ — continuous in every
   stage in this repo (the binary/argmax mode BPBreID also supports is never used here).

Every branch's embedding is L2-normalized independently before leaving the encoder.

## 4. Stage 1 — Per-Part Prompt Learning

`examples/train_relational_prompts.py`, config `configs/stage1_relational_prompts.yaml`. The
backbone/BPAM (from Stage 0) and CLIP's text encoder are both frozen; only a per-identity prompt
context and two small attention blocks train.

**Prompt construction** (`pcr/models/prompt_learner.py`). Each identity owns one learnable context
tensor, `ctx`, covering all $M=1{+}K$ branches uniformly (global/foreground + $K$ parts) — laid out
as one flat sequence so they can all attend to each other. It splices into the fixed template
`"A photo of a [ctx] person."` before the frozen CLIP text encoder ever sees it. (Earlier versions
of this design kept the global/foreground branch as a separate, untrained `fg_ctx` outside the
relational-mixing step entirely; unified into `ctx` so branch 0 receives real gradient like every
part, and CAB/$\mathcal{L}_{\text{align}}$ in Stage 2 don't have to special-case a meaningless
anchor for it.)

**TextualAttentionBlock (TAB)** (`pcr/models/relation_blocks.py`) mixes `ctx`'s $M$ blocks via
self-attention before any branch's prompt is assembled, so one branch's context can be informed by
the others. It has no per-image signal to work with (`ctx` is indexed by identity alone), so it
uses each identity's *mean* per-branch visibility across every cached training image of that
identity (`compute_identity_visibility`) as a soft attention-score bias — an unreliable branch
contributes less as a key to every other branch's post-attention representation. This table is
saved (`identity_visibility.pth`) and reused unchanged by `cache_text_anchors.py`, so TAB's output
stays a deterministic function of identity alone everywhere it's used.

**VisualAttentionBlock (VAB)**, the image-side counterpart, uses the same attention-bias mechanism
but with each image's own real per-branch visibility (available fresh every forward call), also
over all $M$ branches uniformly. A zero-init tanh-gated residual keeps it a no-op at
initialization; VAB is the one Stage 1 module that survives into Stage 2.

**Loss — SupConLoss** (`pcr/loss/clip_supcon_loss.py`), CLIP-ReID's actual multi-positive
contrastive loss, not plain InfoNCE (PK-sampled batches put several images of the same identity in
one batch on purpose; SupCon's positive mask already handles that, InfoNCE would need a
deduplication patch that throws away all but one photo's gradient). Both i2t and t2i directions are
computed for every branch $m \in \{0..K\}$ (global/foreground included), with learnable $\tau$:

$$\mathcal{L}_{\text{SupCon}} = -\sum_{i} \frac{v_i}{\sum_j v_j} \cdot \frac{1}{|P(i)|}\sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a} \exp(z_i \cdot z_a / \tau)}$$

where $P(i)$ is every same-identity row on the other side, and each anchor is weighted by its own
visibility ($v_i$, detached, floored at $10^{-3}$). $\tau$ is **learnable** — parameterized as
$\tau = 1/\exp(s)$ with $s$ (`log_logit_scale`) a trained parameter, clamped so $\exp(s)\le100$
(i.e. $\tau \ge 0.01$), following CLIP's own `logit_scale` convention
(`third_party/clip/model.py`). Initialized at $\tau=0.1$ (not SupCon's original $1.0$, which
produced a near-random-guess loss for the first two epochs against this loss's full-identity-table
scale — see below) and logged every epoch.

**Full-identity-table negative pool.** Both directions compare against the *entire* training set,
not just the current PK batch (751 identities / 12936 images on Market1501) — matching CLIP-ReID's
own original design. i2t achieves this by splicing each batch's fresh, differentiable text row
together with a once-per-epoch snapshot of every *other* identity's text embedding
(`build_text_snapshot`, rebuilt under `no_grad` at the start of each epoch against that epoch's
current TAB/`ctx` weights). t2i simply compares against the whole cached feature set directly,
since the backbone is frozen and `cached_features` never goes stale.

**No upstream visibility filter.** Every cached image trains, unconditionally — the old
image-level filter (`pcr/utils/visibility_filter.py`, deleted) discarded 61% of Market1501's
training images and was found to be driven by an undertrained BPAM signal rather than genuine
occlusion. Reliability is handled entirely by the two mechanisms above: attention-level masking
(VAB/TAB) and loss-level per-row weighting (SupCon's `weights`).

**$\mathcal{L}_{\text{relalign}}$**: both VAB and TAB now return their own attention pattern
alongside their mixed output (`VisualAttentionBlock`/`TextualAttentionBlock.forward` →
`(mixed, attn)`; TAB's raw $M{\cdot}n_{ctx} \times M{\cdot}n_{ctx}$ pattern is averaged down to
$M\times M$ per branch block first, to match VAB's native shape). A KL-divergence term
(`pcr/loss/cross_attn_align_loss.py`, the same function Stage 2's $\mathcal{L}_{\text{crossalign}}$
uses) pulls VAB's per-image pattern toward TAB's per-identity one (detached — this trains VAB, not
TAB), on the same warmup/ramp/flat schedule shape as Stage 2's. Guards against both blocks
converging to a degenerate, near-identical relational pattern across branches — a real failure mode
distinct from what SupCon's own gradient already discourages, since SupCon never looks at either
block's *attention pattern*, only its final mixed output.

**Outputs**: `prompt_learner.pth`, `vab.pth`, `identity_visibility.pth` — consumed by
`examples/cache_text_anchors.py`, which builds two frozen, per-identity lookup tables for Stage 2:

- `text_prototypes.pth`: `[num_identities, M, D]`, each identity's per-branch pooled text embedding
  (L2-normalized), branch 0 (global/foreground) included — a real, SupCon-trained anchor, read
  by Stage 2's $\mathcal{L}_{\text{align}}$ like every other branch.
- `text_self_attention.pth`: `[num_identities, M, M]`, each identity's own CLIP text-transformer
  self-attention among those $M$ branch embeddings (see Stage 2 / CAB, below) — computed by
  replaying the text transformer's last block with `need_weights=True` (third_party/clip hardcodes
  it off) on this short $M$-token sequence. Only valid when the CLIP arch's `transformer_width`
  equals its `embed_dim` (true for ViT-B/16 and ViT-B/32, the two archs this repo actually uses —
  see `pcr/models/clip_text_encoder.py`'s own docstring on why those differ for the RN* archs).

Both tables are deterministic functions of identity alone (`prompt_learner`/TAB/the text encoder
are fully frozen once Stage 1 ends), which is exactly why they're computed once here rather than
recomputed inside Stage 2's training loop.

## 5. Stage 2 — Supervised Backbone Fine-Tuning

`examples/train_relational_finetune.py`, config `configs/stage2_relational_finetune.yaml`. The
backbone + BPAM unfreeze here (initialized from the *same* Stage 0 checkpoint Stage 1 used going
in); VAB continues training from its Stage 1 weights. Stage 1's prompt learner and the CLIP text
encoder are never loaded — only the two frozen tables they produced.

Per iteration: `combined = VAB(f_0..f_K, v_0..v_K)`, one `[B, M, D]` tensor — VAB mixes all $M$
branches uniformly (see Stage 1, above).

**CrossAttentionBlock (CAB)**, new in this design (`pcr/models/relation_blocks.py`). Stage 1's
frozen text anchors describe BPAM's part definitions *as they were at the end of Stage 1* — but
Stage 2 keeps training BPAM further via id/triplet gradients that know nothing about real part
boundaries, so by late Stage 2 the anchors can describe a stale convention. CAB grounds the visual
branches against text every step instead of relying on a single static cosine-similarity pull:

$$\text{vis\_grounded}, A_{i2t} = \text{CAB}(\text{query}=\texttt{combined}, \text{context}=\texttt{prompt\_feats})$$

where `prompt_feats = text_prototypes[targets]` — no live CLIP forward pass needed, since that
table is already the exact per-branch embedding for this batch's real identities. CAB is a
standard multi-head cross-attention block with a zero-init tanh-gated residual (identity function
at init, same device as VAB's gate). `vis_grounded` (not raw `combined`) is what `L_align` reads
below. (A symmetric text-queries-image direction is deliberately not built — it would have no
consumer yet, so it isn't computed at all.)

**$\mathcal{L}_{\text{crossalign}}$** (`pcr/loss/cross_attn_align_loss.py`) keeps CAB's grounding
meaningful rather than an arbitrary learned reweighting: it's a row-wise KL divergence pulling
CAB's own image-queries-text attention pattern $A_{i2t}$ toward CLIP's real internal text
self-attention pattern for that identity (`text_self_attention[targets]`, precomputed in Stage 1 —
see above). Ramped in on a schedule, not from epoch 0, since CAB starts as a no-op (gate$=0$) and
benefits from a few epochs of plain $\mathcal{L}_{\text{align}}$ pressure first:

$$\lambda_{\text{crossalign}}(e) = \begin{cases} 0 & e < 0.25E \\ \lambda_{\max}\cdot\frac{e - 0.25E}{0.15E} & 0.25E \le e < 0.40E \\ \lambda_{\max} & e \ge 0.40E\end{cases}$$

($E$ = total epochs, $\lambda_{\max}=0.3$ by default). Verified directly: during the warmup phase
CAB's gate stays at exactly 0 (no gradient reaches it, since $\lambda_{\text{crossalign}}=0$ and
$\mathcal{L}_{\text{align}}$'s gradient through a gate of exactly 0 vanishes too), then starts
moving the moment the schedule activates.

**The other four loss terms**, matching Algorithm 2's own step 16 (all weights configurable, all
default to an implicit weight of 1 except where noted):

- $\mathcal{L}_{\text{id\_global}}$ — cross-entropy on the global (branch 0) classifier, post-BNNeck.
- $\mathcal{L}_{\text{tri\_global}} + \mathcal{L}_{\text{tri\_parts}}$ — batch-hard triplet, computed
  once on branch 0 alone and once per part branch (summed) — not one fused multi-branch call.
  Visibility gates triplet with a loose boolean floor (`triplet_visibility_min`, default 0.05) since
  batch-hard mining's max/min doesn't compose with continuous weights the way the others do.
- $\mathcal{L}_{\text{align}}$ (`pcr/loss/clip_cosine_align_loss.py`) — softmax classification of
  each branch's `vis_grounded` feature against that branch's full frozen prototype table (every
  other identity is an implicit negative), weighted per-branch by continuous visibility. All
  $M=1{+}K$ branches, global/foreground included — `text_prototypes[:,0,:]` is a real,
  SupCon-trained anchor since Stage 1's unification (see Stage 1, above), not the untrained noise
  an earlier version of this design left it as.
- $\mathcal{L}_{\text{attn}}$ (`BodyPartAttentionLoss`, same pixel-supervision loss as Stage 0) —
  mandatory whenever masks are configured, so BPAM can't drift arbitrarily far from Stage 0/1's
  anatomical convention while everything else keeps training it. Its weight now cosine-decays
  (`bpa_weight_initial → bpa_weight_floor` over `bpa_weight_decay_epochs`) rather than staying flat,
  so this constraint loosens gradually instead of applying uniform pressure for the whole run.

**BoT additions** (Luo et al., CVPRW 2019), not in Algorithm 2's own literal steps: BNNeck (one
BatchNorm1d per branch between the pooled feature and whichever of id/align reads it — triplet
keeps reading the pre-BN feature) and a 10-epoch linear LR warmup before the just-unfrozen backbone
and its newly-interacting losses hit full LR.

**$\mathcal{L}_{\text{i2tce}}$, superseded**: a separate i2t classification loss for the global
branch against a frozen text anchor was considered alongside CAB, but its only candidate anchor
(`text_prototypes[:,0,:]`) was, at the time, untrained noise (`fg_ctx` had no Stage 1 loss term).
Stage 1's branch-0 unification (above) resolves this at the root: $\mathcal{L}_{\text{align}}$
already covers branch 0 with a real anchor now, making a separate $\mathcal{L}_{\text{i2tce}}$
unnecessary.

## 6. Stage 3 — Self-Paced Unsupervised Domain Adaptation

`examples/train_uda.py` / `examples/train_usl.py`, via `pcr/trainers.py`. Optional final stage:
adapts a Stage-2-finetuned model to an unlabeled target domain (UDA) or refines it with no labels
at all (USL). Confirmed present in the current code: DSBN (`pcr/models/dsbn.py`, domain-specific
BatchNorm so source and target statistics don't interfere), a part-aware hybrid memory
(`pcr/models/hm.py::PartHybridMemory`), DBSCAN clustering with Jaccard-distance re-ranking
(`pcr/utils/jaccard_rerank.py`) to generate pseudo-labels each epoch, and self-paced sample
reliability weighting so noisy pseudo-labels count for less. This stage is untouched by the CAB
work above — it loads a Stage 2 checkpoint's `state_dict` only, and never touches VAB/CAB/the text
side at all.

## 7. Fusion Module

`pcr/models/fusion_head.py::FusionHead` — combines the global and part embeddings into one
inference-time descriptor (fixed-weight or learned-gate). **Currently dormant**: it exists and is
unit-testable on its own, but has zero call sites anywhere in the training or evaluation scripts.
Wiring it in is a separate, unscoped piece of future work.

## 8. Evaluation Protocol

`pcr/evaluators.py::Evaluator` — standard person-ReID retrieval evaluation (mAP, CMC/Rank-k) run
periodically during Stage 2/3 and once at the end against the best checkpoint.

## References

- Somers et al., "Body Part-Based Representation Learning for Occluded Person Re-Identification"
  (BPBreID), WACV 2023.
- Li et al., "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without
  Concrete Text Labels", AAAI 2023.
- Ge et al., "Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object
  Re-ID" (SPCL), NeurIPS 2020.
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP),
  ICML 2021.
- Khosla et al., "Supervised Contrastive Learning" (SupCon), NeurIPS 2020.
- Luo et al., "Bag of Tricks and a Strong Baseline for Deep Person Re-identification" (BoT),
  CVPRW 2019.
