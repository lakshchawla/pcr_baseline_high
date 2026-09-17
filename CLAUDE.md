# PCR — Part-based Contrastive/CLIP Re-ID (`pcr_attn` fork)

## What this is

PCR: a part-based person re-identification pipeline combining BPBreID's part-attention feature
pooling with CLIP's frozen visual/text towers (CLIP-ReID-style prompt learning), trained through
four stages. Package is `pcr/`, driver scripts are in `examples/`, per-stage configs in `configs/`.

**This repo is a fork, not the main line.** It was branched from `pcr2`
(`/home/lakshh/workspace/reid/pcr2`) at commit `410d465` — the pre-"audit" baseline that produced
the best mAP/R1 run so far (82.2/91.8 on Market1501, RN50). `pcr2` itself moved on from there
through a large training-recipe overhaul and an attention-architecture rewrite (foreground branch
removed entirely, CAB moved into Stage 1, etc.) — none of that history is in this fork. This repo
exists to test one specific, isolated hypothesis on top of the *known-good* baseline instead of
on top of `pcr2`'s current (unvalidated) state. Origin remote: `github.com/lakshchawla/pcr_attn`,
branch `baseline-410d465`.

## Stage pipeline

0. **`train_bpa_segmentation{,_rn50,_vit}.py`** — pretrain BPBreID's pixel-to-part classifier
   against real part masks (Market1501 only). Produces a checkpoint consumed by every later stage.
1. **`train_relational_prompts.py`** — CLIP backbone + text encoder frozen; only per-identity
   prompt context (`PromptLearner.ctx`), `TextualAttentionBlock` (TAB), `VisualAttentionBlock`
   (VAB), and (this fork) `AttentionPoolingBlock` train, via SupCon loss (full-identity-table
   negative pool). Saves `prompt_learner.pth`, `vab.pth`, `pool.pth`, `identity_visibility.pth`.
2. **`cache_text_anchors.py`** — one-shot: loads Stage 1's frozen `PromptLearner`, builds
   `text_prototypes.pth` (per-identity per-branch text embeddings) for Stage 2 to read.
3. **`train_relational_finetune.py`** — backbone + BPAM unfreeze; VAB/pool continue training from
   Stage 1's weights; `CrossAttentionBlock` (CAB) grounds visual branches against the frozen text
   table. Losses: id/triplet (global + per-part), cosine align, cross-attention align, BPA.
4. **`train_uda.py` / `train_usl.py`** — optional Stage 3 domain adaptation (SpCL-style hybrid
   memory + DBSCAN). Independent of VAB/CAB/pool — not touched by this fork's changes.

## Key modules

- `pcr/models/clip_dense_part_encoder.py`, `clip_rn50_bpam_encoder.py`, `clip_vit_bpam_encoder.py`
  — frozen CLIP visual towers wrapped with BPBreID's `PixelToPartClassifier`. Produces per-branch
  pooled embeddings `f_out [B, M, D]` + visibility `vis [B, M]`, branch order
  `0=foreground, 1..K=parts, [last=CLIP-native global iff RN50]`.
- `pcr/models/relation_blocks.py` — **the module this fork's changes live in.** See below.
- `pcr/models/prompt_learner.py` — per-identity learnable context + TAB, builds per-branch text
  prompts fed through the frozen CLIP text encoder.
- `pcr/loss/` — `clip_supcon_loss.py` (Stage 1), `clip_cosine_align_loss.py`,
  `part_triplet_loss.py`, `cross_attn_align_loss.py` (Stage 2), `body_part_attention_loss.py`
  (Stage 0/BPA supervision).
- `pcr/evaluators.py` — retrieval-time feature extraction; must mirror training's VAB/pool
  pipeline exactly (CAB is *not* replayed at inference — it needs the ground-truth identity
  label, which isn't available for query/gallery images).

## This fork's architecture change (commit `baac26f`)

Rearchitected how the global embedding is built and how CAB compares image/text branches. Before:
`VisualAttentionBlock` (VAB) ran plain self-attention over *all* M branches (foreground + K parts,
uniformly) — foreground sat inside the same attention set as a 6th peer token. Now, via the new
`apply_vab_with_pooling(vab, pool, f_out, vis, has_global)` helper (called from both stage
scripts and `evaluators.py`, replacing the old bare `vab(f_out, vis)` call):

1. **Foreground gates, doesn't attend.** Each of the K part tokens is multiplied by its own
   visibility (`part_tokens[k] *= visibility[k]`) *before* self-attention — reusing the existing
   visibility signal rather than treating foreground as separate semantic content.
2. **VAB** then runs standard self-attention over the K *gated* parts only (unchanged as a class;
   still visibility-biased at the logit level via `_visibility_attn_bias`).
3. **`AttentionPoolingBlock`** (new) — a Set-Transformer/PMA-style single learnable seed query
   attends over the relationally-mixed part tokens to produce the global embedding, replacing the
   old hand-pooled foreground branch. Visibility-biased the same way; zero-init gate returns the
   plain visibility-weighted mean of the parts at init.
4. The result is reassembled with the new pooled global taking over **branch 0's exact position**
   — every downstream consumer (`id_classifiers`, `bn_necks`, per-branch triplet/align loops)
   needed zero changes.
5. **`CrossAttentionBlock` (CAB)** now cosine-normalizes Q/K with a CLIP-style learnable
   temperature (`log_logit_scale`) instead of raw scaled dot-product, plus a learnable diagonal
   bias (only when `Nq == Nk`) so query branch *i* starts with a strong preference for its own
   matched context branch *i*. The diagonal bias is scaled as a *fraction of* `log_logit_scale`'s
   value, not an absolute logit — an absolute bias only won the diagonal ~40-60% of the time once
   `logit_scale` moved off its init value (verified empirically); scaling with it keeps the
   dominance guarantee invariant (100% diagonal-argmax at init after the fix).

`TextualAttentionBlock` (text side) is untouched. `train_relational_prompts.py`'s `L_relalign`
now slices TAB's `[1+K, 1+K]` attention down to its parts-only `[K, K]` sub-block to match VAB's
new shape.

Verification so far is a synthetic shape/gradient wireframe test (both RN50-with-native-global
and ViT-no-global branch layouts), not a live training run — no Stage 0/1 checkpoints exist in
this fork yet (gitignored, never copied over from `pcr2`), so nothing has actually been trained
end-to-end here.

## Conventions

- Branch order is load-bearing throughout: `0=foreground/global-anchor, 1..K=parts[, last=native
  global on RN50]`. Any change to branch composition must preserve this or update every indexed
  consumer (`id_classifiers`, `bn_necks`, `align_bn_necks`, the per-branch loss loops).
- Zero-init gated residuals (`tanh(gate) * delta`) are the standing convention for every new
  trainable block (VAB, CAB, pool) — a block should be a no-op at initialization and only
  learn to deviate once training finds a reason to.
- `has_global` (RN50 only) gates a lot of branch-count logic — always derive it from
  `encoder._has_global`, don't hardcode per-arch assumptions.
- No test suite; verification is `python -m py_compile` across the repo plus synthetic
  shape/gradient smoke tests against the real classes (see `baac26f`'s commit message for the
  pattern) — there's no live GPU/checkpoint harness in this fork yet.
