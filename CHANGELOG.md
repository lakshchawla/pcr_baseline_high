# Changelog

This fork (`pcr_attn`, branch `baseline-410d465`) starts from `pcr2` commit `410d465` — the
pre-audit baseline that produced the best mAP/R1 run so far (82.2/91.8, Market1501, RN50).
Entries below are this fork's own commits on top of that baseline.

## `baac26f` — Attention-pool the global embedding; foreground gates parts; CAB gets diagonal bias + cosine-normalized logits

Rearchitected `pcr/models/relation_blocks.py` and its call sites in
`examples/train_relational_prompts.py`, `examples/train_relational_finetune.py`, and
`pcr/evaluators.py`.

**1. New `AttentionPoolingBlock`** — replaces the old hand-pooled foreground branch. A
Set-Transformer/PMA-style single learnable seed query attends over the K (relationally-mixed)
part tokens, visibility-biased the same way as `VisualAttentionBlock`'s own self-attention, to
produce the global embedding. Zero-init gate: at init it returns the plain visibility-weighted
mean of the part tokens, not an arbitrary pooled vector, and only learns to deviate once training
finds a reason to (same convention as VAB/CAB's own zero-init gates).

**2. Foreground now gates, doesn't attend.** Previously `VisualAttentionBlock` ran self-attention
over all M branches uniformly (foreground/global sat inside the same attention set as a 6th peer
token next to the 5 real parts). Now each part token is multiplied by its own visibility
(`part_tokens[k] *= visibility[k]`) *before* self-attention/pooling — foreground acts purely as a
multiplicative confidence gate, reusing the visibility signal already computed rather than
inventing a separate "foreground confidence" concept. `VisualAttentionBlock` itself is unchanged
as a class; the new `apply_vab_with_pooling(vab, pool, f_out, vis, has_global)` helper does the
gating + calls VAB on the K gated parts only + calls the new pooling block, then reassembles the
branch tensor with the pooled global taking over branch 0's exact position. Every downstream
consumer (`id_classifiers`, `bn_necks`, per-branch triplet/align loops) needed zero changes as a
result — only the one `vab(f_out, vis)` call site in each script changed.

**3. `CrossAttentionBlock` (CAB): cosine-normalized logits + learnable temperature + diagonal
bias.** Previously used raw scaled dot-product (`QK^T / sqrt(d)`) on unnormalized embeddings.
Now Q/K are L2-normalized before the dot product and scaled by a CLIP-style learnable temperature
(`log_logit_scale`, same convention as CLIP's own `logit_scale`), keeping this block's geometry
consistent with how CLIP's weights were actually pretrained (cosine similarity, temperature-
scaled) instead of imposing an unrelated, norm-sensitive convention on top of them. Also added a
learnable diagonal bias (applied only when `Nq == Nk`) so query branch *i* starts with a
preference for its own matched context branch *i*, loosenable by training.

**Bug found and fixed during verification**: the diagonal bias was first implemented as a fixed
absolute logit value. Synthetic testing showed it only won the diagonal-argmax ~40-60% of the
time once `logit_scale` moved away from its initial guess — the bias's *relative* strength
depended on a value it wasn't scaled against. Fixed by making the diagonal bias a fraction *of*
`logit_scale` (`diag_bias * logit_scale`, not `diag_bias` alone), which keeps the
dominance-at-init guarantee invariant regardless of `logit_scale`'s value. Confirmed via a
synthetic test: 100% diagonal-argmax at init after the fix.

**Left untouched**: `TextualAttentionBlock` (text side) — only the visual aggregation and CAB
were in scope. `train_relational_prompts.py`'s `L_relalign` needed one shape fix as a result: TAB
still returns a `[1+K, 1+K]` attention pattern (all branches), now sliced down to its parts-only
`[K, K]` sub-block to match VAB's new (parts-only) shape before the KL-divergence comparison.

**Verification**: full-repo `python -m py_compile`, plus a synthetic shape/gradient wireframe
test against the real classes covering both branch layouts (RN50, with the native CLIP global
branch passed through `apply_vab_with_pooling` unchanged; ViT, no native global branch) — checked
output shapes, unit-norm invariants, and gradient flow into every new/modified parameter
(`vab`/`pool`/`cab`'s gate, `log_logit_scale`, `diag_bias`). Not yet run against real data or a
live checkpoint: this fork has no Stage 0/1 checkpoints (gitignored, never copied over from
`pcr2`), so nothing here has trained end-to-end yet.

Pushed to `origin/baseline-410d465`.
