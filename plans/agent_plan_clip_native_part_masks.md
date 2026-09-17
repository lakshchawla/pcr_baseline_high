# Agent Plan: CLIP-Native Part Masks (MaskCLIP-style) Matched Against Live Text Contexts

## 0. Exact scope, and what this does NOT touch

This plan is independent of the logged attnpool bug (`_attnpool_forward`'s `query=x` issue). It does
not modify `project()`, `project_all()`, `project_global()`, or `_attnpool_forward()` at all — those
stay exactly as they are, bug included, until that's fixed separately. This plan **adds** a new,
parallel method (`project_dense`) that never calls `multi_head_attention_forward` in the first
place, so it has no dependency on that bug or its fix.

**The exact MaskCLIP mechanism, restated precisely:** CLIP's `AttentionPool2d` pools by computing
`softmax(q_proj(mean_token) · k_proj(patches)) @ v_proj(patches)`, then `c_proj(...)` on the result.
MaskCLIP's insight: `v_proj` and `c_proj` are the only two of those four projections that every
patch's information actually passed through on the way to CLIP's real pretrained output (weighted
by attention, but present) — so composing them directly, per patch, with no query/key/softmax step
at all, gives a dense feature map that stays in CLIP's real joint embedding space without needing
any attention computation:

```
dense_feature[patch] = c_proj( v_proj( patch_feature ) )        # two nn.Linear calls, nothing else
```

No mean token, no positional embedding needed for this path (position doesn't matter for a
per-patch affine transform), no `q_proj`/`k_proj` involved anywhere.

## 1. File manifest

| File | Status | Role |
|---|---|---|
| `pcr/models/clip_dense_part_encoder.py` (backbone class file — confirm exact filename) | modified, additive | add `project_dense()` |
| `pcr/models/clip_native_masks.py` | new | the text-matched mask function |
| `pcr/models/clip_dense_part_encoder.py::ClipBPAMEncoder._forward_common` | modified | optional `text_contexts`/`blend_weight` args, default off |
| `examples/train_relational_prompts.py` | modified | pass current batch's live per-identity contexts + scheduled `β` into the encoder call |
| `configs/stage1_relational_prompts.yaml` | modified | add `clip_native_mask` schedule block |

**Stage 2/3 are untouched.** They call the encoder without the new arguments, which default to
`text_contexts=None, blend_weight=0.0` — behavior identical to today, `pixel_classifier` alone.

## 2. Step 1 — `project_dense()` on `ClipRN50DenseBackbone`

```python
def project_dense(self, patch_feats):
    """
    MaskCLIP-style: v_proj -> c_proj per patch, no attention, no query, no positional embedding.
    patch_feats: [B, N, vision_width] (same input project()/project_all() take)
    Returns:     [B, N, embed_dim], already in CLIP's joint space.
    """
    ap = self.attnpool
    x = patch_feats.type(self.dtype)
    v = F.linear(x, ap.v_proj.weight, ap.v_proj.bias)
    dense = F.linear(v, ap.c_proj.weight, ap.c_proj.bias)
    return dense.float()
```

**Verification gate:** unit test comparing `project_dense(patch_feats)[b, n]` against
`project(patch_feats)[b, n]` for a few (b, n) pairs — they will **not** match numerically (that's
expected and correct, since `project()` still has the query-mixing bug) — this test exists only to
confirm `project_dense` runs, returns the right shape (`[B, N, 1024]` for RN50), and produces
finite, non-degenerate values (not all-zero, not NaN, reasonable variance across patches).

## 3. Step 2 — the text-matched mask function

**New file, `pcr/models/clip_native_masks.py`:**

```python
import torch
import torch.nn.functional as F


def clip_native_part_masks(dense_feats: torch.Tensor, text_contexts: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    dense_feats:    [B, N, D]        -- project_dense() output for this batch's images
    text_contexts:  [B, 1+K, D]      -- THIS BATCH'S OWN identities' current per-branch context
                                          embeddings (branch 0 = global/foreground, 1..K = parts),
                                          gathered per-sample by identity -- NOT a shared, generic
                                          template across the whole batch. Each image is matched
                                          only against its own identity's learned contexts.
    temperature:    softmax temperature, share Stage 1's existing learnable tau if convenient,
                     or a separate small constant -- ablate both.
    Returns:        [B, N, 1+K] -- probability each patch belongs to each branch.
    """
    dense_feats = F.normalize(dense_feats, dim=-1)
    text_contexts = F.normalize(text_contexts, dim=-1)
    sims = torch.einsum('bnd,bmd->bnm', dense_feats, text_contexts) / temperature
    return F.softmax(sims, dim=-1)
```

**Why per-sample identity contexts, not a shared generic set:** this repo's contexts are
per-(identity, branch), not generic category names ("torso" in the abstract) the way vanilla
MaskCLIP compares against fixed class-name text. Image `i`'s patches should be matched against
identity `y_i`'s own learned `c_0..c_K` — the same pairing SupCon already uses for that image, just
applied to localization instead of classification.

**Verification gate:** feed a synthetic batch of 2 identities with orthogonal random context
vectors and a dense feature map where the first half of patches is engineered to be near-identical
to identity 1's `c_torso` and the second half near identity 1's `c_legs`; confirm the returned mask
correctly separates the two regions for identity 1's images and does *not* activate meaningfully
for identity 2's contexts on identity 1's images (sanity-checks the per-sample gather is wired
correctly, not accidentally broadcasting one identity's contexts onto another's image).

## 4. Step 3 — wire an optional blend into `ClipBPAMEncoder._forward_common`

Add two new optional parameters, both defaulting to off, so every existing call site
(Stage 2, Stage 3, any script that doesn't pass them) is unaffected:

```python
def _forward_common(self, images, text_contexts=None, blend_weight=0.0):
    # ... existing code down through `probs = F.softmax(pixels_cls_scores, ...)` unchanged ...

    if text_contexts is not None and blend_weight > 0.0:
        dense_for_mask = self.backbone.project_dense(patch_feats)          # [B, N, embed_dim]
        clip_mask = clip_native_part_masks(dense_for_mask, text_contexts, temperature=self.mask_temperature)
        # clip_mask: [B, N, 1+K], same layout as `probs` -- branch 0 first, then K parts
        probs = (1 - blend_weight) * probs + blend_weight * clip_mask

    # ... rest of the function (parts_masks = probs[:, :, 1:], GWAP pooling, visibility, etc.) unchanged,
    #     now operating on the blended `probs` when the new args are supplied ...
```

Import `clip_native_masks` and add `self.mask_temperature` (a constructor arg, or reuse an existing
temperature parameter already in the class) at the top of the file.

**Why this is additive and safe:** `probs` is already the single point every downstream computation
(`parts_masks`, `foreground_masks`, GWAP pooling, visibility) reads from — blending happens *before*
any of that, so nothing downstream needs to know a blend even occurred. `text_contexts=None` (every
existing caller) skips the `if` block entirely — zero behavior change for Stage 2/3.

**Verification gate:** call `_forward_common` twice on identical input, once with
`blend_weight=0.0` (regardless of whether `text_contexts` is passed) and once with the same call
before this change existed at all — confirm bit-identical output. This proves the default path is
truly untouched before testing the new path at all.

## 5. Step 4 — wire it into Stage 1's training loop

**Touch:** `examples/train_relational_prompts.py`.

Locate where the current batch's fresh, differentiable per-identity context embeddings are already
computed each step (per `METHODOLOGY.md`: "each batch's fresh, differentiable text row" — this
already exists for the SupCon i2t loss; do not recompute it a second time). Reshape/gather that
into `[B, 1+K, D]`, indexed per-sample by that sample's identity, and pass it straight into the
encoder call:

```python
# wherever this batch's fresh text embeddings are currently produced for SupCon:
batch_text_ctx = ...  # reshape to [B, 1+K, D], per-sample identity already implied by batch order

blend_weight = clip_native_mask_schedule(current_epoch, total_epochs, cfg.clip_native_mask)
f_out, vis, pixel_scores = encoder.forward_full(images, text_contexts=batch_text_ctx, blend_weight=blend_weight)
```

```python
def clip_native_mask_schedule(epoch, total_epochs, cfg):
    # same warmup/ramp/flat shape as L_crossalign's schedule elsewhere in this pipeline
    warmup_end = cfg.warmup_fraction * total_epochs
    ramp_end = warmup_end + cfg.ramp_fraction * total_epochs
    if epoch < warmup_end:
        return 0.0
    if epoch >= ramp_end:
        return cfg.blend_weight_max
    return cfg.blend_weight_max * (epoch - warmup_end) / (ramp_end - warmup_end)
```

**Config addition, `configs/stage1_relational_prompts.yaml`:**
```yaml
clip_native_mask:
  warmup_fraction: 0.3
  ramp_fraction: 0.2
  blend_weight_max: 0.3    # do NOT start higher than this without first confirming pixel_classifier
                            # alone is still a stable floor -- see verification gate below
  mask_temperature: 0.07   # CLIP's own convention; ablate against Stage 1's learnable tau separately
```

**Important dependency to flag, not to skip past:** this only makes sense to activate once contexts
have had time to become meaningful — at `epoch < warmup_end`, `blend_weight=0`, so this doesn't
interfere with early training the same way `L_crossalign`'s own warmup protects it in Stage 2.
`warmup_end` here should be tuned independently of Stage 2's schedule, not copied verbatim — Stage
1's contexts converge on a different timescale than Stage 2's CAB does.

## 6. Verification gate for the full mechanism, and the specific thing to check first

1. Run Stage 1 with `blend_weight_max: 0.0` (i.e., the schedule computed but always returning 0) —
confirm this reproduces your current numbers exactly (regression check, same as Step 4 above but
end-to-end through a real training run, not just a unit test).
2. Run with `blend_weight_max: 0.3` as configured above. Log, per epoch: (a) the blend weight
actually applied, (b) how much `probs` changes on average when blending activates (`||clip_mask -
pixel_classifier_mask||` per branch, per epoch) — if this stays near zero throughout, the two
sources are agreeing (mildly reassuring, but means this addition isn't doing much); if it's large
and *decreasing* over epochs, that's the signature of the CLIP-native path converging toward
agreement with the supervised prior as contexts mature, which is what you want to see.
3. Compare final Stage 1 SupCon loss curves and downstream Stage 2 mAP/Rank-1 against the
`blend_weight_max: 0.0` baseline from step 1. This is the actual deliverable — don't conclude
anything from step 2's logging alone without this comparison.
4. Only after this is validated: consider whether `blend_weight_max` should be pushed higher, or
whether the schedule should extend into Stage 2 as well (Stage 2 doesn't have live, per-step
differentiable contexts the way Stage 1 does — it would need `text_prototypes[targets]`, the frozen
cached table, substituted for `text_contexts` instead; treat this as a separate follow-up, not part
of this plan, since Stage 2's contexts are frozen and the "text side reacting to current image
state" framing this mechanism relies on doesn't apply there in the same way).
