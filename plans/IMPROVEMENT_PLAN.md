# Improvement Plan

This document is for one specific situation: **you ran Stage 0/1/2 and your mAP / rank-1 numbers
are lower than you'd like.** It walks through the parts of the pipeline most likely responsible,
in priority order, each with: what's happening in plain language, the math behind it, exactly
which file/function to change, and a concrete fix. Real papers are cited where a fix borrows a
published idea, so you can go read the original if you want more depth.

This is a plan, not a change log -- nothing here has been implemented yet. Pick items in priority
order; each is independent of the others except where noted.

## How to read this document

Each section has the same shape:
- **The problem**, in plain language.
- **The math**, only where it clarifies *why* the problem happens (no math for its own sake).
- **Code reference**: exact file and function in this repo as it stands today.
- **The fix**, concretely.
- **Why this fix, not something else** (the paper it's based on, if any, and why that idea applies
  here).

## Priority 0 (do this first, before judging any mAP number at all): finish training Stage 0

Every mAP/rank-1 number produced by this pipeline so far has come from a Stage 0 checkpoint that
was only smoke-tested for **1 epoch**, not the **60 epochs** the pipeline was designed for
(`configs/stage0_bpa_segmentation.yaml`). A diagnostic run this session found the consequence
directly: one body-part branch reads as "invisible" across essentially the entire training set,
another reads as "visible" 95%+ of the time regardless of what's actually in the photo. That is
the signature of an attention head that hasn't learned yet, not of genuine occlusion patterns in
Market1501. Every other item in this document assumes Stage 0 is allowed to actually converge --
none of the fixes below can compensate for a part-attention head that hasn't learned what a "leg"
or "torso" looks like yet. If you haven't already, run:

```bash
python examples/train_bpa_segmentation.py --config configs/stage0_bpa_segmentation.yaml
```

for the full 60 epochs before re-measuring anything downstream.

## 1. Fix the visibility-weight gradient leak (Stage 2's alignment loss) -- IMPLEMENTED 2026-08-28

**The problem**: Stage 2 scales each part's alignment loss by that part's own visibility score --
lower visibility, less weight. The intent is "don't punish the model hard for a part it genuinely
can't see." The problem is that the visibility score is *itself an output of the network being
trained*, and nothing stops the optimizer from changing that score directly, instead of doing the
harder work of actually improving the part's alignment.

**The math**: for a batch of alignment losses `l_1, ..., l_B` (one per sample, `l_i = 1 -
cos(v_i, t_i)`) with weights `w_1, ..., w_B`, the loss is

```
L = (Σ_i w_i * l_i) / (Σ_i w_i)
```

Since `w_i` is a differentiable function of the network's own parameters (through the
part-attention head), the gradient of `L` with respect to `w_j` is:

```
∂L/∂w_j = (l_j - L) / (Σ_i w_i)
```

Whenever a sample's loss `l_j` is worse than the current weighted average `L`, this gradient is
**positive** -- meaning gradient *descent* pushes `w_j` **down**. The optimizer has a standing,
always-available option to reduce `L` by declaring hard samples "less visible," with zero
improvement to the actual visual-text alignment. This is the textbook failure mode of any
self-paced / instance-reweighting scheme where the weight and the loss share a computation graph
(the general fix -- stop gradient into the weight -- is exactly what self-paced learning and
sample-reweighting literature always does; it's not specific to ReID).

**Code reference**:
- `pcr/loss/clip_cosine_align_loss.py`, `CosineAlignLoss.forward`, line `w =
  weights.clamp(min=self.weight_floor)` -- no `.detach()`.
- Fed from `examples/train_relational_finetune.py`, `compute_losses()`: `w = vis[:, branch]`
  (line ~239), where `vis` comes straight out of `encoder.forward_full(imgs)` / `encoder(imgs)`
  inside the live, gradient-tracked forward pass (`build_encoder` sets
  `training_binary_visibility_score = False`, which is what makes `vis` differentiable in the
  first place -- see `third_party/torchreid/models/bpbreid.py`'s `pixels_parts_probabilities
  .amax(dim=(2, 3))` computation).
- Stage 1's loss had the identical pattern in its weights argument. It was `InfoNCELoss` at the
  time this was first written; Stage 1 has since switched back to `SupConLoss`
  (`pcr/loss/clip_supcon_loss.py`, restored 2026-08-28 -- see `progress.md`), which was given the
  same detach directly, from the start, rather than left as a follow-up.

**The fix**: detach the weight before clamping, in both loss classes:

```python
# pcr/loss/clip_cosine_align_loss.py and pcr/loss/clip_supcon_loss.py
w = weights.detach().clamp(min=self.weight_floor)
```

One line, in each of the two files. This makes "visibility informs the loss but never receives
gradient from it" a property of the loss class itself, not something that depends on every future
caller remembering to detach upstream.

**Status**: done in both `CosineAlignLoss` (`pcr/loss/clip_cosine_align_loss.py`) and Stage 1's
`SupConLoss` (`pcr/loss/clip_supcon_loss.py`) -- verified for `CosineAlignLoss` with an isolated
forward/backward check (`weights.grad is None` after `loss.backward()`, confirming the leak is
closed while gradient still reaches `visual_features`) and a real one-epoch Stage 2 smoke run (375
iterations, all loss terms finite throughout, no NaN/Inf). Not currently exploitable in Stage 1
either way (`cached_visibility` is built entirely under `torch.no_grad()`, so there's no live graph
there today), but the detach is now a guarantee of the loss class itself rather than an accident of
the caller.

## 2. Give the "part went silent" case a real signal, not a one-time warning

**The problem**: `PartTripletLoss` returns `None` for a branch when a batch has zero valid
triplets (every image reads that part below `triplet_visibility_min`). Python's default warning
filter only prints a given warning **once per process**, so if this keeps happening every batch
for the rest of training (plausible for the collapsed branch described above), there is currently
no way to see it in the log.

**Code reference**: `pcr/loss/part_triplet_loss.py`, `_hard_mine_triplet_loss`, the
`warnings.warn('PartTripletLoss: no valid triplets in this batch')` branch; `PartTripletLoss`
already computes and returns a `valid_triplets_ratio` (the third element of its return tuple) that
`compute_losses()` in `examples/train_relational_finetune.py` currently discards
(`part_result[0]` is read, `part_result[1]`/`part_result[2]` are not).

**The fix**: accumulate and periodically print a per-branch valid-triplet rate, using data the
loss already computes:

```python
# in compute_losses(), accumulate across the branch loop:
valid_ratios = []
for branch in range(1, num_branches):
    part_result = triplet_loss(combined[:, branch:branch+1, :], targets,
                                parts_visibility=vis_mask[:, branch:branch+1])
    if part_result is not None:
        l_tri_parts = l_tri_parts + part_result[0]
        valid_ratios.append(part_result[2].item())
    else:
        valid_ratios.append(0.0)
log['tri_valid_ratio_per_branch'] = valid_ratios
```

Then print `log['tri_valid_ratio_per_branch']` at the existing `print_freq` cadence. A branch
sitting at 0.0 for many consecutive prints is your early warning that it has effectively dropped
out of triplet training -- exactly the situation this document's Priority 0 item is meant to
prevent, but this gives you a way to *see* it happening in any future run.

## 3. Turn the alignment loss back into a classification, not a pure regression -- IMPLEMENTED 2026-08-28

**The problem**: `CosineAlignLoss` only pulls a part's visual feature toward its own identity's
text anchor. It has no term that pushes it away from *other* identities' anchors. A loss with no
repulsion term can be minimized by collapsing all identities' features toward a similar point --
exactly the opposite of what re-identification needs (different people need to look different).
This is a real regression relative to the original CLIP-ReID design, which used a full
cross-entropy classification against the entire identity-prototype table (every other identity's
prototype acts as an implicit negative, the same softmax mechanism InfoNCE and CLIP's own
contrastive loss both use).

**The math**: currently,

```
L_align = (1/Σw) * Σ_i w_i * (1 - cos(v_i, t_{y_i}))
```

no other identity's prototype `t_j` (j ≠ y_i) ever appears in this formula. The fix reintroduces
them as a softmax denominator, matching the original CLIP-ReID `I2TLoss` shape but keeping this
repo's per-part visibility weighting:

```
L_align' = -(1/Σw) * Σ_i w_i * log [ exp(cos(v_i, t_{y_i}) / τ) / Σ_j exp(cos(v_i, t_j) / τ) ]
```

where the sum in the denominator runs over **every identity's prototype for that branch**
(`text_prototypes[:, branch, :]`, already available in `examples/train_relational_finetune.py` as
the full loaded table -- currently only ever indexed per-sample via `text_prototypes[targets,
branch, :]`, never used as a full table inside the loss itself). `τ` is a temperature, e.g. reuse
Stage 1's `0.07`.

**Code reference**: `pcr/loss/clip_cosine_align_loss.py` (`CosineAlignLoss`), and its call site in
`examples/train_relational_finetune.py`'s `compute_losses()` (currently passes only
`branch_anchors = text_prototypes[targets, branch, :]`, a per-sample gather -- the fix needs the
full `text_prototypes[:, branch, :]` table passed in as well).

**The fix** (sketch):

```python
class CosineAlignLoss(nn.Module):
    def __init__(self, temperature=0.07, weight_floor=1e-3):
        ...
    def forward(self, visual_features, all_text_anchors, targets, weights):
        # visual_features: [B, D]; all_text_anchors: [num_identities, D] (this branch's full table)
        w = weights.detach().clamp(min=self.weight_floor)
        logits = visual_features @ all_text_anchors.t() / self.temperature   # [B, num_identities]
        per_sample = F.cross_entropy(logits, targets, reduction='none')      # [B]
        return (w * per_sample).sum() / w.sum().clamp(min=1e-8)
```

**Why this fix**: this is precisely CLIP-ReID's own original I2TLoss mechanism (already vendored
in this repo's history as `clip_i2t_loss.py`, removed only because "Algorithm 2"'s literal wording
called for the simpler regression form) -- restoring it, with the visibility weighting layered on
top, keeps this repo's occlusion-handling work while fixing the missing-repulsion gap. It's also
the same core idea used by PLIP's identity-based vision-language contrast term (Zuo et al., "PLIP:
Language-Image Pre-training for Person Representation Learning," arXiv:2305.08386) and by
PAB-ReID's part-triplet-supervised local features (Chen & Ge, "Part-Attention Based Model Make
Occluded Person Re-Identification Stronger," IJCNN 2024, arXiv:2404.03443) -- both use a
classification/contrastive-style term with real inter-identity negatives for exactly this reason.

**Status**: implemented in `pcr/loss/clip_cosine_align_loss.py` and wired up in
`examples/train_relational_finetune.py`'s `compute_losses()` (now passes the full
`text_prototypes[:, branch, :]` table plus `targets`, not just the per-sample gathered anchor); a
new `loss.align_temperature: 0.07` config knob was added to `configs/stage2_relational_finetune.yaml`
alongside the pre-existing loss weights. Verified with an isolated shape/gradient check and a real
one-epoch Stage 2 smoke run -- both clean, no NaN/Inf.

**Follow-up this smoke test surfaced**: the new loss's raw scale is much larger than the old
regression version's bounded `[0, 2]` range -- real smoke values were ~35-50 (summed over 5 part
branches, so ~7-10 per branch), driven by the softmax now being computed over all 751 Market1501
identities. `loss.align_weight` (`0.5`, tuned for the old small-scale loss) was not retuned as part
of this fix, since that's a hyperparameter decision best made against real (not 1-epoch smoke)
training curves -- worth revisiting once a longer run is available, e.g. lowering `align_weight` or
scaling the cross-entropy by `1 / log(num_identities)` so its magnitude stays comparable across
dataset sizes.

## 4. Widen (or route around) Stage 1's small negative pool -- IMPLEMENTED 2026-08-28

**The original problem**: `InfoNCELoss` (used briefly in Stage 1, since replaced) deduplicated a
PK-sampled batch down to its unique identities before building its negative set (needed to avoid
treating two photos of the same person as false negatives of each other). With this repo's batch
settings (`batch_size: 32`, `num_instances: 4`), that left only `32 / 4 = 8` unique identities
compared per step, and threw away all but one representative photo per identity in the process.

**First fix (2026-08-28)**: Stage 1 switched back to `SupConLoss` (`pcr/loss/clip_supcon_loss.py`)
-- it needs no deduplication at all, since its positive mask already tells same-identity rows
apart from different-identity rows directly. This removed the self-inflicted information loss, but
the *absolute* comparison size per step was still capped at one PK batch (~32 images, ~8
identities) -- much smaller than CLIP-ReID's own original design, which classifies against the
*whole* identity table every step.

**Second fix, closing the gap (2026-08-28, per direct user request)**: implemented in
`examples/train_relational_prompts.py`. Both directions of the loss are now widened to the full
training set, not just the current batch:

- **i2t** (image anchors against text): compared against **every training identity's** text
  anchor, not just the ~8 in the current batch. `build_text_snapshot()` rebuilds a
  `[num_identities, K, D]` text-anchor table once per epoch (under `no_grad`, against that epoch's
  current `part_ctx`/TAB weights) -- cheap, since it's one extra CLIP-text pass over the identity
  set per epoch, not per iteration. Each iteration, identities in the current batch get their
  fresh, differentiable text row spliced back in (gradient must still reach `part_ctx`/TAB for
  them); every other identity's row comes from the snapshot, detached, as a pure negative.
- **t2i** (text anchors against images): compared against **every cached training image** (12936,
  not one batch's ~32) directly -- no snapshot or splicing needed at all here, since
  `cached_features` never goes stale (the backbone/BPAM are frozen for the entirety of Stage 1).

No new memory-bank/queue infrastructure needed, and no new hyperparameter to tune (the "pool size"
is just `num_identities`/`num_images`, both already known) -- Options A/B below were the originally
sketched alternatives, superseded by this more direct approach once it became clear Stage 1's own
frozen encoder makes a full-dataset comparison this cheap.

**Verified**: isolated checks (`build_text_snapshot`'s output shape/finiteness; the batch/snapshot
splicing produces no overlapping identities between "fresh" and "snapshot" rows) and a real 2-epoch
Stage 1 smoke run -- no NaN/Inf/errors, per-epoch wall time unchanged (~23s, same as before
widening) confirming the added cost is negligible, loss decreasing visibly within the first epoch
against the now much harder (751-way / 12936-way) task. `cache_text_anchors.py` run against the
result, producing a fully finite `text_prototypes.pth` -- the Stage 2 handoff is unaffected.

**Superseded alternatives, kept for reference**:

*Option A -- bigger PK batches.* Simplest, but limited: raising `batch_size` to 64 only reaches 16
identities per step, nowhere near 751 -- not pursued once the snapshot approach made the full
identity set reachable directly.

*Option B -- a MoCo/SimCLR-style FIFO queue of past batches' features.* Would have introduced
staleness (queued features drift out of date as the producing weights keep training) and a new
`queue_size` hyperparameter to tune. Not needed here specifically because Stage 1's visual side
never goes stale at all (frozen encoder) and the text side is cheap enough to fully refresh once
per epoch -- a queue is the right tool when neither of those is true, which is why it's still worth
knowing about for e.g. a hypothetical future Stage 1 with an unfrozen encoder.

A code-light alternative for the *loss formulation itself*, if a plain contrastive loss (rather than
SupCon's multi-positive one) is ever wanted again: **Decoupled Contrastive Learning** (Yeh et al.,
"Decoupled Contrastive Learning," arXiv:2110.06848), which removes the positive pair's own term
from the denominator of the InfoNCE softmax specifically to recover signal lost to small
batches/negative counts -- worth knowing about, though it doesn't apply to `SupConLoss`'s current
mask-based formulation directly.

## 5. Make VAB/TAB's attention visibility-aware -- IMPLEMENTED 2026-08-28

**The problem**: `VisualAttentionBlock` and `TextualAttentionBlock` mix all K body-part tokens via
full, unconditional self-attention. A part that's actually occluded still participates as a full
"key" that every other part's query attends to -- so a badly-occluded leg doesn't just have a bad
loss term of its own, it also pollutes the head/torso/arm tokens that attend to it, *before* any
loss-level visibility weighting (this document's items above) ever gets a chance to matter. This is
the mechanism, not just the existence, behind the currently-open `changes.md` item on VAB/TAB being
unmasked.

**The math**: standard scaled dot-product self-attention computes, for query `i` attending to key
`j`:

```
score_ij = (q_i . k_j) / sqrt(d)
attn_ij  = softmax_j(score_ij)
```

with no dependence on how reliable token `j`'s content actually is. The fix adds a soft,
visibility-dependent bias to the score before the softmax, so an unreliable key contributes less
regardless of how well it happens to correlate with the query in raw dot-product terms:

```
score_ij' = (q_i . k_j) / sqrt(d) + log(v_j + eps)
attn_ij   = softmax_j(score_ij')
```

`log(v_j)` is the standard way to implement a *soft* mask inside a softmax (as `v_j → 0`, `log(v_j)
→ -inf`, which is exactly what a hard `key_padding_mask` does in `nn.TransformerEncoderLayer` --
this is that same mechanism, generalized from a 0/1 mask to a continuous score). `eps` (e.g. 1e-6)
avoids `log(0)`.

**Code reference**: `pcr/models/relation_blocks.py`, wherever `VisualAttentionBlock`/
`TextualAttentionBlock` currently build their `nn.TransformerEncoder` layers and call them
unconditionally on all K tokens. `nn.TransformerEncoderLayer` doesn't expose a way to add a
continuous per-key bias directly (only a hard boolean/float additive mask via `src_mask`, which
*can* actually be used for this -- `src_mask` in PyTorch's MultiheadAttention is added directly to
attention scores before softmax, so `log(v_j + eps)`, broadcast appropriately, can be passed as
`src_mask` without needing a custom attention implementation).

**The fix (sketch)**:

```python
# VisualAttentionBlock.forward, given part_tokens: [B, K, D] and part_visibility: [B, K]
log_vis = torch.log(part_visibility.clamp(min=eps))          # [B, K]
attn_bias = log_vis.unsqueeze(1).expand(-1, K, -1)            # [B, K, K], broadcast over queries
# reshape/repeat per attention head as required by nn.MultiheadAttention's mask shape,
# then pass as src_mask to self.encoder(part_tokens, mask=attn_bias)
```

This requires threading `part_visibility` into `VisualAttentionBlock.forward`'s signature (and the
analogous change in `TextualAttentionBlock`, using the same per-identity visibility used for
Stage 1's InfoNCE weighting), plus updating both call sites in `examples/
train_relational_prompts.py` and `examples/train_relational_finetune.py` to pass the batch's
`b_vis`/`vis` tensor through.

**Why this fix, and why now (not earlier in this repo's history)**: this was deliberately deferred
during the original visibility-weighting refactor as a separate, larger change (see `changes.md`'s
existing item on this) -- it's listed here as the concrete next step once items 1-4 above are in
place and Stage 0 is properly trained (fixing this before Stage 0 converges would mean tuning
attention masking against visibility scores that are still an undertraining artifact, not signal).
The general approach -- letting a topology/relation module know which parts are trustworthy before
mixing them -- is the central idea of **HOReID** (Wang et al., "High-Order Information Matters:
Learning Relation and Topology for Occluded Person Re-Identification," CVPR 2020,
arXiv:2003.08177), whose cross-graph alignment layer explicitly only connects key-point nodes it
estimates as visible, and of **PVPM** (Gao et al., "Pose-guided Visible Part Matching for Occluded
Person ReID," CVPR 2020, arXiv:2004.00230), whose pose-guided attention is gated by a jointly
learned visibility predictor rather than computed unconditionally. **VPM** (Sun et al., "Perceive
Where to Focus: Learning Visibility-aware Part-level Features for Partial Person Re-identification,"
CVPR 2019, arXiv:1904.00537) is the earliest and simplest version of the same idea (only compare
regions visible in both images at matching time) and is worth reading first if the later two feel
too complex to port directly.

**Status**: implemented in `pcr/models/relation_blocks.py` (`_visibility_attn_bias`, shared by both
blocks) essentially as sketched above, with one difference for `TextualAttentionBlock`: since
`PromptLearner.part_ctx` has no per-image signal at all, it uses each identity's *mean* visibility
across every cached training image of that identity instead of a per-image score (computed once by
`examples/train_relational_prompts.py::compute_identity_visibility`, saved as
`identity_visibility.pth` alongside `prompt_learner.pth`/`vab.pth`, and reused unchanged -- not
recomputed -- by `examples/cache_text_anchors.py`, so the final frozen prototypes match what
training actually converged against). `VisualAttentionBlock` uses each call's own real per-image
`vis[:, 1:]` in both Stage 1 and Stage 2, exactly as sketched.

Verified with: (1) isolated forward/backward checks on both blocks (shape correctness, the
zero-gate no-op property still holding for VAB, masking measurably changing output vs. a uniform-
visibility control, finite output even at the zero-visibility edge case); (2) a real end-to-end
smoke run of the full chain -- Stage 1 (375 iterations) -> `cache_text_anchors.py` ->
Stage 2 (375 iterations) -- all clean, no NaN/Inf/errors.

**A real bug found and fixed along the way, unrelated to the masking formula itself**: the first
end-to-end run produced `NaN` in every part-branch text prototype. Traced to PyTorch's
`nn.TransformerEncoderLayer` silently switching to a fused native kernel
(`torch._transformer_encoder_layer_fwd`) whenever a layer is in `.eval()` mode (which
`cache_text_anchors.py` puts `PromptLearner`/TAB into, correctly, for deterministic output) --
that fused kernel produced `NaN` specifically when given a real, non-uniform additive attention
bias, while the identical computation in `.train()` mode, and `.eval()` mode with a uniform (no-op)
bias, both came out finite -- isolating the bug to the fused kernel, not the masking math. Fixed
with `torch.backends.mha.set_fastpath_enabled(False)`, set once at module import in
`pcr/models/relation_blocks.py` (the only place in this repo building an `nn.TransformerEncoder`,
so nothing else is affected).

## 6. Encourage the K parts to actually specialize (longer-term, after 1-5)

**The problem**: nothing in this pipeline currently penalizes two part-attention branches for
converging on the same region. If Stage 0's collapse (Priority 0) recurs even partially after a
full 60-epoch run, a diversity term is a direct, well-established way to discourage it, independent
of getting the visibility-score mechanics right.

**Code reference**: would be a new loss term added to `examples/train_bpa_segmentation.py` (Stage
0) alongside its existing `BodyPartAttentionLoss`, computed over the K attention maps produced by
`pixel_classifier`.

**The fix**: add a diversity-regularization term that penalizes the Gram matrix of the K part
descriptors for being close to a matrix of all-ones (i.e., penalize parts for correlating with each
other), pushing it toward the identity matrix instead. This is the exact mechanism from
**MHSA-Net**'s Feature Diversity Regularization Term (arXiv:2008.04015) and the earlier **Diversity
Regularized Spatiotemporal Attention** paper (arXiv:1803.09882, video-ReID, adapted here to a
single-image, per-part setting) -- both add a term of the form `||Gram(F) - I||_F^2` (Frobenius
norm) over the stacked, L2-normalized part descriptors `F`, and report it directly reduces
part-collapse.

## Suggested order of operations

1. **Finish Stage 0's real 60-epoch run** (Priority 0) -- do not trust any mAP number before this.
2. **Detach the visibility weights** (item 1) -- one line each in two files, no architecture
   change, closes a real gradient-gaming path.
3. **Add the triplet silent-dropout logging** (item 2) -- pure observability, no behavior change,
   lets you actually see whether item 6 is still needed after Stage 0 is retrained.
4. Re-run Stage 1 + Stage 2 smoke tests with 2 and 3 in place; check whether the previously
   collapsed branch's visibility distribution and valid-triplet-ratio look healthier now that
   Stage 0 has real signal.
5. If mAP/rank-1 is still lower than expected after step 4, take up items 3-5 (classification-style
   align loss, wider Stage 1 negative pool, visibility-aware VAB/TAB) roughly in that order --
   item 3 is the cheapest and most directly tied to the original CLIP-ReID design this repo departed
   from; item 5 is the most invasive (touches the attention architecture) and is best attempted
   last, once you can trust the visibility scores it depends on.
6. Item 6 (diversity regularization) is worth adding only if, after a real Stage 0 run, some
   branches are still noticeably collapsing -- it's a hedge against recurrence, not a fix for the
   current under-training itself.

## References

1. Sun, Y., Xu, Q., Li, Y., Zhang, C., Li, Y., Wang, S., Sun, J. "Perceive Where to Focus:
   Learning Visibility-aware Part-level Features for Partial Person Re-identification." CVPR 2019.
   [arXiv:1904.00537](https://arxiv.org/abs/1904.00537)
2. Gao, S., Wang, J., Lu, H., Liu, Z. "Pose-guided Visible Part Matching for Occluded Person
   ReID." CVPR 2020. [arXiv:2004.00230](https://arxiv.org/abs/2004.00230)
3. Wang, G., Yang, S., Liu, H., Wang, Z., Yang, Y., Wang, S., Yu, G., Zhou, E., Sun, J.
   "High-Order Information Matters: Learning Relation and Topology for Occluded Person
   Re-Identification." CVPR 2020. [arXiv:2003.08177](https://arxiv.org/abs/2003.08177)
4. Li, Y. et al. "Diverse Part Discovery: Occluded Person Re-identification with Part-Aware
   Transformer." CVPR 2021. [arXiv:2106.04095](https://arxiv.org/abs/2106.04095)
5. "MHSA-Net: Multi-Head Self-Attention Network for Occluded Person Re-Identification."
   [arXiv:2008.04015](https://arxiv.org/abs/2008.04015)
6. "Diversity Regularized Spatiotemporal Attention for Video-based Person Re-identification."
   [arXiv:1803.09882](https://arxiv.org/abs/1803.09882)
7. Yeh, C.-H., Hong, C.-Y., Hsu, Y.-C., Liu, T.-L., Chen, Y., LeCun, Y. "Decoupled Contrastive
   Learning." [arXiv:2110.06848](https://arxiv.org/abs/2110.06848)
8. Zuo, J. et al. "PLIP: Language-Image Pre-training for Person Representation Learning."
   [arXiv:2305.08386](https://arxiv.org/abs/2305.08386)
9. Chen, Z., Ge, Y. "Part-Attention Based Model Make Occluded Person Re-Identification Stronger."
   IJCNN 2024. [arXiv:2404.03443](https://arxiv.org/abs/2404.03443) -- of everything above, this
   is the closest architectural match to this repo's own part-attention + part-triplet design, and
   the most direct next paper to read in full.
