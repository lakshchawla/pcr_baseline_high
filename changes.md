# Changes

Pending/proposed changes that have been identified and discussed but deliberately not yet
implemented -- held for later decision or action. Once a change here is actually made, it moves
out of this file and gets its own entry in `progress.md` (which records what was done, not what's
still pending).

## Changes to be done

### 1. Stage 1's per-part loss is symmetric (i2t + t2i) -- back to `SupConLoss` as of 2026-08-28

Stage 1 briefly used `InfoNCELoss` (a literal, single-positive loss matching "Algorithm 1" step
15's own wording), then switched back to `SupConLoss` (CLIP-ReID's real, multi-positive loss) --
see `progress.md`'s entries on both changes for the full reasoning: SupCon needs no patch to work
with PK-sampled batches, while InfoNCE needed one that wasted most of a batch's own signal. SupCon
has always been symmetric (loss_i2t + loss_t2i, via two calls with the two feature sets swapped --
see that file's own docstring), which is CLIP-ReID's own original design, not an extra deviation
layered on top. "Algorithm 1"'s own step 15 still specifies one direction only
(`InfoNCE(relation_feats[:,k], t_i^k)`), so the underlying question below is unchanged regardless
of which of these two losses is in use.

**Decision needed**: keep symmetric (current behavior, and CLIP-ReID's own convention), or switch
to single-direction only (visual->text) to match the algorithm literally.

### 2. ~~`VisualAttentionBlock` (and `TextualAttentionBlock`) still perform fully unmasked
self-attention over possibly-invisible part tokens~~ -- FIXED 2026-08-28

Both blocks now use each part's own reliability as a soft attention bias -- see `progress.md`'s
entry on this fix (and "Red flag 6" below, which tracked the same issue in plain language) for the
full mechanism, including a real PyTorch fast-path bug found and fixed along the way.

## Red flags found in review (2026-08-27, plain language)

A code-focused review and a research-methodology review were run against Stage 0, 1, and 2 as they
stand today. Below is what they found, explained simply -- what's wrong, why it matters, and what
to do about it. This section adds to the "Changes to be done" items above; it doesn't replace them
(those are still valid, just written in more technical language). Concrete code-level fixes with
math and paper references are in `plans/IMPROVEMENT_PLAN.md`.

### 1. ~~(Most serious) The alignment loss can "cheat" by hiding a part instead of improving it~~ -- FIXED 2026-08-28

Stage 2's `CosineAlignLoss` now detaches its visibility weight before using it
(`weights.detach().clamp(...)`), closing the gradient path that let the model lower a part's own
visibility score instead of actually improving that part's alignment. See `progress.md`'s entry on
this fix for the full before/after and verification. Stage 1's loss (`SupConLoss`, restored
2026-08-28) does the same detach directly in its own `forward` -- not left as a follow-up, since
the class was rewritten anyway as part of the SupCon restoration.

### 2. A body part can go completely silent in training with no further warning

**What's wrong**: A body part's triplet loss is skipped for a batch when every image in that batch
reads that part as "not visible enough." When that happens, Python prints one warning -- and then,
by default, never prints that same warning again for the rest of the run, no matter how many more
times it happens. The training log still looks normal (numbers just quietly stop including that
part), so nobody would notice one part effectively dropped out of training.

**Where**: `pcr/loss/part_triplet_loss.py`'s `warnings.warn(...)`, called from
`examples/train_relational_finetune.py`'s `compute_losses()`.

**Why it matters**: this is exactly the shape of the pathology already suspected in this repo (one
body-part branch reading as "invisible" almost everywhere, dataset-wide, because Stage 0's
part-attention head is currently only smoke-tested for 1 epoch instead of the intended 60). If it
recurs in a real run, nothing in the log would reveal it.

**Fix**: log how often this happens (a per-part "how many batches had zero valid triplets" counter,
printed periodically) instead of relying on a one-time warning. See `plans/IMPROVEMENT_PLAN.md`
section 2.

### 3. A leftover scratch note sitting inside Stage 0's file

At the bottom of `examples/train_bpa_segmentation.py`, after the code that actually runs, there is
a stray quoted paragraph suggesting a completely different way to train Stage 0. It doesn't
execute or do anything, but it contradicts the file's own docstring and reads like an
accidentally-committed note. Harmless, but worth deleting (or moving into this file's own "Changes
to be done" list if the idea is worth keeping).

### 4. ~~The alignment loss has no way to say "not like this other person"~~ -- FIXED 2026-08-28

`CosineAlignLoss` is now a softmax classification against each branch's full identity-prototype
table (every other identity acts as an implicit negative), restoring the repulsion term the earlier
pure-regression version was missing -- see `progress.md`'s entry on this fix. One new thing to
watch, found while smoke-testing the fix: this loss's raw magnitude is now much larger than the old
`[0, 2]`-bounded regression (real smoke values were ~35-50 summed over 5 parts, vs. the old
version's max of ~10), so `loss.align_weight` (currently `0.5` in
`configs/stage2_relational_finetune.yaml`) may need retuning down once real training starts, or the
align term will dominate the total loss more than "Algorithm 2" intended.

### 5. ~~Stage 1's "who counts as a negative example" pool is much smaller than the original design~~ -- FULLY RESOLVED 2026-08-28

Originally about `InfoNCELoss`'s own patch for surviving PK-sampled batches: it had to shrink its
comparison down to only the *unique* identities in a batch (~8), throwing away most of a batch's
own signal. Stage 1 switched back to `SupConLoss` (fixing the waste), then was widened further to
close the remaining gap this item was still flagging: Stage 1 now compares every image against
*every* training identity's text anchor (751, not ~8) and every text prompt against *every* cached
training image (12936, not one batch's worth) -- matching CLIP-ReID's own original full-identity-
table design exactly. See `plans/IMPROVEMENT_PLAN.md` section 4 and `progress.md`'s entry on this change
for the full mechanism (`examples/train_relational_prompts.py::build_text_snapshot`).

### 6. ~~Unmasked attention blocks spread bad information into good parts, not just their own loss term~~ -- FIXED 2026-08-28

Both `VisualAttentionBlock` and `TextualAttentionBlock` now use each part's own reliability as a
soft attention bias, so a poorly-visible part contributes less as a "key" to every other part's
mixed representation -- see `progress.md`'s entry on this fix for the full mechanism (and item 2 in
"Changes to be done" above, whose scope limit this resolves).

**A real bug found and fixed while implementing this**: PyTorch's `nn.TransformerEncoderLayer`
silently switches to a fused native kernel whenever a layer is in `.eval()` mode, and that fused
kernel produced `NaN` for every part of every identity as soon as a real (non-uniform) attention
bias was used -- confirmed the masking math itself was correct (the identical computation in
`.train()` mode, and `.eval()` mode with a uniform/no-op bias, both came out finite). Fixed with
`torch.backends.mha.set_fastpath_enabled(False)` in `pcr/models/relation_blocks.py` -- a known,
documented PyTorch workaround, not a change to this repo's own masking logic.
