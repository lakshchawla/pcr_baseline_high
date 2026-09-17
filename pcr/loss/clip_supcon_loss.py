"""Ported from CLIP-ReID's ../CLIP-ReID/loss/supcontrast.py::SupConLoss -- CLIP-ReID's actual
Stage-1 loss, and not plain diagonal InfoNCE. Restored here (2026-08-28) after a round-trip through
`InfoNCELoss` (removed): this repo's own PK-sampled batches (multiple images per identity in the
same batch, on purpose -- see examples/train_relational_prompts.py's own docstring) are exactly the
situation SupCon was designed for, and exactly the situation InfoNCE was not. InfoNCELoss's own
docstring already documented the mismatch it had to patch around (deduplicating a PK batch down to
one representative image per identity, to avoid treating a person's own other photos as false
negatives) -- that patch worked, but it meant only one of a person's several photos in a batch ever
shaped the text-side gradient, and the multiple-choice comparison shrank to only the handful of
unique identities in one batch (~8, with this repo's batch_size=32/num_instances=4), not the full
training set. SupCon needs no such patch: the positive mask below is built from real identity
labels, so an anchor's own siblings in the batch are correctly recognized as positives (not
negatives) and all of them contribute to that anchor's gradient in one calculation, and every image
in the batch (not just deduplicated representatives) stays a comparison point.

The positive mask is built from identity equality across the two feature sets (every same-identity
pair is a positive, not only the same-index pair), so an anchor can have multiple positives;
`mask.sum(1)` normalizes the log-prob sum over however many positives that anchor has. Applied
symmetrically (call once per direction with the two feature sets swapped) to get CLIP-ReID's
loss_i2t + loss_t2i -- unlike InfoNCELoss, which folded both directions into one call, this keeps
the original file's own two-call convention, since neither side needs deduplication here.

Requires every anchor to have at least one positive (mask.sum(1) > 0), same precondition the
original relies on without a clamp/epsilon -- guaranteed by construction as long as both label
arguments come from the same batch of identities (this repo's own stage-1 loop always calls this
with `anchor_labels=other_labels=b_labels`, the identical tensor for both sides of one batch, so
every anchor's own same-index counterpart is always a positive).

Per-sample `weights` (required, not optional -- matches this repo's other Stage 1/2 losses'
convention): each row's own per-part visibility score, weighting how much that row's own log-
probability counts toward the batch mean -- a poorly-visible part still participates fully as a
comparison point for every other row (SupCon's multi-positive mechanism is left untouched), it just
counts less as an anchor in its own right. Detached before use (`weights.detach()`) for the same
reason documented in pcr/loss/clip_cosine_align_loss.py's own docstring: Stage 1's `build_encoder`
uses continuous (differentiable, in general) visibility scores, so leaving a live gradient path
through the weight would let the model reduce this loss by declaring a hard-to-align row "less
visible" instead of actually aligning it better. (Not currently exploitable in Stage 1 specifically,
since `cached_visibility` is built entirely under `torch.no_grad()` before training starts -- but
the detach makes that a guarantee of this class, not an accident of the caller, same rationale as
CosineAlignLoss.) Weights are additionally floored (`weight_floor`, default 1e-3) so no anchor is
ever exactly zero-weighted.

Temperature default `0.1` (changed from `1.0` on 2026-08-29 -- see progress.md's entry on this
change): `1.0` was CLIP-ReID's own original value, tuned for comparing against the handful of
identities in one small batch. examples/train_relational_prompts.py later widened this loss's own
comparison pool to the *entire* training set (751 identities / 12936 images per step -- see
build_text_snapshot's own docstring), and a softmax over that many classes needs a sharper
temperature to produce a useful gradient at all -- confirmed directly: a real training run at
`1.0` sat within a few points of the theoretical random-guess floor (`ln(num_identities) * 5 +
ln(num_images) * 5`) for roughly two full epochs before making visible progress; the same run at
`0.1` cleared that floor markedly faster and further, using identical data and iteration count.

Temperature is now **learnable**, not fixed, following third_party/clip/model.py's own
`logit_scale` convention (CLIP's real training code does the same): parameterized as
`log(1/temperature)` and exponentiated in `forward`, so gradient descent can push it in either
direction without risking a negative or zero temperature, and clamped at `exp() <= 100`
(temperature >= 0.01) so a runaway value can't blow the softmax up into numerical instability.
`temperature=0.1` above is now only the *initial* value -- Stage 1 includes this module's own
parameter in its optimizer and logs the converged value every epoch.

`anchor_features`/`other_features` are both expected to already be L2-normalized by the caller
(not enforced inside this class, matching pcr/loss/clip_cosine_align_loss.py's own convention of
normalizing at the call site) -- otherwise `logits = anchor_features @ other_features.t()` is not
a real cosine similarity, and `temperature` no longer means what it's calibrated to mean. Found
directly: this repo's own visual features were always unit-norm (VisualAttentionBlock's own
output, and BPBreIDEncoder's raw embeddings, are both L2-normalized already), but the text side
(ClipTextEncoder's raw output) was not -- real measured norms were ~10-13 and varied noticeably
identity to identity, silently making the effective per-identity temperature inconsistent. Both
call sites in examples/train_relational_prompts.py now normalize the text side explicitly before
calling this class.
"""
import math

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    MAX_LOGIT_SCALE = 100.0  # temperature >= 0.01 -- same clamp value as CLIP's own training code

    def __init__(self, temperature=0.1, weight_floor=1e-3):
        super(SupConLoss, self).__init__()
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.weight_floor = weight_floor

    @property
    def temperature(self):
        return 1.0 / self.log_logit_scale.exp().clamp(max=self.MAX_LOGIT_SCALE)

    def forward(self, anchor_features, other_features, anchor_labels, other_labels, weights):
        """anchor_features: [Ba, D]. other_features: [Bo, D]. anchor_labels: [Ba]. other_labels:
        [Bo]. weights: [Ba] float, each anchor row's own visibility score for this call's part/
        direction."""
        mask = torch.eq(anchor_labels.unsqueeze(1), other_labels.unsqueeze(0)).float()
        logits = (anchor_features @ other_features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        w = weights.detach().clamp(min=self.weight_floor)
        return -(w * mean_log_prob_pos).sum() / w.sum().clamp(min=1e-8)
