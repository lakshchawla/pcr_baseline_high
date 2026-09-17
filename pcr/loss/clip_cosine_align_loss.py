"""Classification-style rewrite of Stage 2's alignment term, replacing the earlier pure-regression
version (`1 - cosine_sim(relation_feats[:,k], frozen_text_anchors[label,k])`, no other identity's
anchor involved at all). Restores CLIP-ReID's own original mechanism -- a softmax classification
against the *entire* frozen prototype table, so every other identity's anchor acts as an implicit
negative -- while keeping this repo's per-part visibility weighting on top. See changes.md's "Red
flag 4" and plans/IMPROVEMENT_PLAN.md section 3 for the full rationale: a loss with no repulsion term at
all can be minimized by letting every identity's features drift toward each other, which is exactly
backwards for re-identification.

Per-sample `weights` (required, not optional -- same convention as pcr/loss/clip_supcon_loss.py's
SupConLoss): a branch with low visibility for a given sample contributes proportionally less to
that branch's own alignment term. Explicitly detached before use (`weights.detach()`) -- plans/
IMPROVEMENT_PLAN.md section 1 / changes.md's "Red flag 1": examples/train_relational_finetune.py's own encoder
construction switches to continuous (differentiable) visibility scores, so without detaching here,
gradient descent has a standing incentive to lower a hard-to-align part's own visibility score
instead of actually improving its alignment -- a shortcut that reduces this loss without the
network doing the work the loss is meant to encourage. Weights are additionally floored
(`weight_floor`, default 1e-3) so no sample is ever exactly zero-weighted.

Both `visual_features` and `text_anchors_table` are expected to already be L2-normalized by the
caller (not enforced here, same convention as SupConLoss) -- otherwise the dot product below isn't
a real cosine similarity and `temperature` doesn't mean what it's calibrated to mean. Fixed
2026-08-29: `visual_features` was already normalized at its call site
(examples/train_relational_finetune.py), but `text_anchors_table` (built by
examples/cache_text_anchors.py) wasn't -- real measured text-embedding norms run ~10-13, so this
loss's dot product was silently running at roughly 10-13x its intended (softer) temperature the
entire time. `cache_text_anchors.py` now normalizes before saving the table, closing the gap.
"""
import torch.nn.functional as F
import torch.nn as nn


class CosineAlignLoss(nn.Module):
    def __init__(self, temperature=0.07, weight_floor=1e-3):
        super(CosineAlignLoss, self).__init__()
        self.temperature = temperature
        self.weight_floor = weight_floor

    def forward(self, visual_features, text_anchors_table, targets, weights):
        """visual_features: [B, D], one branch's pooled+relation-mixed feature per sample.
        text_anchors_table: [num_identities, D], this branch's full frozen prototype table (every
        identity, not just the ones in this batch) -- the source of the negatives this loss was
        previously missing. targets: [B] long, each sample's identity index into that table.
        weights: [B] float, per-sample visibility score for this branch; detached internally.
        Returns the weighted mean cross-entropy of classifying each sample against the full table."""
        w = weights.detach().clamp(min=self.weight_floor)
        logits = visual_features @ text_anchors_table.t() / self.temperature  # [B, num_identities]
        per_sample = F.cross_entropy(logits, targets, reduction='none')  # [B]
        return (w * per_sample).sum() / w.sum().clamp(min=1e-8)
