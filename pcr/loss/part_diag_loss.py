"""Within-identity part contrast (Stage 1): for image part k of one person, the positive is THAT
SAME PERSON's text context k, and the K-1 negatives are that person's OTHER part contexts --
and symmetrically for text part k against that person's K image parts.

Why a separate loss exists at all: SupConLoss's negatives are all cross-identity. It asks "is
this image closer to identity y's text than to other identities'?" and never "is image part k
closer to context k than to context j of the same person?" -- so all K part contexts of one
identity collapsing into a single identity vector is a state SupCon cannot see, let alone
penalize. Measured on a fully-trained Stage 1 checkpoint before this loss existed (2026-09-17,
see progress.md): same-identity different-part text cosine 0.98, i.e. the part prompts carried
identity but essentially no part. This loss adds exactly the missing negative set. Under
collapse every row is uniform and the loss sits at its maximum, log K.

Same idea as the sister repo's L_part_diag (pcr2-baseline-410d465, which reads its
CrossAttentionBlock's attention weights as the class probabilities), computed here from bare
cosine similarity with a fixed temperature instead: no learned q/k projections, deliberately --
a learnable projection pair can satisfy the diagonal by reading "which branch index" off the
tensor structure of both sides without the context itself moving toward its part in the joint
space, and the context is the only thing that survives caching (cache_text_anchors.py). Bare
cosine forces the alignment into ctx.

Both inputs are expected L2-normalized (same convention as every other loss in this package).
Per-sample per-part `weights` (visibility) are detached before use, same reasoning as
SupConLoss/CosineAlignLoss: no incentive to hide a hard part instead of aligning it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartDiagLoss(nn.Module):
    def __init__(self, temperature=0.05, weight_floor=1e-3):
        super(PartDiagLoss, self).__init__()
        self.temperature = temperature
        self.weight_floor = weight_floor

    def _direction(self, anchors, candidates, weights):
        """anchors/candidates: [B, K, D] (index-aligned: anchor k's positive is candidate k of the
        same sample). weights: [B, K]. Returns the visibility-weighted mean over (sample, part) of
        -log softmax_j(cos(anchor_k, candidate_j) / T)[k]."""
        logits = torch.einsum('bkd,bjd->bkj', anchors, candidates) / self.temperature  # [B, K, K]
        log_prob = F.log_softmax(logits, dim=-1)
        diag = torch.diagonal(log_prob, dim1=1, dim2=2)  # [B, K]
        w = weights.detach().clamp(min=self.weight_floor)
        return -(w * diag).sum() / w.sum().clamp(min=1e-8)

    def forward(self, image_parts, text_parts, weights):
        """image_parts/text_parts: [B, K, D], the same person's K image parts and K part
        contexts, both unit-norm. weights: [B, K] per-part visibility. Returns i2t + t2i."""
        return (self._direction(image_parts, text_parts, weights)
                + self._direction(text_parts, image_parts, weights))


def same_identity_part_cosine(text_parts):
    """text_parts: [B, K, D], unit-norm. Mean cosine between DIFFERENT parts of the SAME sample --
    the collapse metric (1.0 = every part prompt identical; the pre-fix checkpoint measured
    0.98). Diagnostic only, no gradient."""
    with torch.no_grad():
        K = text_parts.size(1)
        sim = torch.einsum('bkd,bjd->bkj', text_parts, text_parts)
        off = ~torch.eye(K, dtype=torch.bool, device=sim.device)
        return sim[:, off].mean()
