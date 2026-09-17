"""CLIP-native part masks: MaskCLIP-style (Zhou et al., "Extract Free Dense Labels from CLIP",
ECCV 2022) per-patch text matching, adapted to this repo's per-(identity, branch) prompt contexts.

Vanilla MaskCLIP compares every patch's attention-free dense embedding
(ClipRN50DenseBackbone.project_dense / ClipViTDenseBackbone.project_dense) against a fixed set of
generic class-name text embeddings. Here the "class names" are instead each image's OWN
identity's current per-branch context embeddings (the same PromptLearner outputs SupConLoss
already pairs that image with in Stage 1), so the resulting per-patch assignment localizes
"where in this image is *this identity's* part k" -- text-side localization that stays live and
differentiable w.r.t. the contexts, unlike the frozen pixel_classifier prior it gets blended with
in ClipBPAMEncoder._forward_common.
"""
import torch
import torch.nn.functional as F


def clip_native_part_masks(dense_feats, text_contexts, temperature):
    """dense_feats: [B, N, D] -- project_dense() output for this batch's images (raw, this
    function L2-normalizes). text_contexts: [B, M, D] -- THIS BATCH'S OWN identities' current
    per-branch context embeddings, gathered per-sample by identity (row b pairs with image b),
    NOT one shared template broadcast across the batch. temperature: softmax temperature on the
    cosine similarities (CLIP's own 0.07 convention by default, see the Stage 1 config).
    Returns [B, N, M]: probability each patch belongs to each of the M given branches (softmax
    over M per patch). The caller decides which branches to pass -- ClipBPAMEncoder passes the K
    part contexts only, since the pixel classifier's own channel 0 is *background*, which no text
    context describes (see _forward_common)."""
    dense_feats = F.normalize(dense_feats, p=2, dim=-1)
    text_contexts = F.normalize(text_contexts, p=2, dim=-1)
    sims = torch.einsum('bnd,bmd->bnm', dense_feats, text_contexts) / temperature
    return F.softmax(sims, dim=-1)
