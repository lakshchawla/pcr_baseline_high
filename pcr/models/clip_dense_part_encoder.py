"""Shared pieces for turning one of CLIP's own frozen visual towers into a BPBreID-style,
part-attention encoder -- backbone-agnostic supervised classification (PixelToPartClassifier,
bpbreid's own class, reused verbatim) sitting on top of whichever dense backbone it's given
(pcr/models/clip_rn50_bpam_encoder.py::ClipRN50DenseBackbone or
pcr/models/clip_vit_bpam_encoder.py::ClipViTDenseBackbone). The classifier is pretrained with
real masks by examples/train_bpa_segmentation_rn50.py/_vit.py (Stage 0).

Design (2026-09-17 rewrite -- see progress.md): CLIP-ReID's image path plus per-part branches,
nothing else. CLIP-ReID's RN50 encoder returns (x3, x4, x_proj); this returns those and, for
each of the K body parts, the same two features pooled under that part's mask:

    x3          [B, 1024, H3, W3]   layer3 map (CLIP-ReID's image_features_last)
    x4          [B, N, 2048]        layer4 patches
    x_proj      [B, 1024]           CLIP's own global: attnpool with the mean-token query, as CLIP
                                    trained it
    part_x4     [B, K, 2048]        mask-weighted average of x4 patches per part
    part_xproj  [B, K, 1024]        mask-weighted average of project_dense(x4) per part --
                                    MaskCLIP's attention-free v_proj -> c_proj, which keeps each
                                    patch's own identity (patch-vs-patch cos 0.33 vs 0.83 for the
                                    old every-location-query attnpool that made every part a
                                    copy of the global; measured, progress.md)
    vis         [B, 1+K]            1.0 for the global, max part-mask value per part

Branch order everywhere downstream: 0 = global (x_proj), 1..K = parts. No foreground branch, no
second/third global, no relational attention block: parts pass through as pooled, and each part
is aligned to its own prompt context separately (Stage 1/2 scripts).

Honors the interface Stage 0/3 already use: forward() -> (f_out [B, 1+K, D], vis), forward_full()
adding pixels_cls_scores, plus num_features/num_parts. forward_multi() returns everything above.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreid.models.bpbreid import PixelToPartClassifier

from .clip_native_masks import clip_native_part_masks

# CLIP's own image normalization -- NOT BPBreID's ImageNet stats. Pretrained CLIP weights are
# calibrated for this specific normalization; using the wrong one silently degrades every
# downstream similarity score.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _interpolate_pos_embed(pos_embed, orig_grid, new_grid):
    """pos_embed: [1 + orig_grid*orig_grid, width]. Splits off the leading (CLS/mean-token)
    position and bicubic-interpolates the square grid part to new_grid x new_grid -- CLIP's own
    positional embedding is trained for a fixed square input resolution, and ReID crops are
    neither square nor that resolution. Shared by both dense backbones."""
    lead_pos, grid_pos = pos_embed[:1], pos_embed[1:]
    width = grid_pos.size(-1)
    grid_pos = grid_pos.reshape(1, orig_grid, orig_grid, width).permute(0, 3, 1, 2)
    grid_pos = F.interpolate(grid_pos, size=new_grid, mode='bicubic', align_corners=False)
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(new_grid[0] * new_grid[1], width)
    return torch.cat([lead_pos, grid_pos], dim=0)


def _gwap_pool(mask, feats):
    """mask: [B, N] (per-branch pooling weight per patch). feats: [B, N, D]. Same weighted-average
    formula as bpbreid.py's own GlobalWeightedAveragePoolingHead: sum(mask*feat) / sum(mask)."""
    weighted = mask.unsqueeze(-1) * feats
    return weighted.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)


def _pool_parts(parts_masks, feats):
    """parts_masks: [B, N, K]. feats: [B, N, D]. Returns [B, K, D]."""
    return torch.stack([_gwap_pool(parts_masks[:, :, k], feats) for k in range(parts_masks.size(-1))], dim=1)


class ClipBPAMEncoder(nn.Module):
    """Backbone-agnostic: any dense CLIP backbone exposing forward_multi()->(x3|None, x4 patches),
    project_global()->[B,embed_dim], project_dense()->[B,N,embed_dim], vision_width/embed_dim/
    grid_h/grid_w, plus a real, trained PixelToPartClassifier on top (load its state_dict via
    checkpoint_path before using this for Stage 1/2).

    num_parts stores M = 1+K (matching BPBReIDEncoder's own convention, read by Stage 3); the
    num_parts constructor arg is K alone.

    Optional CLIP-native mask blend (off by default; see _forward_common): a caller may pass this
    batch's own per-identity text contexts plus a blend weight, and the pixel classifier's part
    assignment is blended with a MaskCLIP-style, text-matched one (pcr/models/clip_native_masks.py)
    before pooling. Defaults leave every existing caller bit-identical.
    """

    def __init__(self, backbone, num_parts=5, checkpoint_path=None, device='cuda', mask_temperature=0.07):
        super(ClipBPAMEncoder, self).__init__()
        self.backbone = backbone
        self.mask_temperature = mask_temperature
        self.num_parts = 1 + num_parts  # M, matching BPBReIDEncoder's own convention
        self.num_features = self.backbone.embed_dim
        self._k = num_parts
        self.pixel_classifier = PixelToPartClassifier(self.backbone.vision_width, num_parts).to(device)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            self.pixel_classifier.load_state_dict(state['pixel_classifier'] if 'pixel_classifier' in state else state)
        # Side channel for the (opt-in) mask blend, set by the last forward that actually blended
        # (None otherwise): 'anchor_loss' (scalar with grad, foreground-weighted reverse KL of
        # the text-matched assignment vs the classifier's), 'mask_delta' [K], 'outside_support',
        # 'blended_probs' [B, N, 1+K] (all detached except anchor_loss).
        self.last_blend_stats = None

    def _forward_common(self, images, text_contexts=None, blend_weight=0.0, stop_mask_grad=False):
        """Returns a dict: x3 (or None), x4 [B,N,2048], x_proj [B,D], part_x4 [B,K,2048],
        part_xproj [B,K,D] (unit-norm), vis [B,1+K], pixels_cls_scores [B,1+K,gh,gw], probs
        [B,N,1+K]. x_proj/part_xproj are L2-normalized (they're what every cosine loss and the
        retrieval distance read); x4/part_x4 are raw (id/triplet features, BN-necked by the
        caller, CLIP-ReID style).

        text_contexts: optional [B, 1+K, D] this batch's own identities' text embeddings (0 =
        global, 1..K = parts); only the K part rows are read. blend_weight: fraction of each
        patch's *foreground* mass reassigned by the CLIP-native text-matched assignment.
        stop_mask_grad: detach the classifier's softmax before pooling so a trainable classifier
        gets gradient only through pixels_cls_scores (mask losses), never through the pooled
        embeddings' identity losses."""
        x3, x4 = self.backbone.forward_multi(images)
        B, N, C = x4.shape
        grid = x4.permute(0, 2, 1).reshape(B, C, self.backbone.grid_h, self.backbone.grid_w)
        pixels_cls_scores = self.pixel_classifier(grid)  # [B, 1+K, grid_h, grid_w]
        probs = F.softmax(pixels_cls_scores, dim=1).reshape(B, 1 + self._k, N).permute(0, 2, 1)  # [B,N,1+K]
        if stop_mask_grad:
            probs = probs.detach()

        dense = self.backbone.project_dense(x4)  # [B, N, D], attention-free per-patch joint space

        if text_contexts is not None and blend_weight > 0.0:
            # probs' channel 0 is BACKGROUND; text row 0 is the global -- not the same thing, and
            # no text describes background. Softmax over the K part contexts only, redistributing
            # just the classifier's foreground mass; background untouched, patches still sum to 1.
            part_ctx = text_contexts[:, 1:1 + self._k, :]
            clip_parts = clip_native_part_masks(dense, part_ctx, self.mask_temperature)  # [B, N, K]
            background = probs[:, :, :1]
            classifier_parts = probs[:, :, 1:]
            foreground_mass = 1.0 - background
            blended_parts = (1.0 - blend_weight) * classifier_parts + blend_weight * foreground_mass * clip_parts
            probs = torch.cat([background, blended_parts], dim=-1)
            # Anatomical anchor: reverse KL (text || classifier-conditional-on-foreground,
            # detached) -- mode-seeking, so the text map may sharpen inside part k's region and
            # pays only when its mass leaves it. Foreground-weighted.
            cls_cond = (classifier_parts / foreground_mass.clamp(min=1e-6)).detach().clamp(min=1e-4)
            per_patch_kl = (clip_parts * (clip_parts.clamp(min=1e-8).log() - cls_cond.log())).sum(-1)
            fg_w = foreground_mass.squeeze(-1).detach()
            anchor_loss = (fg_w * per_patch_kl).sum() / fg_w.sum().clamp(min=1e-6)
            with torch.no_grad():
                outside = (clip_parts * (cls_cond < 0.05).float()).sum(-1)
                self.last_blend_stats = {
                    'anchor_loss': None,
                    'mask_delta': (foreground_mass * clip_parts - classifier_parts).abs().mean(dim=(0, 1)),
                    'outside_support': (fg_w * outside).sum() / fg_w.sum().clamp(min=1e-6),
                    'blended_probs': probs.detach(),
                }
            self.last_blend_stats['anchor_loss'] = anchor_loss

        # probs[:, :, 0] (background) is never pooled -- BPBreID's own convention (it carries no
        # identity signal by construction; its only role is supervising the classifier).
        parts_masks = probs[:, :, 1:]  # [B, N, K]

        x_proj = F.normalize(self.backbone.project_global(x4), p=2, dim=-1)        # [B, D]
        part_xproj = F.normalize(_pool_parts(parts_masks, dense), p=2, dim=-1)      # [B, K, D]
        part_x4 = _pool_parts(parts_masks, x4)                                      # [B, K, 2048]

        parts_visibility = parts_masks.amax(dim=1)                                  # [B, K]
        vis = torch.cat([parts_visibility.new_ones(B, 1), parts_visibility], dim=1)  # [B, 1+K]

        return {'x3': x3, 'x4': x4, 'x_proj': x_proj, 'part_x4': part_x4, 'part_xproj': part_xproj,
                'vis': vis, 'pixels_cls_scores': pixels_cls_scores, 'probs': probs}

    @staticmethod
    def joint_branches(out):
        """[B, 1+K, D]: the joint-space branches (0 = x_proj, 1..K = part_xproj) -- what Stage 1
        aligns to text, what Stage 2's align loss reads, and what retrieval matches on."""
        return torch.cat([out['x_proj'].unsqueeze(1), out['part_xproj']], dim=1)

    def forward_multi(self, images, text_contexts=None, blend_weight=0.0, stop_mask_grad=False):
        return self._forward_common(images, text_contexts, blend_weight, stop_mask_grad)

    def forward(self, images, text_contexts=None, blend_weight=0.0, stop_mask_grad=False):
        out = self._forward_common(images, text_contexts, blend_weight, stop_mask_grad)
        return self.joint_branches(out), out['vis']

    def forward_full(self, images, text_contexts=None, blend_weight=0.0, stop_mask_grad=False):
        out = self._forward_common(images, text_contexts, blend_weight, stop_mask_grad)
        return self.joint_branches(out), out['vis'], out['pixels_cls_scores']
