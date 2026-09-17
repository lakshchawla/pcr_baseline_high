"""Shared pieces for turning one of CLIP's own frozen visual towers into a BPBreID-style,
part-attention encoder -- backbone-agnostic supervised classification (PixelToPartClassifier,
bpbreid's own class, reused verbatim) sitting on top of whichever frozen dense backbone it's
given (pcr/models/clip_rn50_bpam_encoder.py::ClipRN50DenseBackbone or
pcr/models/clip_vit_bpam_encoder.py::ClipViTDenseBackbone). Trained with real masks by
examples/train_bpa_segmentation_rn50.py/_vit.py, exactly like Stage 0 does for HRNet/ResNet --
just with a frozen CLIP backbone instead of a trainable CNN one, so only the small classifier
head ever needs training.

Honors BPBReIDEncoder's own interface (forward -> (f_out, vis), forward_full adding
pixels_cls_scores, num_features, num_parts) so it's a drop-in for every existing consumer
(Stage 1/2 training loops, the evaluator, Stage 3).
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
    neither square nor that resolution. Shared by both dense backbones (RN50's AttentionPool2d
    and ViT's own patch grid each have their own version of this same square-grid, fixed-
    resolution positional embedding)."""
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


class ClipBPAMEncoder(nn.Module):
    """Backbone-agnostic: any frozen dense CLIP backbone exposing forward()->[B,N,vision_width],
    project()->[B,N,embed_dim] and grid_h/grid_w (ClipRN50DenseBackbone/ClipViTDenseBackbone),
    plus a real, trained PixelToPartClassifier on top. Load the trained classifier's state_dict
    via checkpoint_path before using this for Stage 1/2.

    num_parts stores M (matching BPBReIDEncoder's own convention, read by Stage 3's
    train_uda.py) -- the num_parts constructor arg is K alone, same as
    BPBReIDModelCfg.masks.parts_num. M is 1+K (foreground + K parts) normally, or 2+K when the
    given backbone also exposes project_global() (currently only ClipRN50DenseBackbone -- see
    that class's own docstring): a real 7th branch, CLIP's own native whole-image embedding
    (out[0] of the same attnpool call project() already makes, discarded there), appended after
    foreground+parts so branch 0's existing role (global ID classifier, etc.) is undisturbed.
    Visibility for this extra branch is a constant 1.0 -- it isn't attention-weighted by any part
    mask, so "visibility" isn't a meaningful concept for it the way it is for foreground/parts;
    always-observed is the accurate semantics, not a placeholder.

    Optional CLIP-native mask blend (Stage 1 only, off by default -- see _forward_common): when
    a caller passes this batch's own per-identity text contexts plus a blend weight, the pixel
    classifier's part assignment is blended with a MaskCLIP-style, text-matched one
    (pcr/models/clip_native_masks.py) before any pooling happens. mask_temperature is that
    text-matching softmax's temperature; it does nothing unless a caller actually opts in.
    """

    def __init__(self, backbone, num_parts=5, checkpoint_path=None, device='cuda', mask_temperature=0.07):
        super(ClipBPAMEncoder, self).__init__()
        self.backbone = backbone
        self.mask_temperature = mask_temperature
        # Side channel for the mask blend, set by the last forward that actually blended (None
        # otherwise); never read by the forward path itself. Keys:
        #   'anchor_loss'  -- scalar WITH gradient: foreground-weighted reverse KL of the
        #                     text-matched part assignment against the classifier's own
        #                     (detached), see _forward_common. Stage 1 adds lambda * this.
        #   'mask_delta'   -- [K], detached: per-part mean |text-matched - classifier| part mass,
        #                     independent of blend_weight (raw disagreement between the sources).
        #   'outside_support' -- scalar, detached: fraction of the text-matched mass sitting on
        #                     (patch, part) cells where the classifier gives that part < 5% --
        #                     "how much of the text map is outside anatomy".
        self.last_blend_stats = None
        self._has_global = hasattr(backbone, 'project_global')
        self.num_parts = (2 if self._has_global else 1) + num_parts  # M, matching BPBReIDEncoder's own convention
        self.num_features = self.backbone.embed_dim
        self._k = num_parts
        self.pixel_classifier = PixelToPartClassifier(self.backbone.vision_width, num_parts).to(device)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            self.pixel_classifier.load_state_dict(state['pixel_classifier'] if 'pixel_classifier' in state else state)

    def _forward_common(self, images, text_contexts=None, blend_weight=0.0):
        """text_contexts: optional [B, num_branches, embed_dim] -- this batch's own identities'
        current per-branch text embeddings in the full branch layout this encoder emits (0 =
        global/foreground, 1..K = parts[, last = native global]); only the K part rows are read.
        blend_weight: fraction of each patch's *foreground* mass reassigned by the CLIP-native,
        text-matched part assignment instead of the pixel classifier's own. Both default to off,
        which leaves every existing caller (Stage 2/3, the evaluator, the Stage 1 feature cache)
        bit-identical to before -- the blend block below is skipped entirely."""
        patch_feats = self.backbone(images)  # [B, N, vision_width], raw
        B, N, D = patch_feats.shape
        grid = patch_feats.permute(0, 2, 1).reshape(B, D, self.backbone.grid_h, self.backbone.grid_w)
        pixels_cls_scores = self.pixel_classifier(grid)  # [B, 1+K, grid_h, grid_w]
        num_pixel_classes = 1 + self._k
        probs = F.softmax(pixels_cls_scores, dim=1).reshape(B, num_pixel_classes, N).permute(0, 2, 1)  # [B,N,1+K]

        if text_contexts is not None and blend_weight > 0.0:
            # probs' channel 0 is BACKGROUND (PixelToPartClassifier's own layout), while
            # text_contexts' row 0 is the global/foreground branch -- the two "index 0"s are not
            # the same thing, and no text context describes background at all. So the CLIP-native
            # assignment is a softmax over the K part contexts only, and it redistributes just the
            # classifier's foreground mass (1 - background) among the parts: background stays
            # exactly as the classifier says, every patch still sums to 1, and downstream
            # (parts_masks / foreground_masks / visibility) reads the blended probs unchanged.
            part_ctx = text_contexts[:, 1:1 + self._k, :]  # [B, K, embed_dim]
            dense_feats = self.backbone.project_dense(patch_feats)  # [B, N, embed_dim], attention-free
            clip_parts = clip_native_part_masks(dense_feats, part_ctx, self.mask_temperature)  # [B, N, K]
            background = probs[:, :, :1]
            classifier_parts = probs[:, :, 1:]
            foreground_mass = 1.0 - background
            blended_parts = (1.0 - blend_weight) * classifier_parts + blend_weight * foreground_mass * clip_parts
            probs = torch.cat([background, blended_parts], dim=-1)

            # Anatomical anchor for the text-matched assignment. Nothing in Stage 1's objective
            # says WHERE part k is except the Stage-0 classifier: SupCon rewards pooling
            # identity-discriminative patches, which are the same patches for every k, so left
            # alone the K text maps have a standing incentive to converge on the same region.
            # Reverse KL, text || classifier-conditional-on-foreground (detached): mode-seeking,
            # so the text map may sharpen freely INSIDE the classifier's region for part k (the
            # refinement the blend exists for) and pays only when its mass lands where the
            # classifier says that part isn't. Foreground-weighted -- background is the
            # classifier's alone and the text softmax over K parts is undefined there.
            cls_cond = (classifier_parts / foreground_mass.clamp(min=1e-6)).detach().clamp(min=1e-4)
            per_patch_kl = (clip_parts * (clip_parts.clamp(min=1e-8).log() - cls_cond.log())).sum(-1)  # [B, N]
            fg_w = foreground_mass.squeeze(-1).detach()
            anchor_loss = (fg_w * per_patch_kl).sum() / fg_w.sum().clamp(min=1e-6)
            with torch.no_grad():
                outside = (clip_parts * (cls_cond < 0.05).float()).sum(-1)  # [B, N]
                self.last_blend_stats = {
                    'anchor_loss': None,  # filled below, outside no_grad
                    'mask_delta': (foreground_mass * clip_parts - classifier_parts).abs().mean(dim=(0, 1)),
                    'outside_support': (fg_w * outside).sum() / fg_w.sum().clamp(min=1e-6),
                }
            self.last_blend_stats['anchor_loss'] = anchor_loss

        # project_all() (RN50 only) shares ONE attnpool call between the per-patch and global
        # outputs -- calling project()/project_global() separately here would silently recompute
        # the identical attention operation twice (deterministic, so not a correctness bug, but
        # real, avoidable waste; see project_all()'s own docstring, added after finding exactly
        # this during a later audit pass).
        if self._has_global:
            raw_joint_feats, raw_global = self.backbone.project_all(patch_feats)
            global_emb = F.normalize(raw_global, p=2, dim=-1)  # [B, embed_dim]
        else:
            raw_joint_feats = self.backbone.project(patch_feats)
        joint_feats = F.normalize(raw_joint_feats, p=2, dim=-1)  # [B, N, embed_dim]

        # probs[:, :, 0] (background) is intentionally never read below -- BPBreID's own native
        # design excludes background from every embedding it ever builds (it carries no identity
        # signal by construction; its only role anywhere in this pipeline is supervising the
        # pixel classifier via BodyPartAttentionLoss, which reads pixels_cls_scores directly, not
        # this softmax'd version of it). Slicing straight to [:, :, 1:] here, rather than binding
        # an unused `background_masks` name first, so this omission reads as intentional instead
        # of looking like a value that was meant to be used and wasn't.
        parts_masks = probs[:, :, 1:]
        foreground_masks = parts_masks.amax(dim=-1)

        foreground_emb = _gwap_pool(foreground_masks, joint_feats)
        part_embs = torch.stack([_gwap_pool(parts_masks[:, :, k], joint_feats) for k in range(self._k)], dim=1)
        f_out = torch.cat([foreground_emb.unsqueeze(1), part_embs], dim=1)

        parts_visibility = parts_masks.amax(dim=1)
        foreground_visibility = parts_visibility.amax(dim=1)
        vis = torch.cat([foreground_visibility.unsqueeze(1), parts_visibility], dim=1)

        if self._has_global:
            f_out = torch.cat([f_out, global_emb.unsqueeze(1)], dim=1)
            global_vis = vis.new_ones(B, 1)
            vis = torch.cat([vis, global_vis], dim=1)

        f_out = F.normalize(f_out, p=2, dim=-1)

        return f_out, vis, pixels_cls_scores

    def forward(self, images, text_contexts=None, blend_weight=0.0):
        f_out, vis, _ = self._forward_common(images, text_contexts, blend_weight)
        return f_out, vis

    def forward_full(self, images, text_contexts=None, blend_weight=0.0):
        return self._forward_common(images, text_contexts, blend_weight)
