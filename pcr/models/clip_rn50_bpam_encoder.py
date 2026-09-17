"""RN50-specific half of the CLIP-frozen-backbone BPAM encoder (see pcr/models/
clip_dense_part_encoder.py for the shared, backbone-agnostic ClipBPAMEncoder wrapper this pairs
with, and pcr/models/clip_vit_bpam_encoder.py for the ViT-family counterpart).

Note for whoever wires a given arch in: the per-branch embeddings ClipBPAMEncoder returns are
re-projected into that CLIP checkpoint's own pretrained joint space (via attnpool here, or ViT's
own ln_post+proj in the other file) -- that space is paired with that SAME checkpoint's own text
tower, not an arbitrary one. Mixing this image tower with a different CLIP checkpoint's text
tower would not actually be "the same CLIP space," defeating the point of this design.
"""
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_dense_part_encoder import ClipBPAMEncoder, _interpolate_pos_embed


class ClipRN50DenseBackbone(nn.Module):
    """Frozen CLIP RN50 (ModifiedResNet), modified at two points so per-location identity survives
    to the output instead of being discarded the way stock CLIP discards it:

    1. layer4's stride is dropped from 2 to 1 (the standard ReID "last-stride" trick -- the same
       one BPBreID's own `last_stride` config applies to its resnet50/hrnet32 backbones), keeping
       the final grid 2x finer for small body parts. CLIP's own Bottleneck (third_party/clip/
       model.py) encodes stride entirely via parameter-free nn.AvgPool2d(stride) layers -- every
       conv in it is stride=1 always -- so this is done by swapping those two AvgPool2d(2)
       instances (the main branch's post-conv2 one and the shortcut's pre-conv one) for
       nn.Identity(); no pretrained weight is touched or needs retraining.
    2. AttentionPool2d's own forward only ever lets ONE token (the spatial mean) serve as query,
       discarding every individual location's own post-attention representation (`query=x[:1]` in
       third_party/clip/model.py -- verified directly, not assumed). `project()` below re-invokes
       that same layer's pretrained q_proj/k_proj/v_proj/c_proj weights, completely unchanged, but
       with every location serving as both query and key/value -- so every patch keeps its own
       joint-space embedding instead of collapsing into one vector.
    """

    def __init__(self, clip_arch='RN50', height=384, width=128, device='cuda'):
        super(ClipRN50DenseBackbone, self).__init__()
        clip_model, _ = clip.load(clip_arch, device=device, jit=False)
        # clip.load() only casts back to fp32 when device=='cpu' (third_party/clip/clip.py) --
        # on cuda it leaves every weight in fp16 (build_model always calls convert_weights,
        # third_party/clip/model.py). Harmless for Stage 1 (frozen, forward-only), but Stage 2
        # unfreezes this backbone and trains it with plain Adam + loss.backward(), no GradScaler
        # -- fp16 leaf parameters trained that way reliably NaN within a handful of steps
        # (confirmed directly: an unfrozen Stage 2 smoke run went NaN by iteration 5 before this
        # fix). Cast to fp32 master weights unconditionally so this class is safe to train,
        # not just to run frozen -- same reasoning already documented in
        # pcr/models/prompt_learner.py's own docstring for ctx (a different fp16-training pitfall
        # in this same pipeline, same fix: fp32 master weights).
        clip_model = clip_model.float()
        visual = clip_model.visual
        assert hasattr(visual, 'attnpool'), (
            "needs an RN-family CLIP arch (conv backbone + AttentionPool2d) -- ViT-based CLIP "
            "visual towers have no such pooling head.")

        self.dtype = clip_model.dtype
        self.conv1, self.bn1, self.relu1 = visual.conv1, visual.bn1, visual.relu1
        self.conv2, self.bn2, self.relu2 = visual.conv2, visual.bn2, visual.relu2
        self.conv3, self.bn3, self.relu3 = visual.conv3, visual.bn3, visual.relu3
        self.avgpool = visual.avgpool
        self.layer1 = visual.layer1
        self.layer2 = visual.layer2
        self.layer3 = visual.layer3
        self.layer4 = visual.layer4
        self._drop_last_stride(self.layer4)

        self.attnpool = visual.attnpool
        self.vision_width = self.attnpool.k_proj.in_features   # 2048 for RN50: channels into attnpool
        self.embed_dim = self.attnpool.c_proj.out_features     # 1024 for RN50: final joint-space dim
        self.num_heads = self.attnpool.num_heads

        # Stock total downsampling is 32x (stem 4x, layer2 2x, layer3 2x, layer4 2x); dropping
        # layer4's stride above makes it 16x -- the grid attnpool's own pretrained
        # positional_embedding was fit to (visual.input_resolution // 32) is smaller than the one
        # actually reached now, same bicubic-interpolation fix needed for any non-square,
        # non-224px input.
        orig_grid = visual.input_resolution // 32
        new_grid = (height // 16, width // 16)
        pos_embed = _interpolate_pos_embed(self.attnpool.positional_embedding.float(), orig_grid, new_grid)
        self.register_buffer('positional_embedding', pos_embed.type(self.dtype))
        self.grid_h, self.grid_w = new_grid

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @staticmethod
    def _drop_last_stride(layer4):
        first_block = layer4[0]
        first_block.avgpool = nn.Identity()
        if first_block.downsample is not None:
            first_block.downsample[0] = nn.Identity()  # the "-1" AvgPool2d(stride) entry

    def forward(self, images):
        """images: [B, 3, H, W], CLIP-normalized (pcr.models.clip_dense_part_encoder.CLIP_MEAN/
        CLIP_STD, not BPBreID's ImageNet stats). Returns patch_feats [B, grid_h*grid_w,
        vision_width] (raw conv features, pre attention-pool) -- every location individually
        carried to the output."""
        x = images.type(self.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, vision_width, grid_h, grid_w]
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1).float()  # [B, N, vision_width]

    def _attnpool_forward(self, patch_feats):
        """Shared by project()/project_global() below -- one attnpool call produces both outputs
        at once (out[0] = mean-token row, out[1:] = per-location rows), so there's no reason to
        pay for two separate attention calls just to get one or the other."""
        x = patch_feats.permute(1, 0, 2).type(self.dtype)  # NLC -> LNC, matches attnpool's own layout
        mean_tok = x.mean(dim=0, keepdim=True)
        x = torch.cat([mean_tok, x], dim=0)  # [1+N, B, C]
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        ap = self.attnpool
        out, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1], num_heads=self.num_heads,
            q_proj_weight=ap.q_proj.weight, k_proj_weight=ap.k_proj.weight, v_proj_weight=ap.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([ap.q_proj.bias, ap.k_proj.bias, ap.v_proj.bias]),
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0,
            out_proj_weight=ap.c_proj.weight, out_proj_bias=ap.c_proj.bias,
            use_separate_proj_weight=True, training=False, need_weights=False)
        return out  # [1+N, B, embed_dim], LND layout

    def project(self, patch_feats):
        """patch_feats: [B, N, vision_width] (raw, from forward()). Re-runs AttentionPool2d's own
        pretrained weights with every location as both query and key/value (see class docstring
        point 2) instead of stock CLIP's mean-token-only query. Cast to float32 to match this
        repo's convention for every embedding a loss touches."""
        out = self._attnpool_forward(patch_feats)
        return out[1:].permute(1, 0, 2).float()  # drop the mean token, LNC -> NLC: [B, N, embed_dim]

    def project_global(self, patch_feats):
        """patch_feats: [B, N, vision_width] (raw, from forward()). Returns out[0] -- the
        mean-token row of the SAME attnpool call project() makes, i.e. exactly what stock CLIP's
        own AttentionPool2d.forward computes as its sole output (query=mean token only, real
        pretrained q/k/v/c_proj weights) -- CLIP's own native, pretrained-calibrated whole-image
        embedding, discarded by project() above since it only keeps the per-location rows.
        [B, embed_dim]. Standalone convenience method -- ClipBPAMEncoder._forward_common calls
        project_all() below instead when it needs both outputs, so it doesn't pay for two
        separate attnpool calls (see that method's own docstring)."""
        out = self._attnpool_forward(patch_feats)
        return out[0].float()  # [B, embed_dim]

    def project_all(self, patch_feats):
        """Same two outputs as project() + project_global(), from ONE shared attnpool call --
        what ClipBPAMEncoder._forward_common actually uses when it needs both (checked directly:
        calling project() then project_global() separately, as an earlier version of this file
        did, silently recomputes the identical attention operation twice -- deterministic, so not
        a correctness bug, but real, avoidable waste this method removes). Returns
        (per_patch [B, N, embed_dim], global [B, embed_dim])."""
        out = self._attnpool_forward(patch_feats)
        return out[1:].permute(1, 0, 2).float(), out[0].float()

    def project_dense(self, patch_feats):
        """MaskCLIP-style dense projection (Zhou et al., "Extract Free Dense Labels from CLIP",
        ECCV 2022): v_proj -> c_proj applied per patch, nothing else. Of AttentionPool2d's four
        projections, v_proj and c_proj are the only two every location's own information
        actually passes through on the way to CLIP's pretrained output (attention-weighted, but
        present) -- composing them directly per patch gives a dense map that stays in CLIP's real
        joint space without any query/key/softmax step, mean token, or positional embedding (a
        per-patch affine transform doesn't care where the patch is). Deliberately does NOT call
        _attnpool_forward()/multi_head_attention_forward at all, so it's independent of
        project()/project_all()'s own attention path. patch_feats: [B, N, vision_width] (raw,
        from forward()). Returns [B, N, embed_dim], float32 like every other projection here."""
        ap = self.attnpool
        x = patch_feats.type(self.dtype)
        v = F.linear(x, ap.v_proj.weight, ap.v_proj.bias)
        return F.linear(v, ap.c_proj.weight, ap.c_proj.bias).float()


def ClipRN50BPAMEncoder(clip_arch='RN50', height=384, width=128, num_parts=5,
                         checkpoint_path=None, device='cuda', mask_temperature=0.07):
    backbone = ClipRN50DenseBackbone(clip_arch, height, width, device)
    return ClipBPAMEncoder(backbone, num_parts, checkpoint_path, device, mask_temperature)
