"""ViT-specific half of the CLIP-frozen-backbone BPAM encoder (see pcr/models/
clip_dense_part_encoder.py for the shared, backbone-agnostic ClipBPAMEncoder wrapper this pairs
with, and pcr/models/clip_rn50_bpam_encoder.py for the RN-family counterpart).

Works unchanged for any ViT-family CLIP arch (ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px) --
patch size, transformer width, depth, and head count are all read from the loaded checkpoint's
own weights at construction time, nothing here is hardcoded to one size. Same note as the RN50
file: the per-branch embeddings ClipBPAMEncoder returns are re-projected into that SAME CLIP
checkpoint's own pretrained joint space (via ln_post+proj here) -- pair this with the identical
checkpoint's own text tower (pcr/models/clip_text_encoder.py's clip_arch), not a different one.
"""
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_dense_part_encoder import ClipBPAMEncoder, _interpolate_pos_embed


class ClipViTDenseBackbone(nn.Module):
    """Frozen CLIP ViT, modified only at the last transformer block (Q/K attention -> V-V
    attention, CLIP Surgery/SCLIP's trick) so per-patch identity survives to the output instead
    of being collapsed toward the class token the way stock CLIP collapses it. Returns raw,
    pre-projection dense tokens at the ViT's own internal transformer width (1024 for ViT-L/14,
    768 for ViT-B/16) -- the same convention BPBreID's own pixel classifier uses (classifies on
    the backbone's raw width, before any dimension reduction). `project()` below applies CLIP's
    own ln_post+proj afterward, landing every token in the final, text-aligned joint space
    (768 for ViT-L/14, 512 for ViT-B/16).

    Note on patch-grid divisibility, checked directly, not assumed: this repo's 384x128 input
    with ViT-L/14's patch_size=14 does NOT divide evenly (384/14=27.43, 128/14=9.14). This isn't a
    crash risk -- nn.Conv2d(kernel_size=stride=patch_size, no padding)'s output spatial size is
    exactly height//patch_size (floor division, verified algebraically: for kernel=stride=P,
    floor((H-P)/P)+1 == floor(H/P) for any H,P), which is exactly what new_grid below computes --
    so patch_feats' actual shape and the interpolated positional embedding's shape always agree.
    The real (minor, not a bug) consequence is a few pixels along the bottom/right edge of every
    image falling outside the patch grid entirely, silently dropped rather than zero-padded.
    """

    def __init__(self, clip_arch='ViT-L/14', height=384, width=128, device='cuda'):
        super(ClipViTDenseBackbone, self).__init__()
        clip_model, _ = clip.load(clip_arch, device=device, jit=False)
        # Same fp16-training pitfall already found and fixed for ClipRN50DenseBackbone (see that
        # file's own comment on this): clip.load() only casts back to fp32 for device=='cpu', so
        # on cuda every weight starts in fp16. Harmless for Stage 1 (frozen, forward-only), but
        # Stage 2 unfreezes and trains this backbone with plain Adam + loss.backward(), no
        # GradScaler -- applying this fix here too, proactively, rather than waiting to
        # rediscover the same NaN failure mode a second time.
        clip_model = clip_model.float()
        visual = clip_model.visual
        assert hasattr(visual, 'transformer'), (
            "needs a ViT-based CLIP arch (has patch tokens to keep) -- RN50-style CLIP visual "
            "towers have no per-patch token sequence to preserve (see ClipRN50DenseBackbone "
            "instead).")

        self.conv1 = visual.conv1
        self.class_embedding = visual.class_embedding
        self.ln_pre = visual.ln_pre
        self.resblocks = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.dtype = clip_model.dtype
        self.num_heads = self.resblocks[-1].attn.num_heads
        self.vision_width = self.conv1.out_channels   # 1024 for ViT-L/14, 768 for ViT-B/16
        self.embed_dim = self.proj.shape[1]            # 768 for ViT-L/14, 512 for ViT-B/16

        patch_size = self.conv1.kernel_size[0]
        orig_grid = visual.input_resolution // patch_size
        new_grid = (height // patch_size, width // patch_size)
        pos_embed = _interpolate_pos_embed(visual.positional_embedding.float(), orig_grid, new_grid)
        self.register_buffer('positional_embedding', pos_embed.type(self.dtype))
        self.grid_h, self.grid_w = new_grid

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, images):
        """images: [B, 3, H, W], CLIP-normalized (pcr.models.clip_dense_part_encoder.CLIP_MEAN/
        CLIP_STD, not BPBreID's ImageNet stats). Returns patch_feats [B, grid_h*grid_w,
        vision_width] (raw, not L2-normalized, CLS dropped) -- every patch individually carried
        to the output, not blurred into one summary vector."""
        x = images.type(self.dtype)
        x = self.conv1(x)  # [B, vision_width, grid_h, grid_w]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, grid_h*grid_w, vision_width]
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)  # [B, 1+grid_h*grid_w, vision_width]
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND, matches CLIP's own internal layout
        for block in self.resblocks[:-1]:
            x = block(x)  # normal MHSA -- unchanged, this is where per-patch content is built

        # Last block only: replace Q/K-based attention with V-V attention (CLIP Surgery / SCLIP)
        # so this final step stops mixing patches into each other -- every patch keeps its own
        # identity through to the output instead of collapsing toward one CLS summary.
        last = self.resblocks[-1]
        normed = last.ln_1(x)  # [L, B, vision_width]
        L, B, D = normed.shape
        Wv = last.attn.in_proj_weight[2 * D:3 * D]
        bv = last.attn.in_proj_bias[2 * D:3 * D]
        v = F.linear(normed, Wv, bv)  # [L, B, D]
        head_dim = D // self.num_heads
        v_heads = v.reshape(L, B * self.num_heads, head_dim).permute(1, 0, 2)  # [B*heads, L, head_dim]
        attn = torch.softmax((v_heads @ v_heads.transpose(-1, -2)) / (head_dim ** 0.5), dim=-1)
        out = attn @ v_heads  # [B*heads, L, head_dim]
        out = out.permute(1, 0, 2).reshape(L, B, D)
        out = last.attn.out_proj(out)
        x = x + out
        x = x + last.mlp(last.ln_2(x))

        x = x.permute(1, 0, 2)  # LND -> NLD
        return x[:, 1:, :].float()  # drop CLS -- raw, pre-projection, vision_width-dim

    def project(self, patch_feats):
        """patch_feats: [..., vision_width] (raw, from forward()). Applies CLIP's own ln_post +
        proj -- the exact op CLIP normally reserves for the CLS token alone -- landing every
        token in the final, text-aligned joint space. Cast back to float32 to match this repo's
        convention for every embedding a loss touches."""
        return (self.ln_post(patch_feats.type(self.dtype)) @ self.proj).float()


def ClipViTBPAMEncoder(clip_arch='ViT-L/14', height=384, width=128, num_parts=5,
                        checkpoint_path=None, device='cuda'):
    backbone = ClipViTDenseBackbone(clip_arch, height, width, device)
    return ClipBPAMEncoder(backbone, num_parts, checkpoint_path, device)
