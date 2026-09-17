"""Frozen CLIP text tower, faithful to CLIP-ReID's TextEncoder
(../CLIP-ReID/model/make_model_clipreid.py, lines 31-48) -- only the text-side transformer,
positional_embedding, ln_final, text_projection and token_embedding are kept. CLIP's own visual
encoder is never loaded/used anywhere in this pipeline: BPBreID's backbone is the sole visual
encoder (pcr/models/bpbreid_encoder.py), CLIP contributes only a frozen semantic anchor via text.

Loaded via the official openai/CLIP package (`pip install git+https://github.com/openai/CLIP.git`,
done in pcr2-run), not CLIP-ReID's own vendored fork under CLIP-ReID/model/clip/ -- that fork is
patched specifically for CLIP-ReID's own ReID visual-encoder resolution/stride interpolation,
irrelevant here since the visual tower is never built.

RN50 is the default arch: its text embedding dim (`embed_dim`, the output of `text_projection` --
what actually gets compared against BPBreID's part embeddings in the alignment losses) is 1024,
exactly matching BPBReIDModelCfg.dim_reduce_output -- no projection layer needed between BPBreID's
part embeddings and CLIP's text embeddings there. Any other CLIP arch works too as long as its own
`embed_dim` is set to match on both sides -- ViT-B/32 and ViT-B/16 are 512, ViT-L/14 and RN50x16 are
768, RN50x4 is 640, RN101 is 512, RN50x64 is 1024 (same as RN50) -- none of the standard archs
produce 2048, so there is no way to reach that size through clip_arch choice alone.

`embed_dim` is NOT the same number as this transformer's own internal width (`transformer_width`,
exposed separately below) -- a real bug this repo hit switching from ViT-B/16 to RN50: ViT-B/16
happens to have `transformer_width == embed_dim` (both 512), which hid the distinction completely;
RN50's are different (512 internal, 1024 final, per `third_party/clip/model.py`'s own `CLIP.
__init__`: `transformer_width` sizes `token_embedding`/`positional_embedding`/the transformer
itself, while `text_projection` is a separate `[transformer_width, embed_dim]` matrix applied only
once, after the transformer, to reach the final joint embedding space). `PromptLearner`'s own
learnable context tokens get concatenated with this class's frozen `token_prefix`/`token_suffix`
(built from `token_embedding`, i.e. `transformer_width`-sized) *before* they ever reach this
class's `forward()` -- so they must be sized to `transformer_width`, not `embed_dim`, or that
concatenation fails outright (see PromptLearner's own docstring for the fix; progress.md has the
full story of finding this).
"""
import clip
import torch
import torch.nn as nn


class ClipTextEncoder(nn.Module):
    def __init__(self, clip_arch='RN50', device='cuda'):
        super(ClipTextEncoder, self).__init__()
        clip_model, _ = clip.load(clip_arch, device=device, jit=False)
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.embed_dim = self.text_projection.shape[1]
        # This transformer's own internal width -- NOT the same as embed_dim above in general (see
        # this file's own module docstring). Callers that build tokens meant to be concatenated
        # with token_embedding's own output (PromptLearner's prefix/suffix splicing) must size
        # those tokens to this, not to embed_dim.
        self.transformer_width = self.text_projection.shape[0]

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, prompts, tokenized_prompts):
        """prompts: [B, 77, ctx_dim] assembled prompt embeddings (from PromptLearner -- NOT
        raw token ids, the embedding lookup already happened). tokenized_prompts: [1, 77] integer
        token ids from the shared template, used only to locate each sequence's EOT position via
        argmax (EOT has the highest token id in CLIP's vocabulary); broadcasts against the batch
        since every prompt shares the same template structure regardless of its learnable
        context. Ported from CLIP-ReID's TextEncoder.forward.
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_branch_prepool(self, prompts, tokenized_prompts):
        """Same steps as forward() through ln_final + EOT-position pooling, but stops before the
        final `@ text_projection` -- returns the transformer's own internal, transformer_width-
        sized representation, not the final embed_dim-sized joint-space one. Needed by
        encode_branch_self_attention below: the last transformer block's own attention always
        operates at transformer_width, so replaying it directly on a post-projection (embed_dim)
        vector only happens to work when the two are equal (ViT-B/16/32); this method makes that
        replay correct for every arch, RN50 included, where they differ (see this class's own
        docstring on transformer_width vs. embed_dim)."""
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]  # [B, transformer_width]

    def encode_branch_self_attention(self, branch_embeddings):
        """branch_embeddings: [B, L, transformer_width] -- e.g. one identity's L=1+K per-branch
        pooled representations from encode_branch_prepool above (not a CLIP-template token
        sequence, and not the final embed_dim-sized text_prototypes -- see that method's own
        docstring for why). Runs the last transformer block's own attention with
        need_weights=True (third_party/clip hardcodes it False) to get a real [B, L, L]
        self-attention matrix over these L tokens, with no attention mask -- used only offline,
        no_grad (see examples/cache_text_anchors.py)."""
        resblock = self.transformer.resblocks[-1]
        x = resblock.ln_1(branch_embeddings.permute(1, 0, 2).type(self.dtype))  # NLD -> LND
        _, attn = resblock.attn(x, x, x, need_weights=True, average_attn_weights=True)
        return attn  # [B, L, L]
