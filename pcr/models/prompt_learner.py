"""Per-identity, per-branch prompt learning, generalized from CLIP-ReID's PromptLearner
(../CLIP-ReID/model/make_model_clipreid.py, lines 191-239): one learnable n_ctx-token context per
(identity, branch), spliced into the fixed template and pushed through the frozen CLIP text
encoder. No mixing across branches (the TextualAttentionBlock that used to sit here was removed
2026-09-17 -- see progress.md: it was the fastest route to the K part prompts collapsing into
one identity vector, and CLIP-ReID's own design needs nothing between ctx and the encoder).
"""
import clip
import torch
import torch.nn as nn


class PromptLearner(nn.Module):
    """n_ctx serves two purposes that CLIP-ReID's original code keeps as two separately-named
    variables (n_ctx, n_cls_ctx) always set to the same literal 4: the number of "X" placeholder
    tokens in the fixed template, and the length of the learnable context spliced into their
    place. One name here instead of two, since they must always be equal for the splicing to
    line up. The frozen prefix's own real length ("SOT + 'a photo of a'") is measured separately
    below, independent of n_ctx -- it used to be assumed equal to n_ctx+1, which only happened to
    hold at n_ctx=4 (that phrase's own real token count) and would silently corrupt the prompt at
    any other n_ctx (the first n_ctx-4 "X" placeholder embeddings would get treated as frozen
    prefix instead of learnable context) -- not something this repo has hit yet (every config so
    far uses n_ctx=4), but fixed at the root rather than left as a landmine for the next config
    that changes it.

    One learnable context tensor, `ctx` ([num_identities, num_branches*n_ctx, ctx_dim]), laid out
    as M=1+K contiguous n_ctx-token blocks (branch 0 = global, 1..K = parts).

    Deviation from CLIP-ReID's original, found by actually running the training loop (back when
    this repo still only had UDA/USL, long before this file existed): CLIP-ReID creates its
    context directly in clip_model.dtype (fp16 on GPU). `torch.amp.GradScaler.step()` hard-errors
    ("Attempting to unscale FP16 gradients") on an fp16 leaf parameter -- mixed-precision
    training requires fp32 master weights. Fixed here: `ctx` is fp32 always, cast to the frozen
    buffers' dtype only inside build_part_prompts() when assembling each prompt -- autograd
    handles the cast's backward correctly (the incoming gradient is cast back to fp32 for
    accumulation into the fp32 parameter).
    """

    def __init__(self, num_identities, num_parts, clip_text_encoder, n_ctx=4, device='cuda'):
        """num_branches = 1 + num_parts: branch 0 is the global (matched to the encoder's x_proj),
        1..K the parts (matched to part_xproj) -- see clip_dense_part_encoder.py."""
        super(PromptLearner, self).__init__()
        # transformer_width, NOT embed_dim -- these tokens get concatenated with token_prefix/
        # token_suffix below (built from clip_text_encoder.token_embedding, which outputs
        # transformer_width-sized vectors) and fed through the transformer itself, which also
        # operates at transformer_width throughout; embed_dim (the *final*, post-text_projection
        # size that must match BPBreID's dim_reduce_output) only applies to ClipTextEncoder's own
        # output, after this class is done. See clip_text_encoder.py's own docstring -- ViT-B/16
        # has transformer_width == embed_dim (both 512), which is why using embed_dim here never
        # broke anything until this repo switched to RN50 (512 vs. 1024).
        ctx_dim = clip_text_encoder.transformer_width
        dtype = clip_text_encoder.dtype

        prefix_text = "A photo of a"
        ctx_init = prefix_text + " " + " ".join(["X"] * n_ctx) + " person."
        tokenized_prompts = clip.tokenize(ctx_init).to(device)
        with torch.no_grad():
            embedding = clip_text_encoder.token_embedding(tokenized_prompts).type(dtype)

        # Real length of "SOT + prefix_text", measured by tokenizing prefix_text on its own and
        # dropping that isolated tokenization's own EOT -- independent of n_ctx, unlike the
        # n_ctx+1 this used to assume (see class docstring). BPE tokenizes each word against its
        # own trailing word-boundary marker, so a word's token id doesn't depend on what follows
        # it -- prefix_text's tokens here are identical to its tokens inside ctx_init above.
        prefix_only = clip.tokenize(prefix_text).to(device)
        prefix_len = int((prefix_only != 0).sum().item()) - 1  # drop the isolated EOT, keep SOT

        num_branches = 1 + num_parts

        # fp32 master weights, cast to the frozen buffers' dtype only in build_part_prompts() --
        # see the class docstring for why (GradScaler forbids fp16 leaf parameters)
        ctx_vectors = torch.empty(num_identities, num_branches * n_ctx, ctx_dim, dtype=torch.float32)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        self.prompt_dtype = dtype

        # not trained, but must move with the module (.cuda()/.to()) -- registered as buffers
        # rather than plain attributes, unlike CLIP-ReID's own hardcoded .cuda() call
        self.register_buffer('tokenized_prompts', tokenized_prompts)  # [1, 77]
        self.register_buffer('token_prefix', embedding[:, :prefix_len, :])
        self.register_buffer('token_suffix', embedding[:, prefix_len + n_ctx:, :])

        self.num_identities = num_identities
        self.num_parts = num_parts
        self.num_branches = num_branches
        self.n_ctx = n_ctx

    def _splice(self, ctx):
        """ctx: [B, n_ctx, ctx_dim], already dtype-cast. Returns [B, 77, ctx_dim]: frozen prefix
        + ctx + frozen suffix."""
        b = ctx.size(0)
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        return torch.cat([prefix, ctx, suffix], dim=1)

    def build_part_prompts(self, labels):
        """labels: [B] identity indices. Returns a list of `num_branches` tensors, each
        [B, 77, ctx_dim], in branch order (0 = global, 1..K = parts): that identity's own context
        block for the branch, spliced between the frozen prefix and suffix -- a pure lookup, no
        cross-branch mixing."""
        ctx = self.ctx[labels].type(self.prompt_dtype)  # [B, num_branches*n_ctx, ctx_dim]
        prompts = []
        for b in range(self.num_branches):
            start = b * self.n_ctx
            prompts.append(self._splice(ctx[:, start:start + self.n_ctx, :]))
        return prompts
