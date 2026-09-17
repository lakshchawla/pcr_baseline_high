"""Relational attention on the text side (TextualAttentionBlock, over PromptLearner's per-branch
learnable context tokens) plus the image side's global aggregator (AttentionPoolingBlock).

Architecture note (2026-09-17, this fork): the image-side self-attention block
(`VisualAttentionBlock`, VAB) and the Stage-2 cross-attention block (`CrossAttentionBlock`, CAB)
are gone. Evidence from the two real runs: VAB's zero-init gate sat at -0.002 after 86 Stage 1
epochs and 0.000 after 58 Stage 2 epochs -- the objective never found a reason to mix the K
pooled part tokens, and mixing them is exactly the part-specificity erosion L_part_diag now
penalizes. CAB's gate did open (-0.46 in Stage 2) but CAB is never run at retrieval (it needs
the ground-truth identity to fetch its text context), so Stage 2's align loss was shaping a
feature the evaluator never computes. The image side is now: masks -> GWAP-pooled parts ->
foreground-gated -> `AttentionPoolingBlock` for the global; the only text/image attention left is
patch-level, inside ClipBPAMEncoder (the CLIP-native mask blend), which is what actually carries
spatial information. See progress.md's 2026-09-17 entries.

Foreground is a coarse "how much of this part is real foreground, not background" signal, not a
semantic region: it GATES each of the K part tokens multiplicatively (`part_tokens[k] *=
visibility[k]`) before pooling, reusing the per-part visibility the encoder already computes.
The "global" embedding is `AttentionPoolingBlock`'s output -- a Set-Transformer/PMA-style
single-learnable-query attention pool over the gated part tokens, visibility-biased -- and takes
over branch 0's exact slot (`apply_part_pooling` below), so every downstream consumer
(id_classifiers, bn_necks, the per-branch loss loops) indexes branches exactly as before.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch's nn.TransformerEncoderLayer automatically switches to a fused, native fast-path kernel
# (torch._transformer_encoder_layer_fwd) whenever a layer is in eval() mode -- found the hard way,
# by hand: cache_text_anchors.py calls TextualAttentionBlock through PromptLearner.eval(), and with
# this session's real per-identity visibility values (a normal, non-uniform additive attn_mask,
# nothing extreme -- no zeros, no huge magnitudes) that fast path produced NaN for every part
# branch, on every identity, deterministically. Confirmed the mask math itself is correct: the
# exact same tensors run through the same module in .train() mode (fast path never engages there)
# and eval() with a *uniform* mask (no real masking effect) both produce finite output -- only the
# combination of eval() mode and a real, non-uniform mask breaks. Disabling this fast path globally
# is the documented way around it (torch.backends.mha docs); this module is the only place in this
# repo that builds an nn.TransformerEncoder, so there's no other fast-path user to slow down, and
# the block here is tiny (M*n_ctx tokens) where the fused kernel's speed advantage is negligible
# next to correctness.
torch.backends.mha.set_fastpath_enabled(False)


def _visibility_attn_bias(visibility, num_heads, eps=1e-6):
    """visibility: [B, L], one reliability score per key token, in (0, 1]. Returns an additive
    attention-score bias [B*num_heads, L, L] suitable for nn.TransformerEncoder's own `mask`
    argument: log(v_j) added to every query's raw attention score for key j, before the softmax --
    the same mechanism nn.TransformerEncoderLayer's own key_padding_mask uses for a hard 0/1 mask
    (log(0) = -inf), generalized here to a continuous score, so an unreliable key contributes less
    regardless of how well it happens to correlate with a query in raw dot-product terms. Identical
    across every query row and attention head: only the key index carries information."""
    B, L = visibility.shape
    log_vis = torch.log(visibility.clamp(min=eps))  # [B, L]
    bias = log_vis.unsqueeze(1).expand(B, L, L)      # [B, L, L], broadcast over queries
    return bias.unsqueeze(1).expand(B, num_heads, L, L).reshape(B * num_heads, L, L)


def _replay_with_attention(encoder, x, mask):
    """Manually replays a norm_first nn.TransformerEncoder's own layers with need_weights=True --
    nn.TransformerEncoderLayer.forward always discards attention weights (need_weights=False,
    hardcoded in its _sa_block). Returns (output, last layer's attention [B, L, L], heads
    averaged) -- TAB uses a single-layer encoder today, so "last layer" is the only
    layer; a deeper stack would only expose its final layer's pattern this way."""
    attn = None
    for layer in encoder.layers:
        normed = layer.norm1(x)
        sa_out, attn = layer.self_attn(normed, normed, normed, attn_mask=mask,
                                        need_weights=True, average_attn_weights=True)
        x = x + layer.dropout1(sa_out)
        x = x + layer._ff_block(layer.norm2(x))
    return x, attn


class AttentionPoolingBlock(nn.Module):
    """Set-to-vector aggregation over the K (foreground-gated) part tokens: a single learnable
    seed query attends over the part set and the resulting weighted sum becomes the "global"
    embedding -- Pooling by Multi-Head Attention (Set Transformer; the same mechanism CoCa and
    Perceiver use to compress a variable token set into one vector). Strictly better fit than a
    fixed stack/mean here, since it lets the weighting be both learned and image-specific (e.g.
    torso-heavy when legs are occluded) instead of treating every part as equally important
    regardless of visibility or discriminativeness.

    Visibility-biased exactly like TextualAttentionBlock's own self-attention
    (_visibility_attn_bias): a low-visibility part contributes less as a key/value to the pooled
    result, automatically shifting the summary toward whichever parts are actually trustworthy
    for this image.

    Zero-init tanh gate: at init, `forward` returns the plain visibility-weighted mean of the
    part tokens (not an arbitrary random pooled vector) -- a safe, sane starting point that
    training only deviates from once it finds a reason to, same convention as TAB's own
    zero-init gates elsewhere in this file."""

    def __init__(self, dim, num_heads=4):
        super(AttentionPoolingBlock, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, part_tokens, part_visibility):
        """part_tokens: [B, K, D]. part_visibility: [B, K]. Returns (global [B, D], attn [B, K]
        averaged over heads, query dim squeezed)."""
        B, K, D = part_tokens.shape
        seed = self.seed.expand(B, -1, -1)  # [B, 1, D]
        q = self.q_proj(seed).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(part_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(part_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, heads, 1, K]
        log_vis = torch.log(part_visibility.clamp(min=1e-6)).view(B, 1, 1, K)
        attn = torch.softmax(logits + log_vis, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, D)
        out = self.out_proj(out)

        vis_w = part_visibility.unsqueeze(-1)  # [B, K, 1]
        mean_fallback = (part_tokens * vis_w).sum(dim=1) / vis_w.sum(dim=1).clamp(min=1e-6)
        pooled = mean_fallback + torch.tanh(self.gate) * out
        return F.normalize(pooled, p=2, dim=-1), attn.mean(dim=1).squeeze(1)


def apply_part_pooling(pool, f_out, vis, has_global):
    """Foreground gates the K parts, `pool` (AttentionPoolingBlock) aggregates the gated parts
    into the global embedding, which takes over branch 0's exact position; the K part tokens
    pass through untouched (no cross-part mixing -- see this module's docstring for why VAB was
    removed). Replaces the old `apply_vab_with_pooling`.

    f_out/vis: [B, M, D]/[B, M], the encoder's own branch order (0=foreground, 1..K=parts, and a
    real CLIP-native global appended last iff has_global -- see clip_dense_part_encoder.py). That
    native global branch, when present, is passed through completely unchanged.

    Returns (combined [B, M, D], pool_attn [B, K]): combined has the exact same shape/branch
    layout as f_out; pool_attn is the pooling block's own visibility-biased weighting over the K
    parts (diagnostic)."""
    foreground_end = 1
    parts_end = foreground_end + (f_out.size(1) - foreground_end - (1 if has_global else 0))
    part_tokens = f_out[:, foreground_end:parts_end, :]
    part_vis = vis[:, foreground_end:parts_end]

    # Foreground gates the parts multiplicatively (a confidence signal, not a peer token) --
    # reuses each part's own visibility, already derived from the same foreground-vs-background
    # pixel classifier that would otherwise feed a separate "foreground" branch.
    gated_parts = part_tokens * part_vis.unsqueeze(-1)
    global_pooled, pool_attn = pool(gated_parts, part_vis)

    pieces = [global_pooled.unsqueeze(1), part_tokens]
    if has_global:
        pieces.append(f_out[:, -1:, :])
    combined = torch.cat(pieces, dim=1)
    return combined, pool_attn


class TextualAttentionBlock(nn.Module):
    """Bidirectional self-attention over a person's M*n_ctx learnable branch-context tokens (text
    side, M=1+K: global/foreground + K parts), run before any single branch's prompt is assembled
    -- so a branch's context can be informed by every other branch's, which the frozen CLIP text
    encoder's own causally-masked self-attention can never provide on its own (a token can only
    attend to earlier tokens there).

    Training-only: exists solely within Stage 1, alongside PromptLearner. Both are frozen and
    discarded once Stage 1 ends -- cache_text_anchors.py reads their state once to build the
    frozen text-prototype table Stage 2 actually uses, and neither is loaded again afterward.

    Zero-init tanh-gated residual (`gated=True`, the default), same convention as every other
    trainable block in this file. This block used to have no gate and no residual ("training-
    only, so no no-op-at-inference concern") -- but "training-only" doesn't make its output
    harmless, because its output is exactly what gets cached as Stage 2's text prototypes.
    Measured on a fully-trained ungated checkpoint (2026-09-17, see progress.md): the mixed ctx
    tokens came out at norm ~34 vs ~0.4 for the learnable ctx and for CLIP's own token
    embeddings around them (the learnable ctx was numerically irrelevant to the prompt), the
    branch-to-branch attention was near-uniform (mean diagonal 0.131 vs 0.143 uniform), and the
    K part prompts of one identity landed at cosine 0.98 in the joint space -- the parts had
    collapsed into one identity vector, through this block. Gating fixes the scale (at init the
    prompt IS ctx, at CLIP token scale) and removes the free one-step collapse path (uniform
    mixing now has to be learned against an objective, not inherited from a random init).
    `gated=False` only exists so cache_text_anchors.py can still replay checkpoints trained
    before this change (their state dict has no `gate` key; an ungated output is not
    representable by the gated form).
    """

    def __init__(self, ctx_dim, n_ctx, num_heads=4, num_layers=1, gated=True):
        super(TextualAttentionBlock, self).__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=ctx_dim, nhead=num_heads, dim_feedforward=ctx_dim * 2,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.n_ctx = n_ctx
        self.num_heads = num_heads
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, ctx_tokens, branch_visibility):
        """ctx_tokens: [B, M*n_ctx, ctx_dim] -- one batch's raw per-identity branch context, laid
        out as M contiguous n_ctx-token blocks (see PromptLearner.build_part_prompts's own
        slicing). branch_visibility: [B, M], that identity's mean per-branch visibility across
        every cached training image of that identity (there is no single per-image signal here --
        ctx has no per-image input at all, see PromptLearner's own docstring) -- expanded so every
        context token belonging to a given branch shares that branch's score. Returns
        (mixed [B, M*n_ctx, ctx_dim], attn [B, M, M]) -- mixed is the same shape, relationally
        mixed; attn is the raw [B, M*n_ctx, M*n_ctx] self-attention pattern reduced to one M x M
        branch-to-branch summary: summed over each key branch's own n_ctx tokens (each query row
        sums to 1 over the full M*n_ctx keys, so summing -- not averaging -- a key block preserves
        that row's total probability mass), then averaged over each query branch's own n_ctx rows
        (a mean of several valid distributions is itself a valid distribution). Kept as a real
        per-row probability distribution (diagnostic only now that L_relalign, its former consumer,
        went with VAB; in train() mode attention dropout leaves rows summing to ~1, not exactly)."""
        M = branch_visibility.size(1)
        token_visibility = branch_visibility.repeat_interleave(self.n_ctx, dim=1)  # [B, M*n_ctx]
        attn_bias = _visibility_attn_bias(token_visibility, self.num_heads)
        relation_out, attn_full = _replay_with_attention(self.encoder, ctx_tokens, attn_bias)
        if self.gated:
            # tanh(gate) * delta: the transformer's own output already carries ctx_tokens via its
            # internal residuals, so the gated quantity is its deviation from ctx, not the whole
            # output -- at gate=0 the prompt is exactly the raw learnable ctx.
            mixed = ctx_tokens + torch.tanh(self.gate) * (relation_out - ctx_tokens)
        else:
            mixed = relation_out
        B = attn_full.size(0)
        attn = attn_full.view(B, M, self.n_ctx, M, self.n_ctx).sum(dim=4).mean(dim=2)
        return mixed, attn


