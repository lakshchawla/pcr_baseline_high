"""Relational attention across a person's branches, on both the visual side
(VisualAttentionBlock, over BPBreID's pooled branch features) and the text side
(TextualAttentionBlock, over PromptLearner's per-branch learnable context tokens), plus
cross-modal grounding (CrossAttentionBlock).

Architecture note (this fork, `pcr_attn`): global/foreground is no longer a peer token inside
self-attention. Two changes, applied together via `apply_vab_with_pooling` below:

  1. Foreground is a coarse "how much of this part is real foreground, not background" signal,
     not a semantic region on the same footing as "torso" or "legs" -- there's no meaningful
     content for a part to attend to when the "other token" is just a confidence score. It now
     GATES each of the K real part tokens multiplicatively (`part_tokens[k] *= visibility[k]`)
     before they enter attention at all, instead of sitting inside the self-attention set as a
     6th token competing for attention weight against real body parts. (Reuses the per-part
     visibility this encoder already computes rather than inventing a separate "foreground
     confidence" signal -- BPBreID's own pixel classifier already derives both from the same
     foreground-vs-background distinction.)
  2. The "global" embedding is no longer a separately hand-pooled branch -- it's now the output
     of AttentionPoolingBlock, a Set-Transformer/PMA-style single-learnable-query attention pool
     over the (gated, relationally-mixed) K part tokens. A visibility-biased *learned* weighting
     of which parts matter most for this specific image is a strictly better aggregator than a
     fixed stack/mean, and it reuses the same visibility signal used everywhere else in this file.

VisualAttentionBlock itself is now self-attention among the K real parts ONLY (no foreground/
global peer) -- everything else about it (visibility-biased attention logits, zero-init gate,
L2-normalized output) is unchanged. TextualAttentionBlock (text side) is untouched by this fork's
changes; only the visual pipeline's aggregation was rearchitected.

Both self-attention blocks stay visibility-aware exactly as before: a poorly-visible key
contributes less to every other key's post-attention representation, closing the contamination
gap loss-level weighting alone can't reach.
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
# both blocks here are tiny (K=5 tokens) where the fused kernel's speed advantage is negligible
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
    averaged) -- both VAB and TAB use single-layer encoders today, so "last layer" is the only
    layer; a deeper stack would only expose its final layer's pattern this way."""
    attn = None
    for layer in encoder.layers:
        normed = layer.norm1(x)
        sa_out, attn = layer.self_attn(normed, normed, normed, attn_mask=mask,
                                        need_weights=True, average_attn_weights=True)
        x = x + layer.dropout1(sa_out)
        x = x + layer._ff_block(layer.norm2(x))
    return x, attn


class VisualAttentionBlock(nn.Module):
    """Bidirectional self-attention over the K real part features ONLY (image side) -- no
    foreground/global peer token (see this module's own docstring for why; apply_vab_with_pooling
    below gates parts by visibility before they ever reach here). Permanent inference-time module:
    trained in Stage 1 (backbone/BPAM frozen, this is one of the few trainable things), then
    carried over and continues training in Stage 2 (jointly with the now-unfrozen backbone) --
    never discarded, unlike TextualAttentionBlock.

    A learned, zero-initialized residual gate keeps this a no-op at initialization
    (`tanh(0) == 0`, so `forward` returns `part_tokens` unchanged the moment training starts) and
    doubles as a training-stability/interpretability device: the converged value of
    `torch.tanh(self.gate)` is a direct read on how much relational mixing training actually
    found useful for this run -- a gate that stays near 0 is a real (negative) result, not a bug.
    The visibility-aware attention bias below doesn't disturb this: at gate=0, `relation_out`'s
    value (masked or not) is multiplied by zero either way.

    Output is L2-normalized before returning (see changes.md's now-resolved entry on this): the
    residual sum above can drift away from unit norm as the gate moves off zero, but every
    consumer of this output (SupConLoss in Stage 1, PartTripletLoss/CosineAlignLoss in Stage 2)
    computes similarity assuming unit-normalized inputs -- matching BPBreIDEncoder's own
    foreground/global embedding, which is already normalized before this block ever sees the part
    embeddings. Normalizing here, once, means every caller gets a consistent invariant rather than
    each loss call site needing to remember it separately. Doesn't change the zero-init no-op
    property: at gate=0 this returns `normalize(part_tokens)`, and `part_tokens` arrives already
    unit-normalized from BPBreIDEncoder, so it's a true no-op (up to floating-point precision),
    not just an approximate one.
    """

    def __init__(self, dim, num_heads=4, num_layers=1, ff_dim=None):
        super(VisualAttentionBlock, self).__init__()
        ff_dim = ff_dim or dim * 2
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=ff_dim,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.gate = nn.Parameter(torch.zeros(1))
        self.num_heads = num_heads

    def forward(self, branch_tokens, branch_visibility):
        """branch_tokens: [B, K, C], the K real part features (gated by visibility already, see
        apply_vab_with_pooling). branch_visibility: [B, K], that same image's own per-part
        visibility score (same order as branch_tokens) -- used as a soft attention bias so a
        poorly-visible part's near-garbage feature contributes less as a key to every other
        part's post-attention representation. Returns (mixed [B, K, C], attn [B, K, K]) -- mixed
        is relationally mixed and L2-normalized; attn is this call's own self-attention pattern
        (see train_relational_prompts.py's L_relalign, which is the only current consumer)."""
        attn_bias = _visibility_attn_bias(branch_visibility, self.num_heads)
        relation_out, attn = _replay_with_attention(self.encoder, branch_tokens, attn_bias)
        mixed = branch_tokens + torch.tanh(self.gate) * relation_out
        return F.normalize(mixed, p=2, dim=-1), attn


class AttentionPoolingBlock(nn.Module):
    """Set-to-vector aggregation over the K (relationally-mixed) part tokens: a single learnable
    seed query attends over the part set and the resulting weighted sum becomes the "global"
    embedding -- Pooling by Multi-Head Attention (Set Transformer; the same mechanism CoCa and
    Perceiver use to compress a variable token set into one vector). Strictly better fit than a
    fixed stack/mean here, since it lets the weighting be both learned and image-specific (e.g.
    torso-heavy when legs are occluded) instead of treating every part as equally important
    regardless of visibility or discriminativeness.

    Visibility-biased exactly like VisualAttentionBlock's own self-attention
    (_visibility_attn_bias): a low-visibility part contributes less as a key/value to the pooled
    result, automatically shifting the summary toward whichever parts are actually trustworthy
    for this image.

    Zero-init tanh gate: at init, `forward` returns the plain visibility-weighted mean of the
    part tokens (not an arbitrary random pooled vector) -- a safe, sane starting point that
    training only deviates from once it finds a reason to, same convention as VAB/CAB's own
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


def apply_vab_with_pooling(vab, pool, f_out, vis, has_global):
    """Replaces the old "VAB mixes all M branches as peers" pipeline: foreground gates the K
    parts, VAB relationally mixes the (gated) parts only, then `pool` (AttentionPoolingBlock)
    aggregates the mixed parts into a new global embedding -- which takes over branch 0's exact
    position (every downstream consumer -- id_classifiers, bn_necks, the per-branch triplet/align
    loops -- keeps indexing branch 0 as "the global anchor" unchanged).

    f_out/vis: [B, M, D]/[B, M], BPBreID's own branch order (0=foreground, 1..K=parts, and a real
    CLIP-native global appended last iff has_global -- see clip_dense_part_encoder.py). That
    native global branch, when present, is passed through completely unchanged: it's CLIP's own
    pretrained whole-image embedding, unrelated to the part-pooling this function replaces.

    Returns (combined [B, M, D], attn [B, K, K]): combined has the exact same shape/branch layout
    as the old vab(f_out, vis) call it replaces, so no downstream code needs to change; attn is
    VAB's own relational pattern among the K parts (L_relalign's consumer, Stage 1 only)."""
    foreground_end = 1
    parts_end = foreground_end + (f_out.size(1) - foreground_end - (1 if has_global else 0))
    part_tokens = f_out[:, foreground_end:parts_end, :]
    part_vis = vis[:, foreground_end:parts_end]

    # Foreground gates the parts multiplicatively (a confidence signal, not a peer token) --
    # reuses each part's own visibility, already derived from the same foreground-vs-background
    # pixel classifier that would otherwise feed a separate "foreground" branch.
    gated_parts = part_tokens * part_vis.unsqueeze(-1)

    mixed_parts, attn = vab(gated_parts, part_vis)
    global_pooled, _ = pool(mixed_parts, part_vis)

    pieces = [global_pooled.unsqueeze(1), mixed_parts]
    if has_global:
        pieces.append(f_out[:, -1:, :])
    combined = torch.cat(pieces, dim=1)
    return combined, attn


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
        (a mean of several valid distributions is itself a valid distribution). Needed as a real
        probability distribution, each row summing to 1, since L_relalign
        (train_relational_prompts.py) feeds this into a KL divergence against VAB's native
        [B, M, M] -- naively averaging over both axes (as a purely-visual heatmap wouldn't need to
        care about) leaves each row summing to 1/n_ctx instead, silently breaking KL's
        non-negativity."""
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


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention: `query_tokens` attends to `context_tokens` from the other
    modality. Tanh-gated, zero-init residual (starts as an identity function). See
    METHODOLOGY.md's Stage 2 / CAB section for how this is used.

    Two deviations from a generic cross-attention block (this fork, `pcr_attn`), both aimed at
    the same failure mode -- every part's query collapsing onto whichever text token is
    generically most useful (e.g. the branch with the strongest average signal) instead of its
    own matched counterpart:

    1. Cosine-normalized logits with a learnable temperature (Swin-V2's "scaled cosine
       attention"), not raw scaled dot-product on unnormalized embeddings. CLIP's own pretraining
       compares image/text purely via normalized cosine similarity, temperature-scaled -- a raw
       QK^T/sqrt(d) on unnormalized vectors introduces a norm-dependence CLIP's weights were never
       calibrated for. `log_logit_scale` follows CLIP's own `logit_scale` convention.
    2. A learnable scalar diagonal bias on the attention logits (only when Nq == Nk, i.e. query
       branch i and context branch i are the same real branch), initialized positive so query i
       starts with a preference for context i -- its own matched branch -- while training is
       free to loosen or even reverse it wherever real cross-talk helps."""

    def __init__(self, dim, num_heads=4, diag_init=2.0):
        super(CrossAttentionBlock, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(1))
        # CLIP-style learnable temperature: logits = cos_sim * exp(log_logit_scale), clamped so
        # the effective temperature never drops below 0.01 (exp(log_logit_scale) <= 100).
        self.log_logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))
        # diag_bias is a fraction OF logit_scale, not an absolute logit value: cosine similarities
        # are bounded in [-1,1], so a diagonal bonus of `diag_init * logit_scale` (diag_init=2.0
        # covers the entire possible off-diagonal range twice over) reliably makes query i's own
        # matched branch i the argmax at init, regardless of what logit_scale itself is -- an
        # absolute bias would need re-tuning every time logit_scale moves (confirmed empirically:
        # a fixed absolute bias of the same initial magnitude only won the diagonal ~40-60% of the
        # time once logit_scale left its initial value). Scaling with logit_scale keeps the ratio,
        # and therefore this init guarantee, invariant as logit_scale is learned.
        self.diag_bias = nn.Parameter(torch.tensor(float(diag_init)))

    def forward(self, query_tokens, context_tokens):
        """query_tokens: [B, Nq, D]. context_tokens: [B, Nk, D]. Returns (updated_query
        [B, Nq, D], attn_weights [B, Nq, Nk] averaged over heads)."""
        B, Nq, D = query_tokens.shape
        Nk = context_tokens.shape[1]
        q = self.q_proj(query_tokens).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context_tokens).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context_tokens).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        logit_scale = self.log_logit_scale.exp().clamp(max=100.0)
        logits = (q @ k.transpose(-1, -2)) * logit_scale  # [B, heads, Nq, Nk], cosine sim in [-1,1]
        if Nq == Nk:
            logits = logits + torch.eye(Nq, device=logits.device, dtype=logits.dtype) * logit_scale * self.diag_bias
        attn = torch.softmax(logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        out = self.out_proj(out)
        updated_query = query_tokens + torch.tanh(self.gate) * out
        return F.normalize(updated_query, p=2, dim=-1), attn.mean(dim=1)
