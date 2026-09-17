"""L_crossalign: aligns CrossAttentionBlock's cross-attention pattern to CLIP's own internal
text self-attention pattern for the same identity (see METHODOLOGY.md's Stage 2 / CAB section)."""
import torch
import torch.nn.functional as F


def cross_attention_alignment_loss(A_cross, A_text_internal):
    """A_cross, A_text_internal: [B, N, N] (rows are softmax distributions). Row-wise KL
    divergence, mean over batch and rows."""
    A_cross_log = torch.log(A_cross.clamp_min(1e-8))
    loss = F.kl_div(A_cross_log, A_text_internal, reduction="none", log_target=False)
    return loss.sum(dim=-1).mean()
