"""Part-generalized version of ICE's hard-instance contrastive loss (ice/loss/contrastive.py
::ViewContrastiveLoss, "L_h_ins" in the ICE paper, Eq. 9). q = student encoder features on
view 1, k = EMA teacher encoder features on view 2; for each anchor, the hardest same-pseudo-ID
sample in k (lowest similarity) is the single positive, every different-ID sample is a negative,
InfoNCE/cross-entropy over [positive, all negatives].

The only generalization from ICE's original (which operates on flat [B, D] vectors) is how the
[B, B] similarity matrix is obtained from BPBReID's [B, M, D] per-branch embeddings: a
visibility-gated mean across branches, mirroring pcr/utils/part_distance.py's convention (bool
visibility -> mutual outer-product mask; continuous -> sqrt-product soft gate) -- but note this
combines *similarity*, not *distance*, so branch-pairs with zero mutual visibility fall back to
0 (neutral), not part_distance.py's "worst observed + 1" sentinel: a -1 fallback here would look
like a maximally-similar... no, maximally-*dissimilar* pair to batch_hard's positive search
(which picks the *lowest* similarity as the hardest positive), consistently and wrongly
preferring pairs with no shared visible evidence over genuinely hard-but-visible ones. The
hard-mining/InfoNCE math below (batch_hard + softmax cross-entropy) is otherwise unchanged from
ICE's original.
"""
import torch
from torch import nn
import torch.nn.functional as F


def _combine_similarity(sim, vis_q, vis_k):
    """sim: [M, B, B] per-branch cosine similarity. vis_q, vis_k: [B, M]. Returns [B, B]."""
    vis_q_t = vis_q.t()  # [M, B]
    vis_k_t = vis_k.t()  # [M, B]
    if vis_q.dtype == torch.bool and vis_k.dtype == torch.bool:
        mask = (vis_q_t.unsqueeze(2) & vis_k_t.unsqueeze(1)).float()
    else:
        mask = torch.sqrt(vis_q_t.float().unsqueeze(2) * vis_k_t.float().unsqueeze(1))
    return (sim * mask).sum(0) / mask.sum(0).clamp_min(1e-6)


class PartViewContrastiveLoss(nn.Module):
    def __init__(self, num_instance=4, T=1.0):
        super(PartViewContrastiveLoss, self).__init__()
        self.num_instance = num_instance
        self.T = T

    def forward(self, q, q_vis, k, k_vis, label):
        """q, k: [B, M, D] (already per-branch L2-normalized by BPBReIDEncoder). q_vis, k_vis:
        [B, M]. label: [B] pseudo-labels, used only to define same-identity pairs."""
        B = q.size(0)
        sim = torch.einsum('bmd,nmd->mbn', q, k)  # [M, B, B]
        mat_sim = _combine_similarity(sim, q_vis, k_vis)  # [B, B]

        mat_eq = label.expand(B, B).eq(label.expand(B, B).t()).float()
        hard_p = self.batch_hard_positive(mat_sim, mat_eq)
        l_pos = hard_p.view(B, 1)

        mat_ne = label.expand(B, B).ne(label.expand(B, B).t())
        negatives = torch.masked_select(mat_sim, mat_ne).view(B, -1)

        out = torch.cat((l_pos, negatives), dim=1) / self.T
        targets = torch.zeros(B, device=q.device, dtype=torch.long)
        log_prob = F.log_softmax(out, dim=1)
        return F.nll_loss(log_prob, targets)

    @staticmethod
    def batch_hard_positive(mat_sim, mat_eq):
        """Lowest-similarity same-identity pair per anchor -- the hardest positive to pull
        together."""
        sorted_sim, _ = torch.sort(mat_sim + 9999999. * (1 - mat_eq), dim=1, descending=False)
        return sorted_sim[:, 0]
