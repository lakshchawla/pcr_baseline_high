"""Single-descriptor fusion for inference/gallery-indexing (plans/reid_pipeline_plan.md section 4),
adapted to this repo's actual encoder convention: BPBReIDEncoder.forward returns one [B, M, D]
tensor (branch 0 = foreground/global, branches 1..K = parts, per-branch L2-normalized already)
and one [B, M] visibility tensor, not the plan's separate global_feat/part_feats/visibility[K]
arguments -- the fusion math below is the same, just indexed against branch 0 instead of a
trailing global slot.

Purely additive and optional: the existing part-distance matching in pcr/evaluators.py and
pcr/utils/part_distance.py::compute_bpb_pairwise_distance remains the default (and only) matching
path used by every training/eval script in this repo. FusionHead exists only for a use case those
scripts don't cover -- exporting one fixed-length vector per identity for fast ANN/FAISS-style
retrieval, per the plan's own framing of when fusion is useful (heavy-occlusion re-ranking
pipelines: fused vector for top-N retrieval, then the native part-wise distance only on the
top-N candidates).

Two modes, both from the plan's own pseudocode:
  - 'fixed_weights': a single scalar weight for the global branch, one shared scalar weight for
    all part branches, gated by that branch's visibility (invisible parts contribute zero).
  - 'learned_gate': a small MLP maps [global_feat, visibility] -> per-branch softmax weights,
    trained jointly wherever FusionHead is wired into a training loop (not wired into any script
    yet -- this stage only builds and verifies the module itself, per the approved plan).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionHead(nn.Module):
    def __init__(self, num_branches, embed_dim, mode='fixed_weights', w_global=1.0, w_part=0.5,
                 gate_hidden_dim=128):
        super(FusionHead, self).__init__()
        if mode not in ('fixed_weights', 'learned_gate'):
            raise ValueError("mode must be 'fixed_weights' or 'learned_gate', got {!r}".format(mode))
        self.num_branches = num_branches
        self.embed_dim = embed_dim
        self.mode = mode

        if mode == 'fixed_weights':
            weights = torch.full((num_branches,), float(w_part))
            weights[0] = w_global
            self.register_buffer('fixed_weights', weights)
            self.gate = None
        else:
            self.fixed_weights = None
            self.gate = nn.Sequential(
                nn.Linear(embed_dim + num_branches, gate_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden_dim, num_branches),
            )

    def forward(self, f_out, visibility):
        """f_out: [B, M, D] per-branch embeddings, branch 0 = global/foreground. visibility:
        [B, M] bool or float. Returns a single L2-normalized [B, M*D] fused descriptor."""
        f_n = F.normalize(f_out, dim=-1)
        vis = visibility.float()

        if self.mode == 'fixed_weights':
            w = self.fixed_weights.unsqueeze(0)                         # [1, M]
            global_w = w[:, :1].expand(f_out.size(0), -1)               # [B, 1], not visibility-gated
            parts_w = w[:, 1:] * vis[:, 1:]                             # [B, M-1], zero when invisible
            w_eff = torch.cat([global_w, parts_w], dim=-1)              # [B, M]
        else:
            gate_input = torch.cat([f_out[:, 0, :], vis], dim=-1)
            gate_weights = F.softmax(self.gate(gate_input), dim=-1)     # [B, M]
            w_eff = torch.cat([gate_weights[:, :1], gate_weights[:, 1:] * vis[:, 1:]], dim=-1)

        weighted = w_eff.unsqueeze(-1) * f_n                            # [B, M, D]
        fused = weighted.flatten(1)                                     # [B, M*D]
        return F.normalize(fused, dim=-1)
