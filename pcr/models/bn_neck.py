"""BNNeck (Luo et al., "Bag of Tricks and a Strong Baseline for Deep Person Re-identification",
CVPRW 2019) -- one BatchNorm1d per branch, inserted between the pooled feature (combined, built in
examples/train_relational_finetune.py::compute_losses from BPBreIDEncoder's global embedding +
VisualAttentionBlock's mixed part embeddings) and whichever loss needs a differently-shaped
version of it.

The problem this solves: without it, Stage 2's triplet loss and its ID/align losses all read the
exact same feature vector, but want it shaped two different, competing ways -- triplet's Euclidean-
margin objective wants features spread out across the embedding space, while ID/align's softmax-
style classification converges better on a more compact, roughly-hyperspherical distribution.
Training one shared feature to satisfy both at once means each loss's gradient partially fights the
other's, every step. The fix: triplet loss keeps reading the feature BEFORE this BN layer
(unchanged -- see compute_losses, which still passes `combined` itself to PartTripletLoss); id/align
loss reads the feature AFTER it. Each loss gets the shape it actually wants, from the exact same
underlying pooled feature, via one extra learnable (but tiny -- 2 x embed_dim scale/shift
parameters per branch) affine transform in between.

Bias disabled on every BatchNorm1d (`bn.bias.requires_grad_(False)`) -- the original BoT paper's
own convention: a following linear classifier already has its own bias-like degree of freedom, so
a learnable BN shift is redundant with it, and BoT's own ablation found leaving it fixed at zero
(pure learned scale, no shift) works better in practice. `PartIdClassifiers`'s own linear layer is
left untouched (still has its own bias) -- that's a separate, additional detail from BoT's full
recipe, not needed for the specific triplet-vs-id/align interference this module fixes.

Scope note, updated after Evaluator was later fixed to apply VisualAttentionBlock at test time
too (pcr/evaluators.py::Evaluator(encoder, vab)): BoT's own paper also recommends using the
*post*-BN feature for retrieval, not just at training time, since the BN-normalized space is
empirically more discriminative for retrieval too. That part is still not wired in -- the
Evaluator now reads VAB-mixed `combined` directly, not this module's post-BN version of it -- a
separate, smaller, still-open gap from the one this module was built to close (the
triplet-vs-id/align gradient interference above), not something addressed here.
"""
import torch.nn as nn


class PartBNNecks(nn.Module):
    def __init__(self, num_branches, embed_dim):
        super(PartBNNecks, self).__init__()
        self.bnnecks = nn.ModuleDict()
        for b in range(num_branches):
            bn = nn.BatchNorm1d(embed_dim)
            bn.bias.requires_grad_(False)
            self.bnnecks[str(b)] = bn

    def forward(self, f_out, branch):
        """f_out: [B, M, D]. branch: int, 0..M-1. Returns [B, D] -- BatchNorm1d applied to that
        one branch's already-pooled feature vector."""
        return self.bnnecks[str(branch)](f_out[:, branch, :])
