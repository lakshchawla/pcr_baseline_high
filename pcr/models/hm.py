"""Part-generalized SpCL HybridMemory: features/[num_samples, D] -> [num_samples, M, D],
M = however many branches the caller's own encoder produces (whole-body foreground + K learnable
parts, +1 more for RN50's real global embedding -- see pcr/models/clip_dense_part_encoder.py).
Ported from spcl/models/hm.py.

Two usage modes in this repo, same class, unchanged code: Stage 3 (examples/train_uda.py) uses it
as originally designed -- num_samples = total instances (source identities' centroids + every
target instance individually), labels updated as pseudo-clusters evolve. Stage 2
(examples/train_relational_finetune.py) uses it differently: num_samples = num_identities,
labels = arange(num_identities) fixed for the whole run, so every row IS one real identity's own
running centroid from the start -- a genuine, always-supervised centroid table, not a stand-in for
evolving pseudo-labels. The `sim_by_label` aggregation below becomes a harmless no-op in that mode
(every label already has exactly one row), not a bug.

Note on why compute_bpb_pairwise_distance (pcr/utils/part_distance.py) is NOT called here:
that function does a two-sided (query AND gallery) visibility-gated combination, meant for
query-gallery matching and Jaccard re-ranking where both sides have their own visibility. The
memory bank only stores per-slot *features*, not per-slot visibility (a memory slot is a running
average across many past sightings of that identity/part, so "is this slot visible" isn't a
well-defined single value). PartHybridMemory therefore uses its own lighter, single-sided
combination -- weighted only by the current batch's (query-side) visibility -- to combine the
per-branch similarities into one [B, N] similarity before the standard masked-softmax + NLL loss.
This is a deliberate distinction, not an oversight.
"""
import torch
import torch.nn.functional as F
from torch import nn, autograd


class PartHM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, visibility, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes, visibility)
        # per-branch similarity: outputs[b, m, n] = inputs[b, m] . features[n, m]
        outputs = torch.einsum('bmd,nmd->bmn', inputs, ctx.features)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes, visibility = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = torch.einsum('bmn,nmd->bmd', grad_outputs, ctx.features)

        # momentum update, skipped per-branch when that branch is invisible for the sample
        for b in range(inputs.size(0)):
            y = indexes[b]
            for m in range(inputs.size(1)):
                if visibility[b, m]:
                    ctx.features[y, m] = ctx.momentum * ctx.features[y, m] + (1. - ctx.momentum) * inputs[b, m]
                    ctx.features[y, m] /= ctx.features[y, m].norm()

        return grad_inputs, None, None, None, None


def part_hm(inputs, indexes, visibility, features, momentum=0.5):
    return PartHM.apply(inputs, indexes, visibility, features,
                         torch.Tensor([momentum]).to(inputs.device))


class PartHybridMemory(nn.Module):
    def __init__(self, num_features, num_parts, num_samples, temp=0.05, momentum=0.2):
        super(PartHybridMemory, self).__init__()
        self.num_features = num_features
        self.num_parts = num_parts
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_parts, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes, visibility):
        # inputs: [B, M, D], indexes: [B], visibility: [B, M] (bool or float in [0, 1])
        sim = part_hm(inputs, indexes, visibility, self.features, self.momentum)  # [B, M, N]
        # out-of-place (unlike SpCL's original in-place `/=`): modern autograd forbids an
        # in-place op on a custom Function's output view
        sim = sim / self.temp
        B = sim.size(0)

        weights = visibility.float().unsqueeze(-1)  # [B, M, 1]
        combined = (weights * sim).sum(1) / weights.sum(1).clamp_min(1e-6)  # [B, N]

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        device = combined.device
        sim_by_label = torch.zeros(labels.max() + 1, B).float().to(device)
        sim_by_label.index_add_(0, labels, combined.t().contiguous())
        nums = torch.zeros(labels.max() + 1, 1).float().to(device)
        nums.index_add_(0, labels, torch.ones(self.num_samples, 1).float().to(device))
        mask = (nums > 0).float()
        sim_by_label /= (mask * nums + (1 - mask)).clone().expand_as(sim_by_label)
        mask = mask.expand_as(sim_by_label)
        masked_sim = masked_softmax(sim_by_label.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)


class PerPartCentroidMemory(nn.Module):
    """Per-part contrast against every identity's SAME part (2026-09-18, see progress.md).

    One momentum centroid per (identity, part): `features[y, m]`. For each branch m
    SEPARATELY, image part m of sample b is classified against all N identities' part-m
    centroids -- softmax over identities of cos(part_m, centroid[:, m]) / temp, cross-entropy at
    the true identity -- and the per-part losses are averaged with the sample's own per-part
    visibility as weights. This is the principle "push my shoe away from everyone else's shoe"
    with the whole identity table (751) as the negative pool, not the ~16 identities in a batch.

    Deliberately NOT PartHybridMemory: that class averages the per-part similarities into one
    [B, N] similarity before its softmax, so a part that disagrees is diluted by parts that
    agree -- exactly the failure the soft-min retrieval rule (pcr/utils/part_distance.py) was
    introduced to remove. Here every part gets its own softmax, so an uninformative part (black
    hair, say) shows up as an irreducibly high loss on that branch -- expected, and logged per
    part -- without touching the others.

    Same PartHM mechanics: einsum similarity + per-(sample, part) momentum update of the
    centroids in backward, skipped for parts whose `update_mask` is False; centroids stay
    unit-norm. Inputs are expected unit-norm (the joint-space branches)."""

    def __init__(self, num_features, num_parts, num_identities, temp=0.05, momentum=0.2):
        super(PerPartCentroidMemory, self).__init__()
        self.num_parts = num_parts
        self.temp = temp
        self.momentum = momentum
        self.register_buffer('features', torch.zeros(num_identities, num_parts, num_features))

    def forward(self, inputs, targets, weights, update_mask):
        """inputs: [B, M, D] unit-norm. targets: [B] identity indices. weights: [B, M] float,
        per-part loss weight (visibility, detached inside). update_mask: [B, M] bool, which
        (sample, part) slots update their centroid. Returns (loss, per_part [M])."""
        sim = part_hm(inputs, targets, update_mask, self.features, self.momentum) / self.temp  # [B, M, N]
        log_prob = F.log_softmax(sim, dim=-1)
        nll = -log_prob.gather(-1, targets.view(-1, 1, 1).expand(-1, self.num_parts, 1)).squeeze(-1)  # [B, M]
        w = weights.detach().clamp(min=1e-3)
        per_part = (w * nll).sum(0) / w.sum(0).clamp(min=1e-8)  # [M]
        return per_part.mean(), per_part.detach()
