"""Ported from bpbreid's torchreid/losses/part_averaged_triplet_loss.py::PartAveragedTripletLoss
(the "part-averaged" combination strategy -- one of 7 in the original, chosen as the default
here since it's bpbreid's own default). Batch-hard triplet loss generalized to K part-branch
embeddings: per-branch distance matrices are combined via a visibility-gated mean before the
standard batch-hard mining, so a pair with no mutually-visible part is excluded from mining
rather than compared on a made-up distance.

Reuses masked_mean/replace_values from pcr/utils/part_distance.py for the visibility-gated
combination step (identical semantics: -1 marks "couldn't be compared"). Does NOT reuse that
file's own _compute_body_parts_dist_matrices, because triplet training backpropagates through
these distances and needs the epsilon-stabilized sqrt below (plain sqrt has an infinite gradient
exactly at distance 0, which matters here since gradients actually flow through this loss --
part_distance.py's callers only ever use its distances for eval/clustering, no backward pass,
so it never needed this).

Dropped from the original: bpbreid's `self.writer.update_*_count(...)` telemetry hooks (dead
without bpbreid's own Writer singleton, not needed here).
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcr.utils.part_distance import masked_mean


class PartTripletLoss(nn.Module):
    def __init__(self, margin=0.3, epsilon=1e-16):
        super(PartTripletLoss, self).__init__()
        self.margin = margin
        self.epsilon = epsilon

    def forward(self, part_based_embeddings, labels, parts_visibility=None):
        """part_based_embeddings: [N, M, D]. labels: [N]. parts_visibility: [N, M] bool or
        float, optional. Returns (triplet_loss, trivial_triplets_ratio, valid_triplets_ratio),
        or None if no valid triplet could be formed in this batch."""
        part_based_pairwise_dist = self._part_based_pairwise_distance_matrix(
            part_based_embeddings.transpose(1, 0))  # [M, N, N]

        if parts_visibility is not None:
            vis = parts_visibility.t()  # [M, N]
            valid_mask = vis.unsqueeze(1) * vis.unsqueeze(2)  # [M, N, N]
            if valid_mask.dtype is not torch.bool:
                valid_mask = torch.sqrt(valid_mask)
            pairwise_dist = masked_mean(part_based_pairwise_dist, valid_mask)
        else:
            pairwise_dist = part_based_pairwise_dist.mean(0)

        return self._hard_mine_triplet_loss(pairwise_dist.unsqueeze(0), labels, self.margin)

    def _part_based_pairwise_distance_matrix(self, embeddings):
        """embeddings: [M, N, D]. ||a-b||^2 = |a|^2 - 2<a,b> + |b|^2, epsilon-stabilized before
        sqrt so exact-zero (self-)distances don't produce an infinite gradient."""
        dot_product = torch.matmul(embeddings, embeddings.transpose(2, 1))
        square_sum = dot_product.diagonal(dim1=1, dim2=2)
        distances = square_sum.unsqueeze(2) - 2 * dot_product + square_sum.unsqueeze(1)
        distances = F.relu(distances)

        zero_mask = torch.eq(distances, 0).float()
        distances = distances + zero_mask * self.epsilon
        distances = torch.sqrt(distances)
        distances = distances * (1 - zero_mask)
        return distances

    @staticmethod
    def _hard_mine_triplet_loss(batch_pairwise_dist, labels, margin):
        """batch_pairwise_dist: [K, N, N] (K=1 for the part-averaged combination used here).
        A value of -1 means no distance could be computed for that pair (excluded from mining)."""
        max_value = torch.finfo(batch_pairwise_dist.dtype).max
        valid_pairwise_dist_mask = (batch_pairwise_dist != -1.)

        mask_anchor_positive = PartTripletLoss._anchor_positive_mask(labels).unsqueeze(0)
        mask_anchor_positive = mask_anchor_positive * valid_pairwise_dist_mask
        valid_positive_dist = (batch_pairwise_dist * mask_anchor_positive.float()
                                - (~mask_anchor_positive).float())
        hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=-1)  # [K, N]

        mask_anchor_negative = PartTripletLoss._anchor_negative_mask(labels).unsqueeze(0)
        mask_anchor_negative = mask_anchor_negative * valid_pairwise_dist_mask
        valid_negative_dist = (batch_pairwise_dist * mask_anchor_negative.float()
                                + (~mask_anchor_negative).float() * max_value)
        hardest_negative_dist, _ = torch.min(valid_negative_dist, dim=-1)  # [K, N]

        valid_triplets_mask = (hardest_positive_dist != -1) & (hardest_negative_dist != max_value)
        hardest_dist = torch.stack([hardest_positive_dist, hardest_negative_dist], 2)  # [K, N, 2]
        valid_hardest_dist = hardest_dist[valid_triplets_mask, :]  # [num_valid, 2]

        if valid_hardest_dist.nelement() == 0:
            warnings.warn('PartTripletLoss: no valid triplets in this batch')
            return None

        triplet_losses = F.relu(valid_hardest_dist[:, 0] - valid_hardest_dist[:, 1] + margin)
        triplet_loss = triplet_losses.mean()
        trivial_ratio = (triplet_losses == 0.).float().mean()
        valid_ratio = valid_triplets_mask.float().mean()
        return triplet_loss, trivial_ratio, valid_ratio

    @staticmethod
    def _anchor_positive_mask(labels):
        not_self = ~torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
        return not_self & same_label

    @staticmethod
    def _anchor_negative_mask(labels):
        return labels.unsqueeze(0) != labels.unsqueeze(1)
