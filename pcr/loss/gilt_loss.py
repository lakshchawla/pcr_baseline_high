"""pcr2-native adaptation of bpbreid's GiLt (Global-identity Local-triplet) loss
(torchreid/losses/GiLt_loss.py), combining a per-branch weighted id + triplet loss.

Not a direct port: GiLt's own "id" component classifies through BPBReID's fixed parametric
classifier heads (BNClassifier, sized at model-construction time), which is a poor fit here --
DBSCAN's cluster count and numbering both change every epoch, so a fixed classifier layer would
keep reassigning what each output neuron "means" across epochs. Instead the id component reuses
this repo's other GiLt-flavored piece, pcr/loss/crossentropy.py::CrossEntropyLabelSmooth,
classifying against per-epoch cluster centers (exactly ICE's own ccloss mechanism) instead of a
persistent classifier layer -- so "GiLt" and "CrossEntropyLabelSmooth as in ICE" are not two
separate things here, they're the same id component viewed from two angles.

Branch weighting mirrors bpbreid's own default_losses_weights for the two branch-types this
repo's encoder actually exposes (foreground + parts; no separate global/concat_parts branches):
id loss on the foreground branch only (index 0), triplet loss across all M branches (foreground
+ parts) -- matching bpbreid's yaml (`foreg: id=1,tr=1`, `parts: id=0,tr=1`).

The id loss is divided by log(num_clusters) before being weighted into the total loss. Its raw
scale (CrossEntropyLabelSmooth's chance-level baseline is ~log(num_clusters)) is set entirely by
DBSCAN's cluster count for that epoch, not by how well the model is learning -- and that count is
observed to swing wildly (e.g. 335 -> 9 -> 1 across consecutive epochs on an under-trained
encoder, purely from clustering noise). Left unnormalized, a fixed id_weight would pull the total
gradient balance around for reasons unrelated to learning progress. The *unnormalized* value is
still what's returned/logged (id_loss_val below), since the raw cross-entropy is the
interpretable "how confident is the classifier" number -- only the term actually added to `loss`
is rescaled.
"""
import math

import torch.nn as nn

from .crossentropy import CrossEntropyLabelSmooth
from .part_triplet_loss import PartTripletLoss


class PartGiLtLoss(nn.Module):
    def __init__(self, num_clusters, tau_c=0.5, triplet_margin=0.3, id_weight=1.0, triplet_weight=1.0):
        super(PartGiLtLoss, self).__init__()
        self.tau_c = tau_c
        self.id_weight = id_weight
        self.triplet_weight = triplet_weight
        self.id_loss = CrossEntropyLabelSmooth(num_clusters) if id_weight > 0 else None
        self.triplet_loss = PartTripletLoss(margin=triplet_margin) if triplet_weight > 0 else None
        self.id_loss_norm = math.log(max(num_clusters, 2))

    def forward(self, f_out, vis, centers, targets):
        """f_out: [B, M, D] (branch 0 = foreground, per BPBReIDEncoder's convention). vis: [B, M].
        centers: [num_clusters, M, D], per-epoch cluster centers (see examples/train_usl.py).
        targets: [B] pseudo-labels, indexing into `centers`' first dimension."""
        loss = f_out.new_zeros(())
        id_loss_val = triplet_loss_val = None

        if self.id_loss is not None:
            logits = (f_out[:, 0, :] @ centers[:, 0, :].t()) / self.tau_c  # [B, num_clusters]
            id_loss_val = self.id_loss(logits, targets)
            loss = loss + self.id_weight * (id_loss_val / self.id_loss_norm)

        if self.triplet_loss is not None:
            result = self.triplet_loss(f_out, targets, parts_visibility=vis)
            if result is not None:
                triplet_loss_val = result[0]
                loss = loss + self.triplet_weight * triplet_loss_val

        return loss, id_loss_val, triplet_loss_val
