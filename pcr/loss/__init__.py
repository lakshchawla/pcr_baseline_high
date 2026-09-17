from __future__ import absolute_import

from .contrastive import PartViewContrastiveLoss
from .crossentropy import CrossEntropyLabelSmooth
from .part_triplet_loss import PartTripletLoss
from .gilt_loss import PartGiLtLoss
from .body_part_attention_loss import BodyPartAttentionLoss
from .clip_supcon_loss import SupConLoss
from .clip_cosine_align_loss import CosineAlignLoss
from .cross_attn_align_loss import cross_attention_alignment_loss

__all__ = [
    'PartViewContrastiveLoss',
    'CrossEntropyLabelSmooth',
    'PartTripletLoss',
    'PartGiLtLoss',
    'BodyPartAttentionLoss',
    'SupConLoss',
    'CosineAlignLoss',
    'cross_attention_alignment_loss',
]
