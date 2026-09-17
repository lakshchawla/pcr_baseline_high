"""Adapter wrapping bpbreid's BPBreID model as a drop-in PCR encoder.

Requires torchreid (bpbreid's own package, not deep-person-reid's) importable, i.e.
`pip install -e /path/to/bpbreid` into this repo's environment -- see README.md.
"""
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreid.models.bpbreid import BPBreID
from torchreid.utils import load_pretrained_weights
from torchreid.utils.constants import BN_FOREGROUND, BN_PARTS, FOREGROUND, PARTS

# resnet50 (torchreid/models/resnet.py) downloads its ImageNet weights automatically via
# torch.utils.model_zoo from a stable pytorch.org URL, cached under ~/.cache/torch/hub -- no
# setup needed. hrnet32 has no such URL anywhere in bpbreid's own repo or docs (only a vague
# "download on our Google Drive" comment with no fixed file/folder ID given) -- auto-downloading
# it would mean guessing a link, which risks silently fetching the wrong file. So hrnet32 is
# checked for locally and fails fast with copy-pasteable instructions instead.
_HRNET_WEIGHTS_FILENAME = 'hrnetv2_w32_imagenet_pretrained.pth'
_HRNET_DOWNLOAD_INSTRUCTIONS = (
    "HRNet32 ImageNet weights not found at '{path}', and there's no stable URL to auto-download "
    "them from (bpbreid's own repo only points to a Google Drive folder with no fixed file ID). "
    "Either (1) download '{filename}' yourself from "
    "https://github.com/HRNet/HRNet-Image-Classification or bpbreid's Google Drive "
    "(https://drive.google.com/drive/folders/1aUjpSXXVGtAh2nzV0RVsCq0tTXuDZWoH) and place it at "
    "'{path}', or (2) pass --backbone resnet50 instead, which downloads automatically."
)


@dataclass
class BPBReIDModelCfg:
    """Plain-Python stand-in for bpbreid's yacs model_cfg -- only the ~13 attributes
    BPBreID.__init__/forward actually read. Not a training-hyperparameter config; it just
    describes the frozen architecture matching whatever checkpoint gets loaded.

    parts_num=5 is correct for the market1501/dukemtmc bpbreid configs: masks.preprocess=
    'five_v' in those yamls auto-resolves parts_num to 5 via
    torchreid/data/masks_transforms/__init__.py::compute_parts_num_and_names, even though the
    yaml's own explicit masks.parts_num default is 1.
    """

    class Masks:
        parts_num: int = 5

    masks: Masks = field(default_factory=Masks)
    shared_parts_id_classifier: bool = False
    test_use_target_segmentation: str = 'none'
    training_binary_visibility_score: bool = True
    testing_binary_visibility_score: bool = True
    backbone: str = 'hrnet32'  # or 'resnet50' -- see BPBReIDEncoder's module docstring on why
                                # resnet50 is the zero-setup choice and hrnet32 needs a manual
                                # download the first time.
    last_stride: int = 1
    dim_reduce: str = 'after_pooling'
    # 1024, not bpbreid's own 512 default -- must equal whatever CLIP text embedding size Stage
    # 1/2's own clip.arch config produces (no projection layer bridges the two -- see
    # pcr/models/clip_text_encoder.py's own docstring). Loading a genuinely external bpbreid
    # checkpoint trained at the original 512 would need this overridden back down for that one
    # load (Stage 3's train_uda.py/train_usl.py have no CLI flag for this, only this shared
    # default -- see BPBReIDModelCfg's own construction there).
    dim_reduce_output: int = 1024
    hrnet_pretrained_path: str = 'pretrained_models'  # matches bpbreid's own yacs default;
                                                        # only read when backbone == 'hrnet32'
    learnable_attention_enabled: bool = True
    normalization: str = 'identity'
    pooling: str = 'gwap'


class BPBReIDEncoder(nn.Module):
    """learnable_attention_enabled=True means the pixel-to-part classifier is learned
    end-to-end from the ID/triplet/contrastive gradient -- no external masks needed for
    forward, so source and target domain images are on identical footing here.

    num_classes=1 is a placeholder: PCR never uses BPBreID's own ID classifier heads, only
    the returned embeddings + visibility scores.

    Backbone ImageNet pretraining: requested (`pretrained=True` passed down to
    torchreid.models.build_model) whenever no full bpbreid `checkpoint_path` is given -- a full
    checkpoint's weights get loaded over the backbone afterward anyway via
    load_pretrained_weights (shape-matched keys only), so ImageNet init would just be immediately
    overwritten there; skipping it in that case avoids a pointless download/load. When training
    truly from scratch (no checkpoint), ImageNet init is the sensible default rather than the
    prior behaviour of a fully random backbone.
    """

    def __init__(self, model_cfg, checkpoint_path=None):
        super(BPBReIDEncoder, self).__init__()
        self.model_cfg = model_cfg
        pretrained = not checkpoint_path
        if pretrained and model_cfg.backbone == 'hrnet32':
            self._check_hrnet_weights_available(model_cfg.hrnet_pretrained_path)
        self.model = BPBreID(num_classes=1, pretrained=pretrained, loss='part_based', model_cfg=model_cfg)
        if checkpoint_path:
            load_pretrained_weights(self.model, checkpoint_path)
        self.num_parts = 1 + model_cfg.masks.parts_num
        self.num_features = model_cfg.dim_reduce_output

    @staticmethod
    def _check_hrnet_weights_available(pretrained_dir):
        weights_path = os.path.join(pretrained_dir, _HRNET_WEIGHTS_FILENAME)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(_HRNET_DOWNLOAD_INSTRUCTIONS.format(
                path=weights_path, filename=_HRNET_WEIGHTS_FILENAME))

    def forward(self, images):
        embeddings, visibility_scores, _, _, _, _ = self.model(images)

        foreground_emb = embeddings[BN_FOREGROUND]  # [B, D]
        parts_emb = embeddings[BN_PARTS]  # [B, K, D]
        foreground_vis = visibility_scores[FOREGROUND]  # [B]
        parts_vis = visibility_scores[PARTS]  # [B, K]

        f_out = torch.cat([foreground_emb.unsqueeze(1), parts_emb], dim=1)  # [B, 1+K, D]
        f_out = F.normalize(f_out, p=2, dim=-1)  # each of the M branches normalized independently

        vis = torch.cat([foreground_vis.unsqueeze(1), parts_vis], dim=1)  # [B, 1+K]

        return f_out, vis

    def forward_full(self, images):
        """Like forward(), but also returns pixels_cls_scores [B, K, Hf, Wf] (needed by
        BodyPartAttentionLoss). A separate method rather than an always-3-tuple forward() so
        every existing caller (trainers, evaluators, memory init) keeps working unchanged --
        the tradeoff is that this method isn't nn.DataParallel-aware: call it on the unwrapped
        module (`model.module.forward_full(...)` if `model` is DataParallel-wrapped), which
        runs on a single device. Fine for this repo's single-GPU dev setup; would need
        revisiting (e.g. always returning the 3-tuple from forward() instead) to actually
        scale BPA loss across multiple GPUs.
        """
        embeddings, visibility_scores, _, pixels_cls_scores, _, _ = self.model(images)

        foreground_emb = embeddings[BN_FOREGROUND]
        parts_emb = embeddings[BN_PARTS]
        foreground_vis = visibility_scores[FOREGROUND]
        parts_vis = visibility_scores[PARTS]

        f_out = torch.cat([foreground_emb.unsqueeze(1), parts_emb], dim=1)
        f_out = F.normalize(f_out, p=2, dim=-1)
        vis = torch.cat([foreground_vis.unsqueeze(1), parts_vis], dim=1)

        return f_out, vis, pixels_cls_scores


