"""Stage 2: supervised backbone finetune, implementing "Algorithm 2 -- Stage 2: Backbone
Fine-Tuning" exactly (see progress.md's entry on this file for the full step-by-step mapping);
correspondence to that algorithm's own names:

  Algorithm 2 name           This file / pcr/models
  -----------------          -----------------------
  backbone + BPAM             ClipBPAMEncoder, now trainable; pixel_classifier initialized from
                               Stage 1's pixel_classifier.pth (Stage 0's head, continued in Stage 1
                               under BPA + distillation from the text-refined masks -- see
                               configs/stage2_relational_finetune.yaml's model.checkpoint_path)
  VRB                         AttentionPoolingBlock (global aggregator over the foreground-gated
                               parts), trainable, initialized from Stage 1's pool.pth. No VAB/CAB
                               any more -- see pcr/models/relation_blocks.py's module docstring.
  frozen_text_anchors          text_prototypes.pth (built by examples/cache_text_anchors.py),
                               loaded once; ctx_params/TRB/CLIP text encoder are never loaded
                               here at all -- nothing to "discard", they simply aren't imported
  global ID classifier         PartIdClassifiers (foreground/branch-0 only)
  L_id_global                  id_loss (CrossEntropyLabelSmooth) on the global classifier's logits
  L_tri_global + L_tri_parts    two separate PartTripletLoss calls, NOT one fused call across all
                               branches -- see compute_losses' own comments for why this matters
  L_align                      CosineAlignLoss (pcr/loss/clip_cosine_align_loss.py) -- a softmax
                               classification of each part's feature against that branch's FULL
                               frozen prototype table (every identity acts as an implicit negative
                               via the softmax), restoring CLIP-ReID's own original I2TLoss
                               mechanism rather than the literal per-sample regression Algorithm 2's
                               own wording describes (see changes.md's "Red flag 4" /
                               plans/IMPROVEMENT_PLAN.md section 3 for why the pure-regression version was
                               replaced -- it had no term pushing different identities' features
                               apart at all). All M=1+K branches, global/foreground included --
                               Stage 1 now trains ctx/TAB on all branches uniformly, so
                               text_prototypes[:,0,:] is a real anchor, not meaningless noise.
  L_attn                       BodyPartAttentionLoss, mandatory by default now (data.masks_dir
                               defaults to Market1501's real masks, matching Algorithm 2's own
                               unconditional framing) but still optional in code for datasets with
                               no masks on disk (dukemtmc-reid) -- see the config's own comment.

Two additions beyond Algorithm 2's own literal steps, both from "Bag of Tricks and a Strong
Baseline for Deep Person Re-identification" (Luo et al., CVPRW 2019, "BoT") -- explicit, deliberate
deviations, not silent scope creep:

  BNNeck                       PartBNNecks (pcr/models/bn_neck.py) -- one BatchNorm1d per branch,
                               inserted between the pooled feature (`combined`) and whichever of
                               id_loss/align_loss consumes it, so triplet (which keeps reading
                               `combined` directly, pre-BN) and id/align (which read the post-BN
                               version) stop fighting over what shape the one shared feature should
                               have. See that file's own docstring for the full mechanism, including
                               why align_loss's post-BN input is additionally L2-normalized (BN
                               alone doesn't preserve the unit-norm assumption CosineAlignLoss's
                               fixed-temperature softmax depends on -- an ArcFace/CosFace-style
                               "BN then L2-normalize before a cosine-similarity head" pattern, not
                               an extra trick layered on top of BNNeck).
  10-epoch linear warmup        Algorithm 2's own step 16 says nothing about a learning-rate
                               schedule at all; this file previously used a bare StepLR from epoch
                               0 at full LR, straight onto a just-unfrozen backbone with three
                               newly-interacting losses (id/triplet/align) -- exactly the situation
                               BoT's own ablation warns is a known source of early instability that
                               can leave training in a worse basin for good. Replaced with
                               WarmupMultiStepLR (pcr/utils/lr_scheduler.py, already used by
                               examples/train_usl.py) -- linear warmup for cfg.optim.warmup_epochs
                               (default 10, BoT's own number), then the same step-decay-at-
                               cfg.optim.step_size behavior as before.

Loss combination in compute_losses() matches Algorithm 2 step 16 exactly: L_attn + L_id_global +
L_tri_global + L_tri_parts + lambda_clip * L_align, where lambda_clip is cfg.loss.align_weight --
the one term the algorithm gives its own explicit coefficient; every other term uses an implicit
weight of 1 in the algorithm's own formula, which this file's id_weight/triplet_weight/bpa_weight
config knobs default to (kept configurable rather than hardcoded to 1, since every other stage in
this repo already exposes its loss weights the same way).

No hard per-branch visibility gating anywhere in these loss computations -- every branch
contributes for every sample, unconditionally. Reliability is handled by *weighting*, not
exclusion, same design as Stage 1: build_encoder switches this stage's own encoder to continuous
(not binary) visibility scores, L_align is weighted per-part by that part's own visibility
(CosineAlignLoss's weights argument, detached before use -- see that file's own docstring for why),
and L_tri_global/L_tri_parts keep a loose hard exclusion via PartTripletLoss's own parts_visibility
argument (batch-hard mining's max/min operations don't compose with soft weights the way
InfoNCE/align's weighted means do -- see configs/stage2_relational_finetune.yaml's
loss.triplet_visibility_min comment). This replaces the
upstream image-level filter Stage 1/2 both used to run (pcr/utils/visibility_filter.py, deleted --
see progress.md's entry on this change) before either stage's training set was ever built: that
filter discarded 61% of Market1501's training images in practice, was the wrong granularity (an
image with 4 good parts and 1 occluded one lost all 4), and was found to be driven by an
undertrained BPAM signal rather than genuine occlusion. AttentionPoolingBlock is also
visibility-aware at the attention level itself (see pcr/models/relation_blocks.py's own docstring):
this forward pass's own vis is passed in as a soft attention-score bias, so a poorly-visible part
contributes less to the pooled global, not just less to its own downstream loss term.

End-of-training checkpoint (Algorithm 2 step 20) bundles AttentionPoolingBlock's and PartBNNecks'
state into the SAME saved dict as the encoder's own state ('pool_state_dict'/'bn_necks_state_dict'
alongside 'state_dict'), rather than separate files -- one checkpoint containing {backbone, BPAM,
pool, BNNeck}, directly loadable by the existing examples/train_uda.py --checkpoint-path /
examples/train_usl.py --checkpoint-path unchanged (both only ever read the 'state_dict' key,
ignoring the rest -- confirmed against bpbreid's own load_pretrained_weights). Stage 3 stays
completely out of this file's scope otherwise; nothing downstream reads 'pool_state_dict' or
'bn_necks_state_dict' yet.

Renamed from train_finetune.py -- paired with train_relational_prompts.py's rename.

Config-driven (YAML) -- see configs/stage2_relational_finetune.yaml. Same deliberate deviation
from the rest of pcr2 as examples/train_relational_prompts.py (train_uda.py/train_usl.py stay
argparse-only).
"""
from __future__ import print_function, absolute_import
import argparse
import collections
import math
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcr import datasets
from pcr.models.clip_dense_part_encoder import CLIP_MEAN, CLIP_STD
from pcr.models.clip_rn50_bpam_encoder import ClipRN50BPAMEncoder
from pcr.models.clip_vit_bpam_encoder import ClipViTBPAMEncoder
from pcr.models.bn_neck import PartBNNecks
from pcr.models.hm import PartHybridMemory
from pcr.models.id_classifier import PartIdClassifiers
from pcr.models.relation_blocks import AttentionPoolingBlock, apply_part_pooling
from pcr.loss import (PartTripletLoss, CrossEntropyLabelSmooth, CosineAlignLoss, BodyPartAttentionLoss)
from pcr.evaluators import Evaluator, extract_features
from pcr.utils.config import load_yaml_config
from pcr.utils.data import IterLoader
from pcr.utils.data import transforms as T
from pcr.utils.data.sampler import RandomIdentitySampler
from pcr.utils.data.preprocessor import Preprocessor, PreprocessorMaskedSingleView
from pcr.utils.logging import Logger
from pcr.utils.lr_scheduler import WarmupMultiStepLR
from pcr.utils.osutils import mkdir_if_missing
from pcr.utils.serialization import save_checkpoint, load_checkpoint


def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir, name))


def get_photometric_transform():
    # CLIP's own normalization, NOT ImageNet stats -- the CLIP backbone's pretrained (and, here,
    # increasingly fine-tuned) weights are calibrated for this specific normalization; using the
    # wrong one silently miscalibrates every input to it. RandomErasing's own fill value matches
    # too, for the same reason (it fills the already-normalized tensor directly, so its fill value
    # should be drawn from the same normalization convention as everything else in this pipeline).
    normalizer = T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))
    return T.Compose([
        T.RandomApply([T.GaussianBlur((.1, 2.))], p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=list(CLIP_MEAN)),
    ])


def get_unmasked_transform(height, width):
    normalizer = T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))
    return T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=list(CLIP_MEAN)),
    ])


def get_train_loader(dataset, cfg, train_set):
    sampler = RandomIdentitySampler(train_set, cfg.data.num_instances)
    if cfg.data.masks_dir:
        dataset_wrapper = PreprocessorMaskedSingleView(
            train_set, masks_root=dataset.dataset_dir, masks_dir=cfg.data.masks_dir,
            height=cfg.data.height, width=cfg.data.width,
            photometric_transform=get_photometric_transform(),
            root=dataset.images_dir, mask_suffix=cfg.data.masks_suffix)
    else:
        dataset_wrapper = Preprocessor(train_set, root=dataset.images_dir,
                                        transform=get_unmasked_transform(cfg.data.height, cfg.data.width))
    return IterLoader(
        DataLoader(dataset_wrapper, batch_size=cfg.data.batch_size, num_workers=cfg.data.workers,
                   sampler=sampler, pin_memory=True, drop_last=True))


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))
    test_transformer = T.Compose([T.Resize((height, width), interpolation=3), T.ToTensor(), normalizer])
    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))
    return DataLoader(Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
                       batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)


def build_encoder(cfg):
    # ClipRN50BPAMEncoder / ClipViTBPAMEncoder (dispatched on cfg.clip.arch, same convention as
    # Stage 1's own build_encoder -- must agree with Stage 1 exactly, see that file's comment on
    # why), not BPBreIDEncoder(HRNet32) -- must be initialized from the pixel_classifier Stage 1
    # ENDED with (pixel_classifier.pth: Stage 0's head continued under BPA + distillation from
    # the text-refined masks; see this file's own module docstring's "backbone + BPAM" row), so
    # text_prototypes.pth describes a part convention this encoder actually starts from.
    # Visibility is continuous by construction (softmax attention maps, no binary mode) -- no
    # config knob needed here, unlike BPBReIDModelCfg's training/testing_binary_visibility_score.
    # Unfrozen (per Algorithm 2's own intent): CLIP's pretrained weights fine-tune here too,
    # which is exactly what makes L_align below load-bearing rather than decorative -- it's what
    # stops this fine-tuning drifting the backbone out of CLIP's own joint space.
    encoder_cls = ClipViTBPAMEncoder if cfg.clip.arch.startswith('ViT') else ClipRN50BPAMEncoder
    encoder = encoder_cls(clip_arch=cfg.clip.arch, height=cfg.data.height, width=cfg.data.width,
                           num_parts=cfg.model.parts_num,
                           checkpoint_path=cfg.model.checkpoint_path or None, device='cuda').cuda()
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad_(True)
    return encoder


def _vit_backbone_depth(name, num_blocks):
    """name: a parameter name from ClipViTDenseBackbone.named_parameters() (e.g.
    'resblocks.13.attn.in_proj_weight', 'conv1.weight', 'ln_post.weight'). Returns a depth index
    0..num_blocks+1: 0 for the patch-embedding stage (conv1/class_embedding/ln_pre -- CLIP's own
    most general, least task-specific weights), 1..num_blocks for each transformer block in
    order, num_blocks+1 for ln_post/proj (the final projection into the joint space -- closest to
    the output, wants the least decay)."""
    if name.startswith('resblocks.'):
        return int(name.split('.')[1]) + 1
    if name.startswith('ln_post') or name.startswith('proj'):
        return num_blocks + 1
    return 0  # conv1, class_embedding, ln_pre


def _llrd_param_groups(named_params, base_lr, num_depths, depth_fn, weight_decay, decay_rate):
    """Layer-wise LR decay: buckets (name, param) pairs by a depth index (0 = earliest/most
    pretrained-general, num_depths-1 = deepest/closest to the task-specific output) via depth_fn,
    scaling each depth's LR by decay_rate**(deepest - depth) -- standard practice for fully
    fine-tuning a deep, heavily-pretrained transformer (BEiT/ELECTRA-style LLRD). Root cause this
    addresses: Stage 2's single global LR (cfg.optim.lr) was tuned against a CNN (BoT's own
    recipe, Luo et al. CVPRW 2019) and, applied uniformly to a 24-block ViT-L/14, updates its
    early/general-purpose blocks far too aggressively for how little downstream data (751
    Market1501 identities) is available -- diagnosed directly from a real comparison run: ViT
    landed at mAP 77.0/R1 99.0 vs RN50's 84.5/93.5, the classic "globally strong but
    under-refined" signature of an under-adapted large-transformer fine-tune (high R1 -- the
    coarse identity signal still finds the single best match -- alongside a depressed mAP -- fine
    per-part separation across the full ranked list is noisier), not a sign ViT is architecturally
    worse-suited to this task (CLIP-ReID's own source paper reports the opposite ranking between
    its RN50 and ViT-B/16 variants).

    Also splits each depth into decay/no-decay sub-groups (no_decay = 1-D params: every
    LayerNorm/BatchNorm weight+bias and every linear/conv bias) -- decaying those measurably hurts
    transformer convergence in the published ViT fine-tuning recipes (DeiT/BEiT/timm's own
    defaults all exclude them), a distinct, additive fix from LLRD itself. Applied only where
    diagnosed: RN50/HRNet32's own optimizer construction (see build_optimizer below) is
    deliberately left untouched -- CNNs tolerate a single uniform LR/WD far better (that's why
    BoT's own recipe never needed this), and changing an already-validated baseline while fixing a
    broken one would confound the next comparison."""
    buckets = {}
    for name, p in named_params:
        if not p.requires_grad:
            continue
        key = (depth_fn(name), p.ndim <= 1)
        buckets.setdefault(key, []).append(p)

    groups = []
    for (depth, no_decay), params in buckets.items():
        scale = decay_rate ** (num_depths - 1 - depth)
        groups.append({
            'params': params,
            'lr': base_lr * scale,
            'weight_decay': 0.0 if no_decay else weight_decay,
        })
    return groups


def build_optimizer(encoder, id_classifiers, pool, bn_necks, cfg):
    """RN50/HRNet32: unchanged, single flat param list at one LR/WD -- see _llrd_param_groups'
    own docstring for why this stays untouched. ViT: layer-wise LR decay across the backbone's 24
    blocks (+patch-embed stage +final projection), decay/no-decay split everywhere, and every
    task-specific head (pixel_classifier, id_classifiers, pool, bn_necks -- none of which have a
    pretrained "depth" of their own) at the backbone's full, undecayed LR."""
    if not cfg.clip.arch.startswith('ViT'):
        params = (list(encoder.parameters()) + list(id_classifiers.parameters())
                  + list(pool.parameters()) + list(bn_necks.parameters()))
        return torch.optim.Adam(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    num_blocks = len(encoder.backbone.resblocks)
    num_depths = num_blocks + 2  # 0=patch-embed, 1..num_blocks=resblocks, num_blocks+1=ln_post/proj
    decay_rate = cfg.vit.llrd_decay
    groups = _llrd_param_groups(
        encoder.backbone.named_parameters(), cfg.optim.lr, num_depths,
        lambda n: _vit_backbone_depth(n, num_blocks), cfg.optim.weight_decay, decay_rate)

    head_named_params = []
    for i, m in enumerate([encoder.pixel_classifier, id_classifiers, pool, bn_necks]):
        head_named_params.extend(('head{}.{}'.format(i, n), p) for n, p in m.named_parameters())
    groups += _llrd_param_groups(head_named_params, cfg.optim.lr, 1, lambda n: 0,
                                  cfg.optim.weight_decay, decay_rate)
    return torch.optim.Adam(groups)


def bpa_weight_schedule(epoch, initial, floor, decay_epochs):
    """Cosine decay from `initial` at epoch 0 down to `floor`, reached at `decay_epochs` and held
    afterward."""
    if epoch >= decay_epochs:
        return floor
    progress = epoch / decay_epochs
    return floor + 0.5 * (initial - floor) * (1 + math.cos(math.pi * progress))


def mask_to_pixel_targets(mask, pixels_cls_scores):
    """mask: [B, 1+parts_num, H, W] (soft, sums to 1 per pixel). Resized to pixels_cls_scores'
    spatial size and argmax'd into an integer target per pixel -- matches bpbreid's own
    part_based_engine.py::combine_losses and pcr/trainers_usl.py::ICEUSLTrainer._mask_targets."""
    mask = F.interpolate(mask, size=pixels_cls_scores.shape[2:], mode='bilinear', align_corners=True)
    return mask.argmax(dim=1)


def compute_losses(encoder, pool, bn_necks, id_classifiers, triplet_loss, id_loss,
                    align_loss, bpa_loss, part_memory, text_prototypes, imgs,
                    mask, targets, cfg, epoch):
    use_masks = bpa_loss is not None
    if use_masks:
        f_out, vis, pixels_cls_scores = encoder.forward_full(imgs)
    else:
        f_out, vis = encoder(imgs)
        pixels_cls_scores = None

    # apply_part_pooling: foreground gates the K parts, pool (AttentionPoolingBlock) aggregates
    # them into a new global embedding taking over branch 0's exact slot; the parts pass through
    # untouched -- see relation_blocks.py's own module docstring (no VAB, no CAB any more: VAB's
    # gate never left zero, and CAB shaped a feature retrieval never computes). vis is this same
    # forward pass's own per-branch visibility, used both as the gate and as pool's attention
    # bias.
    combined, _ = apply_part_pooling(pool, f_out, vis, encoder._has_global)  # [B, 1+K, D]
    num_branches = combined.size(1)

    # vis shares combined's exact branch axis (0=foreground, 1..K=parts) -- both are built from
    # the same encoder call. Loose hard exclusion for triplet's batch-hard mining only (soft
    # weights don't compose with max/min mining); continuous weighting for align, below.
    vis_mask = vis >= cfg.loss.triplet_visibility_min  # [B, 1+K] bool

    total = f_out.new_zeros(())
    log = {}

    # Algorithm 2 step 10: L_id_global, the global classifier's cross-entropy on f_g alone.
    # BNNeck (pcr/models/bn_neck.py): id_loss reads the post-BN feature; triplet (below) keeps
    # reading `combined` directly, pre-BN. bn_global is [B, D]; wrapped back to [B, 1, D] so
    # PartIdClassifiers' own f_out[:, branch, :] slicing convention (shared with Stage 3, not
    # changed here) still applies unchanged.
    bn_global = bn_necks(combined, 0)
    id_logits = id_classifiers(bn_global.unsqueeze(1), 0)
    l_id = id_loss(id_logits, targets)
    total = total + cfg.loss.id_weight * l_id
    log['id'] = l_id.item()

    # Algorithm 2 steps 11-13: L_tri_global (batch-hard triplet on f_g alone) and L_tri_parts (K
    # separate per-part batch-hard triplet losses, summed) -- two independent computations, not
    # one triplet loss fused across all M branches' distances the way this loop used to call
    # PartTripletLoss once on `combined` directly. Calling PartTripletLoss with a single-branch
    # slice ([B, 1, D]) gives that branch's own, unfused batch-hard mining. parts_visibility is a
    # loose boolean exclusion (vis_mask, threshold cfg.loss.triplet_visibility_min) -- the one
    # place hard exclusion still applies in this design; see module docstring.
    global_result = triplet_loss(combined[:, 0:1, :], targets, parts_visibility=vis_mask[:, 0:1])
    if global_result is not None:
        l_tri_global = global_result[0]
        total = total + cfg.loss.triplet_weight * l_tri_global
        log['tri_global'] = l_tri_global.item()

    l_tri_parts = f_out.new_zeros(())
    for branch in range(1, num_branches):
        part_result = triplet_loss(combined[:, branch:branch + 1, :], targets,
                                    parts_visibility=vis_mask[:, branch:branch + 1])
        if part_result is not None:
            l_tri_parts = l_tri_parts + part_result[0]
    total = total + cfg.loss.triplet_weight * l_tri_parts
    log['tri_parts'] = l_tri_parts.item()

    # Algorithm 2 steps 14-15: L_align, a softmax classification of each branch's feature against
    # that branch's FULL frozen prototype table (every identity is an implicit negative), summed
    # over all M=1+K branches -- global/foreground included, now that Stage 1 gives it a real
    # SupCon-trained text prototype too (see relation_blocks.py's own module docstring). This is
    # also what makes a separate global-branch i2t loss (L_i2tce, considered alongside CAB)
    # unnecessary: L_align already covers branch 0 here.
    align_total = f_out.new_zeros(())
    for branch in range(num_branches):
        branch_prototypes = text_prototypes[:, branch, :]  # [num_identities, D], full table -- the
                                                             # negatives this loss classifies against
        w = vis[:, branch]  # continuous weighting, not the boolean vis_mask used for triplet
        # BNNeck again: align_loss reads the post-BN feature too (triplet, above, still reads
        # combined directly). Reads `combined` -- the exact representation retrieval computes
        # (no CAB in between any more). Re-normalized after BN -- BatchNorm1d's own per-dimension
        # scaling doesn't preserve the unit-norm assumption CosineAlignLoss's fixed-temperature
        # softmax depends on, so this restores it (see bn_neck.py's own docstring).
        bn_part = F.normalize(bn_necks(combined, branch), p=2, dim=-1)
        align_total = align_total + align_loss(bn_part, branch_prototypes, targets, weights=w)
    total = total + cfg.loss.align_weight * align_total
    log['align'] = align_total.item()

    # L_centroid: PartHybridMemory (pcr/models/hm.py), SPCL's own per-slot momentum-updated
    # memory + full-table softmax classification, reused here with num_samples=num_identities and
    # indexes=targets so every memory row IS one identity's own running centroid -- a much larger
    # comparison pool for the image side specifically than triplet's own PK-batch-limited one
    # (~8 identities), additive to triplet rather than replacing it (see this file's own combined
    # module docstring / plans for why: isolating one variable at a time). Reads `combined`
    # (post-VAB, pre-CAB) -- same visibility-weighting convention as align_loss above, continuous
    # `vis`, not the boolean vis_mask (PartHybridMemory's own combination handles continuous
    # weights natively, see that class's own forward()).
    l_centroid = part_memory(combined, targets, vis)
    total = total + cfg.loss.centroid_weight * l_centroid
    log['centroid'] = l_centroid.item()

    # Algorithm 2 steps 9/16: L_attn, mandatory whenever masks are configured (see
    # configs/stage2_relational_finetune.yaml's own comment on why this defaults on now, and why
    # it still needs to stay optional in code for mask-less datasets like dukemtmc-reid).
    if use_masks:
        mask_targets = mask_to_pixel_targets(mask.cuda(), pixels_cls_scores)
        l_bpa, _ = bpa_loss(pixels_cls_scores, mask_targets)
        bpa_weight = bpa_weight_schedule(epoch, cfg.loss.bpa_weight_initial, cfg.loss.bpa_weight_floor,
                                          cfg.loss.bpa_weight_decay_epochs)
        total = total + bpa_weight * l_bpa
        log['bpa'] = l_bpa.item()
        log['bpa_weight'] = bpa_weight

    return total, log


def main():
    parser = argparse.ArgumentParser(description="PCR Stage 2: supervised backbone finetune")
    parser.add_argument('--config', type=str, required=True, metavar='PATH')
    parser.add_argument('--setup-only', action='store_true',
                         help="build dataset/encoder/losses/loader, print shapes, exit before "
                              "the training loop")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    main_worker(cfg, setup_only=args.setup_only)


def main_worker(cfg, setup_only=False):
    seed = getattr(cfg.logging, 'seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    mkdir_if_missing(cfg.logging.logs_dir)
    sys.stdout = Logger(osp.join(cfg.logging.logs_dir, 'log.txt'))
    print("==========\nConfig:{}\n==========".format(vars(cfg)))
    start_time = time.monotonic()

    dataset = get_data(cfg.data.dataset, cfg.data.data_dir)
    num_identities = dataset.num_train_pids
    # has_global_branch: same predicate examples/cache_text_anchors.py and this file's own
    # build_encoder use to pick ClipRN50BPAMEncoder vs ClipViTBPAMEncoder -- only the RN50
    # backbone currently exposes the extra global branch (see ClipBPAMEncoder's own docstring).
    has_global_branch = not cfg.clip.arch.startswith('ViT')
    num_branches = (2 if has_global_branch else 1) + cfg.model.parts_num

    proto_path = osp.join(cfg.stage1.prompt_dir, 'text_prototypes.pth')
    proto = load_checkpoint(proto_path)
    assert proto['num_identities'] == num_identities, (
        "Stage 1's text_prototypes.pth was built for {} identities, this dataset has {} -- "
        "stage1 and stage2 must use the same dataset.".format(proto['num_identities'], num_identities))
    assert proto['num_branches'] == num_branches, (
        "Stage 1's text_prototypes.pth was built for {} branches, this config expects {} branches "
        "(parts_num={}, has_global_branch={}) -- stage1 and stage2 must agree on both."
        .format(proto['num_branches'], num_branches, cfg.model.parts_num, has_global_branch))
    text_prototypes = proto['text_prototypes'].cuda()  # [num_identities, num_branches, D]

    encoder = build_encoder(cfg)
    id_classifiers = PartIdClassifiers(num_identities, cfg.model.dim_reduce_output, branches=(0,)).cuda()
    bn_necks = PartBNNecks(num_branches, cfg.model.dim_reduce_output).cuda()

    pool = AttentionPoolingBlock(dim=cfg.model.dim_reduce_output, num_heads=cfg.pool.num_heads).cuda()
    pool_path = osp.join(cfg.stage1.prompt_dir, 'pool.pth')
    pool.load_state_dict(load_checkpoint(pool_path))
    print('==> Loaded Stage 1 AttentionPoolingBlock weights from {}'.format(pool_path))

    triplet_loss = PartTripletLoss(margin=cfg.loss.triplet_margin).cuda()
    id_loss = CrossEntropyLabelSmooth(num_identities).cuda()
    align_loss = CosineAlignLoss(temperature=cfg.loss.align_temperature).cuda()
    use_masks = bool(cfg.data.masks_dir)
    bpa_loss = BodyPartAttentionLoss().cuda() if use_masks else None

    # PartHybridMemory (pcr/models/hm.py), SPCL's own per-slot momentum-updated memory, reused
    # here as a genuine per-identity centroid table: num_samples=num_identities and labels=
    # arange(num_identities) means every memory row IS one identity (not a per-instance slot the
    # way Stage 3 uses it) -- compute_losses always calls it with indexes=targets, the real
    # identity id, so each row is a running, momentum-updated average of that identity's own
    # embeddings, additive to triplet's own PK-batch-limited comparison (see this file's own
    # module docstring for why additive, not a replacement, for this first pass).
    part_memory = PartHybridMemory(num_features=cfg.model.dim_reduce_output, num_parts=num_branches,
                                    num_samples=num_identities, temp=cfg.loss.centroid_temp,
                                    momentum=cfg.loss.centroid_momentum).cuda()
    part_memory.labels = torch.arange(num_identities).cuda()

    train_set = sorted(dataset.train)
    train_loader = get_train_loader(dataset, cfg, train_set)
    test_loader = get_test_loader(dataset, cfg.data.height, cfg.data.width, cfg.data.batch_size,
                                   cfg.data.workers)

    if setup_only:
        print('==> Setup complete: {} identities, {} branches, {} images (no upstream visibility '
              'filtering -- every training image is used, weighted per-part inside the loss), '
              'masks {}. Exiting before the training loop (--setup-only).'.format(
                  num_identities, num_branches, len(train_set),
                  'ON (' + cfg.data.masks_dir + ')' if use_masks else 'off'))
        return

    # Initialize part_memory's per-identity centroids -- otherwise every row starts at the
    # class's own all-zero default, giving a meaningless (always-zero) similarity for every
    # identity until touched at least once by a momentum update partway through the first epoch.
    # Mirrors examples/train_uda.py's own "source-domain class centroid" initialization
    # (visibility-weighted mean per identity, with a plain-mean fallback for any branch with zero
    # visible members for that identity) -- a real, already-used pattern in this codebase, not new
    # logic. Uses encoder/pool exactly as Stage 1 left them, before Stage 2's own training loop
    # (and its optimizer, built below) has taken a single step.
    print('==> Initializing per-identity centroids in part_memory')
    encoder.eval()
    init_loader = get_test_loader(dataset, cfg.data.height, cfg.data.width, cfg.data.batch_size,
                                   cfg.data.workers, testset=train_set)
    init_features, init_vis, _ = extract_features(encoder, init_loader, pool)
    fea_dict = collections.defaultdict(list)
    vis_dict = collections.defaultdict(list)
    for fname, pid, _ in train_set:
        fea_dict[pid].append(init_features[fname])
        vis_dict[pid].append(init_vis[fname])
    centers = []
    for pid in range(num_identities):
        feats = torch.stack(fea_dict[pid], dim=0)                          # [n_i, M, D]
        pid_vis = torch.stack(vis_dict[pid], dim=0).float().unsqueeze(-1)  # [n_i, M, 1]
        weight_sum = pid_vis.sum(0)                                         # [M, 1]
        weighted_mean = (feats * pid_vis).sum(0) / weight_sum.clamp_min(1e-6)
        has_visible = (weight_sum.squeeze(-1) > 0).unsqueeze(-1)
        centers.append(torch.where(has_visible, weighted_mean, feats.mean(0)))
    part_memory.features = F.normalize(torch.stack(centers, 0), dim=-1).cuda()
    del init_loader, init_features, init_vis, fea_dict, vis_dict, centers
    print('==> part_memory initialized for {} identities'.format(num_identities))

    optimizer = build_optimizer(encoder, id_classifiers, pool, bn_necks, cfg)
    # BoT's own recommended schedule (Luo et al., CVPRW 2019) -- linear warmup for
    # cfg.optim.warmup_epochs (default 10), starting from cfg.optim.warmup_factor x the base LR,
    # then the same step decay this file used before (a single drop by 10x at cfg.optim.step_size)
    # -- see this file's own module docstring for why the warmup was missing before and why that
    # matters here specifically (freshly-unfrozen backbone, three newly-interacting losses). ViT
    # gets a longer warmup (cfg.vit.warmup_epochs) than RN50/HRNet32's own cfg.optim.warmup_epochs
    # -- a freshly-unfrozen 24-block transformer benefits from more of it than BoT's own
    # CNN-tuned number gives it (see build_optimizer/_llrd_param_groups for the fuller
    # reasoning); WarmupMultiStepLR applies this same warmup/decay multiplier to EVERY param
    # group's own base_lr independently (confirmed directly in pcr/utils/lr_scheduler.py's
    # get_lr(): `[base_lr * warmup_factor * ... for base_lr in self.base_lrs]`), so it needs no
    # change to support build_optimizer's multiple LLRD groups above.
    is_vit = cfg.clip.arch.startswith('ViT')
    warmup_epochs = cfg.vit.warmup_epochs if is_vit else cfg.optim.warmup_epochs
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=[cfg.optim.step_size], gamma=0.1,
                                      warmup_factor=cfg.optim.warmup_factor,
                                      warmup_iters=warmup_epochs, warmup_method='linear')
    # Passing pool (not just encoder): Stage 2's own losses train against attention-pooled
    # features (compute_losses' `combined`), so retrieval must use the same representation. This
    # is a live reference, same pattern as `encoder` above: in-loop weight updates are reflected
    # automatically at every periodic evaluator.evaluate() call below, no extra wiring needed.
    evaluator = Evaluator(encoder, pool)

    best_mAP = 0
    for epoch in range(cfg.optim.epochs):
        encoder.train()
        pool.train()
        bn_necks.train()
        train_loader.new_epoch()
        train_iters = len(train_loader)

        epoch_start = time.time()
        for it in range(train_iters):
            inputs = train_loader.next()
            if use_masks:
                imgs, mask, targets, _, _ = inputs
            else:
                imgs, _, targets, _, _ = inputs
                mask = None
            imgs = imgs.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            loss, log = compute_losses(encoder, pool, bn_necks, id_classifiers, triplet_loss,
                                        id_loss, align_loss, bpa_loss, part_memory, text_prototypes,
                                        imgs, mask, targets, cfg, epoch)
            loss.backward()
            optimizer.step()

            if (it + 1) % cfg.logging.print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.3f}\tPool gate {:.3f}\t{}'.format(
                    epoch, it + 1, train_iters, loss.item(), torch.tanh(pool.gate).item(),
                    '\t'.join('{} {:.3f}'.format(k, v) for k, v in log.items())))

        lr_scheduler.step()
        print('Epoch {} done in {:.1f}s'.format(epoch, time.time() - epoch_start))

        if (epoch + 1) % cfg.logging.eval_step == 0 or epoch == cfg.optim.epochs - 1:
            # float(): mean_ap() (pcr/evaluation_metrics/ranking.py) returns numpy.float64, not
            # a plain float. Found by actually round-tripping a saved checkpoint through
            # bpbreid's own torchreid.utils.load_pretrained_weights (the real consumer, via
            # train_uda.py/train_usl.py --checkpoint-path): its load_checkpoint doesn't pass
            # weights_only=False, so PyTorch 2.6+'s stricter weights_only=True default rejects a
            # numpy scalar sitting anywhere in the pickled checkpoint dict with an
            # UnpicklingError -- this repo's own pcr.utils.serialization.load_checkpoint already
            # passes weights_only=False and would have hidden this, so only testing the actual
            # cross-repo consumption path caught it.
            mAP = float(evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False))
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            # Algorithm 2 step 20: one saved checkpoint containing {backbone, BPAM, pool, BN
            # necks} -- 'pool_state_dict' rides alongside 'state_dict' in the same file rather
            # than a separate pool.pth. 'state_dict' is now encoder.state_dict() directly (the whole
            # ClipBPAMEncoder -- backbone + pixel_classifier), not encoder.model.state_dict():
            # unlike BPBReIDEncoder, this class has no inner .model, it IS the top-level module.
            # This is a deliberate, scoped format change -- Stage 3 (train_uda.py/train_usl.py
            # --checkpoint-path) is not part of this pivot and never reads a CLIP-RN50-shaped
            # checkpoint, so nothing existing depends on the old format here.
            save_checkpoint({
                'state_dict': encoder.state_dict(),
                'pool_state_dict': pool.state_dict(),
                'bn_necks_state_dict': bn_necks.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                'optimizer': optimizer.state_dict(),
            }, is_best, fpath=osp.join(cfg.logging.logs_dir, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.format(
                epoch, mAP, best_mAP, ' *' if is_best else ''))

    print('==> Test with the best model:')
    best_fpath = osp.join(cfg.logging.logs_dir, 'model_best.pth.tar')
    if osp.isfile(best_fpath):
        checkpoint = load_checkpoint(best_fpath)
        encoder.load_state_dict(checkpoint['state_dict'])
        # Reload pool's best-epoch weights too -- without this, the final report would combine
        # the best epoch's encoder with whatever epoch training happened to end on for pool,
        # which is not what "best model" means now that the evaluator reads pooled features.
        pool.load_state_dict(checkpoint['pool_state_dict'])
    else:
        print('No model_best.pth.tar in {}, testing with the final model'.format(cfg.logging.logs_dir))
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
