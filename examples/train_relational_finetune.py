"""Stage 2: supervised backbone finetune -- CLIP-ReID's Stage 2 (RN50 recipe) extended to K
body-part branches. Rewritten 2026-09-17 (see progress.md); the reference is
../CLIP-ReID/model/make_model_clipreid.py + processor_clipreid_stage2.py + configs/person/
cnn_clipreid.yml.

Per image the encoder (pcr/models/clip_dense_part_encoder.py) gives CLIP-ReID's three features
and their per-part versions:

  global:  x3 (layer3, avg-pooled)   x4 (layer4, avg-pooled)   x_proj (CLIP's attnpool global)
  parts:                             part_x4[k] (mask-pooled)  part_xproj[k] (mask-pooled dense)

Losses, mirroring CLIP-ReID's own composition and adding the parts as extra branches:
  L_id      CE (label-smoothed) on BN(x4), BN(x_proj)  [CLIP-ReID]  + BN(part_x4[k]) per part,
            each part's term weighted by that part's visibility
  L_tri     batch-hard triplet on x3, x4, x_proj      [CLIP-ReID]  + K separate per-part
            batch-hard triplets on part_x4[k] (each part mines its OWN hard negatives, so a part
            that differs is never averaged away during training) + one combined part triplet
            over part_x4 under the same soft-min rule retrieval scores with (cfg.eval.part_combine)
  L_cen     per-part centroid contrast (pcr/models/hm.py::PerPartCentroidMemory): image branch m
            vs ALL identities' momentum centroids of branch m, softmax over identities -- "my shoe
            vs everyone else's shoe" with the full 751-identity negative pool, visibility-weighted,
            logged per part (an uninformative part shows as an irreducibly high cen_k)
  L_align   x_proj vs identity y's text prototype 0   [CLIP-ReID's I2T, as a softmax over the
            full prototype table]  + part_xproj[k] vs prototype k per part, visibility-weighted
            -- "each part index faces its own alignment with its own prompt context"
  L_bpa     pixel classifier vs real PifPaf masks (Stage 0's loss, continued), decayed

Branch order everywhere: 0 = global, 1..K = parts. No VAB/TAB/CAB/pool, no centroid memory.
Retrieval (pcr/evaluators.py) matches on the joint-space branches [x_proj, part_xproj]: per-part
distances combined by a visibility-weighted log-sum-exp (soft-max of part distances = soft-min of
part similarities, pcr/utils/part_distance.py::combine_part_distances) instead of BPBReID's mean
-- so one part that disagrees (two people in all black, different shoes) penalizes the whole
score instead of being averaged away by the parts that agree.

Two additions from "Bag of Tricks" (Luo et al., CVPRW 2019), both also in CLIP-ReID: BNNeck
before every id classifier, and linear LR warmup. Config-driven: configs/stage2_relational_finetune.yaml.
"""
from __future__ import print_function, absolute_import
import argparse
import math
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcr import datasets
from pcr.models.clip_dense_part_encoder import CLIP_MEAN, CLIP_STD
from pcr.models.clip_rn50_bpam_encoder import ClipRN50BPAMEncoder
from pcr.models.clip_vit_bpam_encoder import ClipViTBPAMEncoder
from pcr.models.bn_neck import PartBNNecks
from pcr.models.id_classifier import PartIdClassifiers
from pcr.models.hm import PerPartCentroidMemory
from pcr.evaluators import extract_features
from pcr.loss import PartTripletLoss, CrossEntropyLabelSmooth, CosineAlignLoss, BodyPartAttentionLoss
from pcr.evaluators import Evaluator
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
    """ClipRN50BPAMEncoder / ClipViTBPAMEncoder, dispatched on cfg.clip.arch (must agree with
    Stage 1). pixel_classifier initialized from cfg.model.checkpoint_path (Stage 0). Unfrozen:
    CLIP's pretrained weights fine-tune here (CLIP-ReID's own Stage 2), with L_align as the
    tether to CLIP's joint space."""
    encoder_cls = ClipViTBPAMEncoder if cfg.clip.arch.startswith('ViT') else ClipRN50BPAMEncoder
    encoder = encoder_cls(clip_arch=cfg.clip.arch, height=cfg.data.height, width=cfg.data.width,
                           num_parts=cfg.model.parts_num,
                           checkpoint_path=cfg.model.checkpoint_path or None, device='cuda').cuda()
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad_(True)
    return encoder


class Heads(nn.Module):
    """Everything trainable besides the encoder: BNNecks + id classifiers for the x4-space
    branches (global + K parts, 2048-d) and for CLIP's projected global (1024-d) -- CLIP-ReID's
    `bottleneck`/`classifier` and `bottleneck_proj`/`classifier_proj`, plus one neck+classifier
    per part."""

    def __init__(self, num_identities, num_parts, x4_dim, proj_dim):
        super(Heads, self).__init__()
        branches = list(range(1 + num_parts))
        self.bn_x4 = PartBNNecks(1 + num_parts, x4_dim)
        self.cls_x4 = PartIdClassifiers(num_identities, x4_dim, branches=branches)
        self.bn_proj = PartBNNecks(1, proj_dim)
        self.cls_proj = PartIdClassifiers(num_identities, proj_dim, branches=(0,))


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


def build_optimizer(encoder, heads, cfg):
    """RN50: single flat param list at one LR/WD (CLIP-ReID's own cnn recipe). ViT: layer-wise
    LR decay across the backbone (see _llrd_param_groups) with every head at the full LR."""
    if not cfg.clip.arch.startswith('ViT'):
        params = list(encoder.parameters()) + list(heads.parameters())
        return torch.optim.Adam(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    num_blocks = len(encoder.backbone.resblocks)
    num_depths = num_blocks + 2
    decay_rate = cfg.vit.llrd_decay
    groups = _llrd_param_groups(
        encoder.backbone.named_parameters(), cfg.optim.lr, num_depths,
        lambda n: _vit_backbone_depth(n, num_blocks), cfg.optim.weight_decay, decay_rate)
    head_named_params = [('pixel_classifier.' + n, p) for n, p in encoder.pixel_classifier.named_parameters()]
    head_named_params += [('heads.' + n, p) for n, p in heads.named_parameters()]
    groups += _llrd_param_groups(head_named_params, cfg.optim.lr, 1, lambda n: 0,
                                  cfg.optim.weight_decay, decay_rate)
    return torch.optim.Adam(groups)


def bpa_weight_schedule(epoch, initial, floor, decay_epochs):
    """Cosine decay from `initial` at epoch 0 down to `floor`, reached at `decay_epochs`."""
    if epoch >= decay_epochs:
        return floor
    progress = epoch / decay_epochs
    return floor + 0.5 * (initial - floor) * (1 + math.cos(math.pi * progress))


def mask_to_pixel_targets(mask, pixels_cls_scores):
    """mask: [B, 1+parts_num, H, W] (soft, sums to 1 per pixel). Resized to pixels_cls_scores'
    spatial size and argmax'd into an integer target per pixel (bpbreid's own convention)."""
    mask = F.interpolate(mask, size=pixels_cls_scores.shape[2:], mode='bilinear', align_corners=True)
    return mask.argmax(dim=1)


def weighted_id_loss(logits, targets, weights, num_classes, epsilon=0.1):
    """Label-smoothed CE (CrossEntropyLabelSmooth's formula) with a per-sample weight -- a
    part's id term counts in proportion to that part's visibility. weights detached."""
    log_probs = F.log_softmax(logits, dim=1)
    smooth = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    smooth = (1 - epsilon) * smooth + epsilon / num_classes
    per_sample = -(smooth * log_probs).sum(1)
    w = weights.detach().clamp(min=1e-3)
    return (w * per_sample).sum() / w.sum().clamp(min=1e-8)


def compute_losses(encoder, heads, id_loss, triplet_loss, align_loss, bpa_loss, part_memory, text_prototypes,
                   imgs, mask, targets, cfg, epoch):
    out = encoder.forward_multi(imgs)
    K = encoder._k
    num_identities = text_prototypes.size(0)
    vis = out['vis']                                                   # [B, 1+K]
    g_x4 = out['x4'].mean(dim=1)                                       # [B, 2048]  CLIP-ReID img_feature
    x4_branches = torch.cat([g_x4.unsqueeze(1), out['part_x4']], dim=1)         # [B, 1+K, 2048]
    proj_branches = torch.cat([out['x_proj'].unsqueeze(1), out['part_xproj']], dim=1)  # [B, 1+K, D]

    total = imgs.new_zeros(())
    log = {}

    # L_id: CLIP-ReID's two global classifiers (BN(x4), BN(x_proj)) + one per part on BN(part_x4),
    # visibility-weighted.
    bn_x4_all = torch.stack([heads.bn_x4(x4_branches, b) for b in range(1 + K)], dim=1)   # [B, 1+K, 2048]
    bn_proj_global = heads.bn_proj(proj_branches[:, :1], 0).unsqueeze(1)                    # [B, 1, D]
    l_id = id_loss(heads.cls_x4(bn_x4_all, 0), targets) + id_loss(heads.cls_proj(bn_proj_global, 0), targets)
    l_id_parts = imgs.new_zeros(())
    for k in range(1, 1 + K):
        l_id_parts = l_id_parts + weighted_id_loss(heads.cls_x4(bn_x4_all, k), targets, vis[:, k], num_identities)
    total = total + cfg.loss.id_weight * (l_id + l_id_parts)
    log['id'] = l_id.item()
    log['id_parts'] = l_id_parts.item()

    # L_tri: CLIP-ReID's triplet on each of x3 / x4 / x_proj (raw, un-normalized, as there) + one
    # BPBreID-style part-based triplet over part_x4 (mean of the visible parts' distances).
    vis_mask = vis >= cfg.loss.triplet_visibility_min
    l_tri = imgs.new_zeros(())
    global_feats = [g_x4, out['x_proj']]
    if out['x3'] is not None:
        global_feats.insert(0, out['x3'].mean(dim=(2, 3)))
    for feat in global_feats:
        result = triplet_loss(feat.unsqueeze(1), targets)
        if result is not None:
            l_tri = l_tri + result[0]
    # Per-part triplets: part k mined on ITS OWN distances (a single-branch call), so one part
    # that separates two people is never diluted by the parts that don't -- the training-side
    # counterpart of the soft-min retrieval rule.
    l_tri_parts = imgs.new_zeros(())
    for k in range(K):
        r = triplet_loss(out['part_x4'][:, k:k + 1], targets, parts_visibility=vis_mask[:, k + 1:k + 2])
        if r is not None:
            l_tri_parts = l_tri_parts + r[0]
    # Combined part triplet under the retrieval rule (cfg.eval.part_combine): trains the
    # combination the evaluator actually scores with.
    part_result = triplet_loss(out['part_x4'], targets, parts_visibility=vis_mask[:, 1:])
    l_tri_lse = part_result[0] if part_result is not None else imgs.new_zeros(())
    total = total + cfg.loss.triplet_weight * (l_tri + l_tri_parts + l_tri_lse)
    log['tri'] = l_tri.item()
    log['tri_parts'] = l_tri_parts.item()
    log['tri_parts_lse'] = l_tri_lse.item()

    # L_cen: per-part contrast against every identity's same-part centroid (see module
    # docstring). On the joint-space branches -- the exact features retrieval matches on.
    # Centroids update only from parts visible enough to trust (triplet_visibility_min).
    l_cen, cen_parts = part_memory(proj_branches, targets, vis, vis_mask)
    total = total + cfg.loss.centroid_weight * l_cen
    log['cen'] = l_cen.item()
    for m, v in enumerate(cen_parts.tolist()):
        log['cen%d' % m] = v

    # L_align (CLIP-ReID's I2T): each joint-space branch classified against ITS OWN branch's
    # full frozen prototype table (every identity an implicit negative) -- global at weight 1,
    # each part weighted by its visibility. BNNeck is deliberately NOT applied here: this is the
    # cosine geometry CLIP was trained in, and it's what retrieval matches on.
    l_align = align_loss(proj_branches[:, 0], text_prototypes[:, 0], targets, weights=vis[:, 0])
    l_align_parts = imgs.new_zeros(())
    for k in range(1, 1 + K):
        l_align_parts = l_align_parts + align_loss(proj_branches[:, k], text_prototypes[:, k], targets,
                                                  weights=vis[:, k])
    total = total + cfg.loss.align_weight * (l_align + l_align_parts)
    log['align'] = l_align.item()
    log['align_parts'] = l_align_parts.item()

    if bpa_loss is not None:
        mask_targets = mask_to_pixel_targets(mask.to(imgs.device), out['pixels_cls_scores'])
        l_bpa, _ = bpa_loss(out['pixels_cls_scores'], mask_targets)
        bpa_weight = bpa_weight_schedule(epoch, cfg.loss.bpa_weight_initial, cfg.loss.bpa_weight_floor,
                                          cfg.loss.bpa_weight_decay_epochs)
        total = total + bpa_weight * l_bpa
        log['bpa'] = l_bpa.item()
        log['bpa_w'] = bpa_weight

    with torch.no_grad():
        parts = proj_branches[:, 1:]
        off = ~torch.eye(K, dtype=torch.bool, device=parts.device)
        log['part_cos'] = torch.einsum('bkd,bjd->bkj', parts, parts)[:, off].mean().item()

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
    num_parts = cfg.model.parts_num
    num_branches = 1 + num_parts
    train_set = sorted(dataset.train)

    proto_path = osp.join(cfg.stage1.prompt_dir, 'text_prototypes.pth')
    proto = load_checkpoint(proto_path)
    assert proto['num_identities'] == num_identities, (
        "Stage 1's text_prototypes.pth was built for {} identities, this dataset has {}".format(
            proto['num_identities'], num_identities))
    assert proto['num_branches'] == num_branches, (
        "Stage 1's text_prototypes.pth was built for {} branches, this config expects {} "
        "(parts_num={}) -- stage1 and stage2 must agree.".format(
            proto['num_branches'], num_branches, num_parts))
    text_prototypes = proto['text_prototypes'].cuda()  # [num_identities, 1+K, D]

    encoder = build_encoder(cfg)
    heads = Heads(num_identities, num_parts, encoder.backbone.vision_width, encoder.num_features).cuda()

    # Same part-combination rule as retrieval (cfg.eval): the part triplet mines its hard
    # negatives under the metric the model is tested with.
    triplet_loss = PartTripletLoss(margin=cfg.loss.triplet_margin, combine=cfg.eval.part_combine,
                                   temperature=cfg.eval.lse_temperature).cuda()
    id_loss = CrossEntropyLabelSmooth(num_identities).cuda()
    align_loss = CosineAlignLoss(temperature=cfg.loss.align_temperature).cuda()
    use_masks = bool(cfg.data.masks_dir)
    bpa_loss = BodyPartAttentionLoss().cuda() if use_masks else None

    part_memory = PerPartCentroidMemory(encoder.num_features, num_branches, num_identities,
                                        temp=cfg.loss.centroid_temp, momentum=cfg.loss.centroid_momentum).cuda()

    train_loader = get_train_loader(dataset, cfg, train_set)
    test_loader = get_test_loader(dataset, cfg.data.height, cfg.data.width, cfg.data.batch_size, cfg.data.workers)

    if setup_only:
        print('==> Setup complete: {} identities, {} branches, text_prototypes {}, BPA {}. Exiting '
              'before the training loop (--setup-only).'.format(
                  num_identities, num_branches, tuple(text_prototypes.shape),
                  'ON (' + cfg.data.masks_dir + ')' if use_masks else 'off'))
        return

    # Centroid init: one no-grad pass over the training set with the encoder as Stage 0/1 left
    # it, visibility-weighted mean per (identity, branch) (plain mean where a branch is never
    # visible for an identity) -- otherwise every row starts at zero and the softmax is
    # meaningless until each identity has been seen once.
    print('==> Initializing per-(identity, part) centroids')
    encoder.eval()
    init_loader = get_test_loader(dataset, cfg.data.height, cfg.data.width, cfg.data.batch_size,
                                   cfg.data.workers, testset=train_set)
    init_features, init_vis, _ = extract_features(encoder, init_loader)
    sums = torch.zeros(num_identities, num_branches, encoder.num_features)
    wsum = torch.zeros(num_identities, num_branches, 1)
    plain = torch.zeros_like(sums)
    counts = torch.zeros(num_identities, 1, 1)
    for fname, pid, _ in train_set:
        f, v = init_features[fname], init_vis[fname].float().unsqueeze(-1)
        sums[pid] += f * v
        wsum[pid] += v
        plain[pid] += f
        counts[pid] += 1
    centers = torch.where(wsum > 0, sums / wsum.clamp(min=1e-6), plain / counts.clamp(min=1))
    part_memory.features = F.normalize(centers, dim=-1).cuda()
    del init_loader, init_features, init_vis, sums, wsum, plain, counts, centers
    print('==> centroids initialized for {} identities x {} branches'.format(num_identities, num_branches))

    optimizer = build_optimizer(encoder, heads, cfg)
    is_vit = cfg.clip.arch.startswith('ViT')
    warmup_epochs = cfg.vit.warmup_epochs if is_vit else cfg.optim.warmup_epochs
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=list(cfg.optim.milestones), gamma=0.1,
                                      warmup_factor=cfg.optim.warmup_factor,
                                      warmup_iters=warmup_epochs, warmup_method='linear')
    evaluator = Evaluator(encoder, part_combine=cfg.eval.part_combine, temperature=cfg.eval.lse_temperature)

    best_mAP = 0
    for epoch in range(cfg.optim.epochs):
        encoder.train()
        heads.train()
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
            loss, log = compute_losses(encoder, heads, id_loss, triplet_loss, align_loss, bpa_loss,
                                       part_memory, text_prototypes, imgs, mask, targets, cfg, epoch)
            loss.backward()
            optimizer.step()

            if (it + 1) % cfg.logging.print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.3f}\t{}'.format(
                    epoch, it + 1, train_iters, loss.item(),
                    '\t'.join('{} {:.3f}'.format(k, v) for k, v in log.items())))

        lr_scheduler.step()
        print('Epoch {} done in {:.1f}s'.format(epoch, time.time() - epoch_start))

        if (epoch + 1) % cfg.logging.eval_step == 0 or epoch == cfg.optim.epochs - 1:
            # float(): mean_ap() returns numpy.float64; a numpy scalar in the pickled checkpoint
            # breaks bpbreid's weights_only=True loader downstream (Stage 3).
            mAP = float(evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False))
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': encoder.state_dict(),
                'heads_state_dict': heads.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                'optimizer': optimizer.state_dict(),
            }, is_best, fpath=osp.join(cfg.logging.logs_dir, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.format(
                epoch, mAP, best_mAP, ' *' if is_best else ''))

    print('==> Test with the best model:')
    best_fpath = osp.join(cfg.logging.logs_dir, 'model_best.pth.tar')
    if osp.isfile(best_fpath):
        encoder.load_state_dict(load_checkpoint(best_fpath)['state_dict'])
    else:
        print('No model_best.pth.tar in {}, testing with the final model'.format(cfg.logging.logs_dir))
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
