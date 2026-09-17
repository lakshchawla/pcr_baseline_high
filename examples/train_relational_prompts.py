from __future__ import print_function, absolute_import
import argparse
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
from pcr.models.clip_text_encoder import ClipTextEncoder
from pcr.models.prompt_learner import PromptLearner
from pcr.models.relation_blocks import VisualAttentionBlock, AttentionPoolingBlock, apply_vab_with_pooling
from pcr.loss.clip_supcon_loss import SupConLoss
from pcr.loss.part_diag_loss import PartDiagLoss, same_identity_part_cosine
from pcr.loss.cross_attn_align_loss import cross_attention_alignment_loss
from pcr.utils.config import load_yaml_config
from pcr.utils.data import transforms as T
from pcr.utils.data.preprocessor import Preprocessor
from pcr.utils.logging import Logger
from pcr.utils.lr_scheduler import WarmupCosineLR
from pcr.utils.osutils import mkdir_if_missing


def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir, name))


def get_test_transform(height, width):
    # CLIP's own normalization, NOT ImageNet stats -- the frozen CLIP backbone's pretrained
    # weights are calibrated for this specific normalization; using the wrong one silently
    # miscalibrates every input to it (a real bug found this way once already in this pipeline's
    # history, before this branch existed -- checked directly rather than assumed here).
    normalizer = T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))
    return T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])


def get_cache_loader(dataset_list, root, height, width, batch_size, workers):
    return DataLoader(
        Preprocessor(dataset_list, root=root, transform=get_test_transform(height, width)),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)


def get_batch_image_loader(dataset_list, root, height, width, batches, workers):
    """Re-reads the images behind this epoch's already-drawn PK batches (build_pk_batches'
    output, indices into `dataset_list` -- the same sorted list the feature cache was built
    over, so index i means the same image in both), in that exact batch order, for the live
    encoder forward the CLIP-native mask blend needs. Same test-time transform as the cache
    pass: the frozen backbone must see the identical pixels it was cached on, so a blend_weight
    of 0 reproduces the cached features exactly."""
    return DataLoader(
        Preprocessor(dataset_list, root=root, transform=get_test_transform(height, width)),
        batch_sampler=[b.tolist() for b in batches], num_workers=workers, pin_memory=True)


def cache_part_features(encoder, data_loader):
    """Single full-dataset forward pass under no_grad, caching every image's part embeddings,
    visibility, and real identity label -- mirrors CLIP-ReID's own stage-1 full-dataset feature
    cache, generalized to BPBreID's [M, D] per-branch embeddings."""
    encoder.eval()
    features, visibilities, labels = [], [], []
    with torch.no_grad():
        for imgs, _, pids, _, _ in data_loader:
            f_out, vis = encoder(imgs.cuda())
            features.append(f_out.cpu())
            visibilities.append(vis.cpu())
            labels.append(pids)
    return torch.cat(features, 0), torch.cat(visibilities, 0), torch.cat(labels, 0)


def compute_identity_visibility(cached_visibility, cached_labels, num_identities):
    """cached_visibility: [N, 1+K] (per-image, from cache_part_features). cached_labels: [N].
    Returns [num_identities, 1+K]: each identity's mean visibility across every cached image of
    that identity. TextualAttentionBlock has no per-image signal available (PromptLearner.ctx
    is indexed by identity alone -- see relation_blocks.py's own docstring), so this is the
    per-identity substitute passed into it as its attention bias; saved to disk
    (identity_visibility.pth) so examples/cache_text_anchors.py can reuse the exact same values
    when it rebuilds the final frozen prompts, keeping TAB's output a true, consistent function of
    identity alone rather than depending on which script last computed it."""
    num_branches = cached_visibility.size(1)
    sums = torch.zeros(num_identities, num_branches, device=cached_visibility.device)
    sums.index_add_(0, cached_labels, cached_visibility)
    counts = torch.zeros(num_identities, device=cached_visibility.device)
    counts.index_add_(0, cached_labels, torch.ones_like(cached_labels, dtype=cached_visibility.dtype))
    return sums / counts.unsqueeze(1).clamp(min=1)


def build_text_snapshot(prompt_learner, text_encoder, num_identities, num_branches,
                         identity_visibility, id_batch):
    """Full-dataset text-anchor snapshot, [num_identities, num_branches, D] (branch 0 =
    global/foreground, 1..K = parts), rebuilt once at the start of every epoch (not every
    iteration -- see main_worker's own call site) under no_grad, using the model's CURRENT
    ctx/TextualAttentionBlock weights. This is what widens Stage 1's negative pool from "the ~8
    identities in one PK batch" to "every identity in the training
    set", matching CLIP-ReID's own original Stage 1 design (full-identity-table classification,
    not a batch-restricted one) -- see plans/IMPROVEMENT_PLAN.md section 4 and progress.md's entry on
    this change for the full reasoning.

    Only used as a *negative* pool for identities NOT present in the current PK batch (see
    main_worker's own per-iteration splicing) -- an identity that IS in the current batch gets a
    fresh, differentiable re-encoding instead, since gradient must still reach ctx/TAB for it.
    A once-per-epoch refresh (rather than once per iteration) keeps this affordable: rebuilding it
    costs one extra CLIP-text forward pass over the whole identity set, not per training step, and
    a whole epoch's worth of iterations (hundreds) is far more than enough for the small
    per-iteration drift in ctx/TAB to matter for what is, after all, only a negative-comparison
    pool, not something being directly optimized against.

    TextualAttentionBlock's own attention only mixes tokens *within* one identity's own K*n_ctx-
    token sequence (standard transformer batching never attends across the batch dimension), so
    building this in chunks of `id_batch` identities at a time is exactly equivalent to building
    every identity one at a time -- no cross-identity leakage or batching-order sensitivity."""
    prompt_learner.eval()
    D = text_encoder.embed_dim
    snapshot = torch.zeros(num_identities, num_branches, D, device='cuda')
    with torch.no_grad():
        for start in range(0, num_identities, id_batch):
            ids = torch.arange(start, min(start + id_batch, num_identities), device='cuda')
            branch_vis = identity_visibility[ids]
            prompts, _ = prompt_learner.build_part_prompts(ids, branch_vis)
            for b in range(num_branches):
                text_feat = text_encoder(prompts[b], prompt_learner.tokenized_prompts).float()
                # L2-normalized before storing -- see this file's own module docstring (the
                # "SupCon" mapping-table entry) for why: SupConLoss's dot product only behaves as
                # a real cosine similarity, matching its temperature's calibration, if both sides
                # are unit-norm -- branch_visual (the image side) already is; this was the one place
                # text wasn't.
                snapshot[ids, b] = F.normalize(text_feat, p=2, dim=-1)
    return snapshot


def build_pk_batches(cached_labels, num_instances, batch_size):
    """Groups the cached feature set's indices by identity, then partitions all identities into
    PK batches for one epoch: batch_size // num_instances identities per batch, num_instances
    cached images per identity (sampled with replacement if that identity has fewer than
    num_instances cached images). Algorithm 1 step 6 ("Sample a PK batch of pre-filtered
    images") -- see this file's own module docstring for why SupConLoss's multi-positive mechanism
    depends on this. A final partial group of identities (fewer than batch_size // num_instances
    left over) is dropped, matching this repo's other PK samplers' drop_last convention
    (pcr/utils/data/sampler.py::RandomIdentitySampler)."""
    labels_np = cached_labels.cpu().numpy()
    id_to_indices = {}
    for idx, pid in enumerate(labels_np):
        id_to_indices.setdefault(int(pid), []).append(idx)
    pids = list(id_to_indices.keys())
    random.shuffle(pids)

    num_pids_per_batch = max(1, batch_size // num_instances)
    batches = []
    for start in range(0, len(pids), num_pids_per_batch):
        batch_pids = pids[start:start + num_pids_per_batch]
        if len(batch_pids) < num_pids_per_batch:
            break
        batch_idx = []
        for pid in batch_pids:
            pool = id_to_indices[pid]
            replace = len(pool) < num_instances
            chosen = np.random.choice(pool, size=num_instances, replace=replace)
            batch_idx.extend(int(i) for i in chosen)
        batches.append(torch.tensor(batch_idx, dtype=torch.long, device=cached_labels.device))
    return batches


def ramp_schedule(epoch, total_epochs, warmup_fraction, ramp_fraction, max_value):
    """0 during warmup, then a linear ramp up to max_value, then flat -- same shape as
    examples/train_relational_finetune.py's crossalign_schedule. Shared by L_relalign's lambda
    and the CLIP-native mask blend weight (each with its own config block: the two ramp on
    independent timescales, so their fractions are never copied from one another)."""
    warmup_end = warmup_fraction * total_epochs
    ramp_end = warmup_end + ramp_fraction * total_epochs
    if epoch < warmup_end:
        return 0.0
    if epoch >= ramp_end:
        return max_value
    progress = (epoch - warmup_end) / (ramp_end - warmup_end)
    return max_value * progress


def build_encoder(cfg):
    # CLIP's own frozen visual tower + BPBreID's part-attention head (pcr/models/
    # clip_rn50_bpam_encoder.py / clip_vit_bpam_encoder.py) -- replaces BPBreIDEncoder(HRNet32)
    # here so Stage 1's ctx/TAB train against the SAME frozen visual encoder Stage 2 will continue
    # fine-tuning from (Stage 2's own docstring already assumes this invariant; see this file's
    # own module docstring for why the two stages must agree on this). Dispatched on cfg.clip.arch
    # -- 'ViT-*' picks the ViT dense backbone (needs the V-V attention surgery, a real
    # architectural difference), anything else (RN50/RN101/RN50x*) picks the RN-family one
    # (AttentionPool2d re-invocation instead) -- see each backbone's own module docstring for why
    # the fix differs by family. Visibility is continuous by construction in ClipBPAMEncoder
    # (softmax attention maps, no binary mode) -- no config knob needed here, unlike
    # BPBReIDModelCfg's training/testing_binary_visibility_score.
    encoder_cls = ClipViTBPAMEncoder if cfg.clip.arch.startswith('ViT') else ClipRN50BPAMEncoder
    encoder = encoder_cls(clip_arch=cfg.clip.arch, height=cfg.data.height, width=cfg.data.width,
                           num_parts=cfg.model.parts_num,
                           checkpoint_path=cfg.model.checkpoint_path or None, device='cuda',
                           mask_temperature=cfg.clip_native_mask.mask_temperature).cuda()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def main():
    parser = argparse.ArgumentParser(description="PCR Stage 1: per-part CLIP prompt learning")
    parser.add_argument('--config', type=str, metavar='PATH', default="configs/stage1_relational_prompts.yaml")
    parser.add_argument('--setup-only', action='store_true',
                         help="build dataset/encoder/prompt-learner/cache, print shapes, exit "
                              "before the training loop")
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

    # build_encoder (above) IS Algorithm 1's "load pretrained CLIP image encoder... freeze" step
    # now -- ClipRN50BPAMEncoder wraps CLIP's own RN50 visual tower directly, so there's no
    # separate, unused image_encoder placeholder to construct alongside it any more (see this
    # file's module docstring history / progress.md for the dead-code version this replaces).
    encoder = build_encoder(cfg)
    # has_global_branch: read off the already-built encoder rather than re-deriving "which arch
    # has this" separately here -- ClipBPAMEncoder._has_global is True only for
    # ClipRN50BPAMEncoder currently (see that class's own docstring: a genuine 7th branch, CLIP's
    # own native whole-image embedding). PromptLearner needs to match this exactly, or its own
    # ctx/TAB branch count disagrees with what the image side actually produces.
    has_global_branch = getattr(encoder, '_has_global', False)
    num_branches = (2 if has_global_branch else 1) + num_parts
    text_encoder = ClipTextEncoder(clip_arch=cfg.clip.arch, device='cuda').cuda()
    prompt_learner = PromptLearner(num_identities, num_parts, text_encoder, n_ctx=cfg.clip.n_ctx,
                                    tab_num_heads=cfg.tab.num_heads, tab_num_layers=cfg.tab.num_layers,
                                    device='cuda', has_global_branch=has_global_branch).cuda()
    vab = VisualAttentionBlock(dim=cfg.model.dim_reduce_output, num_heads=cfg.vab.num_heads,
                               num_layers=cfg.vab.num_layers).cuda()
    # AttentionPoolingBlock: turns the K (VAB-mixed) part tokens into the global embedding --
    # replaces foreground's old peer-token role. See relation_blocks.py's own module docstring.
    pool = AttentionPoolingBlock(dim=cfg.model.dim_reduce_output, num_heads=cfg.vab.num_heads).cuda()

    train_set = sorted(dataset.train)
    print("==> Caching part-embeddings for the full training set (frozen encoder, single pass, "
          "no upstream filtering -- every image enters training, weighted per-part inside the "
          "loss instead)")
    cache_loader = get_cache_loader(train_set, dataset.images_dir, cfg.data.height, cfg.data.width,
                                     cfg.data.cache_batch_size, cfg.data.workers)
    cached_features, cached_visibility, cached_labels = cache_part_features(encoder, cache_loader)
    cached_features = cached_features.cuda()
    cached_visibility = cached_visibility.cuda()
    cached_labels = cached_labels.cuda()
    num_images = cached_labels.size(0)
    print("==> Cached {} images across {} identities, {} branches".format(
        num_images, num_identities, num_branches))

    if setup_only:
        print('==> Setup complete: {} branches, {} cached images, ctx shape {} (trainable). '
              'Exiting before the training loop (--setup-only).'.format(
                  num_branches, num_images, tuple(prompt_learner.ctx.shape)))
        return

    # Per-identity mean visibility -- TextualAttentionBlock's attention bias (see
    # compute_identity_visibility's own docstring and relation_blocks.py for why this differs from
    # VisualAttentionBlock's per-image one).
    identity_visibility = compute_identity_visibility(cached_visibility, cached_labels, num_identities)

    mask_cfg = cfg.clip_native_mask
    supcon = SupConLoss(temperature=cfg.loss.temperature).cuda()
    # Within-identity part contrast -- the negative set SupCon lacks (a person's own other parts).
    # See pcr/loss/part_diag_loss.py for the collapse it exists to prevent.
    part_diag = PartDiagLoss(temperature=cfg.part_diag.temperature).cuda()
    # ctx (all M=1+K branches, global/foreground included), TextualAttentionBlock
    # (prompt_learner.tab), VisualAttentionBlock, and SupConLoss's own learnable temperature
    # (see that file's own docstring) all train.
    trainable_params = ([prompt_learner.ctx] + list(prompt_learner.tab.parameters())
                         + list(vab.parameters()) + list(pool.parameters()) + list(supcon.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.optim.lr,
                                  weight_decay=cfg.optim.weight_decay)
    scheduler = WarmupCosineLR(optimizer, max_epochs=cfg.optim.epochs,
                                warmup_epochs=cfg.optim.warmup_epochs,
                                warmup_lr_init=cfg.optim.warmup_lr_init,
                                lr_min=cfg.optim.lr_min)
    # GradScaler, not raw fp16 backward -- the CLIP text tower runs in fp16 (matches CLIP-ReID's
    # own dtype exactly, see pcr/models/clip_text_encoder.py's docstring), and CLIP-ReID's own
    # stage-1 loop always wraps its backward in a GradScaler to guard against fp16 gradient
    # underflow through the text transformer -- ported faithfully rather than assuming raw fp16
    # backward is fine. VisualAttentionBlock and PromptLearner's own parameters run in fp32
    # (GradScaler is harmless for fp32 leaves), so one scaler covers everything trainable.
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(cfg.optim.epochs):
        # Refresh the full-identity text-anchor snapshot once per epoch, against this epoch's
        # ctx/TAB weights -- see build_text_snapshot's own docstring. Puts prompt_learner in
        # eval() mode briefly; explicitly switched back to train() below before any real training
        # step runs.
        text_snapshot = build_text_snapshot(prompt_learner, text_encoder, num_identities, num_branches,
                                             identity_visibility, cfg.data.cache_batch_size)
        prompt_learner.train()
        vab.train()
        pool.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        # Algorithm 1 step 6: a fresh PK partition of the cached feature set every epoch, not a
        # plain random sub-batch -- see build_pk_batches' and this file's own module docstring.
        batches = build_pk_batches(cached_labels, cfg.data.num_instances, cfg.data.batch_size)
        iters_per_epoch = len(batches)

        # CLIP-native mask blend (pcr/models/clip_native_masks.py, blended inside
        # ClipBPAMEncoder._forward_common): once this epoch's blend weight is > 0, each batch's
        # images are re-run through the (still frozen) encoder live, with this batch's own fresh
        # per-identity text contexts steering part of the part assignment -- so the pooled
        # features, and through them SupCon's gradient, depend on the contexts' current state.
        # At blend weight 0 the cached features are that same forward's exact output, so the
        # images aren't re-read at all (identical behavior to before this mechanism existed).
        blend_weight = ramp_schedule(epoch, cfg.optim.epochs, mask_cfg.warmup_fraction,
                                     mask_cfg.ramp_fraction, mask_cfg.blend_weight_max)
        blend_active = blend_weight > 0.0
        if blend_active:
            image_batches = iter(get_batch_image_loader(train_set, dataset.images_dir, cfg.data.height,
                                                        cfg.data.width, batches, cfg.data.workers))
        # Anchor weight scales with the blend weight itself (zero whenever the blend is off, so a
        # blend_weight_max of 0 still reproduces the pre-blend script exactly).
        lambda_anchor = mask_cfg.anchor_weight * (blend_weight / mask_cfg.blend_weight_max
                                                  if mask_cfg.blend_weight_max > 0 else 0.0)
        lambda_part_diag = ramp_schedule(epoch, cfg.optim.epochs, cfg.part_diag.warmup_fraction,
                                         cfg.part_diag.ramp_fraction, cfg.part_diag.lambda_max)
        mask_delta_sum = torch.zeros(num_parts, device='cuda')
        outside_sum = 0.0
        part_cos_sum = 0.0

        for it, b_idx in enumerate(batches):
            b_labels = cached_labels[b_idx]
            b_features = cached_features[b_idx]  # [b, 1+K, D], already L2-normalized per branch
            b_vis = cached_visibility[b_idx]     # [b, 1+K]

            optimizer.zero_grad()

            # Algorithm 1 steps 10-14, extended to all M=1+K branches (global/foreground + K
            # parts, uniformly -- see relation_blocks.py's own module docstring): every branch's
            # prompt is built and pushed through the frozen CLIP text encoder. All branches' text
            # rows are built up front (not one at a time inside the loss loop below) because the
            # mask blend needs every part's context before the image side can be computed.
            id_vis = identity_visibility[b_labels]  # [b, 1+K], TAB's per-identity bias
            prompts, A_text = prompt_learner.build_part_prompts(b_labels, id_vis)  # list of 1+K tensors
            # L2-normalized -- see build_text_snapshot's own comment on why: the visual side is
            # already unit-norm, and SupConLoss's dot product only behaves as a real cosine
            # similarity, matching its own temperature, if both sides are.
            branch_texts = torch.stack([
                F.normalize(text_encoder(prompts[m], prompt_learner.tokenized_prompts).float(), p=2, dim=-1)
                for m in range(num_branches)], dim=1)  # [b, 1+K, D], fresh + differentiable

            if blend_active:
                imgs, _, img_pids, _, _ = next(image_batches)
                assert torch.equal(img_pids.to(b_labels.device), b_labels), \
                    "image loader fell out of step with this epoch's PK batches"
                # Frozen encoder, but NOT under no_grad: the blend is the one path through which
                # gradient reaches ctx/TAB from the image side (via the text-matched masks ->
                # pooled features); the backbone itself contributes no graph (requires_grad off).
                b_features, b_vis, _ = encoder.forward_full(imgs.cuda(non_blocking=True),
                                                            text_contexts=branch_texts,
                                                            blend_weight=blend_weight)
                blend_stats = encoder.last_blend_stats
                mask_delta_sum += blend_stats['mask_delta']
                outside_sum += blend_stats['outside_support'].item()
                # This batch's images now have a live (blended) feature row that supersedes their
                # cached (unblended) one -- splice it in for the t2i comparison set below, exactly
                # as i2t already splices fresh text rows over the snapshot for in-batch identities.
                cached_others = torch.ones(num_images, dtype=torch.bool, device=b_idx.device)
                cached_others[b_idx] = False
                t2i_features = torch.cat([b_features, cached_features[cached_others]], dim=0)
                t2i_labels = torch.cat([b_labels, cached_labels[cached_others]], dim=0)
            else:
                t2i_features, t2i_labels = cached_features, cached_labels

            # apply_vab_with_pooling (not bare vab()): foreground gates the K parts, VAB mixes the
            # gated parts only, then pool (AttentionPoolingBlock) aggregates them into a new
            # global -- see relation_blocks.py's own module docstring. branch_visual keeps the
            # exact same [b, 1+K, D] shape/layout as before (global at branch 0).
            branch_visual, A_vis = apply_vab_with_pooling(vab, pool, b_features, b_vis, has_global_branch)

            # Identities NOT in this batch -- their text row comes from this epoch's (detached)
            # snapshot instead of a fresh re-encoding, widening i2t's negative pool to the full
            # training set. See build_text_snapshot's own docstring / plans/IMPROVEMENT_PLAN.md section 4.
            in_batch = torch.zeros(num_identities, dtype=torch.bool, device=b_labels.device)
            in_batch[b_labels] = True
            other_ids = in_batch.logical_not().nonzero(as_tuple=True)[0]

            # Algorithm 1 step 15 (loss_i2t + loss_t2i), via SupConLoss's own two-call convention
            # (see that file's docstring), summed over all M=1+K branches.
            loss = b_features.new_zeros(())
            for m in range(num_branches):
                branch_text = branch_texts[:, m, :]
                visual_m = branch_visual[:, m, :]
                w_m = b_vis[:, m]

                # i2t: full num_identities-way classification, not just this batch's ~8 -- this
                # batch's own identities keep their fresh, differentiable text row (gradient must
                # reach ctx/TAB for them); every other identity is a detached negative from this
                # epoch's text_snapshot.
                other_text = torch.cat([branch_text, text_snapshot[other_ids, m, :]], dim=0)
                other_text_labels = torch.cat([b_labels, other_ids], dim=0)
                loss = loss + supcon(visual_m, other_text, b_labels, other_text_labels, w_m)

                # t2i: full-dataset classification -- every cached image, not just this batch's,
                # is a comparison point. The cache never goes stale (backbone/BPAM are frozen for
                # the whole of Stage 1); only when the mask blend is active do this batch's own
                # rows get superseded by their live, blended versions (spliced above).
                loss = loss + supcon(branch_text, t2i_features[:, m, :], b_labels, t2i_labels, w_m)

            # L_relalign: pushes VAB's own branch-to-branch attention pattern (A_vis, per-image)
            # toward TAB's (A_text, per-identity, detached -- this loss trains VAB, not TAB) -- a
            # direct regularizer against both blocks converging to degenerate, near-identical
            # relational patterns across branches (prompt/branch-embedding collapse), on top of
            # whatever the SupCon gradient above already does. Ramped in on a schedule since both
            # blocks' patterns are meaningless before SupCon has shaped them at all.
            #
            # A_vis is now [b,K,K] (VAB mixes the K real parts only, see relation_blocks.py's own
            # module docstring); TAB is untouched by this fork's changes, so A_text is still
            # [b,1+K,1+K] (foreground included) -- sliced to its own parts-only [1:,1:] sub-block
            # (both axes, since this is self-attention on both sides) to match shapes.
            lambda_relalign = ramp_schedule(epoch, cfg.optim.epochs, cfg.relalign.warmup_fraction,
                                            cfg.relalign.ramp_fraction, cfg.relalign.lambda_max)
            # Renormalized after slicing: the parts-only sub-block of a row-stochastic [M, M]
            # matrix has rows summing to < 1 (the global columns' mass is gone), and a KL against
            # a sub-normalized target reads negative (seen in real logs: relalign ~ -0.2). The
            # gradient direction was already right (the optimum is the renormalized target
            # either way); this makes the logged value a real KL >= 0.
            a_text_target = A_text.detach()[:, 1:1 + num_parts, 1:1 + num_parts]
            a_text_target = a_text_target / a_text_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            l_relalign = cross_attention_alignment_loss(A_vis, a_text_target)
            loss = loss + lambda_relalign * l_relalign

            # L_part_diag: image part k vs the SAME person's K part contexts (and the reverse) --
            # the within-identity negatives SupCon never sees. Global branches excluded (no
            # sibling parts to contrast against). Uses the VAB-mixed image parts, i.e. the same
            # tensor SupCon's i2t reads, so both losses shape the same feature.
            text_parts = branch_texts[:, 1:1 + num_parts, :]
            l_part_diag = part_diag(branch_visual[:, 1:1 + num_parts, :], text_parts,
                                    b_vis[:, 1:1 + num_parts])
            loss = loss + lambda_part_diag * l_part_diag
            part_cos_sum += same_identity_part_cosine(text_parts).item()

            # L_anchor: keeps the text-matched part maps inside the classifier's anatomy while
            # the blend is active -- see ClipBPAMEncoder._forward_common. Zero-cost when off.
            if blend_active:
                loss = loss + lambda_anchor * blend_stats['anchor_loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if (it + 1) % cfg.logging.print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.3f}\tLR {:.2e}\tVAB gate {:.3f}\tTAB gate {:.3f}'
                      '\tPool gate {:.3f}\trelalign {:.4f} (x{:.2f})\tpart_diag {:.4f} (x{:.2f})'
                      '\tanchor {} (x{:.3f})'.format(
                    epoch, it + 1, iters_per_epoch, loss.item(), optimizer.param_groups[0]['lr'],
                    torch.tanh(vab.gate).item(), torch.tanh(prompt_learner.tab.gate).item(),
                    torch.tanh(pool.gate).item(), l_relalign.item(), lambda_relalign,
                    l_part_diag.item(), lambda_part_diag,
                    '{:.4f}'.format(blend_stats['anchor_loss'].item()) if blend_active else 'off',
                    lambda_anchor))

        scheduler.step()
        # Per-part mean |text-matched - classifier| part mass (independent of the blend weight),
        # averaged over the epoch: near zero means the CLIP-native assignment agrees with the
        # supervised prior (the blend isn't doing much); large and shrinking over epochs is the
        # text side converging toward it as contexts mature; GROWING over epochs means the
        # contexts are pulling the masks away from anatomy -- stop and lower blend_weight_max.
        mask_delta = (mask_delta_sum / iters_per_epoch).tolist() if blend_active else None
        # part-ctx cos: mean cosine between DIFFERENT parts' contexts of the SAME identity, in the
        # joint space -- the "are the part prompts actually part-specific" metric (1.0 = fully
        # collapsed; 0.98 on the pre-fix checkpoint). outside-support: fraction of the text map's
        # mass the classifier puts at < 5% for that part -- "how much of it is outside anatomy".
        print('Epoch {} done in {:.1f}s, avg loss {:.4f}, SupCon temperature {:.4f}, '
              'part-ctx cos {:.3f}, clip-mask blend {:.3f}, mask delta/part {}, outside-support {}'.format(
            epoch, time.time() - epoch_start, epoch_loss / iters_per_epoch, supcon.temperature.item(),
            part_cos_sum / iters_per_epoch, blend_weight,
            None if mask_delta is None else ['{:.4f}'.format(d) for d in mask_delta],
            '{:.3f}'.format(outside_sum / iters_per_epoch) if blend_active else None))

    torch.save(prompt_learner.state_dict(), osp.join(cfg.logging.logs_dir, 'prompt_learner.pth'))
    torch.save(vab.state_dict(), osp.join(cfg.logging.logs_dir, 'vab.pth'))
    torch.save(pool.state_dict(), osp.join(cfg.logging.logs_dir, 'pool.pth'))
    torch.save(identity_visibility.cpu(), osp.join(cfg.logging.logs_dir, 'identity_visibility.pth'))
    print('==> Saved prompt_learner.pth, vab.pth, pool.pth and identity_visibility.pth to {}. Run '
          'examples/cache_text_anchors.py next to build text_prototypes.pth for Stage 2.'.format(
              cfg.logging.logs_dir))

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
