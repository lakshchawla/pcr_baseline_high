"""Stage 1: per-identity, per-branch CLIP prompt learning (CLIP-ReID's Stage 1, extended to K
body-part branches). The image encoder (CLIP RN50 + Stage 0's pixel classifier) and the CLIP
text encoder are frozen; only PromptLearner.ctx and SupConLoss's temperature train.

Image side (pcr/models/clip_dense_part_encoder.py): branch 0 = CLIP's own global x_proj,
branches 1..K = mask-pooled part_xproj, all in the joint space -- cached once for the whole
training set (frozen encoder), then trained on in PK batches. Text side: ctx[y, branch] spliced
into "A photo of a [ctx] person." -> frozen text encoder. No TAB/VAB/pool (removed 2026-09-17;
see progress.md).

Losses per branch m: SupCon i2t (image branch m vs every identity's text m -- this batch's fresh
rows plus a per-epoch detached snapshot of all the others) + SupCon t2i (text m vs every cached
image's branch m). Plus L_part_diag: image part k vs the SAME person's K part texts and the
reverse -- the within-identity negatives SupCon lacks, which is what keeps the K part prompts
from collapsing into one identity vector (pcr/loss/part_diag_loss.py).

Outputs (logging.logs_dir): prompt_learner.pth. Run examples/cache_text_anchors.py against the
same config afterward to build text_prototypes.pth for Stage 2.
"""
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
from pcr.loss.clip_supcon_loss import SupConLoss
from pcr.loss.part_diag_loss import PartDiagLoss, same_identity_part_cosine
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
    # weights are calibrated for this specific normalization.
    normalizer = T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))
    return T.Compose([T.Resize((height, width), interpolation=3), T.ToTensor(), normalizer])


def get_cache_loader(dataset_list, root, height, width, batch_size, workers):
    return DataLoader(
        Preprocessor(dataset_list, root=root, transform=get_test_transform(height, width)),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)


def cache_part_features(encoder, data_loader):
    """Single full-dataset forward pass under no_grad: every image's [1+K, D] joint-space
    branches, [1+K] visibility, and identity label -- CLIP-ReID's own stage-1 feature cache,
    generalized to the part branches."""
    encoder.eval()
    features, visibilities, labels = [], [], []
    with torch.no_grad():
        for imgs, _, pids, _, _ in data_loader:
            f_out, vis = encoder(imgs.cuda())
            features.append(f_out.cpu())
            visibilities.append(vis.cpu())
            labels.append(pids)
    return torch.cat(features, 0), torch.cat(visibilities, 0), torch.cat(labels, 0)


def build_text_snapshot(prompt_learner, text_encoder, num_identities, num_branches, id_batch):
    """[num_identities, num_branches, D], unit-norm, rebuilt once per epoch under no_grad from
    the current ctx: the detached negative pool for i2t (every identity, not just the batch's --
    CLIP-ReID's own full-table classification). In-batch identities get a fresh, differentiable
    row instead (see the training loop)."""
    prompt_learner.eval()
    device = prompt_learner.ctx.device
    snapshot = torch.zeros(num_identities, num_branches, text_encoder.embed_dim, device=device)
    with torch.no_grad():
        for start in range(0, num_identities, id_batch):
            ids = torch.arange(start, min(start + id_batch, num_identities), device=device)
            prompts = prompt_learner.build_part_prompts(ids)
            for b in range(num_branches):
                text_feat = text_encoder(prompts[b], prompt_learner.tokenized_prompts).float()
                snapshot[ids, b] = F.normalize(text_feat, p=2, dim=-1)
    return snapshot


def build_pk_batches(cached_labels, num_instances, batch_size):
    """A fresh PK partition of the cached set every epoch: batch_size // num_instances identities
    per batch, num_instances images each (with replacement if an identity has fewer); a final
    partial group is dropped (same drop_last convention as RandomIdentitySampler)."""
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
            chosen = np.random.choice(pool, size=num_instances, replace=len(pool) < num_instances)
            batch_idx.extend(int(i) for i in chosen)
        batches.append(torch.tensor(batch_idx, dtype=torch.long, device=cached_labels.device))
    return batches


def ramp_schedule(epoch, total_epochs, warmup_fraction, ramp_fraction, max_value):
    """0 during warmup, then a linear ramp up to max_value, then flat."""
    warmup_end = warmup_fraction * total_epochs
    ramp_end = warmup_end + ramp_fraction * total_epochs
    if epoch < warmup_end:
        return 0.0
    if epoch >= ramp_end:
        return max_value
    return max_value * (epoch - warmup_end) / (ramp_end - warmup_end)


def build_encoder(cfg):
    """Frozen CLIP visual tower + Stage 0's pixel classifier; dispatched on cfg.clip.arch
    ('ViT-*' -> ClipViTBPAMEncoder, else ClipRN50BPAMEncoder). Fully frozen in this stage."""
    encoder_cls = ClipViTBPAMEncoder if cfg.clip.arch.startswith('ViT') else ClipRN50BPAMEncoder
    encoder = encoder_cls(clip_arch=cfg.clip.arch, height=cfg.data.height, width=cfg.data.width,
                           num_parts=cfg.model.parts_num,
                           checkpoint_path=cfg.model.checkpoint_path or None, device='cuda').cuda()
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
    num_branches = 1 + num_parts

    encoder = build_encoder(cfg)
    text_encoder = ClipTextEncoder(clip_arch=cfg.clip.arch, device='cuda').cuda()
    prompt_learner = PromptLearner(num_identities, num_parts, text_encoder, n_ctx=cfg.clip.n_ctx,
                                    device='cuda').cuda()

    train_set = sorted(dataset.train)
    print("==> Caching joint-space branches for the full training set (frozen encoder, single pass)")
    cache_loader = get_cache_loader(train_set, dataset.images_dir, cfg.data.height, cfg.data.width,
                                     cfg.data.cache_batch_size, cfg.data.workers)
    cached_features, cached_visibility, cached_labels = cache_part_features(encoder, cache_loader)
    cached_features = cached_features.cuda()
    cached_visibility = cached_visibility.cuda()
    cached_labels = cached_labels.cuda()
    num_images = cached_labels.size(0)
    print("==> Cached {} images across {} identities, {} branches".format(
        num_images, num_identities, num_branches))
    with torch.no_grad():
        print("==> Frozen-encoder same-image part-vs-part cosine: {:.3f} (parts must be distinct "
              "for anything below to matter)".format(same_identity_part_cosine(cached_features[:, 1:]).item()))

    if setup_only:
        print('==> Setup complete: {} branches, {} cached images, ctx shape {} (trainable). '
              'Exiting before the training loop (--setup-only).'.format(
                  num_branches, num_images, tuple(prompt_learner.ctx.shape)))
        return

    supcon = SupConLoss(temperature=cfg.loss.temperature).cuda()
    part_diag = PartDiagLoss(temperature=cfg.part_diag.temperature).cuda()
    optimizer = torch.optim.Adam([prompt_learner.ctx] + list(supcon.parameters()),
                                  lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = WarmupCosineLR(optimizer, max_epochs=cfg.optim.epochs,
                                warmup_epochs=cfg.optim.warmup_epochs,
                                warmup_lr_init=cfg.optim.warmup_lr_init,
                                lr_min=cfg.optim.lr_min)
    # GradScaler: the CLIP text tower runs in fp16 (CLIP-ReID's own dtype); its stage-1 loop
    # wraps backward in a GradScaler against fp16 gradient underflow through the transformer.
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(cfg.optim.epochs):
        text_snapshot = build_text_snapshot(prompt_learner, text_encoder, num_identities,
                                             num_branches, cfg.data.cache_batch_size)
        prompt_learner.train()
        epoch_loss = 0.0
        part_cos_sum = 0.0
        epoch_start = time.time()
        batches = build_pk_batches(cached_labels, cfg.data.num_instances, cfg.data.batch_size)
        iters_per_epoch = len(batches)
        lambda_part_diag = ramp_schedule(epoch, cfg.optim.epochs, cfg.part_diag.warmup_fraction,
                                         cfg.part_diag.ramp_fraction, cfg.part_diag.lambda_max)

        for it, b_idx in enumerate(batches):
            b_labels = cached_labels[b_idx]
            b_features = cached_features[b_idx]  # [b, 1+K, D], unit-norm per branch
            b_vis = cached_visibility[b_idx]     # [b, 1+K]

            optimizer.zero_grad()

            prompts = prompt_learner.build_part_prompts(b_labels)
            branch_texts = torch.stack([
                F.normalize(text_encoder(prompts[m], prompt_learner.tokenized_prompts).float(), p=2, dim=-1)
                for m in range(num_branches)], dim=1)  # [b, 1+K, D], fresh + differentiable

            in_batch = torch.zeros(num_identities, dtype=torch.bool, device=b_labels.device)
            in_batch[b_labels] = True
            other_ids = in_batch.logical_not().nonzero(as_tuple=True)[0]

            loss = b_features.new_zeros(())
            for m in range(num_branches):
                branch_text = branch_texts[:, m, :]
                visual_m = b_features[:, m, :]
                w_m = b_vis[:, m]
                # i2t: this batch's fresh text rows + every other identity's snapshot row
                other_text = torch.cat([branch_text, text_snapshot[other_ids, m, :]], dim=0)
                other_text_labels = torch.cat([b_labels, other_ids], dim=0)
                loss = loss + supcon(visual_m, other_text, b_labels, other_text_labels, w_m)
                # t2i: every cached image is a comparison point
                loss = loss + supcon(branch_text, cached_features[:, m, :], b_labels, cached_labels, w_m)

            # L_part_diag: within-identity part contrast (see module docstring)
            text_parts = branch_texts[:, 1:, :]
            l_part_diag = part_diag(b_features[:, 1:, :], text_parts, b_vis[:, 1:])
            loss = loss + lambda_part_diag * l_part_diag
            part_cos_sum += same_identity_part_cosine(text_parts).item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if (it + 1) % cfg.logging.print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.3f}\tLR {:.2e}\tpart_diag {:.4f} (x{:.2f})'.format(
                    epoch, it + 1, iters_per_epoch, loss.item(), optimizer.param_groups[0]['lr'],
                    l_part_diag.item(), lambda_part_diag))

        scheduler.step()
        # part-ctx cos: mean cosine between DIFFERENT parts' prompts of the SAME identity in the
        # joint space -- the "are the part prompts part-specific" metric (1.0 = collapsed).
        print('Epoch {} done in {:.1f}s, avg loss {:.4f}, SupCon temperature {:.4f}, part-ctx cos {:.3f}'.format(
            epoch, time.time() - epoch_start, epoch_loss / iters_per_epoch, supcon.temperature.item(),
            part_cos_sum / iters_per_epoch))

    torch.save(prompt_learner.state_dict(), osp.join(cfg.logging.logs_dir, 'prompt_learner.pth'))
    print('==> Saved prompt_learner.pth to {}. Run examples/cache_text_anchors.py next to build '
          'text_prototypes.pth for Stage 2.'.format(cfg.logging.logs_dir))

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
