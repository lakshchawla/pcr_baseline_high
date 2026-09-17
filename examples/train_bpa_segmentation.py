"""Stage 0: pretrain BPBreID's pixel-to-part classifier (BPA, `pixel_classifier` inside
third_party/torchreid/models/bpbreid.py) as a plain supervised segmentation task, before Stage 1
ever runs -- see progress.md's entry on this change for the full motivation.

Why this exists: without it, BPA's K-way spatial split is shaped only by whatever the downstream
id/triplet/CLIP-alignment gradient happens to reward (Stage 2, once the backbone is unfrozen).
Nothing in that signal ties branch index k to any specific real anatomical region -- and since
Stage 1's per-branch prompts are built from learned placeholder context tokens ("a photo of a
X X X X person"), not real body-part words, CLIP's text encoder never supplies that correspondence
either. The only thing that ever anchors a branch to a real body part is BodyPartAttentionLoss
(pixel-wise cross-entropy against PifPaf/MaskRCNN-derived ground-truth masks) -- already used
inside Stage 2, but only as one of four losses, jointly with id/triplet/align, and off by default.
This stage isolates that one signal and runs it alone, first, so BPA already has a stable,
real-part-anchored spatial split by the time Stage 1's prompts and Stage 2's joint losses build on
top of it.

Scope, matching the same limitation the optional Stage 2 BPA loss already has: this stage is only
usable on datasets with ground-truth part masks on disk (Market1501; DukeMTMC-reID has none, per
progress.md's confirmed on-disk check). No id/triplet/CLIP-alignment loss runs here at all -- pure
segmentation pretraining, nothing else.

Produces a checkpoint of just the BPBreID model's own state dict (matching Stage 2's own save
format exactly), loadable via any of this repo's `model.checkpoint_path`/`--checkpoint-path`
fields -- Stage 1, Stage 2, or even Stage 3 directly.
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
from pcr.models.bpbreid_encoder import BPBReIDEncoder, BPBReIDModelCfg
from pcr.loss import BodyPartAttentionLoss
from pcr.utils.config import load_yaml_config
from pcr.utils.data import transforms as T
from pcr.utils.data.preprocessor import PreprocessorMaskedSingleView
from pcr.utils.logging import Logger
from pcr.utils.osutils import mkdir_if_missing
from pcr.utils.serialization import save_checkpoint


def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir, name))


def get_photometric_transform():
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.RandomApply([T.GaussianBlur((.1, 2.))], p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
    ])


def get_train_loader(dataset, cfg):
    dataset_wrapper = PreprocessorMaskedSingleView(
        dataset.train, masks_root=dataset.dataset_dir, masks_dir=cfg.data.masks_dir,
        height=cfg.data.height, width=cfg.data.width,
        photometric_transform=get_photometric_transform(),
        root=dataset.images_dir, mask_suffix=cfg.data.masks_suffix)
    # Plain shuffle, not PK/RandomIdentitySampler -- no id/triplet loss here, so there's no
    # per-batch identity structure to preserve; every image just needs to be seen eventually.
    return DataLoader(dataset_wrapper, batch_size=cfg.data.batch_size, num_workers=cfg.data.workers,
                       shuffle=True, pin_memory=True, drop_last=True)


def build_encoder(cfg):
    model_cfg = BPBReIDModelCfg(backbone=cfg.model.backbone)
    model_cfg.masks.parts_num = cfg.model.parts_num
    model_cfg.dim_reduce_output = cfg.model.dim_reduce_output
    encoder = BPBReIDEncoder(model_cfg, checkpoint_path=cfg.model.checkpoint_path or None).cuda()
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad_(True)
    return encoder


def mask_to_pixel_targets(mask, pixels_cls_scores):
    """mask: [B, 1+parts_num, H, W] (soft, sums to 1 per pixel). Resized to pixels_cls_scores'
    spatial size and argmax'd into an integer target per pixel -- matches Stage 2's own
    identically-named helper in examples/train_relational_finetune.py."""
    mask = F.interpolate(mask, size=pixels_cls_scores.shape[2:], mode='bilinear', align_corners=True)
    return mask.argmax(dim=1)


def main():
    parser = argparse.ArgumentParser(description="PCR Stage 0: BPA segmentation pretraining")
    parser.add_argument('--config', type=str, metavar='PATH', default="configs/stage0_bpa_segmentation.yaml")
    parser.add_argument('--setup-only', action='store_true',
                         help="build dataset/encoder/loader, print shapes, exit before the "
                              "training loop")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    main_worker(cfg, setup_only=args.setup_only)


def main_worker(cfg, setup_only=False):
    if not cfg.data.masks_dir:
        raise ValueError(
            "Stage 0 requires data.masks_dir -- there is nothing to segment against without "
            "ground-truth part masks (this stage has no id/triplet/align loss to fall back on).")

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
    encoder = build_encoder(cfg)
    bpa_loss = BodyPartAttentionLoss().cuda()
    train_loader = get_train_loader(dataset, cfg)

    if setup_only:
        print('==> Setup complete: {} training images with masks_dir={}. Exiting before the '
              'training loop (--setup-only).'.format(len(dataset.train), cfg.data.masks_dir))
        return

    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.optim.step_size, gamma=0.1)

    for epoch in range(cfg.optim.epochs):
        encoder.train()
        epoch_start = time.time()
        epoch_loss, epoch_acc = 0.0, 0.0
        for it, (imgs, mask, _, _, _) in enumerate(train_loader):
            imgs = imgs.cuda()

            optimizer.zero_grad()
            _, _, pixels_cls_scores = encoder.forward_full(imgs)
            mask_targets = mask_to_pixel_targets(mask.cuda(), pixels_cls_scores)
            loss, acc = bpa_loss(pixels_cls_scores, mask_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

            if (it + 1) % cfg.logging.print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.3f}\tPixel-Acc {:.3f}'.format(
                    epoch, it + 1, len(train_loader), loss.item(), acc))

        lr_scheduler.step()
        print('Epoch {} done in {:.1f}s, avg loss {:.4f}, avg pixel-acc {:.4f}'.format(
            epoch, time.time() - epoch_start, epoch_loss / len(train_loader),
            epoch_acc / len(train_loader)))

    save_checkpoint({
        'state_dict': encoder.model.state_dict(),
        'epoch': cfg.optim.epochs,
    }, True, fpath=osp.join(cfg.logging.logs_dir, 'checkpoint.pth.tar'))
    print('==> Saved BPA-pretrained checkpoint to {}. Pass it as model.checkpoint_path in Stage '
          "1's config (or --checkpoint-path for Stage 2/3) instead of ImageNet init.".format(
              osp.join(cfg.logging.logs_dir, 'model_best.pth.tar')))

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()



"""
SUGGESTED CHANGES:

Train it similar to BPBREID, contrastively using all triplet loss, id loss, etc
and VAB. 
"""