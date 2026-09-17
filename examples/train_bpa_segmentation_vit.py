"""Stage 0, ViT-backbone variant: trains only PixelToPartClassifier (BN + 1x1 conv, bpbreid's own
class) on top of a frozen CLIP ViT (pcr/models/clip_vit_bpam_encoder.py::ClipViTBPAMEncoder),
against the same real Market1501 masks train_bpa_segmentation.py uses. Same role as that script
(anchor branch index to a real anatomical region before Stage 1/2 build on top of it), same loss
(BodyPartAttentionLoss), same mask targets -- only the backbone changes, and unlike the HRNet/
ResNet version, that backbone never trains: CLIP's pretrained weights are the reason Stage 1's
text-aligned joint space is worth having, so nothing here is allowed to drift them. Only the small
classifier head has an optimizer.

examples/train_bpa_segmentation.py and train_bpa_segmentation_rn50.py are untouched; this is a
parallel script for the ViT path.
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
from pcr.models.clip_vit_bpam_encoder import ClipViTBPAMEncoder
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
    normalizer = T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))
    return T.Compose([
        T.RandomApply([T.GaussianBlur((.1, 2.))], p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=list(CLIP_MEAN)),
    ])


def get_train_loader(dataset, cfg):
    dataset_wrapper = PreprocessorMaskedSingleView(
        dataset.train, masks_root=dataset.dataset_dir, masks_dir=cfg.data.masks_dir,
        height=cfg.data.height, width=cfg.data.width,
        photometric_transform=get_photometric_transform(),
        root=dataset.images_dir, mask_suffix=cfg.data.masks_suffix)
    return DataLoader(dataset_wrapper, batch_size=cfg.data.batch_size, num_workers=cfg.data.workers,
                       shuffle=True, pin_memory=True, drop_last=True)


def build_encoder(cfg):
    encoder = ClipViTBPAMEncoder(
        clip_arch=cfg.model.clip_arch, height=cfg.data.height, width=cfg.data.width,
        num_parts=cfg.model.parts_num, checkpoint_path=cfg.model.checkpoint_path or None,
        device='cuda').cuda()
    # Backbone is frozen inside ClipViTDenseBackbone.__init__ (requires_grad_(False)) and has no
    # BatchNorm/Dropout of its own (CLIP's ResidualAttentionBlock builds nn.MultiheadAttention
    # with its default dropout=0.0), so calling .train() on the whole module only puts the
    # classifier's own BatchNorm2d into train mode -- nothing to special-case here (unlike the
    # RN50 variant, whose frozen backbone has real BatchNorm and needs an extra .backbone.eval()
    # call -- see train_bpa_segmentation_rn50.py's own comment on this).
    encoder.train()
    return encoder


def mask_to_pixel_targets(mask, pixels_cls_scores):
    """mask: [B, 1+parts_num, H, W] (soft, sums to 1 per pixel). Resized to pixels_cls_scores'
    spatial size and argmax'd into an integer target per pixel -- matches
    train_bpa_segmentation.py's own identically-named helper."""
    mask = F.interpolate(mask, size=pixels_cls_scores.shape[2:], mode='bilinear', align_corners=True)
    return mask.argmax(dim=1)


def estimate_class_weights(train_loader, num_classes, grid_size, num_batches=50):
    """Background dominates these masks (measured: ~51-78% of pixels depending on grid size, vs.
    1-15% per part) -- plain unweighted CE converges toward mostly predicting background
    everywhere (verified directly, once already for the RN50 variant: trained accuracy landed
    *below* the "always background" trivial baseline). Estimates each class's true pixel
    frequency from `num_batches` real batches (mask loading only, no model forward pass, so this
    is cheap), then returns inverse-sqrt-frequency weights normalized to sum to num_classes --
    gentler than full inverse frequency (which would weight the rarest class far more and risk
    its own instability)."""
    counts = torch.zeros(num_classes)
    seen = 0
    for mask, in ((b[1],) for b in train_loader):
        fake_scores = torch.zeros(mask.size(0), num_classes, *grid_size)
        targets = mask_to_pixel_targets(mask, fake_scores)
        for k in range(num_classes):
            counts[k] += (targets == k).sum().item()
        seen += 1
        if seen >= num_batches:
            break
    freq = counts / counts.sum()
    weight = 1.0 / freq.sqrt()
    weight = weight * (num_classes / weight.sum())
    return weight, freq


def main():
    parser = argparse.ArgumentParser(description="PCR Stage 0 (ViT variant): BPA segmentation pretraining")
    parser.add_argument('--config', type=str, metavar='PATH', default="configs/stage0_bpa_segmentation_vit.yaml")
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
            "ground-truth part masks.")

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
    train_loader = get_train_loader(dataset, cfg)

    if setup_only:
        print('==> Setup complete: {} training images with masks_dir={}, classifier trainable '
              'params={}. Exiting before the training loop (--setup-only).'.format(
                  len(dataset.train), cfg.data.masks_dir,
                  sum(p.numel() for p in encoder.pixel_classifier.parameters())))
        return

    num_classes = 1 + cfg.model.parts_num
    grid_size = (encoder.backbone.grid_h, encoder.backbone.grid_w)
    class_weight, class_freq = estimate_class_weights(train_loader, num_classes, grid_size)
    trivial_baseline = class_freq.max().item()  # accuracy of always predicting the majority class
    print('==> Ground-truth class frequency [bg, parts 1..{}]: {}'.format(cfg.model.parts_num, class_freq.tolist()))
    print('==> Inverse-sqrt-frequency class weights: {}'.format(class_weight.tolist()))
    print('==> Trivial "always predict majority class" baseline accuracy: {:.4f} -- pixel-acc '
          'below this means the classifier learned nothing beyond the class prior.'.format(trivial_baseline))
    bpa_loss = BodyPartAttentionLoss(weight=class_weight.cuda()).cuda()

    optimizer = torch.optim.Adam(encoder.pixel_classifier.parameters(), lr=cfg.optim.lr,
                                  weight_decay=cfg.optim.weight_decay)
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
        avg_acc = epoch_acc / len(train_loader)
        print('Epoch {} done in {:.1f}s, avg loss {:.4f}, avg pixel-acc {:.4f} ({:+.4f} vs. '
              'trivial baseline {:.4f})'.format(
                  epoch, time.time() - epoch_start, epoch_loss / len(train_loader), avg_acc,
                  avg_acc - trivial_baseline, trivial_baseline))

    save_checkpoint({
        'pixel_classifier': encoder.pixel_classifier.state_dict(),
        'epoch': cfg.optim.epochs,
    }, True, fpath=osp.join(cfg.logging.logs_dir, 'checkpoint.pth.tar'))
    print('==> Saved BPA-pretrained classifier to {}. Pass it as model.checkpoint_path to '
          'ClipViTBPAMEncoder in Stage 1.'.format(osp.join(cfg.logging.logs_dir, 'model_best.pth.tar')))

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
