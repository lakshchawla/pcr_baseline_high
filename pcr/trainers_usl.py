"""ICE-style USL trainer. Base signal is still PartViewContrastiveLoss (ice/trainers.py::
ImageTrainer's "Loss_hard_instance", ICE's cluster-classification/cross-camera/KL terms are
still dropped -- see pcr/loss/contrastive.py). This adds two optional, independently-weighted
terms on top: PartGiLtLoss (id-via-cluster-centers + part-triplet, pcr/loss/gilt_loss.py) and
BodyPartAttentionLoss (mask-supervised attention, pcr/loss/body_part_attention_loss.py) -- both
off by default (constructor args `None`), matching this repo's "weight=0 / criterion=None means
the term doesn't run" convention. Still single target domain, no DSBN.
"""
from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.meters import AverageMeter


class ICEUSLTrainer(object):
    def __init__(self, encoder, encoder_ema, vcl_criterion, alpha=0.999,
                 bpa_criterion=None, bpa_weight=1.0):
        super(ICEUSLTrainer, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.vcl_criterion = vcl_criterion
        self.alpha = alpha
        self.bpa_criterion = bpa_criterion
        self.bpa_weight = bpa_weight
        self.use_masks = bpa_criterion is not None

    def train(self, epoch, data_loader_target, optimizer, print_freq=10, train_iters=400,
              gilt_criterion=None, centers=None):
        self.encoder.train()
        self.encoder_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_vcl = AverageMeter()
        losses_gilt_id = AverageMeter()
        losses_gilt_tri = AverageMeter()
        losses_bpa = AverageMeter()
        vis_rate = AverageMeter()  # per-branch mean visibility -- cheap diagnostic for
                                    # part-attention collapse (learnable_attention_enabled=True
                                    # means the pixel classifier is trained from indirect gradient
                                    # alone whenever BPA loss isn't active/dominant; a branch whose
                                    # visibility rate collapses toward 0 or 1 across an epoch is
                                    # attending to nothing or everything, not a distinct part)

        student = self.encoder.module if isinstance(self.encoder, nn.DataParallel) else self.encoder

        end = time.time()
        for i in range(train_iters):
            inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            if self.use_masks:
                img1, img2, mask, targets = self._parse_data_masked(inputs)
            else:
                img1, img2, targets = self._parse_data(inputs)
            b = img1.size(0)
            shuffle_ids, reverse_ids = self._get_shuffle_ids(b)

            if self.use_masks:
                # bypasses nn.DataParallel's scatter/gather -- see forward_full's own docstring
                q, q_vis, pixels_cls_scores = student.forward_full(img1)
            else:
                q, q_vis = self.encoder(img1)

            with torch.no_grad():
                k, k_vis = self.encoder_ema(img2[shuffle_ids])
                k, k_vis = k[reverse_ids], k_vis[reverse_ids]

            loss_vcl = self.vcl_criterion(q, q_vis, k, k_vis, targets)
            loss = loss_vcl
            losses_vcl.update(loss_vcl.item())
            vis_rate.update(q_vis.float().mean(dim=0).detach().cpu())

            if gilt_criterion is not None:
                loss_gilt, id_loss_val, triplet_loss_val = gilt_criterion(q, q_vis, centers, targets)
                loss = loss + loss_gilt
                if id_loss_val is not None:
                    losses_gilt_id.update(id_loss_val.item())
                if triplet_loss_val is not None:
                    losses_gilt_tri.update(triplet_loss_val.item())

            if self.use_masks:
                mask_targets = self._mask_targets(mask, pixels_cls_scores)
                loss_bpa, _ = self.bpa_criterion(pixels_cls_scores, mask_targets)
                loss = loss + self.bpa_weight * loss_bpa
                losses_bpa.update(loss_bpa.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema(epoch * train_iters + i)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_hard_instance {:.3f} ({:.3f})\t'
                      'Loss_gilt_id {:.3f} ({:.3f})\t'
                      'Loss_gilt_tri {:.3f} ({:.3f})\t'
                      'Loss_bpa {:.3f} ({:.3f})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_vcl.val, losses_vcl.avg,
                              losses_gilt_id.val, losses_gilt_id.avg,
                              losses_gilt_tri.val, losses_gilt_tri.avg,
                              losses_bpa.val, losses_bpa.avg))

        print('Epoch {} mean part-visibility rate per branch (branch 0 = foreground): {}'.format(
            epoch, ['{:.2f}'.format(v) for v in vis_rate.avg.tolist()]))

    @staticmethod
    def _mask_targets(mask, pixels_cls_scores):
        """mask: [B, 1+parts_num, H, W] (soft, sums to 1 per pixel). Resized to
        pixels_cls_scores' spatial size and argmax'd into an integer target per pixel --
        exactly bpbreid's own part_based_engine.py::combine_losses does it."""
        mask = F.interpolate(mask, size=pixels_cls_scores.shape[2:], mode='bilinear', align_corners=True)
        return mask.argmax(dim=1)

    def _update_ema(self, global_step):
        for ema_param, param in zip(self.encoder_ema.parameters(), self.encoder.parameters()):
            ema_param.data.mul_(self.alpha).add_(param.data, alpha=1 - self.alpha)

    def _parse_data(self, inputs):
        img1, img2, pids, _ = inputs
        return img1.cuda(), img2.cuda(), pids.cuda()

    def _parse_data_masked(self, inputs):
        img1, img2, mask, pids, _ = inputs
        return img1.cuda(), img2.cuda(), mask.cuda(), pids.cuda()

    @staticmethod
    def _get_shuffle_ids(bsz):
        """ShuffleBN (MoCo): decorrelates BatchNorm statistics between the student's forward
        pass and the EMA teacher's, so BN can't leak batch-level shortcuts between the two
        views being contrasted."""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds
