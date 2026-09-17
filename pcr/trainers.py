from __future__ import print_function, absolute_import
import time

import torch
import torch.nn.functional as F
from torch import nn

from .utils.meters import AverageMeter


class PCRTrainer_UDA(object):
    """Part-generalized SpCLTrainer_UDA (spcl/trainers.py). Same per-iteration structure
    (DSBN device-count reshape trick splitting source/target 1:1 per batch, joint forward
    pass through the shared encoder, de-arrange, contrastive loss against the hybrid
    memory) -- only difference is `_forward` returns (embeddings, visibility) tuples that
    need the same device-reshape/split treatment applied to both tensors.

    Optional GiLt (id + triplet) and BPA loss terms, matching the plan already drafted in
    progress.md's 2026-08-19 15:58 entry and executed for pcr/trainers_usl.py::ICEUSLTrainer at
    the time -- this closes the same gap for the UDA path. Unlike ICEUSLTrainer's GiLt (which
    classifies against per-epoch DBSCAN cluster centers, since USL's pseudo-label count/numbering
    changes every epoch), source's identity labels here are real and static for the whole run, so
    id loss uses a persistent nn.Linear classifier (pcr/models/id_classifier.py::
    PartIdClassifiers, the same module Stage 2's supervised finetune uses) instead -- no per-epoch
    rebuild needed, `id_classifiers`/`id_criterion`/`triplet_criterion` are constructor args, not
    per-`.train()`-call arguments. Id loss is source-only (source has real labels; target's
    pseudo-labels/outlier-singletons are not a fit for a fixed-size classifier head). Triplet loss
    applies to both source (real labels) and target (the actual DBSCAN pseudo-cluster label,
    `t_pseudo_labels` below -- NOT `t_indexes`, which is a per-image PartHybridMemory slot index,
    unique per image and therefore never shared between two different images of the same
    pseudo-identity; using it for triplet mining would make every anchor's positive search fail
    except by the sampler coincidentally drawing the same image twice, a real bug caught during
    this stage's own smoke run by noticing Loss_gilt_tri_t stayed suspiciously exactly 0.000
    every iteration). Both source and target triplet calls are otherwise direct: PartTripletLoss
    takes any integer labels, no cluster-center indirection needed either way.

    BPA loss is source-only (needs ground-truth part masks; masks are confirmed source-only on
    disk, see progress.md) and forces a different forward path: it needs pixels_cls_scores from
    BPBReIDEncoder.forward_full, which -- like ICEUSLTrainer's own masked path -- bypasses
    nn.DataParallel's scatter/gather (documented in forward_full's own docstring), so it only
    correctly parallelizes on a single device. Critically, it must still be fed the *whole* joint
    [source; target] batch in one call (not a source-only sub-batch): DSBN's split
    (pcr/models/dsbn.py) is purely positional -- it always treats the first half of whatever batch
    it's given as source, the second half as target -- so calling forward_full on a source-only
    batch would silently misroute half of those images through DSBN's target BatchNorm branch and
    corrupt its running statistics.

    Because forward_full always runs single-device, the masked path builds its input batch
    directly as `torch.cat((s_inputs, t_inputs), dim=0)` -- *not* via the device_num-way
    reshape/cat trick the unmasked path uses. That trick exists solely so nn.DataParallel's own
    per-device scatter preserves an even source/target split on every device; reusing it here
    (an earlier version of this file did) is actively wrong whenever device_num > 1, since the
    flattened result interleaves as [dev0_source, dev0_target, dev1_source, dev1_target, ...]
    and a flat first-half slice would then mix source and target images instead of isolating
    source. The plain concatenation keeps the source/target boundary at s_inputs.size(0)
    unambiguous regardless of device_num.
    """

    def __init__(self, encoder, memory, source_classes, id_classifiers=None, id_criterion=None,
                 triplet_criterion=None, id_weight=1.0, triplet_weight=1.0,
                 bpa_criterion=None, bpa_weight=1.0):
        super(PCRTrainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes
        self.id_classifiers = id_classifiers
        self.id_criterion = id_criterion
        self.triplet_criterion = triplet_criterion
        self.id_weight = id_weight
        self.triplet_weight = triplet_weight
        self.bpa_criterion = bpa_criterion
        self.bpa_weight = bpa_weight
        self.use_gilt = id_classifiers is not None
        self.use_masks = bpa_criterion is not None

    def train(self, epoch, data_loader_source, data_loader_target,
              optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()
        losses_gilt_id = AverageMeter()
        losses_gilt_tri_s = AverageMeter()
        losses_gilt_tri_t = AverageMeter()
        losses_bpa = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            if self.use_masks:
                s_inputs, s_mask, s_targets, _ = self._parse_data_masked(source_inputs)
            else:
                s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, t_pseudo_labels, t_indexes = self._parse_data(target_inputs)

            if self.use_masks:
                # _forward_full always runs on a single device (bypasses nn.DataParallel's
                # scatter/gather, see _forward_full's docstring), so the device_num-way
                # reshape/cat trick below -- which exists only to lay the batch out so
                # DataParallel's *own* per-device scatter preserves an even source/target split
                # on every device -- is not just unnecessary here, it's actively wrong: with
                # device_num > 1 it would interleave the flattened batch as [dev0_source,
                # dev0_target, dev1_source, dev1_target, ...], and a flat first-half slice
                # would then grab a mix of source and target images, not source-only. A plain
                # [source; target] concatenation keeps the boundary at s_inputs.size(0)
                # unambiguous regardless of device_num.
                B_s = s_inputs.size(0)
                inputs = torch.cat((s_inputs, t_inputs), dim=0)
                f_out, vis, pixels_cls_scores = self._forward_full(inputs)
                f_out_s, f_out_t = f_out[:B_s], f_out[B_s:]
                vis_s, vis_t = vis[:B_s], vis[B_s:]
                pixels_cls_scores_s = pixels_cls_scores[:B_s]
            else:
                # arrange batch for domain-specific BN
                device_num = torch.cuda.device_count()
                B, C, H, W = s_inputs.size()

                def reshape(inputs):
                    return inputs.view(device_num, -1, C, H, W)

                s_inputs_r, t_inputs_r = reshape(s_inputs), reshape(t_inputs)
                inputs = torch.cat((s_inputs_r, t_inputs_r), 1).view(-1, C, H, W)

                f_out, vis = self._forward(inputs)  # f_out: [N, M, D], vis: [N, M]

                # de-arrange batch
                M, D = f_out.size(1), f_out.size(2)
                f_out = f_out.view(device_num, -1, M, D)
                vis = vis.view(device_num, -1, M)
                f_out_s, f_out_t = f_out.split(f_out.size(1) // 2, dim=1)
                vis_s, vis_t = vis.split(vis.size(1) // 2, dim=1)
                f_out_s = f_out_s.contiguous().view(-1, M, D)
                f_out_t = f_out_t.contiguous().view(-1, M, D)
                vis_s = vis_s.contiguous().view(-1, M)
                vis_t = vis_t.contiguous().view(-1, M)

            # compute loss with the part hybrid memory
            loss_s = self.memory(f_out_s, s_targets, vis_s)
            loss_t = self.memory(f_out_t, t_indexes + self.source_classes, vis_t)

            id_loss_val = tri_loss_s_val = tri_loss_t_val = bpa_loss_val = None

            if self.use_gilt:
                id_logits = self.id_classifiers(f_out_s, 0)
                id_loss_val = self.id_criterion(id_logits, s_targets)
                loss_s = loss_s + self.id_weight * id_loss_val

                result_s = self.triplet_criterion(f_out_s, s_targets, parts_visibility=vis_s)
                if result_s is not None:
                    tri_loss_s_val = result_s[0]
                    loss_s = loss_s + self.triplet_weight * tri_loss_s_val

                # t_pseudo_labels (not t_indexes) -- triplet mining needs the actual DBSCAN
                # pseudo-cluster label shared across multiple images of one pseudo-identity;
                # t_indexes is a per-image memory-slot index, unique per image, correct for
                # addressing PartHybridMemory's slots but never shared between two different
                # images, which would make every anchor's "positive" search fail except by
                # coincidence (the sampler drawing the exact same image twice)
                result_t = self.triplet_criterion(f_out_t, t_pseudo_labels, parts_visibility=vis_t)
                if result_t is not None:
                    tri_loss_t_val = result_t[0]
                    loss_t = loss_t + self.triplet_weight * tri_loss_t_val

            if self.use_masks:
                # pixels_cls_scores_s (source's slice) was already computed above, at the
                # point where the batch layout unambiguously puts source first regardless of
                # device_num -- see the forward-path comment
                mask_targets = self._mask_targets(s_mask, pixels_cls_scores_s)
                bpa_loss_val, _ = self.bpa_criterion(pixels_cls_scores_s, mask_targets)
                loss_s = loss_s + self.bpa_weight * bpa_loss_val

            loss = loss_s + loss_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())
            if id_loss_val is not None:
                losses_gilt_id.update(id_loss_val.item())
            if tri_loss_s_val is not None:
                losses_gilt_tri_s.update(tri_loss_s_val.item())
            if tri_loss_t_val is not None:
                losses_gilt_tri_t.update(tri_loss_t_val.item())
            if bpa_loss_val is not None:
                losses_bpa.update(bpa_loss_val.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                extra = ''
                if self.use_gilt:
                    extra += ('\tLoss_gilt_id {:.3f} ({:.3f})\tLoss_gilt_tri_s {:.3f} ({:.3f})'
                              '\tLoss_gilt_tri_t {:.3f} ({:.3f})').format(
                        losses_gilt_id.val, losses_gilt_id.avg,
                        losses_gilt_tri_s.val, losses_gilt_tri_s.avg,
                        losses_gilt_tri_t.val, losses_gilt_tri_t.avg)
                if self.use_masks:
                    extra += '\tLoss_bpa {:.3f} ({:.3f})'.format(losses_bpa.val, losses_bpa.avg)
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_s {:.3f} ({:.3f})\t'
                      'Loss_t {:.3f} ({:.3f}){}'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_s.val, losses_s.avg,
                              losses_t.val, losses_t.avg, extra))

    @staticmethod
    def _mask_targets(mask, pixels_cls_scores):
        """mask: [B, 1+parts_num, H, W] (soft, sums to 1 per pixel). Resized to
        pixels_cls_scores' spatial size and argmax'd into an integer target per pixel -- same
        convention as pcr/trainers_usl.py::ICEUSLTrainer._mask_targets."""
        mask = F.interpolate(mask, size=pixels_cls_scores.shape[2:], mode='bilinear', align_corners=True)
        return mask.argmax(dim=1)

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_masked(self, inputs):
        imgs, mask, pids, _, indexes = inputs
        return imgs.cuda(), mask.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

    def _forward_full(self, inputs):
        # bypasses nn.DataParallel's scatter/gather -- see class docstring and
        # BPBReIDEncoder.forward_full's own docstring
        student = self.encoder.module if isinstance(self.encoder, nn.DataParallel) else self.encoder
        return student.forward_full(inputs)
