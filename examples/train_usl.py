"""ICE-style unsupervised (single-domain) training driver: no source domain, just a target
dataset and per-epoch DBSCAN pseudo-labels (ice/examples/unsupervised_train.py's own flow --
one DBSCAN pass, DBSCAN outliers dropped from that epoch's training set entirely, no SpCL-style
self-paced R_indep/R_comp filtering). Trained with only the hard-instance contrastive loss
(pcr/loss/contrastive.py::PartViewContrastiveLoss) via the EMA teacher-student trainer in
pcr/trainers_usl.py -- no cluster-classification head, no cross-camera loss, no KL consistency
term, and no DSBN (there's no domain split to make BN specific to).
"""
from __future__ import print_function, absolute_import
import argparse
import copy
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import numpy as np
from sklearn.cluster import DBSCAN

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from pcr import datasets
from pcr.models.bpbreid_encoder import BPBReIDEncoder, BPBReIDModelCfg
from pcr.loss import PartViewContrastiveLoss, PartGiLtLoss, BodyPartAttentionLoss
from pcr.trainers_usl import ICEUSLTrainer
from pcr.evaluators import Evaluator, extract_features
from pcr.utils.data import IterLoader
from pcr.utils.data import transforms as T
from pcr.utils.data.sampler import MoreCameraSampler
from pcr.utils.data.preprocessor import Preprocessor, Preprocessor2View, PreprocessorMasked
from pcr.utils.logging import Logger
from pcr.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from pcr.utils.lr_scheduler import WarmupMultiStepLR
from pcr.utils.jaccard_rerank import compute_jaccard_distance
from pcr.utils.part_distance import compute_bpb_pairwise_distance

best_mAP = 0


def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir, name))


def compute_cluster_centers(cf, cf_vis, labels, num_ids):
    """Per-branch, visibility-weighted mean embedding per pseudo-cluster -- same pattern as
    examples/train_uda.py's source-center init, computed fresh every epoch since DBSCAN's
    cluster count/numbering both change epoch to epoch. cf: [N, M, D]. cf_vis: [N, M] bool or
    float. labels: [N] long, values in [0, num_ids) for cluster members (outliers, label -1,
    never match any `cid` below so they're naturally excluded)."""
    centers = []
    for cid in range(num_ids):
        member_idx = labels == cid
        feats = cf[member_idx]
        vis = cf_vis[member_idx].float()
        weights = vis.unsqueeze(-1)
        weight_sum = weights.sum(0)
        weighted_mean = (feats * weights).sum(0) / weight_sum.clamp_min(1e-6)
        has_visible = (weight_sum.squeeze(-1) > 0).unsqueeze(-1)
        centers.append(torch.where(has_visible, weighted_mean, feats.mean(0)))
    return F.normalize(torch.stack(centers, 0), dim=-1)


def get_view2_transform(height, width):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.GaussianBlur((.1, 2.))], p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.6, mean=[0.485, 0.456, 0.406]),
    ])


def get_photometric_transform():
    """View-1 transform used when masks are enabled: PreprocessorMasked already applies the
    geometric steps (resize/pad/crop/flip) jointly to image+mask, so this only covers the
    colour-only augmentation + tensor conversion that comes after."""
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.RandomApply([T.GaussianBlur((.1, 2.))], p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.6, mean=[0.485, 0.456, 0.406]),
    ])


def get_train_loader(dataset, height, width, batch_size, workers, num_instances, iters, trainset,
                      masks_dir=None, masks_suffix='.npy'):
    view2_transform = get_view2_transform(height, width)
    train_set = sorted(trainset)
    sampler = MoreCameraSampler(train_set, num_instances)

    if masks_dir:
        dataset_wrapper = PreprocessorMasked(
            train_set, masks_root=dataset.dataset_dir, masks_dir=masks_dir, height=height, width=width,
            photometric_transform=get_photometric_transform(), view2_transform=view2_transform,
            root=dataset.images_dir, mask_suffix=masks_suffix)
    else:
        dataset_wrapper = Preprocessor2View(train_set, root=dataset.images_dir, transform=view2_transform)

    return IterLoader(
        DataLoader(dataset_wrapper, batch_size=batch_size, num_workers=workers, sampler=sampler,
                   pin_memory=True, drop_last=True), length=iters)


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([T.Resize((height, width), interpolation=3), T.ToTensor(), normalizer])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    return DataLoader(Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
                       batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)


def create_model(args):
    model_cfg = BPBReIDModelCfg(backbone=args.backbone)
    model = BPBReIDEncoder(model_cfg, checkpoint_path=args.checkpoint_path)
    # deepcopy, not a second independent construction: an EMA teacher must start byte-identical
    # to the student (MoCo/BYOL/ICE all require this) -- two separate BPBReIDEncoder(...) calls
    # each do their own random init for every non-backbone layer (dim-reduce, pixel classifier,
    # pooling heads), which silently broke this. Confirmed empirically before fixing: 15 of 409
    # parameter tensors differed between two such calls, including pixel_classifier.weight --
    # the part-attention mechanism itself.
    model_ema = copy.deepcopy(model)
    model.cuda()
    model_ema.cuda()
    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)
    for param in model_ema.parameters():
        param.requires_grad_(False)
    return model, model_ema


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)


def main_worker(args):
    global best_mAP
    start_time = time.monotonic()
    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    iters = args.iters if args.iters > 0 else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.test_batch_size, args.workers)

    model, model_ema = create_model(args)

    params = [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}
              for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1,
                                      warmup_factor=0.1, warmup_iters=args.warmup_step)

    use_gilt = args.gilt_id_weight > 0 or args.gilt_triplet_weight > 0
    bpa_criterion = BodyPartAttentionLoss().cuda() if args.masks_dir else None

    evaluator = Evaluator(model_ema)
    vcl_criterion = PartViewContrastiveLoss(num_instance=args.num_instances, T=args.tau_v)
    trainer = ICEUSLTrainer(model, model_ema, vcl_criterion, alpha=args.alpha,
                             bpa_criterion=bpa_criterion, bpa_weight=args.bpa_weight)

    for epoch in range(args.epochs):
        cluster_loader = get_test_loader(dataset_target, args.height, args.width,
                                          args.test_batch_size, args.workers, testset=sorted(dataset_target.train))
        features, visibilities, _ = extract_features(model_ema, cluster_loader, print_freq=50)
        train_set = sorted(dataset_target.train)
        cf = torch.stack([features[f] for f, _, _ in train_set]).cuda()
        cf_vis = torch.stack([visibilities[f] for f, _, _ in train_set]).cuda()

        print('==> Create pseudo labels for the unlabeled target domain')
        base_dist = compute_bpb_pairwise_distance(cf, cf_vis)
        rerank_dist = compute_jaccard_distance(base_dist, k1=args.k1, k2=args.k2)
        del base_dist

        cluster = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='precomputed', n_jobs=-1)
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)

        gilt_criterion, centers = None, None
        if use_gilt and num_ids > 0:
            centers = compute_cluster_centers(cf, cf_vis, torch.from_numpy(labels).cuda(), num_ids)
            gilt_criterion = PartGiLtLoss(num_clusters=num_ids, tau_c=args.gilt_tau_c,
                                           triplet_margin=args.gilt_triplet_margin,
                                           id_weight=args.gilt_id_weight,
                                           triplet_weight=args.gilt_triplet_weight).cuda()
        del cf, cf_vis

        pseudo_labeled_dataset = [(fname, int(label), cid) for (fname, _, cid), label
                                   in zip(train_set, labels) if label != -1]
        print('Epoch {} has {} labeled samples of {} ids and {} unlabeled samples'
              .format(epoch, len(pseudo_labeled_dataset), num_ids,
                      len(train_set) - len(pseudo_labeled_dataset)))

        # DBSCAN clustering can legitimately collapse to very few clusters (seen in practice on
        # an under-trained encoder) -- MoreCameraSampler yields exactly num_ids*num_instances
        # indices (every surviving cluster has >= min_samples members, all landing in
        # pseudo_labeled_dataset), and with drop_last=True that can be too few to assemble even
        # one batch. IterLoader's own StopIteration retry isn't guarded against a *second*
        # consecutive StopIteration in that case, so this would otherwise crash the whole run.
        # Skip training for this epoch and re-cluster next epoch instead of crashing.
        if num_ids * args.num_instances < args.batch_size:
            print('==> Epoch {}: only {} clusters x {} instances (< batch size {}), skipping '
                  'training this epoch'.format(epoch, num_ids, args.num_instances, args.batch_size))
            continue

        train_loader_target = get_train_loader(dataset_target, args.height, args.width, args.batch_size,
                                                args.workers, args.num_instances, iters, pseudo_labeled_dataset,
                                                masks_dir=args.masks_dir, masks_suffix=args.masks_suffix)
        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_target, optimizer, print_freq=args.print_freq,
                      train_iters=len(train_loader_target), gilt_criterion=gilt_criterion, centers=centers)
        lr_scheduler.step()

        if (epoch + 1) % args.eval_step == 0 or epoch == args.epochs - 1:
            mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'
                  .format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    print('==> Test with the best model:')
    best_fpath = osp.join(args.logs_dir, 'model_best.pth.tar')
    if osp.isfile(best_fpath):
        checkpoint = load_checkpoint(best_fpath)
        copy_state_dict(checkpoint['state_dict'], model_ema)
    else:
        print('No model_best.pth.tar in {}, testing with the final model'.format(args.logs_dir))
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    print('Total running time:', timedelta(seconds=time.monotonic() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PCR: BPBReID under ICE's unsupervised hard-instance contrastive strategy")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501', choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                         help="each minibatch consists of (batch_size // num_instances) identities, "
                              "each with num_instances instances")
    # cluster
    parser.add_argument('--eps', type=float, default=0.55, help="DBSCAN max neighbor distance")
    parser.add_argument('--min-samples', type=int, default=4, help="DBSCAN min samples per cluster")
    parser.add_argument('--k1', type=int, default=30, help="Jaccard re-ranking k1")
    parser.add_argument('--k2', type=int, default=6, help="Jaccard re-ranking k2")
    # model / loss
    parser.add_argument('--checkpoint-path', type=str, default='', metavar='PATH',
                         help="BPBReID checkpoint to initialize both the student and EMA teacher "
                              "from; if omitted, the backbone starts from ImageNet-pretrained "
                              "weights instead of a fully random init")
    parser.add_argument('--backbone', type=str, default='hrnet32', choices=['hrnet32', 'resnet50'],
                         help="resnet50's ImageNet weights auto-download; hrnet32's need a one-time "
                              "manual download (instructions printed if missing) -- see progress.md")
    parser.add_argument('--alpha', type=float, default=0.999, help="EMA momentum for the teacher model")
    parser.add_argument('--tau-v', type=float, default=0.1, help="temperature for the hard-instance contrastive loss")
    # GiLt (id-via-cluster-centers + part-triplet)
    parser.add_argument('--gilt-id-weight', type=float, default=1.0,
                         help="weight for GiLt's id component (CrossEntropyLabelSmooth against "
                              "per-epoch cluster centers, foreground branch only); 0 disables it")
    parser.add_argument('--gilt-triplet-weight', type=float, default=1.0,
                         help="weight for GiLt's triplet component (PartTripletLoss across all "
                              "branches); 0 disables it")
    parser.add_argument('--gilt-tau-c', type=float, default=0.5, help="temperature for the id component")
    parser.add_argument('--gilt-triplet-margin', type=float, default=0.3)
    # body part attention (BPA) loss -- needs PifPaf masks, only usable if the chosen dataset has them
    parser.add_argument('--masks-dir', type=str, default='',
                         help="PifPaf masks subdirectory under <dataset-dir>/masks/ (e.g. "
                              "pifpaf_maskrcnn_filtering); empty (default) disables BPA loss entirely")
    parser.add_argument('--masks-suffix', type=str, default='.npy')
    parser.add_argument('--bpa-weight', type=float, default=0.35, help="matches bpbreid's own yaml default")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[], help="LR decay epochs")
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, '..', 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    main()
