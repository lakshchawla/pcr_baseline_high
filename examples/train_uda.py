from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from pcr import datasets
from pcr.models.bpbreid_encoder import BPBReIDEncoder, BPBReIDModelCfg
from pcr.models.dsbn import convert_dsbn
from pcr.models.hm import PartHybridMemory
from pcr.models.id_classifier import PartIdClassifiers
from pcr.loss import PartTripletLoss, CrossEntropyLabelSmooth, BodyPartAttentionLoss
from pcr.trainers import PCRTrainer_UDA
from pcr.evaluators import Evaluator, extract_features
from pcr.utils.data import IterLoader
from pcr.utils.data import transforms as T
from pcr.utils.data.sampler import RandomMultipleGallerySampler
from pcr.utils.data.preprocessor import Preprocessor, PreprocessorMaskedSingleView
from pcr.utils.logging import Logger
from pcr.utils.serialization import load_checkpoint, save_checkpoint
from pcr.utils.jaccard_rerank import compute_jaccard_distance
from pcr.utils.part_distance import compute_bpb_pairwise_distance


start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_photometric_transform():
    """Colour-only augmentation + tensor conversion -- used for the masks-aware source loader,
    where PreprocessorMaskedSingleView already applies the geometric steps (resize/pad/crop/flip)
    jointly to image and mask; matches examples/train_relational_finetune.py's own
    get_photometric_transform."""
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
    ])


def get_train_loader(args, dataset, height, width, batch_size, workers,
                      num_instances, iters, trainset=None, masks_dir=None, masks_suffix='.npy'):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    sampler = RandomMultipleGallerySampler(train_set, num_instances) if rmgs_flag else None

    if masks_dir:
        dataset_wrapper = PreprocessorMaskedSingleView(
            train_set, masks_root=dataset.dataset_dir, masks_dir=masks_dir, height=height,
            width=width, photometric_transform=get_photometric_transform(),
            root=dataset.images_dir, mask_suffix=masks_suffix)
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            normalizer,
            T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
        dataset_wrapper = Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer)

    train_loader = IterLoader(
        DataLoader(dataset_wrapper, batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model_cfg = BPBReIDModelCfg(backbone=args.backbone)
    model = BPBReIDEncoder(model_cfg, checkpoint_path=args.checkpoint_path)
    # adopt domain-specific BN
    convert_dsbn(model)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if args.iters > 0 else None
    print("==> Load source-domain dataset")
    dataset_source = get_data(args.dataset_source, args.data_dir)
    print("==> Load target-domain dataset")
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            masks_dir=args.masks_dir or None, masks_suffix=args.masks_suffix)
    source_classes = dataset_source.num_train_pids

    # Create model
    model = create_model(args)

    # Create part hybrid memory
    memory = PartHybridMemory(model.module.num_features, model.module.num_parts,
                               source_classes + len(dataset_target.train),
                               temp=args.temp, momentum=args.momentum).cuda()

    # Initialize source-domain class centroids
    print("==> Initialize source-domain class centroids in the part hybrid memory")
    sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                           args.batch_size, args.workers, testset=sorted(dataset_source.train))
    source_features, source_vis, _ = extract_features(model, sour_cluster_loader, print_freq=50)
    sour_fea_dict = collections.defaultdict(list)
    sour_vis_dict = collections.defaultdict(list)
    for f, pid, _ in sorted(dataset_source.train):
        sour_fea_dict[pid].append(source_features[f])
        sour_vis_dict[pid].append(source_vis[f])

    source_centers = []
    for pid in sorted(sour_fea_dict.keys()):
        feats = torch.stack(sour_fea_dict[pid], dim=0)  # [n_i, M, D]
        vis = torch.stack(sour_vis_dict[pid], dim=0).float()  # [n_i, M]
        weights = vis.unsqueeze(-1)  # [n_i, M, 1]
        weight_sum = weights.sum(0)  # [M, 1]
        weighted_mean = (feats * weights).sum(0) / weight_sum.clamp_min(1e-6)  # [M, D]
        # branches with zero visible members for this pid fall back to a plain unweighted
        # mean over the pid's members, to avoid a div-by-zero / all-invisible center
        has_visible = (weight_sum.squeeze(-1) > 0).unsqueeze(-1)  # [M, 1]
        center = torch.where(has_visible, weighted_mean, feats.mean(0))
        source_centers.append(center)
    source_centers = torch.stack(source_centers, 0)  # [num_pids, M, D], sorted PIDs
    source_centers = F.normalize(source_centers, dim=-1)

    # Initialize target-domain instance features
    print("==> Initialize target-domain instance features in the part hybrid memory")
    tgt_cluster_loader = get_test_loader(dataset_target, args.height, args.width,
                                          args.batch_size, args.workers, testset=sorted(dataset_target.train))
    target_features, _, _ = extract_features(model, tgt_cluster_loader, print_freq=50)
    # already per-branch L2-normalized by BPBReIDEncoder.forward
    target_features = torch.stack([target_features[f] for f, _, _ in sorted(dataset_target.train)], 0)
    memory.features = torch.cat((source_centers, target_features), dim=0).cuda()
    del tgt_cluster_loader, source_centers, target_features, sour_cluster_loader, sour_fea_dict, sour_vis_dict

    # Evaluator
    evaluator = Evaluator(model)

    # Optional GiLt (id + triplet) and BPA loss terms -- see pcr/trainers.py::PCRTrainer_UDA's
    # own docstring for why id loss is source-only (real, static labels) while triplet loss
    # applies to both source and target, and why BPA loss forces a different (single-GPU-only)
    # forward path. use_gilt mirrors examples/train_usl.py's exact convention (either weight
    # being positive turns the whole GiLt term on).
    use_gilt = args.gilt_id_weight > 0 or args.gilt_triplet_weight > 0
    id_classifiers = (PartIdClassifiers(source_classes, model.module.num_features, branches=(0,)).cuda()
                       if use_gilt else None)
    id_criterion = CrossEntropyLabelSmooth(source_classes).cuda() if use_gilt else None
    triplet_criterion = PartTripletLoss(margin=args.gilt_triplet_margin).cuda() if use_gilt else None
    bpa_criterion = BodyPartAttentionLoss().cuda() if args.masks_dir else None

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    if id_classifiers is not None:
        params += [{"params": [value]} for _, value in id_classifiers.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    indep_thres = None

    # Trainer
    trainer = PCRTrainer_UDA(model, memory, source_classes, id_classifiers=id_classifiers,
                              id_criterion=id_criterion, triplet_criterion=triplet_criterion,
                              id_weight=args.gilt_id_weight, triplet_weight=args.gilt_triplet_weight,
                              bpa_criterion=bpa_criterion, bpa_weight=args.bpa_weight)

    if args.setup_only:
        print('==> Setup complete: encoder ({} features, {} branches), memory {}x{}x{}, GiLt {}, '
              'BPA {}, optimizer/scheduler/trainer ready. Exiting before training loop '
              '(--setup-only).'.format(
                  model.module.num_features, model.module.num_parts,
                  memory.features.size(0), memory.features.size(1), memory.features.size(2),
                  'on' if use_gilt else 'off',
                  'on ({})'.format(args.masks_dir) if args.masks_dir else 'off'))
        return

    for epoch in range(start_epoch, args.epochs):
        # Calculate distance
        print('==> Create pseudo labels for unlabeled target domain with self-paced policy')
        target_features = memory.features[source_classes:].clone()
        # The memory bank has no per-slot visibility bookkeeping (a slot is a running
        # momentum average across many past sightings, so "is this slot visible" isn't a
        # single well-defined value) -- simplest correct choice: treat every branch of every
        # memory slot as visible for the purpose of the epoch-level base distance, mirroring
        # how the memory itself has no visibility bookkeeping. This avoids an extra
        # full-dataset extraction pass every epoch.
        target_vis = torch.ones(target_features.size(0), target_features.size(1),
                                 dtype=torch.bool, device=target_features.device)
        base_dist = compute_bpb_pairwise_distance(target_features, target_vis)
        rerank_dist = compute_jaccard_distance(base_dist, k1=args.k1, k2=args.k2)
        del target_features, target_vis, base_dist

        if epoch == start_epoch:
            # DBSCAN cluster
            eps = args.eps
            eps_tight = eps - args.eps_gap
            eps_loose = eps + args.eps_gap
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'
                  .format(eps, eps_tight, eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)

        # select & cluster images as training set of this epoch
        pseudo_labels = cluster.fit_predict(rerank_dist)
        pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
        pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
        num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)

        # generate new dataset and calculate cluster centers
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset_target.train), cluster_id)):
                if id != -1:
                    labels.append(source_classes + id)
                else:
                    labels.append(source_classes + num + outliers)
                    outliers += 1
            return torch.Tensor(labels).long()

        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
        pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
        pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)

        # compute R_indep and R_comp
        N = pseudo_labels.size(0)
        label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
        label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

        R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1)
        R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1)
        assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
        assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

        cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
        cluster_img_num = collections.defaultdict(int)
        for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
            cluster_R_comp[label.item() - source_classes].append(comp.item())
            cluster_R_indep[label.item() - source_classes].append(indep.item())
            cluster_img_num[label.item() - source_classes] += 1

        cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
        cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys()))
                                  if cluster_img_num[num] > 1]
        if indep_thres is None:
            indep_thres = np.sort(cluster_R_indep_noins)[
                min(len(cluster_R_indep_noins) - 1, np.round(len(cluster_R_indep_noins) * 0.9).astype('int'))]

        pseudo_labeled_dataset = []
        outliers = 0
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), pseudo_labels)):
            indep_score = cluster_R_indep[label.item() - source_classes]
            comp_score = R_comp[i]
            if (indep_score <= indep_thres) and (comp_score.item() <= cluster_R_comp[label.item() - source_classes]):
                pseudo_labeled_dataset.append((fname, label.item(), cid))
            else:
                pseudo_labeled_dataset.append((fname, source_classes + len(cluster_R_indep) + outliers, cid))
                pseudo_labels[i] = source_classes + len(cluster_R_indep) + outliers
                outliers += 1

        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()] += 1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
              .format(epoch, (index2label > 1).sum(), (index2label == 1).sum(), 1 - indep_thres))

        memory.labels = torch.cat((torch.arange(source_classes), pseudo_labels)).cuda()
        train_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                                                args.batch_size, args.workers, args.num_instances, iters,
                                                trainset=pseudo_labeled_dataset)

        train_loader_source.new_epoch()
        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_source, train_loader_target, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_target))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'indep_thres': indep_thres,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model on the target domain:')
    best_fpath = osp.join(args.logs_dir, 'model_best.pth.tar')
    if osp.isfile(best_fpath):
        checkpoint = load_checkpoint(best_fpath)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('No model_best.pth.tar in {}, testing with the final model'.format(args.logs_dir))
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PCR: BPBReID under SpCL's self-paced UDA strategy")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc-reid',
                         choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                         choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                         help="each minibatch consist of "
                              "(batch_size // num_instances) identities, and "
                              "each identity has num_instances instances")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6, help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                         help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30, help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6, help="hyperparameter for jaccard distance")
    # model / memory
    parser.add_argument('--momentum', type=float, default=0.2,
                         help="update momentum for the hybrid memory")
    parser.add_argument('--temp', type=float, default=0.05,
                         help="temperature for scaling the contrastive loss")
    parser.add_argument('--checkpoint-path', type=str, required=True, metavar='PATH',
                         help="BPBReID checkpoint (pretrained on the source domain via "
                              "bpbreid's own torchreid/scripts/main.py) to load into the encoder")
    parser.add_argument('--backbone', type=str, default='hrnet32', choices=['hrnet32', 'resnet50'],
                         help="must match whatever backbone --checkpoint-path was pretrained with")
    # GiLt (id + triplet) and BPA loss terms -- same flags/defaults as examples/train_usl.py,
    # for consistency; see pcr/trainers.py::PCRTrainer_UDA's docstring for source/target scoping
    parser.add_argument('--gilt-id-weight', type=float, default=1.0,
                         help="source-only id loss weight (0 disables); either this or "
                              "--gilt-triplet-weight being > 0 turns GiLt on")
    parser.add_argument('--gilt-triplet-weight', type=float, default=1.0,
                         help="source+target part-triplet loss weight (0 disables)")
    parser.add_argument('--gilt-triplet-margin', type=float, default=0.3)
    parser.add_argument('--masks-dir', type=str, default='',
                         help="e.g. 'pifpaf_maskrcnn_filtering' -- enables BPA loss on the "
                              "source domain only (masks are source-only on disk); empty = off")
    parser.add_argument('--masks-suffix', type=str, default='.npy')
    parser.add_argument('--bpa-weight', type=float, default=0.35,
                         help="matches bpbreid's own yaml default; only used when --masks-dir is set")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--setup-only', action='store_true',
                         help="initialize datasets, model, memory, optimizer and trainer, "
                              "then exit before the training loop")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                         default=osp.join(working_dir, '..', 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                         default=osp.join(working_dir, 'logs'))
    main()
