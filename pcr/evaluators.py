from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.part_distance import compute_bpb_pairwise_distance


def extract_part_features(model, inputs):
    """f_out [B, 1+K, D]: the encoder's joint-space branches (0 = CLIP's global x_proj, 1..K =
    mask-pooled part_xproj), vis [B, 1+K]. Nothing else is applied at test time -- no text, no
    relational block; compute_bpb_pairwise_distance uses vis for its per-branch masking."""
    inputs = inputs.cuda()
    f_out, vis = model(inputs)
    return f_out.data.cpu(), vis.data.cpu()


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()  # fname -> [M, D]
    visibilities = OrderedDict()  # fname -> [M]
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            f_out, vis = extract_part_features(model, imgs)
            for fname, emb, v, pid in zip(fnames, f_out, vis, pids):
                features[fname] = emb
                visibilities[fname] = v
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, visibilities, labels


def pairwise_distance(features, visibilities, query=None, gallery=None):
    """Stacks query/gallery part embeddings + visibility and defers to
    compute_bpb_pairwise_distance (pcr/utils/part_distance.py) instead of a raw
    2-2*x@y.T computation, since distances here are part-based, not flat vectors."""
    if query is None and gallery is None:
        fnames = list(features.keys())
        x = torch.stack([features[f] for f in fnames], dim=0)
        xv = torch.stack([visibilities[f] for f in fnames], dim=0)
        return compute_bpb_pairwise_distance(x, xv)

    x = torch.stack([features[f] for f, _, _ in query], dim=0)
    xv = torch.stack([visibilities[f] for f, _, _ in query], dim=0)
    y = torch.stack([features[f] for f, _, _ in gallery], dim=0)
    yv = torch.stack([visibilities[f] for f, _, _ in gallery], dim=0)
    dist_m = compute_bpb_pairwise_distance(x, xv, y, yv)
    return dist_m, x, y


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                  query_ids=None, gallery_ids=None,
                  query_cams=None, gallery_cams=None,
                  cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if not cmc_flag:
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                            single_gallery_shot=False,
                            first_match_break=True),
    }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                             query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k - 1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    """Retrieval on the encoder's own joint-space branches with visibility-aware part-wise
    distance (pcr/utils/part_distance.py). Nothing text-side runs here: retrieval has no identity
    label to fetch a text context with (and the training identity set is disjoint from
    query/gallery anyway) -- everything the text side contributed is baked into the weights."""

    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False):
        features, visibilities, _ = extract_features(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, visibilities, query, gallery)
        return evaluate_all(query_features, gallery_features, distmat,
                             query=query, gallery=gallery, cmc_flag=cmc_flag)
