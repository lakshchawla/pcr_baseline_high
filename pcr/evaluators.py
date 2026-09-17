from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .models.relation_blocks import apply_vab_with_pooling
from .utils.meters import AverageMeter
from .utils.part_distance import compute_bpb_pairwise_distance


def extract_part_features(model, inputs, vab=None, pool=None):
    """f_out is optionally mixed/pooled by VAB+AttentionPoolingBlock before being cached -- both
    need only this image's own real per-branch visibility (no label, no text), so they're fully
    computable at test time, unlike CrossAttentionBlock (CAB, see Evaluator's own docstring for
    why CAB can't run here at all). vis itself is left untouched: it's BPBreID's own visibility
    output, unaffected by VAB/pool's mixing, and compute_bpb_pairwise_distance still needs the
    real per-branch visibility for its masking."""
    inputs = inputs.cuda()
    f_out, vis = model(inputs)
    if vab is not None:
        f_out, _ = apply_vab_with_pooling(vab, pool, f_out, vis, getattr(model, '_has_global', False))
    return f_out.data.cpu(), vis.data.cpu()


def extract_features(model, data_loader, vab=None, pool=None, print_freq=50):
    model.eval()
    if vab is not None:
        vab.eval()
        pool.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()  # fname -> [M, D]
    visibilities = OrderedDict()  # fname -> [M]
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            f_out, vis = extract_part_features(model, imgs, vab, pool)
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
    """vab/pool: optional VisualAttentionBlock + AttentionPoolingBlock, same live instances the
    caller is training (pass both None for Stage 3/UDA and USL, whose encoders never use VAB/
    pool/CAB at all -- their behavior is unchanged either way). When given, retrieval features
    are VAB-mixed and attention-pooled exactly like apply_vab_with_pooling does in training --
    otherwise Stage 2 would train on those features but evaluate on BPBreID's raw, unmixed output.

    CrossAttentionBlock (CAB) is deliberately never applied here, and this is not an oversight:
    CAB's forward call needs `text_prototypes[targets]`, i.e. the batch's ground-truth identity
    label, to fetch its text context. At retrieval time that label is exactly the value being
    inferred -- it isn't known -- and even if it were, `text_prototypes` is sized for the
    training identity set only (751 for Market1501), which is disjoint from query/gallery
    identities under the standard evaluation protocol, so there's no row to index even in
    principle. CAB is a training-time cross-modal grounding regularizer: its effect on retrieval
    is already baked into VAB/pool/the backbone's learned weights via the gradient it shapes
    during training, not something applied again at inference.
    """

    def __init__(self, model, vab=None, pool=None):
        super(Evaluator, self).__init__()
        self.model = model
        self.vab = vab
        self.pool = pool

    def evaluate(self, data_loader, query, gallery, cmc_flag=False):
        features, visibilities, _ = extract_features(self.model, data_loader, self.vab, self.pool)
        distmat, query_features, gallery_features = pairwise_distance(features, visibilities, query, gallery)
        return evaluate_all(query_features, gallery_features, distmat,
                             query=query, gallery=gallery, cmc_flag=cmc_flag)
