from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F

from .evaluation_metrics import cmc, mean_ap
from .models.relation_blocks import apply_vab_with_pooling
from .utils.meters import AverageMeter
from .utils.part_distance import compute_bpb_pairwise_distance


def extract_part_features(model, inputs, vab=None, pool=None, bn_x4=None):
    """f_out is optionally mixed/pooled by VAB+AttentionPoolingBlock before being cached -- both
    need only this image's own real per-branch visibility (no label, no text), so they're fully
    computable at test time, unlike CrossAttentionBlock (CAB, see Evaluator's own docstring for
    why CAB can't run here at all). vis itself is left untouched: it's BPBreID's own visibility
    output, unaffected by VAB/pool's mixing, and compute_bpb_pairwise_distance still needs the
    real per-branch visibility for its masking.

    Also returns x4 [B, vision_width]: CLIP-ReID's own `img_feature` (layer4 average-pooled),
    passed through the Stage 2 BNNeck `bn_x4` when given (CLIP-ReID tests on the post-BN
    feature, neck_feat='after') and L2-normalized (its evaluator's feat_norm='yes'). None when
    the encoder doesn't expose forward_multi (Stage 3's BPBreID encoders)."""
    inputs = inputs.cuda()
    if hasattr(model, 'forward_multi'):
        f_out, vis, _, x4 = model.forward_multi(inputs)
        if bn_x4 is not None:
            x4 = bn_x4(x4)
        x4 = F.normalize(x4, p=2, dim=-1).data.cpu()
    else:
        f_out, vis = model(inputs)
        x4 = None
    if vab is not None:
        f_out, _ = apply_vab_with_pooling(vab, pool, f_out, vis, getattr(model, '_has_global', False))
    return f_out.data.cpu(), vis.data.cpu(), x4


def extract_features(model, data_loader, vab=None, pool=None, bn_x4=None, print_freq=50):
    """Returns (features, visibilities, labels, x4_features): the first three as before (fname ->
    [M, D] / [M] / pid); x4_features is fname -> [vision_width] (see extract_part_features), or an
    empty dict for encoders without forward_multi."""
    model.eval()
    if vab is not None:
        vab.eval()
        pool.eval()
    if bn_x4 is not None:
        bn_x4.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()  # fname -> [M, D]
    visibilities = OrderedDict()  # fname -> [M]
    labels = OrderedDict()
    x4_features = OrderedDict()  # fname -> [vision_width]

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            f_out, vis, x4 = extract_part_features(model, imgs, vab, pool, bn_x4)
            for j, (fname, emb, v, pid) in enumerate(zip(fnames, f_out, vis, pids)):
                features[fname] = emb
                visibilities[fname] = v
                labels[fname] = pid
                if x4 is not None:
                    x4_features[fname] = x4[j]

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, visibilities, labels, x4_features


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


def x4_pairwise_distance(x4_features, query, gallery):
    """Plain euclidean distance between the (already BN'd + unit-normalized) x4 features --
    CLIP-ReID's own test-time metric on its `img_feature`. [Nq, Ng]."""
    x = torch.stack([x4_features[f] for f, _, _ in query], dim=0)
    y = torch.stack([x4_features[f] for f, _, _ in gallery], dim=0)
    return torch.cdist(x, y)


def fuse_distances(dist_parts, dist_x4, num_branches, x4_weight=1.0):
    """Treats x4 as one more branch alongside the M joint-space ones: dist_parts is already the
    mean over the (visible) M branches of per-branch euclidean distances on unit vectors, and
    dist_x4 is the same kind of distance on one more unit vector, so
    (M * dist_parts + w * dist_x4) / (M + w) is the visibility-agnostic "M+1 branch mean" (exact
    when every branch is visible; with occlusions the part term is a mean over fewer branches,
    which this keeps rather than re-weighting -- a deliberate, simple fusion, not a tuned one)."""
    return (num_branches * dist_parts + x4_weight * dist_x4) / (num_branches + x4_weight)


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

    def __init__(self, model, vab=None, pool=None, bn_x4=None, x4_weight=1.0):
        """bn_x4: the Stage 2 BNNeck on the x4 feature (live instance), or None to evaluate on
        the raw x4. x4_weight: weight of the x4 distance relative to one joint-space branch in
        fuse_distances (0 disables the x4 term entirely)."""
        super(Evaluator, self).__init__()
        self.model = model
        self.vab = vab
        self.pool = pool
        self.bn_x4 = bn_x4
        self.x4_weight = x4_weight

    def evaluate(self, data_loader, query, gallery, cmc_flag=False):
        """Prints mAP for the joint-space branches alone, x4 alone, and the fusion; returns the
        fusion's result (the descriptor that is the model's actual output) -- the two components
        are logged so a run's log shows which side is carrying the retrieval."""
        features, visibilities, _, x4_features = extract_features(self.model, data_loader, self.vab,
                                                                  self.pool, self.bn_x4)
        distmat, query_features, gallery_features = pairwise_distance(features, visibilities, query, gallery)
        if not x4_features or self.x4_weight <= 0:
            return evaluate_all(query_features, gallery_features, distmat,
                                 query=query, gallery=gallery, cmc_flag=cmc_flag)

        dist_x4 = x4_pairwise_distance(x4_features, query, gallery)
        num_branches = query_features.size(1)
        fused = fuse_distances(distmat, dist_x4, num_branches, self.x4_weight)
        print('  [joint-space branches only]', end=' ')
        evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=False)
        print('  [x4 only]', end=' ')
        evaluate_all(query_features, gallery_features, dist_x4, query=query, gallery=gallery, cmc_flag=False)
        print('  [fused: branches + x4]', end=' ')
        return evaluate_all(query_features, gallery_features, fused,
                             query=query, gallery=gallery, cmc_flag=cmc_flag)
