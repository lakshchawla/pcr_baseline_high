"""BPBReID-style part-based pairwise distance.

Ported from bpbreid's torchreid/metrics/distance.py
(compute_distance_matrix_using_bp_features + _compute_body_parts_dist_matrices) and
torchreid/utils/tensortools.py (masked_mean/replace_values). Dropped: Writer telemetry
hooks (bpbreid-internal debugging, not needed here).

This is THE explicit part-based matching function reused by jaccard_rerank.py (as the base
distance for k-reciprocal re-ranking) and evaluators.py (query-gallery matching at eval time).
"""
import torch
from torch.nn import functional as F

DEFAULT_BATCH_SIZE = 1024


def replace_values(input, mask, value):
    return input * (~mask) + mask * value


def masked_mean(input, mask):
    """output -1 where the mean couldn't be computed (no visible branch in common)."""
    valid_input = input * mask
    mean_weights = mask.sum(0)
    mean_weights = mean_weights + (mean_weights == 0)  # avoid division by 0
    pairwise_dist = valid_input.sum(0) / mean_weights
    invalid_pairs = (mask.sum(dim=0) == 0)
    return replace_values(pairwise_dist, invalid_pairs, -1)


def _compute_body_parts_dist_matrices(qf, gf, metric='euclidean'):
    """qf: [Nq, M, D], gf: [Ng, M, D] -> [M, Nq, Ng] per-branch distance matrices."""
    qf = qf.transpose(1, 0)  # [M, Nq, D]
    gf = gf.transpose(1, 0)  # [M, Ng, D]
    if metric == 'euclidean':
        dot_product = torch.matmul(qf, gf.transpose(2, 1))
        qf_square_sum = qf.pow(2).sum(dim=-1)
        gf_square_sum = gf.pow(2).sum(dim=-1)
        distances = qf_square_sum.unsqueeze(2) - 2 * dot_product + gf_square_sum.unsqueeze(1)
        distances = F.relu(distances)
        distances = torch.sqrt(distances)
    elif metric == 'cosine':
        distances = 1 - torch.matmul(qf, gf.transpose(2, 1))
    else:
        raise ValueError('Unknown distance metric: {}. Use "euclidean" or "cosine"'.format(metric))
    return distances


def combine_part_distances(part_dist, weights, strat='mean', temperature=0.2):
    """Collapses per-part distances into one distance per pair. part_dist: [M, ...] (one distance
    matrix per part, first axis = parts). weights: [M, ...] float >= 0, the pair's per-part
    visibility weight (0 = that part is not mutually visible and drops out exactly). Returns
    [...], with -1 wherever no part has any weight (the caller replaces that sentinel).

    'mean'  -- weighted mean (BPBReID's default). A single strongly different part is diluted by
               the parts that agree: two people in all black with different shoes still come out
               "similar" because 5 of 6 branches agree.
    'lse'   -- weighted log-sum-exp over the parts, i.e. a SOFT MAXIMUM of the part distances --
               equivalently a soft MINIMUM of the part similarities (with d = 1 - cos it is
               exactly 1 - softmin(cos)):
                   D = T * log( sum_k w_k exp(d_k / T) / sum_k w_k )
               Every part's disagreement enters, weighted by exp(d_k / T): one part whose
               similarity collapses drags the whole score with it (the "dynamic penalty"), while
               parts that agree contribute only their share. T -> inf recovers the weighted mean,
               T -> 0 the hard max; equal d_k give D = d_k regardless of T. On unit-norm
               features euclidean d is in [0, 2] and typical between-part gaps are 0.1-0.5, so
               T = 0.2 makes a 0.3 gap count ~4.5x -- max-leaning without being brittle.
    'max'   -- hard maximum over the mutually visible parts.
    """
    if strat == 'mean':
        return masked_mean(part_dist, weights)
    total_w = weights.sum(dim=0)
    invalid = total_w == 0
    if strat == 'max':
        masked = part_dist.masked_fill(weights == 0, float('-inf')).max(dim=0)[0]
        return masked.masked_fill(invalid, -1)
    if strat == 'lse':
        log_w = torch.log(weights.clamp(min=1e-12)).masked_fill(weights == 0, float('-inf'))
        lse = torch.logsumexp(log_w + part_dist / temperature, dim=0)  # -inf where nothing is visible
        combined = temperature * (lse - torch.log(total_w.clamp(min=1e-12)))
        return combined.masked_fill(invalid, -1)
    raise ValueError('Body parts distance combination strategy "{}" not supported'.format(strat))


def _combine_chunk(body_part_dist, qf_vis_t, gf_vis_chunk_t, dist_combine_strat, is_bool, temperature):
    """body_part_dist: [M, Nq, chunk]. qf_vis_t: [M, Nq]. gf_vis_chunk_t: [M, chunk]. Hard
    (bool) visibility gives 0/1 weights; soft visibility the geometric mean sqrt(v_q * v_g)."""
    pair_vis = qf_vis_t.unsqueeze(2) * gf_vis_chunk_t.unsqueeze(1)  # [M, Nq, chunk]
    weights = pair_vis.float() if is_bool else torch.sqrt(pair_vis)
    return combine_part_distances(body_part_dist, weights, dist_combine_strat, temperature)


def compute_bpb_pairwise_distance(qf, qf_vis, gf=None, gf_vis=None, dist_combine_strat='mean',
                                   metric='euclidean', batch_size=DEFAULT_BATCH_SIZE, temperature=0.2):
    """Part-based pairwise distance between two sets of BPBReID embeddings.

    dist_combine_strat / temperature: how the M per-part distances collapse into one -- see
    combine_part_distances ('mean' is BPBReID's default and what Stage 3 still uses; the CLIP
    stages pass 'lse').

    qf, gf: [N, M, D] per-branch (foreground + K parts) embeddings.
    qf_vis, gf_vis: [N, M] visibility, either bool (hard) or float in [0, 1] (soft).
    If gf/gf_vis are omitted, computes the self-distance qf vs qf (used by Jaccard re-ranking).

    Processes the gallery in chunks of `batch_size` so the [M, Nq, Ng] per-branch distance
    tensor is never fully materialized at once -- the target-domain self-distance used by
    Jaccard re-ranking runs over the entire target training set (tens of thousands of images),
    and a dense one-shot [M, N, N] pass OOMs well before that on a consumer GPU (verified: a
    16522x6x512 self-distance needs ~6.5GB for that one intermediate, more than an 8GB card has
    free alongside everything else). The "sentinel = max observed distance + 1" value for
    zero-mutual-visibility pairs is computed as a running max across chunks so it stays a true
    global max, not a per-chunk one. Only the combined [Nq, Ng] result is returned -- unlike
    bpbreid's own two-return-value API, nothing here needs the per-branch distance matrix.

    Returns: Tensor[Nq, Ng] distance matrix.
    """
    if gf is None:
        gf, gf_vis = qf, qf_vis

    is_bool = qf_vis.dtype == torch.bool and gf_vis.dtype == torch.bool
    if not is_bool:
        qf_vis = qf_vis.float()
        gf_vis = gf_vis.float()

    qf_vis_t = qf_vis.t()  # [M, Nq]
    Nq, Ng = qf.size(0), gf.size(0)

    # Written in-place chunk-by-chunk instead of accumulated in a Python list + torch.cat:
    # holding all ~Ng/batch_size chunks alive until a final cat roughly doubles peak memory
    # right when it matters most (this runs concurrently with the resident model/optimizer).
    pairwise_dist = torch.empty(Nq, Ng, device=qf.device, dtype=qf.dtype)
    running_max = torch.zeros((), device=qf.device, dtype=qf.dtype)
    for start in range(0, Ng, batch_size):
        end = min(start + batch_size, Ng)
        gf_chunk = gf[start:end]
        gf_vis_chunk_t = gf_vis[start:end].t()  # [M, chunk]

        body_part_dist = _compute_body_parts_dist_matrices(qf, gf_chunk, metric)  # [M, Nq, chunk]
        running_max = torch.maximum(running_max, body_part_dist.max())

        pairwise_dist[:, start:end] = _combine_chunk(body_part_dist, qf_vis_t, gf_vis_chunk_t,
                                                       dist_combine_strat, is_bool, temperature)
        del body_part_dist

    max_value = running_max + 1
    pairwise_dist.masked_fill_(pairwise_dist == -1, max_value)
    return pairwise_dist
