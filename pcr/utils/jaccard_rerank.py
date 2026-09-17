"""Plain (non-camera-aware, non-FAISS) k-reciprocal Jaccard re-ranking distance.

Ported from SpCL's spcl/utils/faiss_rerank.py (compute_jaccard_distance + k_reciprocal_neigh,
plain path only -- CA-Jaccard branch and FAISS GPU search are not ported). The one deliberate
structural change from the SpCL original: this takes a precomputed [N, N] base distance matrix
directly (built by the caller, e.g. via pcr.utils.part_distance.compute_bpb_pairwise_distance)
instead of computing `2 - 2 * f @ f.T` internally over raw features via FAISS -- this lets the
base distance be either a plain feature distance or a part-based one without touching this file.
The k-reciprocal expansion / V-matrix / Jaccard math below is unchanged from the published
algorithm (Zhong et al., CVPR 2017 k-reciprocal re-ranking, as adapted by SpCL).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(base_dist, k1=30, k2=6, print_flag=True):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    if isinstance(base_dist, np.ndarray):
        base_dist = torch.from_numpy(base_dist)
    base_dist = base_dist.float()

    N = base_dist.size(0)
    mat_type = np.float32

    initial_rank = base_dist.topk(k1, dim=1, largest=False).indices.cpu().numpy()

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index))
                    > 2 / 3 * len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  # element-wise unique
        dist = base_dist[i, k_reciprocal_expansion_index].unsqueeze(0)
        V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex) == N

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time() - end))

    return jaccard_dist
