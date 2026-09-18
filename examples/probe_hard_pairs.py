"""The "two people in all black, different shoes" probe, as a number.

Loads a Stage 2 checkpoint (or the frozen Stage 0 encoder when --checkpoint is omitted), extracts
the joint-space branches for query + gallery, and reports:

  1. retrieval mAP/R1 under each part-combination rule (mean / lse at several T / max) and per
     branch alone (global, each part) -- does any single part carry information the others lack;
  2. the HARD-PAIR PROBE: the top-`--hard-quantile` most similar CROSS-identity query/gallery
     pairs under the global branch (the "look identical overall" pairs) vs all same-identity
     pairs -- per-branch mean distance on each set, and the combined distance under each rule.
     The principle being tested: on the hard cross-identity pairs at least one part should sit
     FAR above its same-identity level (the shoes), and the soft-min rule should turn that one
     part into a larger combined gap than the mean does. Gap is reported in units of the
     same-identity spread so runs are comparable.

Usage:
  python examples/probe_hard_pairs.py --config configs/stage2_relational_finetune.yaml \\
         --checkpoint examples/logs/stage2_finetune_clip_rn50/model_best.pth.tar
"""
from __future__ import print_function, absolute_import
import argparse
import contextlib
import io
import sys

import torch

from pcr import datasets
from pcr.evaluators import extract_features, pairwise_distance, evaluate_all
from pcr.utils.part_distance import _compute_body_parts_dist_matrices, combine_part_distances
from pcr.utils.serialization import load_checkpoint

sys.path.insert(0, 'examples')
import train_relational_finetune as s2  # noqa: E402  (loader + encoder builders)


def quiet_eval(feats, vis, query, gallery, combine, temperature=0.2):
    with contextlib.redirect_stdout(io.StringIO()):
        dist, qf, gf = pairwise_distance(feats, vis, query, gallery, combine=combine, temperature=temperature)
        cmc, mAP = evaluate_all(qf, gf, dist, query=query, gallery=gallery, cmc_flag=True)
    return mAP, cmc[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None, help="Stage 2 checkpoint; omit for the frozen Stage 0 encoder")
    parser.add_argument('--hard-quantile', type=float, default=0.99)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.5, 0.2, 0.1])
    args = parser.parse_args()
    cfg = s2.load_yaml_config(args.config)

    dataset = s2.get_data(cfg.data.dataset, cfg.data.data_dir)
    encoder = s2.build_encoder(cfg)
    if args.checkpoint:
        encoder.load_state_dict(load_checkpoint(args.checkpoint)['state_dict'])
        print('==> loaded', args.checkpoint)
    else:
        print('==> no checkpoint: probing the frozen encoder (Stage 0 weights)')
    encoder.eval()
    loader = s2.get_test_loader(dataset, cfg.data.height, cfg.data.width, cfg.data.batch_size, cfg.data.workers)
    feats, vis, _ = extract_features(encoder, loader)
    query, gallery = dataset.query, dataset.gallery
    M = next(iter(feats.values())).size(0)

    print('\n== retrieval by combination rule')
    rows = [('mean', 0.2)] + [('lse', t) for t in args.temperatures] + [('max', 0.2)]
    for combine, t in rows:
        mAP, r1 = quiet_eval(feats, vis, query, gallery, combine, t)
        print('   %-12s mAP %5.2f%%  R1 %5.2f%%' % (combine + (' T=%.2f' % t if combine == 'lse' else ''), 100 * mAP, 100 * r1))
    print('== retrieval per branch alone (0 = global, 1.. = parts)')
    for b in range(M):
        fb = {k: v[b:b + 1] for k, v in feats.items()}
        vb = {k: v[b:b + 1] for k, v in vis.items()}
        mAP, r1 = quiet_eval(fb, vb, query, gallery, 'mean')
        print('   branch %d      mAP %5.2f%%  R1 %5.2f%%' % (b, 100 * mAP, 100 * r1))

    qf = torch.stack([feats[f] for f, _, _ in query]); gf = torch.stack([feats[f] for f, _, _ in gallery])
    qv = torch.stack([vis[f] for f, _, _ in query]); gv = torch.stack([vis[f] for f, _, _ in gallery])
    qid = torch.tensor([p for _, p, _ in query]); gid = torch.tensor([p for _, p, _ in gallery])
    g_sim = qf[:, 0] @ gf[:, 0].t()
    cross = qid[:, None] != gid[None, :]
    sample = g_sim[cross]
    thr = torch.quantile(sample[torch.randperm(sample.numel())[:2_000_000]], args.hard_quantile)
    hard = cross & (g_sim >= thr)
    same = ~cross
    pd = _compute_body_parts_dist_matrices(qf, gf)  # [M, Nq, Ng]
    w = torch.sqrt(qv.t().unsqueeze(2) * gv.t().unsqueeze(1))

    print('\n== hard-pair probe: cross-identity pairs with global cosine >= %.3f (top %.0f%%, n=%d) vs same-identity pairs (n=%d)'
          % (thr, 100 * (1 - args.hard_quantile), int(hard.sum()), int(same.sum())))
    print('   %-8s %8s %8s %8s' % ('branch', 'hard', 'same-id', 'hard-same'))
    for b in range(M):
        h, s_ = pd[b][hard].mean().item(), pd[b][same].mean().item()
        print('   %-8d %8.3f %8.3f %+8.3f%s' % (b, h, s_, h - s_, '   <- this part separates them' if h - s_ > 0.05 else ''))
    hard_max = pd[1:][:, hard].max(0)[0].mean().item(); same_max = pd[1:][:, same].max(0)[0].mean().item()
    print('   max-over-parts: hard %.3f  same-id %.3f  (%+.3f)' % (hard_max, same_max, hard_max - same_max))
    print('   combined distance under each rule:')
    for combine, t in rows:
        D = combine_part_distances(pd, w, combine, t)
        h, s_, sd = D[hard].mean().item(), D[same].mean().item(), D[same].std().item()
        print('   %-12s hard %.3f | same-id %.3f | gap %+.3f  (%+.2f same-id std)'
              % (combine + (' T=%.2f' % t if combine == 'lse' else ''), h, s_, h - s_, (h - s_) / sd))
    print('\nRead: a positive, growing gap under lse relative to mean = the disagreeing part is being'
          ' allowed to count. A negative gap = the parts do not yet encode the difference; no rule can fix that.')


if __name__ == '__main__':
    main()
