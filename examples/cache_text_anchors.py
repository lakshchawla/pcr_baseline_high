"""Runs once, after examples/train_relational_prompts.py finishes: loads Stage 1's trained
PromptLearner, builds every identity's per-branch text embedding (0 = global, 1..K = parts),
and saves the frozen [num_identities, num_branches, embed_dim] lookup table Stage 2
(examples/train_relational_finetune.py) reads as its alignment target.

A separate script rather than the tail of Stage 1: PromptLearner is frozen the moment Stage 1
ends, and its output for an identity is a deterministic function of that identity alone -- so
Stage 2 is a pure supervised loop against a fixed table, and nothing after this script ever
loads prompt_learner.pth again.
"""
from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from pcr import datasets
from pcr.models.clip_text_encoder import ClipTextEncoder
from pcr.models.prompt_learner import PromptLearner
from pcr.utils.config import load_yaml_config
from pcr.utils.serialization import load_checkpoint


def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir, name))


def compute_text_prototypes(prompt_learner, text_encoder, num_identities, num_branches, id_batch):
    """[num_identities, num_branches, embed_dim], L2-normalized rows -- CosineAlignLoss's dot
    product is only a real cosine similarity (matching its 0.07 temperature) if both sides are."""
    prompt_learner.eval()
    device = prompt_learner.ctx.device
    text_prototypes = torch.zeros(num_identities, num_branches, text_encoder.embed_dim,
                                   dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, num_identities, id_batch):
            ids = torch.arange(start, min(start + id_batch, num_identities), device=device)
            prompts = prompt_learner.build_part_prompts(ids)
            for branch, prompt in enumerate(prompts):
                text_feat = text_encoder(prompt, prompt_learner.tokenized_prompts)
                text_prototypes[ids, branch] = F.normalize(text_feat.float(), p=2, dim=-1)
    return text_prototypes


def main():
    parser = argparse.ArgumentParser(
        description="Build Stage 2's frozen text-prototype table from a trained Stage-1 checkpoint")
    parser.add_argument('--config', type=str, required=True, metavar='PATH',
                         help="the same config Stage 1 was trained with -- reads its "
                              "logging.logs_dir for prompt_learner.pth and writes "
                              "text_prototypes.pth there too")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    dataset = get_data(cfg.data.dataset, cfg.data.data_dir)
    num_identities = dataset.num_train_pids
    num_parts = cfg.model.parts_num
    num_branches = 1 + num_parts

    text_encoder = ClipTextEncoder(clip_arch=cfg.clip.arch, device='cuda').cuda()
    prompt_learner = PromptLearner(num_identities, num_parts, text_encoder, n_ctx=cfg.clip.n_ctx,
                                    device='cuda').cuda()
    prompt_learner_path = osp.join(cfg.logging.logs_dir, 'prompt_learner.pth')
    prompt_learner.load_state_dict(load_checkpoint(prompt_learner_path))
    print('==> Loaded {}'.format(prompt_learner_path))

    print('==> Building text-prototype table for {} identities, {} branches'.format(
        num_identities, num_branches))
    text_prototypes = compute_text_prototypes(prompt_learner, text_encoder, num_identities,
                                               num_branches, cfg.data.cache_batch_size)
    parts = text_prototypes[:, 1:]
    off = ~torch.eye(num_parts, dtype=torch.bool, device=parts.device)
    print('==> Same-identity different-part prototype cosine: {:.3f} (1.0 = collapsed)'.format(
        torch.einsum('bkd,bjd->bkj', parts, parts)[:, off].mean().item()))

    out_path = osp.join(cfg.logging.logs_dir, 'text_prototypes.pth')
    torch.save({'text_prototypes': text_prototypes.cpu(), 'num_identities': num_identities,
                'num_branches': num_branches}, out_path)
    print('==> Saved {}'.format(out_path))


if __name__ == '__main__':
    main()
