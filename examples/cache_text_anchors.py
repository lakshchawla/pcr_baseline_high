"""Runs once, after examples/train_relational_prompts.py finishes: loads Stage 1's trained
PromptLearner (which owns TextualAttentionBlock as a submodule) and its saved
identity_visibility.pth (TextualAttentionBlock's per-identity attention bias -- see
relation_blocks.py's own docstring and train_relational_prompts.py's compute_identity_visibility;
reused unchanged, not recomputed, so this script's prompts match training exactly), builds every
identity's per-branch text embedding, and saves the frozen [num_identities, num_branches,
embed_dim] lookup table Stage 2 (examples/train_relational_finetune.py) actually reads.

Why this is a separate script, not inlined at the end of train_relational_prompts.py the way the
pre-relational-attention version of this pipeline did it: PromptLearner and TextualAttentionBlock are
both frozen and fully done training the moment Stage 1 ends -- their output for any given
identity is now a fixed, deterministic function of that identity's id alone. Recomputing that
forward pass inside every Stage-2 training step would waste compute on something that never
changes. Running it once here and saving the result makes Stage 2 a pure supervised-training loop
against a fixed table, and makes it obvious in the codebase that PromptLearner/TextualAttentionBlock's
useful life ends exactly here -- nothing after this script ever loads prompt_learner.pth again.
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


def compute_text_prototypes(prompt_learner, text_encoder, num_identities, num_branches, id_batch,
                             identity_visibility):
    """identity_visibility: [num_identities, num_branches], the exact table
    examples/train_relational_prompts.py computed and saved -- reused unchanged (not
    recomputed) so TextualAttentionBlock's attention bias here matches training exactly, keeping
    ctx/TAB's output a deterministic function of identity alone (see relation_blocks.py's own
    docstring)."""
    prompt_learner.eval()
    text_prototypes = torch.zeros(num_identities, num_branches, text_encoder.embed_dim,
                                   dtype=torch.float32, device='cuda')
    with torch.no_grad():
        for start in range(0, num_identities, id_batch):
            ids = torch.arange(start, min(start + id_batch, num_identities), device='cuda')
            branch_vis = identity_visibility[ids]  # [b, num_branches], same as training
            prompts, _ = prompt_learner.build_part_prompts(ids, branch_vis)  # list of num_branches tensors
            for branch, prompt in enumerate(prompts):
                text_feat = text_encoder(prompt, prompt_learner.tokenized_prompts)
                # L2-normalized -- matches examples/train_relational_prompts.py's own text-side
                # normalization (see that file's build_text_snapshot / pcr/loss/clip_supcon_loss.py's
                # docstring for why): CosineAlignLoss's own visual-side input is already
                # normalized (examples/train_relational_finetune.py's compute_losses), so leaving
                # this table unnormalized would make that loss's dot product not a real cosine
                # similarity, silently softening its align_temperature far below its intended
                # (CLIP-standard) value.
                text_prototypes[ids, branch] = F.normalize(text_feat.float(), p=2, dim=-1)
    return text_prototypes


def compute_text_preproj(prompt_learner, text_encoder, num_identities, num_branches, id_batch,
                          identity_visibility):
    """Same loop as compute_text_prototypes above, but calls encode_branch_prepool instead of the
    text encoder's own forward() -- returns [num_identities, num_branches, transformer_width]
    (NOT embed_dim -- see ClipTextEncoder.encode_branch_prepool's own docstring), the input
    compute_text_self_attention below actually needs. Not saved to disk itself (only
    text_self_attention.pth, built from it, is) -- this table has no other consumer."""
    prompt_learner.eval()
    preproj = torch.zeros(num_identities, num_branches, text_encoder.transformer_width,
                           dtype=torch.float32, device='cuda')
    with torch.no_grad():
        for start in range(0, num_identities, id_batch):
            ids = torch.arange(start, min(start + id_batch, num_identities), device='cuda')
            branch_vis = identity_visibility[ids]
            prompts, _ = prompt_learner.build_part_prompts(ids, branch_vis)
            for branch, prompt in enumerate(prompts):
                preproj[ids, branch] = text_encoder.encode_branch_prepool(
                    prompt, prompt_learner.tokenized_prompts).float()
    return preproj


def compute_text_self_attention(text_encoder, text_preproj, id_batch):
    """text_preproj: [num_identities, num_branches, transformer_width] (compute_text_preproj
    above). Returns [num_identities, num_branches, num_branches]: each identity's own CLIP
    text-side self-attention among its branch representations -- CrossAttentionBlock's
    L_crossalign target in Stage 2 (see pcr/loss/cross_attn_align_loss.py). Precomputed here,
    once, for the same reason text_prototypes is: prompt_learner/TAB/the text encoder are frozen
    after Stage 1, so this is a deterministic function of identity alone."""
    num_identities, num_branches, _ = text_preproj.shape
    attn_table = torch.zeros(num_identities, num_branches, num_branches,
                              dtype=torch.float32, device='cuda')
    with torch.no_grad():
        for start in range(0, num_identities, id_batch):
            end = min(start + id_batch, num_identities)
            attn_table[start:end] = text_encoder.encode_branch_self_attention(
                text_preproj[start:end]).float()
    return attn_table


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
    # This script never touches the image encoder (purely a text-side replay), so
    # has_global_branch can't be read off a live encoder object the way train_relational_prompts.
    # py/train_relational_finetune.py do -- derived instead from the same predicate those two
    # files use to pick ClipRN50BPAMEncoder vs ClipViTBPAMEncoder in the first place (only the
    # RN50 backbone currently exposes the extra global branch -- see ClipBPAMEncoder's own
    # docstring). Must stay in sync with that dispatch logic if it ever changes. Also must match
    # whatever Stage 1 actually used, or prompt_learner.load_state_dict below fails outright on a
    # ctx shape mismatch (a safe, loud failure, not a silent one).
    has_global_branch = not cfg.clip.arch.startswith('ViT')
    num_branches = (2 if has_global_branch else 1) + num_parts

    text_encoder = ClipTextEncoder(clip_arch=cfg.clip.arch, device='cuda').cuda()
    prompt_learner = PromptLearner(num_identities, num_parts, text_encoder, n_ctx=cfg.clip.n_ctx,
                                    tab_num_heads=cfg.tab.num_heads, tab_num_layers=cfg.tab.num_layers,
                                    device='cuda', has_global_branch=has_global_branch).cuda()
    prompt_learner_path = osp.join(cfg.logging.logs_dir, 'prompt_learner.pth')
    prompt_learner.load_state_dict(load_checkpoint(prompt_learner_path))
    print('==> Loaded {}'.format(prompt_learner_path))

    identity_visibility_path = osp.join(cfg.logging.logs_dir, 'identity_visibility.pth')
    identity_visibility = load_checkpoint(identity_visibility_path).cuda()

    print('==> Building text-prototype table for {} identities, {} branches'.format(
        num_identities, num_branches))
    text_prototypes = compute_text_prototypes(prompt_learner, text_encoder, num_identities,
                                               num_branches, cfg.data.cache_batch_size,
                                               identity_visibility)

    out_path = osp.join(cfg.logging.logs_dir, 'text_prototypes.pth')
    torch.save({'text_prototypes': text_prototypes.cpu(), 'num_identities': num_identities,
                'num_branches': num_branches}, out_path)
    print('==> Saved {}'.format(out_path))

    print('==> Building text self-attention table for CrossAttentionBlock (Stage 2)')
    text_preproj = compute_text_preproj(prompt_learner, text_encoder, num_identities, num_branches,
                                         cfg.data.cache_batch_size, identity_visibility)
    text_self_attention = compute_text_self_attention(text_encoder, text_preproj,
                                                        cfg.data.cache_batch_size)
    attn_path = osp.join(cfg.logging.logs_dir, 'text_self_attention.pth')
    torch.save({'text_self_attention': text_self_attention.cpu(), 'num_identities': num_identities,
                'num_branches': num_branches}, attn_path)
    print('==> Saved {}'.format(attn_path))


if __name__ == '__main__':
    main()
