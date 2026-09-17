"""Thin sequential orchestrator: Stage 0 (optional BPA segmentation pretraining) -> Stage 1
(prompt learning) -> cache_text_anchors (build Stage 2's frozen alignment target from Stage 1's
checkpoint) -> Stage 2 (backbone finetune) -> Stage 3 (UDA or USL domain adaptation, user-selected)
-> eval, per plans/reid_pipeline_plan.md section 6. Each stage script already evaluates internally at
the end of its own run (via pcr/evaluators.py::Evaluator), so no separate eval step exists here --
"eval" in the plan's own section 6 is what Stage 3 already does.

Shells out to each stage's own script as a separate process rather than importing them as library
functions -- train_bpa_segmentation.py/train_relational_prompts.py/cache_text_anchors.py/
train_relational_finetune.py/train_uda.py/train_usl.py are all `if __name__ == '__main__':`-driven
scripts with process-global state (stdout redirection via Logger, CUDA context, argparse), never
designed to be called twice in one Python process. This file's only real job: run the selected
stages in order, stop on the first failure, and compute the one piece of information that crosses
the YAML/argparse boundary between Stage 2 and Stage 3 -- the checkpoint path Stage 2 wrote, which
Stage 3's --checkpoint-path needs. Everything else is passed straight through to whichever scripts
run; Stage 0 -> Stage 1 -> cache_text_anchors -> Stage 2 wiring needs no such help since it's
already config-file-driven -- this script only checks that Stage 0's expected checkpoint path
matches Stage 1's model.checkpoint_path (a warning, like the existing Stage1/Stage2 logs_dir
check, not an auto-rewrite of either config) rather than actually injecting one config's output
path into the next.
"""
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import subprocess
import sys

import yaml


EXAMPLES_DIR = osp.dirname(osp.abspath(__file__))
STAGE_SCRIPTS = {
    'bpa': osp.join(EXAMPLES_DIR, 'train_bpa_segmentation.py'),
    'prompts': osp.join(EXAMPLES_DIR, 'train_relational_prompts.py'),
    'cache_anchors': osp.join(EXAMPLES_DIR, 'cache_text_anchors.py'),
    'finetune': osp.join(EXAMPLES_DIR, 'train_relational_finetune.py'),
    'uda': osp.join(EXAMPLES_DIR, 'train_uda.py'),
    'usl': osp.join(EXAMPLES_DIR, 'train_usl.py'),
}


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def flag_present(extra_args, flag):
    return any(a == flag or a.startswith(flag + '=') for a in extra_args)


def run(cmd):
    print('==> Running: {}'.format(' '.join(cmd)))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print('==> Stage failed (exit code {}), aborting pipeline: {}'.format(
            result.returncode, ' '.join(cmd)))
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="PCR pipeline orchestrator: Stage 0 (BPA segmentation, optional) -> Stage 1 "
                    "(prompts) -> Stage 2 (finetune) -> Stage 3 (UDA or USL). Any argument this "
                    "parser doesn't recognize is passed "
                    "straight through to the Stage 3 script (train_uda.py/train_usl.py) -- e.g. "
                    "`run_pipeline.py --stage3 uda -ds market1501 -dt dukemtmc-reid`. Do not use "
                    "a `--` separator before the pass-through args: argparse.parse_known_args() "
                    "does not strip it, and the Stage 3 script's own parser -- with no positional "
                    "arguments of its own -- rejects a literal '--' as an unrecognized argument.")
    parser.add_argument('--stage0-config', type=str, default=None, metavar='PATH',
                         help="configs/stage0_bpa_segmentation.yaml-style path; omit to skip "
                              "Stage 0 (optional -- most runs start from Stage 1's ImageNet init "
                              "instead). Requires a dataset with masks on disk (Market1501); "
                              "see examples/train_bpa_segmentation.py.")
    parser.add_argument('--stage1-config', type=str, default=None, metavar='PATH',
                         help="configs/stage1_relational_prompts.yaml-style path; omit to skip "
                              "Stage 1 and cache_text_anchors.py (e.g. prompt learning was "
                              "already run separately)")
    parser.add_argument('--stage2-config', type=str, default=None, metavar='PATH',
                         help="configs/stage2_relational_finetune.yaml-style path; omit to skip Stage 2")
    parser.add_argument('--stage3', type=str, choices=['uda', 'usl', 'none'], default='none',
                         help="which Stage-3 domain-adaptation driver to run after Stage 1/2, "
                              "or 'none' to stop after Stage 2's checkpoint")
    parser.add_argument('--python', type=str, default=sys.executable,
                         help="python interpreter to invoke each stage script with")
    args, stage3_extra_args = parser.parse_known_args()

    if args.stage0_config and args.stage1_config:
        stage0_out = osp.join(load_yaml(args.stage0_config)['logging']['logs_dir'], 'model_best.pth.tar')
        stage1_checkpoint = load_yaml(args.stage1_config)['model']['checkpoint_path']
        if osp.normpath(stage0_out) != osp.normpath(stage1_checkpoint or ''):
            print("==> WARNING: stage0 config's expected checkpoint output ({}) does not match "
                  "stage1 config's model.checkpoint_path ({}) -- Stage 1 will not load the "
                  "checkpoint this run of Stage 0 is about to produce (empty means ImageNet init "
                  "instead). Continuing anyway (not a hard error, in case this is "
                  "intentional).".format(stage0_out, stage1_checkpoint or '<empty>'))

    if args.stage1_config and args.stage2_config:
        stage1_out = load_yaml(args.stage1_config)['logging']['logs_dir']
        stage2_in = load_yaml(args.stage2_config)['stage1']['prompt_dir']
        if osp.normpath(stage1_out) != osp.normpath(stage2_in):
            print("==> WARNING: stage1 config's logging.logs_dir ({}) does not match stage2 "
                  "config's stage1.prompt_dir ({}) -- Stage 2 will not read the "
                  "text_prototypes.pth this run of Stage 1 is about to produce. Continuing "
                  "anyway (not a hard error, in case this is intentional).".format(stage1_out, stage2_in))

    if args.stage0_config:
        run([args.python, STAGE_SCRIPTS['bpa'], '--config', args.stage0_config])
    else:
        print('==> Skipping Stage 0 (no --stage0-config given)')

    if args.stage1_config:
        run([args.python, STAGE_SCRIPTS['prompts'], '--config', args.stage1_config])
        run([args.python, STAGE_SCRIPTS['cache_anchors'], '--config', args.stage1_config])
    else:
        print('==> Skipping Stage 1 and cache_text_anchors.py (no --stage1-config given)')

    stage2_checkpoint = stage2_backbone = None
    if args.stage2_config:
        run([args.python, STAGE_SCRIPTS['finetune'], '--config', args.stage2_config])
        stage2_cfg = load_yaml(args.stage2_config)
        stage2_checkpoint = osp.join(stage2_cfg['logging']['logs_dir'], 'model_best.pth.tar')
        stage2_backbone = stage2_cfg['model']['backbone']
    else:
        print('==> Skipping Stage 2 (no --stage2-config given)')

    if args.stage3 == 'none':
        print('==> Pipeline complete (--stage3 none, stopping after Stage 2).')
        return

    stage3_cmd = [args.python, STAGE_SCRIPTS[args.stage3]]
    if stage2_checkpoint is not None and not flag_present(stage3_extra_args, '--checkpoint-path'):
        if not osp.isfile(stage2_checkpoint):
            print('==> WARNING: expected Stage 2 checkpoint not found at {} -- Stage 3 will '
                  'likely fail unless --checkpoint-path is supplied manually.'.format(stage2_checkpoint))
        stage3_cmd += ['--checkpoint-path', stage2_checkpoint]
    if stage2_backbone is not None and not flag_present(stage3_extra_args, '--backbone'):
        stage3_cmd += ['--backbone', stage2_backbone]
    stage3_cmd += stage3_extra_args

    run(stage3_cmd)
    print('==> Pipeline complete.')


if __name__ == '__main__':
    main()
