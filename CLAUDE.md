# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# PCR — Part-based CLIP Re-ID (`pcr_baseline_high`)

## What this is

Part-based person re-identification: BPBreID's pixel-to-part attention pooling on top of CLIP's
frozen RN50 visual tower, with CLIP-ReID-style per-identity, per-part prompt learning. Package
`pcr/`, driver scripts `examples/`, YAML configs `configs/`. Started as a snapshot of `pcr2` at its
best baseline and then reduced to **CLIP-ReID's image path + one branch per body part**, after
the original design was found to pool every part from an attention path that made all parts
copies of the global (part-vs-part cosine 0.976 in the frozen encoder; every branch alone scored
the same mAP as all seven together). Origin:
`github.com/lakshchawla/pcr_baseline_high`, branch `main`. History is a single `initial_commit`;
`progress.md` holds the dated rationale for everything since.

## Commands

Everything runs from the repo root; every path in `configs/*.yaml` is repo-root-relative
(`data.data_dir: ../datasets`, `examples/logs/...`). Conda env `pcr2-run` (torch 2.13/cu129); its
editable `pcr` install points at a sibling repo, so run with `PYTHONPATH=third_party:.` to get
*this* tree's `clip`/`torchreid`.

```bash
pip install -e .                                   # vendored third_party/{clip,torchreid} included
export PYTHONPATH=third_party:.

python examples/train_bpa_segmentation_rn50.py --config configs/stage0_bpa_segmentation_rn50.yaml
python examples/train_relational_prompts.py    --config configs/stage1_relational_prompts.yaml
python examples/cache_text_anchors.py          --config configs/stage1_relational_prompts.yaml   # same config as Stage 1
python examples/train_relational_finetune.py   --config configs/stage2_relational_finetune.yaml
./run_full_pipeline.sh                              # all four; STAGE{0,1,2}_CONFIG env overrides

python examples/train_relational_finetune.py --config ... --setup-only    # build everything, exit before training
find pcr examples -name '*.py' -not -path '*/__pycache__/*' | xargs python -m py_compile
```

No test suite. Verification = `py_compile` sweep + synthetic shape/gradient gates against the real
classes on CPU + a `--setup-only` run + (ideally) a 1-epoch smoke with a scratch `logs_dir`. The
`Logger` opens `log.txt` with `'w'`: **any run against a config overwrites that `logs_dir`'s
log**, so use a scratch `logging.logs_dir` for smoke runs. Check `pgrep -af train_relational`
before touching configs or `examples/logs/` — a long run is often live on the single 8 GB GPU.
`examples/logs/`, `*.pth`, `*.pth.tar` are gitignored; `__pycache__` is tracked (don't `git add -A`).

## Pipeline

| Stage | Script | Trains | Reads → writes |
|---|---|---|---|
| 0 | `train_bpa_segmentation_rn50.py` | `pixel_classifier` (BN + 1×1 conv) on PifPaf masks, backbone frozen | → `stage0_bpa_rn50/model_best.pth.tar` |
| 1 | `train_relational_prompts.py` | `PromptLearner.ctx` + SupCon temperature only | Stage 0 ckpt → `prompt_learner.pth` |
| — | `cache_text_anchors.py` | nothing (replay) | Stage 1 dir → `text_prototypes.pth` |
| 2 | `train_relational_finetune.py` | backbone + `pixel_classifier` + BN necks + id heads | Stage 0 ckpt, `text_prototypes.pth` → `model_best.pth.tar` |
| 3 | `train_uda.py` / `train_usl.py` | optional SpCL-style adaptation, argparse-driven, untouched | |

Branch order is load-bearing everywhere: `0 = global (x_proj), 1..K=5 = parts`. `encoder.num_parts`
is M = 1+K (Stage 3 reads it).

## Architecture (2026-09-17 rewrite: CLIP-ReID + per-part branches, nothing else)

- `clip_rn50_bpam_encoder.py::ClipRN50DenseBackbone` — frozen-loadable CLIP RN50, layer4 stride
  dropped (24×8 grid at 384×128). `forward_multi` → `(x3, x4 patches)`; `project_global` → CLIP's
  real global (mean-token attnpool, `x_proj`); `project_dense` → MaskCLIP `c_proj(v_proj(patch))`.
  **Never pool parts from the attention path** (`project`/`project_all`, kept for reference):
  every-location-query attnpool made every patch ≈ the same image-wide average (part-vs-part
  cos 0.976 frozen); `project_dense` gives 0.76–0.80.
- `clip_dense_part_encoder.py::ClipBPAMEncoder.forward_multi` → dict `x3, x4, x_proj, part_x4
  [B,K,2048], part_xproj [B,K,1024], vis [B,1+K], pixels_cls_scores`. `forward()` →
  `(joint_branches [B,1+K,1024], vis)` = what retrieval and Stage 1 use. The CLIP-native mask
  blend (`text_contexts`, `blend_weight`, `stop_mask_grad`, `last_blend_stats`) is still there,
  opt-in, verified no-op by default — not wired into any script now.
- `prompt_learner.py` — ctx → template → frozen text encoder. **No TAB, no VAB, no CAB, no pool**
  (`relation_blocks.py` deleted): VAB's gate never left 0, CAB never ran at retrieval, TAB was the
  fastest route to prompt collapse. Each part aligns to its own prompt, separately.
- `evaluators.py` — `Evaluator(model, part_combine, temperature)`, `extract_features(model,
  loader)`; part-wise visibility-aware distance over the joint-space branches. Per-part distances
  are combined by **`combine_part_distances`** (`pcr/utils/part_distance.py`): `'lse'` = weighted
  log-sum-exp = soft-max of part distances = soft-min of part similarities (one disagreeing part
  penalizes the whole score — the "all black, different shoes" case), `'mean'` = BPBReID's default
  (still what Stage 3 uses). Stage 2's `PartTripletLoss` mines under the same rule (`eval.*`).

**Stage 1 losses:** SupCon i2t/t2i per branch (cross-identity negatives, full table/cache) +
`PartDiagLoss` (`pcr/loss/part_diag_loss.py`: image part *k* vs the same person's other parts —
the anti-collapse term). **Stage 2 losses** (= CLIP-ReID's + parts): id on BN(x4), BN(x_proj),
BN(part_x4[k]) vis-weighted; triplet on x3/x4/x_proj + one BPBreID part-triplet over part_x4;
align (CLIP-ReID's I2T) x_proj↔prototype 0 and part_xproj[k]↔prototype k; BPA. Recipe = CLIP-ReID
RN50: lr 3.5e-4, batch 64, 120 ep, ×0.1 at 40/70, 10-ep warmup. (**5e-6 is the ViT recipe** — it
gave 75.9 mAP here.)

**Metrics to read:** Stage 1 `part-ctx cos` (want ≪ 0.9; 0.98 = collapsed), Stage 2 `part_cos`
(train-mode BN, so higher than eval), `align_parts`, `tri_parts`. Per-branch retrieval ablation
(global-only vs parts-only vs all) is the decisive check that parts carry information — see
progress.md 2026-09-17 (4) for the script pattern.

## Conventions

- Every embedding a loss touches is L2-normalized first; temperatures are calibrated for cosines.
- Visibility weights inside losses are detached (no "hide the part instead of aligning it").
- Configs carry the *reason* for each non-default value in a comment; update it with the value.
- `progress.md` is append-only and is where design rationale lives (code stays comment-light on
  the *why*); `changes.md` is the pending list; `plans/` holds the plan docs implemented here.
