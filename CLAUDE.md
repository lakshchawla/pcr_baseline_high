# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# PCR — Part-based CLIP Re-ID (`pcr_baseline_high`)

## What this is

Part-based person re-identification: BPBreID's pixel-to-part attention pooling on top of CLIP's
frozen RN50 visual tower, with CLIP-ReID-style per-identity, per-part prompt learning. Package
`pcr/`, driver scripts `examples/`, YAML configs `configs/`. Started as a snapshot of `pcr2` at its
best baseline (82.2/91.8 mAP/R1 on Market1501) and then reworked around one thesis: **part
prompts must be tied to anatomy, or they collapse into identity vectors** (measured: 0.98 cosine
between one identity's five part prompts under the old design). Origin:
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
| 1 | `train_relational_prompts.py` | `PromptLearner.ctx`, TAB, `AttentionPoolingBlock`, SupCon temperature; **`pixel_classifier` while the blend is active** | Stage 0 ckpt → `prompt_learner.pth`, `pool.pth`, `identity_visibility.pth`, **`pixel_classifier.pth`** |
| — | `cache_text_anchors.py` | nothing (replay) | Stage 1 dir → `text_prototypes.pth` |
| 2 | `train_relational_finetune.py` | backbone + `pixel_classifier` + pool + BN necks + id head | `text_prototypes.pth`, `pool.pth`, **`pixel_classifier.pth`** → `model_best.pth.tar` |
| 3 | `train_uda.py` / `train_usl.py` | optional SpCL-style adaptation, argparse-driven, untouched by all of the above | |

Branch order is load-bearing everywhere: `0 = global (pooled), 1..K=5 = parts, 6 = CLIP-native
global (RN50 only)`. `has_global` comes from `encoder._has_global`; never hardcode it.

## Image side (`pcr/models/`)

- `clip_rn50_bpam_encoder.py::ClipRN50DenseBackbone` — frozen CLIP RN50, last stride dropped,
  `_attnpool_forward` with **`query=x`** (every location as query; `x[:1]` was tried and returns
  an empty `project()` — don't reintroduce it). `project_all` → per-patch joint feats + native
  global; `project_dense` → MaskCLIP-style `c_proj(v_proj(patch))`, attention-free.
- `clip_dense_part_encoder.py::ClipBPAMEncoder._forward_common(images, text_contexts=None,
  blend_weight=0.0, stop_mask_grad=False)` — classifier softmax → optional **CLIP-native mask
  blend** (`clip_native_masks.py`: each patch cosine-matched against *its own identity's* K part
  texts; only the classifier's *foreground* mass is redistributed, background untouched, probs
  still sum to 1) → GWAP pooling → visibility. All defaults off ⇒ bit-identical to the pre-blend
  encoder (verified). Blend diagnostics + the reverse-KL **anchor loss** + `blended_probs` come
  back through `encoder.last_blend_stats`. `stop_mask_grad` detaches the classifier from the
  pooled features so it learns only from mask losses.
- `relation_blocks.py` — `TextualAttentionBlock` (TAB, **gated**, zero-init; `gated=False` only to
  replay pre-gate checkpoints) and `AttentionPoolingBlock` + `apply_part_pooling`. **VAB and CAB
  are gone** (VAB gate never left 0 in two real runs; CAB never ran at retrieval). The K part
  tokens are never mixed with each other on the image side.
- `evaluators.py` — `extract_features(model, loader, pool=None)`, `Evaluator(model, pool=None)`.
  Nothing text-side at retrieval.

## Text side + Stage 1 losses

`ctx[y] → TAB → "A photo of a [ctx_k] person." → frozen CLIP text encoder → branch_texts`, a
function of identity alone — **that is what lets `cache_text_anchors.py` freeze it**. Any
image-conditioned change to the text path breaks the train/cache equality; a cross-attention
that *modifies* the prompt must condition on per-identity centroids, not per-image features.

| Loss | Negatives | Config |
|---|---|---|
| SupCon i2t / t2i (per branch) | other identities (full table / full cache) | `loss.temperature` (learnable) |
| `PartDiagLoss` (`pcr/loss/part_diag_loss.py`) | the **same person's other parts** — the set SupCon lacks; the anti-collapse term | `part_diag:` |
| `L_anchor` (reverse KL text-map ‖ classifier map) | — | `clip_native_mask.anchor_weight` (× β/β_max) |
| `L_bpa` + `L_distil` (`pcr/utils/mask_targets.py`) | — | `bpam:` (only while β > 0) |

Once β > 0, each PK batch's images + masks are re-read (`get_batch_image_loader`, deterministic
transform = the cache transform) and run live; the full feature cache is rebuilt every
`bpam.recache_every` epochs. `blend_weight_max: 0` reproduces the pre-blend script exactly.

**Epoch-line metrics to read:** `part-ctx cos` (same-identity different-part text cosine; 0.98 =
collapsed, want well below 0.9), `mask delta/part` (text map vs classifier, β-independent),
`outside-support`, `bpa`, `distil`. Iteration line: `TAB gate`, `Pool gate`, `part_diag`,
`anchor`.

## Conventions

- Zero-init `tanh(gate) * delta` residual for every trainable block; a block is a no-op at init.
- Every embedding a loss touches is L2-normalized first; temperatures are calibrated for cosines.
- Visibility weights inside losses are detached (no "hide the part instead of aligning it").
- Configs carry the *reason* for each non-default value in a comment; update it with the value.
- `progress.md` is append-only and is where design rationale lives (code stays comment-light on
  the *why*); `changes.md` is the pending list; `plans/` holds the plan docs implemented here.
