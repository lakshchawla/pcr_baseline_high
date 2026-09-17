# PCR

PCR combines three techniques into one pipeline, per `plans/reid_pipeline_plan.md`:

1. **BPBreID** part-attention feature pooling (K learnable body parts + visibility scores, no
   external masks needed at forward time).
2. **CLIP-ReID**-style two-stage prompt learning -- a frozen CLIP text encoder regularizes
   training via per-(identity, body-part) learnable prompts (Stage 1), then the backbone is
   fine-tuned against those frozen prompts' text embeddings (Stage 2).
3. **SPCL**'s self-paced hybrid-memory strategy (Jaccard/k-reciprocal re-ranking -> DBSCAN ->
   self-paced reliability filtering -> contrastive memory-bank loss -> DSBN) for unsupervised
   domain adaptation, or an ICE-style single-domain unsupervised alternative.
4. **Part-relational attention** -- bidirectional self-attention across a person's K body-part
   tokens, on both the visual side (VisualAttentionBlock) and the text side (TextualAttentionBlock),
   in Stage 1/2 only (Stage 3 is untouched by this). See `pcr/models/relation_blocks.py` and
   `progress.md` for the design.

Repo layout mirrors SpCL's own shape (`examples/` + a `pcr/` package with `models/`, `loss/`,
`utils/`, `datasets/`, `evaluation_metrics/`), not BPBReID/torchreid's. Every hyperparameter for
the pre-existing UDA/USL drivers is an inline argparse default (no config files); the two newer
CLIP training stages are YAML-driven instead -- see `configs/`.

## Install

```bash
pip install -e .
```

`clip` and `torchreid` (bpbreid) are vendored under `third_party/` and wired into `setup.py` via
`package_dir` -- the single `pip install -e .` above builds both from this repo's own local source,
no separate `pip install -e /path/to/bpbreid` or git-installed CLIP package needed.

## Pipeline stages

```
Stage 0                Stage 1                cache_text_anchors.py    Stage 2                    Stage 3
(train_bpa_             (train_relational        (build Stage 2's       (train_relational          (train_uda.py or
 segmentation.py,         _prompts.py)              alignment target)     _finetune.py)              train_usl.py)
 optional)
BPA pixel-classifier   per-part CLIP prompt   -->                   --> supervised backbone   --> domain adaptation
pretrained against     + relation blocks,                               finetune + CLIP           (UDA) or single-
real part masks        backbone frozen                                  alignment + VAB           domain unsupervised
```

Run the whole thing with `examples/run_pipeline.py`, or any subset of stages directly. Stage 0 is
optional and stands alone -- its output checkpoint feeds into Stage 1's `model.checkpoint_path`
(or Stage 2/3 directly) in place of ImageNet init; `run_pipeline.py` doesn't wire it in yet.

### Stage 0 -- BPA segmentation pretraining (optional, masks required)

```bash
python examples/train_bpa_segmentation.py --config configs/stage0_bpa_segmentation.yaml
```

Trains BPBreID's own pixel-to-part classifier (`pixel_classifier`, the mechanism behind `A_m` in
`METHODOLOGY.md` section 2.1) via plain supervised segmentation against ground-truth PifPaf/
MaskRCNN part masks -- no id/triplet/CLIP-alignment loss at all, just `BodyPartAttentionLoss`
alone. Motivation: without this, nothing in Stage 1/2's downstream signal ties a given branch
index to a real anatomical region (Stage 1's per-branch prompts are learned placeholder tokens,
not real body-part words, so CLIP never supplies that correspondence either) -- see `progress.md`
for the full rationale. Only usable on datasets with masks on disk (`data.masks_dir` defaults to
Market1501's real `pifpaf_maskrcnn_filtering` directory, unlike Stage 1/2's optional, empty-by-
default `masks_dir` -- this stage has no other loss to fall back on). DukeMTMC-reID has no masks,
so this stage only runs against Market1501. Produces a checkpoint in Stage 2's own save format,
loadable via `model.checkpoint_path` in Stage 1/2's configs or `--checkpoint-path` for Stage 3.

### Stage 1 -- per-part CLIP prompt learning + relational attention

```bash
python examples/train_relational_prompts.py --config configs/stage1_relational_prompts.yaml
```

Frozen BPBreID encoder (ImageNet-init by default, or set `model.checkpoint_path` to an
externally-pretrained BPBreID checkpoint, switched to continuous visibility scores by this
script's own `build_encoder`) + frozen CLIP text encoder. Trainable: `PromptLearner`
(per-(identity, branch) learnable prompt context, owning a `TextualAttentionBlock` that mixes the K
part branches' context together before any part's prompt is built) and a `VisualAttentionBlock`
(mixes the K part branches' pooled visual features the same way) -- both blocks' own self-attention
is visibility-aware too (a poorly-visible part counts less as a key others attend to), not just the
loss below. Every training image is used, no upstream filtering -- each part's `SupConLoss` term is
weighted by that part's own per-image visibility score instead (see `pcr/loss/clip_supcon_loss.py`;
`progress.md` has the full rationale for replacing the earlier image-level filter). Produces
`prompt_learner.pth`, `vab.pth`, and `identity_visibility.pth` (TextualAttentionBlock's per-identity
attention bias) under `logging.logs_dir`.

```bash
python examples/cache_text_anchors.py --config configs/stage1_relational_prompts.yaml
```

Run once, after Stage 1 finishes, against the *same* config. Loads `prompt_learner.pth` and
builds `text_prototypes.pth` (a frozen per-identity, per-branch text-embedding lookup table) --
the only one of Stage 1's outputs Stage 2 actually reads.

### Stage 2 -- supervised backbone finetune

```bash
python examples/train_relational_finetune.py --config configs/stage2_relational_finetune.yaml
```

Set `stage1.prompt_dir` in the config to Stage 1's `logging.logs_dir` (this is where both
`text_prototypes.pth` and `vab.pth` are read from). Trains the full BPBreID encoder with real
identity labels: id loss (foreground branch only) + triplet loss + an alignment loss against
Stage 1's frozen text prototypes, plus an optional BPA (body-part-attention) loss if
`data.masks_dir` is set (masks are source-domain-only on disk for Market1501; DukeMTMC-reID has
none). `VisualAttentionBlock` continues training here (loaded from Stage 1's `vab.pth`, jointly
with the now-unfrozen backbone) -- unlike `PromptLearner`/`TextualAttentionBlock`, which are frozen
and discarded after Stage 1. Same no-upstream-filtering design as Stage 1: every training image is
used; the alignment loss (`CosineAlignLoss`) is weighted per-part by that part's own visibility,
while the triplet loss keeps a loose hard exclusion (`loss.triplet_visibility_min`) since
batch-hard mining doesn't compose with soft weights. Produces a checkpoint directly loadable by
Stage 3's `--checkpoint-path`, unchanged (Stage 3 is untouched by VisualAttentionBlock -- its own
weights save separately and aren't consumed yet).

### Stage 3 -- domain adaptation

Either UDA (source labeled + target unlabeled, SpCL-style self-paced hybrid memory):

```bash
python examples/train_uda.py \
    -ds dukemtmc-reid -dt market1501 \
    --checkpoint-path /path/to/stage2_or_external_checkpoint.pth.tar \
    --data-dir /path/to/datasets \
    --logs-dir logs/duke2market
```

...or USL (single target domain, no source, ICE-style EMA teacher-student):

```bash
python examples/train_usl.py \
    -dt market1501 \
    --checkpoint-path /path/to/stage2_or_external_checkpoint.pth.tar \
    --data-dir /path/to/datasets \
    --logs-dir logs/market_usl
```

`--data-dir` should contain `<data-dir>/market1501/{bounding_box_train,query,bounding_box_test}`
and `<data-dir>/dukemtmc-reid/DukeMTMC-reID/{bounding_box_train,query,bounding_box_test}`.
`--checkpoint-path` is required for `train_uda.py`, optional (ImageNet-init fallback) for
`train_usl.py`. Both accept `--setup-only` for a dry run that builds the model/memory/optimizer
and exits before training, to sanity-check shapes. Both also accept `--gilt-id-weight`/
`--gilt-triplet-weight`/`--masks-dir`/`--bpa-weight` to enable GiLt (id+triplet) and BPA loss
terms alongside the base memory/contrastive loss (`train_uda.py`'s id loss is source-only via a
persistent classifier; `train_usl.py`'s classifies against per-epoch cluster centers instead,
since its pseudo-label numbering changes every epoch).

### Orchestrator

```bash
python examples/run_pipeline.py \
    --stage0-config configs/stage0_bpa_segmentation.yaml \
    --stage1-config configs/stage1_relational_prompts.yaml \
    --stage2-config configs/stage2_relational_finetune.yaml \
    --stage3 uda \
    -ds dukemtmc-reid -dt market1501 --data-dir /path/to/datasets --logs-dir logs/full_run
```

Runs the selected stages in order (Stage 1 automatically triggers `cache_text_anchors.py`
afterward, against the same `--stage1-config`), stopping on the first failure. `--stage0-config`
is optional (most runs start from Stage 1's ImageNet init instead) -- unlike the Stage2->Stage3
handoff below, Stage 0's checkpoint is *not* auto-filled into Stage 1's config; if
`--stage0-config` and `--stage1-config` are both given but Stage 1's own `model.checkpoint_path`
doesn't already point at Stage 0's expected output, this script only warns (matching the
Stage1/Stage2 `logs_dir` consistency check), it doesn't rewrite either YAML file for you. Any
argument this script doesn't recognize (like `-ds`/`-dt`/`--data-dir`/`--logs-dir` above) is passed
straight through to the Stage 3 script; `--checkpoint-path`/`--backbone` are auto-filled from
Stage 2's checkpoint and config unless already given explicitly. Omit `--stage0-config`/
`--stage1-config`/`--stage2-config` to skip those stages (e.g. to run Stage 3 alone against an
externally-pretrained checkpoint); pass `--stage3 none` to stop after Stage 2.

### Alternative: external BPBreID source pretraining

Skipping Stages 1-2 entirely and pretraining with bpbreid's own pipeline instead is still
supported -- any checkpoint works for `--checkpoint-path`, this repo doesn't require it to come
from Stage 2:

```bash
cd /path/to/bpbreid
python torchreid/scripts/main.py --config-file configs/bpbreid/bpbreid_market1501_train.yaml
```

(or `bpbreid_dukemtmc_train.yaml` for the other source domain). If PifPaf masks aren't
available, pass `loss.part_based.weights.pixls.ce=0` on the CLI so the pixel-supervision loss is
skipped and the attention trains unsupervised, purely from the ID/triplet gradient.

## Fusion module

`pcr/models/fusion_head.py::FusionHead` fuses BPBreID's per-branch embeddings into a single
L2-normalized descriptor (fixed-weight or learned-gate visibility gating), for ANN/FAISS-style
retrieval. It's a standalone module, not wired into any training/eval script -- every script
above matches via `pcr/utils/part_distance.py::compute_bpb_pairwise_distance`'s native part-wise
distance instead, which is the default (and generally more accurate) matching strategy.
