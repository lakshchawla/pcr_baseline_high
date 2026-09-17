# PCR: Part-based Contrastive Re-id — BPBReID under SpCL's self-paced UDA strategy

## Context

PCR is a new, standalone technique/repo: BPBReID's part-attention feature pooling (this repo,
`/home/lakshh/workspace/reid/bpbreid`), trained under SpCL's original self-paced contrastive UDA
strategy (`/home/lakshh/workspace/reid/SpCL`, baseline pipeline: `examples/spcl_train_uda.py` +
`spcl/trainers.py::SpCLTrainer_UDA` + `spcl/models/hm.py::HybridMemory`), for cross-domain person
re-ID between Market1501 and DukeMTMC-reID (either direction).

Why the *baseline* SpCL pipeline and not the repo's own heavily-customized
`ca_uda_trainer.py`/`CrossDomainMemory`/`HybridCameraMemory` pipeline: that pipeline is SpCL's own
research extension (camera-aware CA-Jaccard clustering, EMA teacher-student, camera-proxy memories),
not "the SpCL training strategy" itself. PCR targets the original technique — supervised source
pretraining, then per-epoch (1) traditional camera-blind Jaccard/k-reciprocal re-ranking distance on
the target domain, (2) three-way self-paced DBSCAN (R_indep/R_comp reliability filtering), (3) a
single unified `HybridMemory` holding fixed source-class slots + per-target-instance slots,
contrastive InfoNCE loss against it, domain-specific BatchNorm (DSBN) for source/target statistics —
generalized so every step operates on BPBReID's K part-embeddings + visibility scores instead of one
global vector.

This is a **new, separate repo**, not a modification of SpCL or BPBReID. Layout mirrors SpCL's own
repo shape (`examples/` + a package dir with `models/`, `utils/`, `datasets/`, `evaluation_metrics/`,
top-level `trainers.py`/`evaluators.py`), not BPBReID/torchreid's. It only contains the pieces this
plan needs plus the small generic utilities they depend on — nothing else is ported over.

**Naming note**: `/home/lakshh/workspace/reid/pcr` already exists as an unrelated, unaffected repo (a
CLIP-ReID fork with its own history, most recently "purge pose-guided integration, back to clean
CLIP-ReID fork"). To avoid any collision, this new repo lives at a sibling path,
`/home/lakshh/workspace/reid/pcr2` (directory created, currently empty) — rename later if desired.
The *technique* itself is still called PCR; the Python package inside the repo is `pcr/` (mirroring
how the `SpCL` repo contains the `spcl/` package).

**Scope note on CA-Jaccard / camera-aware memory**: explicitly out of scope for this build.
Clustering uses traditional (camera-blind) Jaccard re-ranking only. No camera-proxy or camera-aware
memory component is included.

## Repo layout (mirrors SpCL's shape)

```
pcr2/
  examples/
    train_uda.py                  # main UDA driver: -ds/-dt for either direction
  pcr/
    __init__.py
    trainers.py                   # PCRTrainer_UDA: part-generalized SpCLTrainer_UDA
    evaluators.py                 # part-aware Evaluator (M2) / plain Evaluator (M1)
    models/
      __init__.py                 # factory dict, registers 'bpbreid'
      bpbreid_encoder.py          # adapter wrapping torchreid.models.bpbreid.BPBreID
      dsbn.py                     # ported near-verbatim from spcl/models/dsbn.py (generic, no changes needed)
      hm.py                       # PartHM autograd fn + PartHybridMemory
    utils/
      jaccard_rerank.py           # traditional Jaccard re-ranking, built to consume a precomputed distance matrix
      part_distance.py            # compute_bpb_pairwise_distance, ported from torchreid/metrics/distance.py
      meters.py                   # ported near-verbatim (AverageMeter)
      logging.py                  # ported near-verbatim (Logger)
      serialization.py            # ported near-verbatim (save/load_checkpoint, copy_state_dict)
      data/
        preprocessor.py           # ported (img, fname, pid, camid[, index] tuples)
        sampler.py                # ported (RandomMultipleGallerySampler or equivalent PK sampler)
        transforms.py             # ported
    datasets/
      __init__.py
      market1501.py               # ported dataset-list builder: flat (img_path, pid, camid) tuples, no masks
      dukemtmc.py                 # same
    evaluation_metrics/
      __init__.py
      cmc.py, ranking.py          # ported near-verbatim (needed by evaluators.py)
  setup.py
  README.md
```

Everything under `utils/`, `datasets/`, `evaluation_metrics/`, `meters.py`, `logging.py`,
`serialization.py` is a near-verbatim, generic port from `spcl/utils/*` / `spcl/datasets/*` /
`spcl/evaluation_metrics/*` — small, dependency-light, and not SpCL-strategy-specific (they're
plumbing: checkpoint I/O, PK sampling, image transforms, CMC/mAP math). Deliberately **not** reusing
BPBReID's own `torchreid/data/datasets/image/market1501.py`/`dukemtmcreid.py`, since those are
masks-aware (`ImageDataset` base class with `masks_path` resolution) and Milestone 1's whole premise
is that masks aren't needed for the forward pass — SpCL's own simpler flat-tuple dataset convention
is the right fit here and keeps PCR decoupled from BPBReID's dataset/mask directory layout.

## Milestone 1 — single-branch drop-in (validate the ported SpCL strategy end-to-end)

Goal: confirm the ported baseline `SpCLTrainer_UDA`/`HybridMemory`/DSBN/traditional-Jaccard pipeline
works correctly with a BPBReID backbone producing one embedding branch (`bn_foreg`), before adding
the part/visibility generalization. This isolates "did the port work" from "does part-based help."

1. **Source-pretrain a BPBReID checkpoint**, reusing BPBReID's own training pipeline completely
   unmodified (external to this repo): `python torchreid/scripts/main.py --config-file
   configs/bpbreid/bpbreid_market1501_train.yaml` (or `bpbreid_dukemtmc_train.yaml`, whichever is
   the chosen source domain). If PifPaf masks aren't available for whichever Market1501/DukeMTMC
   copy will be used, set `loss.part_based.weights.pixls.ce=0` via CLI opts so
   `BodyPartAttentionLoss` is skipped and attention trains unsupervised, purely from ID+triplet
   gradient — a supported BPBReID config, no new code needed for this step.

2. **`pcr/models/bpbreid_encoder.py`**: a minimal plain-Python `model_cfg` object (dataclass, not
   yacs) exposing just the ~13 attributes `BPBreID.__init__`/`forward` actually read
   (`masks.parts_num`, `shared_parts_id_classifier`, `test_use_target_segmentation='none'`,
   `training_binary_visibility_score`, `testing_binary_visibility_score`, `backbone`, `last_stride`,
   `dim_reduce`, `dim_reduce_output`, `hrnet_pretrained_path`, `learnable_attention_enabled=True`,
   `normalization='identity'`, `pooling='gwap'`). `class BPBReIDEncoder(nn.Module)` constructs
   `torchreid.models.bpbreid.BPBreID` with a placeholder `num_classes`, loads the Milestone-1-step-1
   checkpoint via `torchreid.utils.load_pretrained_weights` inside its own constructor.
   `forward(self, images) -> Tensor[B,D]`: runs the wrapped model, takes the `bn_foreg` branch from
   the returned `embeddings` dict, L2-normalizes — matches SpCL's `extract_cnn_feature`'s plain-tensor
   expectation exactly, so it's a drop-in for `models.create(...)`. Requires `pip install -e
   /home/lakshh/workspace/reid/bpbreid` into pcr2's env so `torchreid` is importable (no name
   collision — this repo's package is `pcr`).

3. **`pcr/models/__init__.py`**: register `'bpbreid'` in a factory dict (`create(name, **kwargs)`),
   same pattern SpCL itself uses for backbone registration.

4. **`pcr/models/dsbn.py`**: port `convert_dsbn`/`DSBN2d`/`DSBN1d` from `spcl/models/dsbn.py`
   unchanged — fully generic (recursively replaces any `BatchNorm1d`/`BatchNorm2d` in a module tree
   with a source/target-split twin), applies cleanly to BPBReID's backbone + pooling-head BN layers.
   Splits the batch exactly in half by convention (source first, target second) — the training loop
   must always feed 1:1 source:target batches, matching baseline SpCL.

5. **`pcr/models/hm.py`**: port `HM`/`hm`/`HybridMemory` from `spcl/models/hm.py` unchanged for
   Milestone 1 (single `[num_samples, D]` buffer, per-slot momentum update in the autograd
   `backward`, masked-softmax + NLL loss in `forward`) — no part/visibility generalization yet.

6. **`pcr/trainers.py::PCRTrainer_UDA`**: port `SpCLTrainer_UDA` unchanged (reshape-for-DSBN
   device-count trick, joint forward pass, `loss = memory(f_out_s, s_targets) +
   memory(f_out_t, t_indexes + source_classes)`).

7. **`pcr/utils/jaccard_rerank.py::compute_jaccard_distance(features: Tensor[N,D], k1, k2) ->
   ndarray[N,N]`**: port the plain (non-camera-aware) k-reciprocal Jaccard re-ranking algorithm from
   `spcl/utils/faiss_rerank.py:28-136`. For Milestone 1, drop the FAISS GPU kNN search
   (`search_raw_array_pytorch`) in favor of a plain `torch.cdist`/matmul top-k over the (small,
   single-domain, few-thousand-image) target set — this removes the `faiss-gpu` dependency entirely
   from the new repo (not needed at this scale) and, more importantly, is written from the start in
   the shape Milestone 2 needs: internally build `initial_rank` from a `[N,N]` base distance matrix
   (computed here as `2 - 2 * features @ features.T`), so that in Milestone 2 the base distance can
   simply be swapped for a precomputed part-combined one instead of patching two separate raw-vector
   call sites later.

8. **`examples/train_uda.py`**: port `main_worker` from `examples/spcl_train_uda.py:115-323`
   unchanged in structure — `create_model` (encoder + `convert_dsbn` + cuda + DataParallel),
   `HybridMemory(D, source_classes + len(target.train))`, source-center / target-instance
   initialization via a full-dataset extraction pass, per-epoch: `compute_jaccard_distance` on
   `memory.features[source_classes:]` → three-way DBSCAN (`eps`/`eps±eps_gap`) → R_indep/R_comp
   self-paced reliability filter → `memory.labels` rebuild → `PCRTrainer_UDA.train(...)`. CLI
   `--dataset-source`/`--dataset-target` accepting `market1501`/`dukemtmc-reid` (symmetric — either
   direction is just a flag swap; only the Milestone-1-step-1 checkpoint needs regenerating per
   chosen source domain).

**Verify M1**: `--setup-only`-style dry run first (confirm shapes: memory size =
`source_classes + len(target.train)`, encoder output dim matches `dim_reduce_output`), then a short
smoke run (small `--iters`, few epochs, check loss doesn't NaN and mAP computes), then full runs in
both directions. Compare target-domain mAP/CMC against SpCL's own `resnet_ibn50a` baseline on
`spcl_train_uda.py` as a sanity reference point (not an apples-to-apples target, since backbone and
codebase differ, but useful to confirm the port isn't badly broken).

## Milestone 2 — true part-based PCR (visibility-gated, all new files; M1 stays intact as a fallback)

9. **`pcr/utils/part_distance.py::compute_bpb_pairwise_distance(qf: Tensor[N,M,D], vis: Tensor[N,M],
   dist_combine_strat='mean') -> Tensor[N,N]`**: port from `torchreid/metrics/distance.py:87-247`
   (`compute_distance_matrix_using_bp_features` + `_compute_body_parts_dist_matrices`) together with
   `masked_mean`/`replace_values` from `torchreid/utils/tensortools.py` — pure PyTorch, no cfg
   dependency, ports directly. Per-part distance `[M,N,N]` combined via visibility-gated masked-mean
   (bool visibility: outer-product mask; continuous: sqrt-product soft gate), with a
   "max-observed-distance + 1" sentinel for pairs with zero mutual visibility.

10. **`BPBReIDEncoder.forward_parts(self, images) -> (Tensor[B,M,D], Tensor[B,M])`**: new method on
    the Milestone-1 adapter, stacking BPBReID's default `test_embeddings` branches (`bn_foreg` +
    `bn_parts`, M = 1+K) and their visibility scores.

11. **`pcr/models/hm.py::PartHM`/`PartHybridMemory`**: generalize `HM`/`HybridMemory` to
    `features`/`visibility` buffers of shape `[num_samples, M, D]` / `[num_samples, M]`.
    - `forward`: per-branch similarity `inputs[:,m,:] @ features[:,m,:].T`, combined into one
      `[B, num_samples]` similarity via a visibility-gated weighted average across branches (reuse
      `masked_mean`-style logic from step 9's port) — then the *rest* of `HybridMemory.forward`
      (masked-softmax over cluster/class slots, NLL loss) is unchanged, since only the
      similarity-computation step changes shape.
    - `backward` (momentum update): update `features[y, m]` per branch only for samples where part
      `m` is visible in that forward call (skip the update for invisible parts, rather than blending
      in a zero/garbage embedding), then L2-renormalize each updated branch independently.
    - Source-center init (in `examples/train_uda.py`'s Milestone-2 variant): per-branch mean over
      member samples' embeddings, L2-normalized per branch, visibility-weighted (an invisible part
      on a given sample contributes 0 to that branch's mean, matching GiLt's own visibility-gating
      convention in this repo's `torchreid/losses/GiLt_loss.py`).

12. **`pcr/utils/jaccard_rerank.py`**: no structural change needed beyond step 7's design —
    Milestone 2's driver computes `base_dist = compute_bpb_pairwise_distance(target_features,
    target_visibility)` (step 9) and passes it straight into the same `compute_jaccard_distance`
    entry point in place of the raw `2 - 2 * features @ features.T` computation, since step 7 was
    already written to consume a precomputed `[N,N]` base distance internally.

13. **`pcr/evaluators.py::Evaluator`**: Milestone 1's evaluator can be the plain SpCL-style one
    (flat `[D]` vectors via `bn_foreg`, ported near-verbatim from `spcl/evaluators.py`). Milestone
    2 needs its own `extract_features_parts` (calling `forward_parts`) feeding
    `compute_bpb_pairwise_distance` (step 9) for the final query-gallery distance matrix at
    evaluation time — keep both variants in `evaluators.py`, selected by which milestone's driver is
    running.

**Verify M2**: standalone numerical check first — `compute_bpb_pairwise_distance` (step 9) should
match `torchreid.metrics.distance.compute_distance_matrix_using_bp_features` on a small synthetic
batch (it's a direct port, should be identical). Then dry-run/shape check, short smoke run, full runs
in both directions. Compare final target-domain mAP/CMC across three points to isolate where any gain
comes from: (a) SpCL's own baseline (`resnet_ibn50a` via `spcl_train_uda.py`), (b) Milestone 1
(BPBReID single branch under the ported baseline strategy), (c) Milestone 2 (full part-based
memory + part-based Jaccard clustering).

## Deferred / explicitly out of scope for this build
- CA-Jaccard (camera-aware clustering) and any camera-proxy/camera-aware memory component —
  traditional Jaccard only, per current instruction.
- Part-based triplet loss variants (`torchreid/losses/part_*_triplet_loss.py`) — Milestone 2 relies
  solely on the generalized `PartHybridMemory` InfoNCE signal, matching baseline SpCL's own reliance
  on `HybridMemory` alone (no auxiliary triplet term in the baseline strategy).
- EMA teacher-student model — baseline SpCL trains a single model; the memory bank itself serves as
  the running feature cache (`memory.features[source_classes:]`), no separate teacher network needed.

## Critical source files to port from / reference
- SpCL baseline (`/home/lakshh/workspace/reid/SpCL`): `examples/spcl_train_uda.py`,
  `spcl/trainers.py::SpCLTrainer_UDA`, `spcl/models/hm.py`, `spcl/models/dsbn.py`,
  `spcl/utils/faiss_rerank.py:28-136`, `spcl/evaluators.py`, `spcl/utils/data/preprocessor.py`,
  `spcl/datasets/market1501.py`, `spcl/datasets/dukemtmc.py`.
- BPBReID (this repo): `torchreid/models/bpbreid.py`, `torchreid/metrics/distance.py`,
  `torchreid/utils/tensortools.py`, `torchreid/losses/GiLt_loss.py` (visibility-gating convention
  reference only, not reused directly), `configs/bpbreid/bpbreid_market1501_train.yaml`,
  `configs/bpbreid/bpbreid_dukemtmc_train.yaml`.
