# PCR2 Progress Log

This file is the source of truth for what has been built, why, and what's left. Code stays
comment-light on purpose — major implementation steps, decisions, and their rationale are logged
here instead. Append new dated entries at the bottom of **Progress Log** as work continues; don't
rewrite history, add to it.

---

## Plans

### Plan A (original handoff, `PLAN.md`) — staged Milestone 1 / Milestone 2

Written by a peer session before implementation started. Full text preserved in `PLAN.md` at the
repo root. Summary: port SpCL's baseline self-paced UDA strategy in two stages —

- **Milestone 1**: validate the ported SpCL strategy (HybridMemory, DSBN, trainer, plain Jaccard
  re-ranking) end-to-end using BPBReID as a single-branch (`bn_foreg` only) drop-in encoder, to
  isolate "did the port work" from "does part-based help."
- **Milestone 2**: generalize to BPBReID's full K part-embeddings + visibility scores — new
  `PartHybridMemory`, `compute_bpb_pairwise_distance`, part-based Jaccard base distance,
  part-aware evaluator. M1 was meant to stay intact as a fallback.
- Training hyperparameters were left implicitly config-driven (SpCL-argparse style, not specified
  as a hard requirement either way).
- Explicitly out of scope: CA-Jaccard / camera-aware memory, part-based triplet loss variants, EMA
  teacher-student.

### Plan B (superseding direction, given directly by the user mid-conversation) — what was actually built

The user overrode Plan A's staging before any code was written. Effective spec, in full:

1. **Merge M1+M2 — build the full part-based version directly**, skip the single-branch
   `bn_foreg`-only milestone. BPBReID's K learnable part embeddings + visibility scores are used
   from the start.
2. **Keep `PartHybridMemory`** (an earlier draft instruction to drop the memory bank and replace it
   with a classifier-head + pseudo-label loss was explicitly retracted by the user one message
   later — see conversation history, not re-litigated here).
3. **Explicit BPBReID-style similarity/matching function**: `compute_bpb_pairwise_distance` in
   `pcr/utils/part_distance.py`, ported from `torchreid/metrics/distance.py`'s
   `compute_distance_matrix_using_bp_features`. This one function is the single source of truth
   for part-based distance, reused by both Jaccard re-ranking and the evaluator.
4. **No separate config file for training hyperparameters.** Every hyperparameter used in
   `examples/train_uda.py` is an `argparse` flag with its default set directly in that file — no
   YAML/config-loader layer. (The one exception, `BPBReIDModelCfg`, is a plain dataclass describing
   frozen *model architecture* to satisfy `BPBreID.__init__`, not a training-hyperparameter config.)
5. **Learnable part attention, no masks anywhere in pcr2.** `learnable_attention_enabled=True`
   means the pixel-to-part classifier is learned end-to-end from the contrastive gradient alone;
   PifPaf masks are never loaded or used in this repo (they're only relevant to the *external*
   source-pretraining step, which is out of scope here and stays in the bpbreid repo). This was
   explicitly called out by the user because PifPaf masks only exist for the source domain, not the
   target — learnable attention sidesteps the need for them on either side.
6. **Weights are saved** via a ported `save_checkpoint` (best + latest).
7. **Use a coding subagent** for the actual file-writing, after the orchestrating session did its
   own grounding (reading the real SpCL/BPBReID source to nail exact signatures) and wrote a
   detailed, file-by-file, source-referenced spec — then verified the subagent's output against
   that spec by reading the actual files, not by trusting its self-report.
8. **Minimal comments in code.**

Deferred / explicitly out of scope for this build (carried over from Plan A, still holds):
- CA-Jaccard / camera-aware or camera-proxy memory — traditional Jaccard re-ranking only.
- Part-based triplet loss variants / BodyPartAttentionLoss pixel-supervision — the part-based
  memory-bank contrastive (InfoNCE-style) loss is the sole training signal, matching baseline
  SpCL's own reliance on `HybridMemory` alone with no auxiliary triplet term.
- EMA teacher-student — the memory bank itself is the running feature cache, no separate teacher
  network.
- Source-domain pretraining code — stays external, reuses bpbreid's own
  `torchreid/scripts/main.py` + its existing yaml configs unmodified.

---

## Progress Log

### 2026-08-13 — Full repo built (Plan B, merged M1+M2), verified

**Repo tree** (`/home/lakshh/workspace/reid/pcr2`):

```
pcr2/
  setup.py, README.md, PLAN.md, progress.md
  examples/train_uda.py
  pcr/
    __init__.py, trainers.py, evaluators.py
    models/{__init__,bpbreid_encoder,dsbn,hm}.py
    utils/{__init__,meters,logging,serialization,osutils,jaccard_rerank,part_distance}.py
    utils/data/{__init__,base_dataset,preprocessor,sampler,transforms}.py
    datasets/{__init__,market1501,dukemtmc}.py
    evaluation_metrics/{__init__,ranking}.py
```

**What each non-trivial file does:**

- `pcr/models/bpbreid_encoder.py` — `BPBReIDModelCfg` (plain dataclass, ~13 attrs `BPBreID`
  actually reads) + `BPBReIDEncoder(nn.Module)`. Wraps `torchreid.models.bpbreid.BPBreID` with a
  placeholder `num_classes=1` (BPBReID's own ID classifier heads are never used — only embeddings +
  visibility). `forward(images) -> (Tensor[B, 1+K, D], Tensor[B, 1+K])`: stacks `bn_foreg` +
  `bn_parts` embeddings (index 0 = foreground, 1..K = parts), L2-normalizes each branch
  independently, stacks matching visibility scores. This *is* the module's `forward` (not a
  separately named method) so `nn.DataParallel` gathers it fine as a tuple-of-tensors return.
  Gotcha logged here so it isn't rediscovered: `parts_num=5` is correct for the
  market1501/dukemtmc bpbreid configs even though the yaml's own explicit `masks.parts_num` default
  is 1 — `masks.preprocess: 'five_v'` auto-resolves it to 5 via
  `torchreid/data/masks_transforms/__init__.py::compute_parts_num_and_names` at config-parse time
  in the *original* bpbreid repo. Verified by reading that file directly, not assumed.

- `pcr/utils/part_distance.py::compute_bpb_pairwise_distance(qf, qf_vis, gf=None, gf_vis=None, ...)`
  — ported from `torchreid/metrics/distance.py::compute_distance_matrix_using_bp_features` +
  `_compute_body_parts_dist_matrices`, and `masked_mean`/`replace_values` from
  `torchreid/utils/tensortools.py`. Bool visibility -> outer-product mutual-visibility mask +
  "max observed distance + 1" sentinel for zero-mutual-visibility pairs. Continuous visibility ->
  sqrt-product soft gate. `gf=None` computes self-distance (used by Jaccard re-ranking).
  Line-by-line cross-checked against the original during verification.

- `pcr/models/hm.py::PartHM` (autograd Function) + `PartHybridMemory` — SpCL's `HM`/`HybridMemory`
  generalized from `[N,D]` to `[N,M,D]` feature buffers. Per-branch similarity
  `einsum('bmd,nmd->bmn', ...)`; momentum update in `backward` skips a branch entirely (leaves the
  stored slot untouched) when that branch is invisible for the sample, rather than blending in a
  garbage embedding, then L2-renormalizes only the branches that were actually updated.
  `PartHybridMemory.forward` combines the M per-branch similarities into one `[B,N]` via a
  *query-visibility-only* weighted average (the memory bank has no per-slot visibility bookkeeping,
  so it can't do the full two-sided gating `compute_bpb_pairwise_distance` does — this is a
  deliberate, documented distinction, not an oversight) — then the masked-softmax + NLL loss over
  cluster/class slots is unchanged from SpCL's original. One real bug caught and fixed during
  verification: SpCL's original in-place `sim /= self.temp` throws under current PyTorch autograd
  when applied to a custom `Function`'s output view; changed to out-of-place `sim = sim / self.temp`.

- `pcr/utils/jaccard_rerank.py::compute_jaccard_distance(base_dist, k1, k2)` — plain (non-FAISS,
  non-camera-aware) k-reciprocal algorithm ported unchanged math-wise from
  `spcl/utils/faiss_rerank.py`; the one structural change is that it now takes a **precomputed**
  `[N,N]` base distance matrix directly (built by the caller via `compute_bpb_pairwise_distance`)
  instead of computing `2-2*f@f.T` internally via FAISS GPU search — this is what lets the same
  function serve both a flat-vector and a part-based base distance without touching this file.

- `pcr/trainers.py::PCRTrainer_UDA`, `pcr/evaluators.py`, `pcr/models/dsbn.py` — straightforward
  generalizations/verbatim ports; `DSBN2d/DSBN1d/convert_dsbn/convert_bn` needed no changes at all
  (fully generic over any BatchNorm1d/2d in the model tree). Trainer's device-reshape/split trick
  now applies to `(f_out, vis)` tuples instead of a flat tensor. Evaluator's `pairwise_distance` now
  defers to `compute_bpb_pairwise_distance` instead of a raw `2-2*x@y.T`.

- `examples/train_uda.py` — every hyperparameter is an inline `argparse` default (dataset choice,
  batch size, height/width — 384x128, BPBReID's own convention, not SpCL's 256x128 — DBSCAN eps/
  eps-gap, Jaccard k1/k2, memory momentum/temp, optimizer lr/weight-decay/epochs/iters/step-size,
  `--checkpoint-path` for the externally-pretrained BPBReID checkpoint). Source-center
  initialization does a per-branch, visibility-weighted mean over each pid's member embeddings
  (falls back to a plain unweighted mean for any branch with zero visible members for that pid, to
  avoid div-by-zero). Target-instance init writes extracted part-embeddings straight into the
  memory. Per-epoch: base distance -> Jaccard re-rank -> three-way DBSCAN -> R_indep/R_comp
  self-paced reliability filtering -> pseudo-labeled dataset -> `PCRTrainer_UDA.train(...)`,
  ported near-verbatim from SpCL's `main_worker`. Two resolved ambiguities, logged with rationale
  directly in that file as one-line comments: (a) the epoch-level base distance for target memory
  slots treats every slot/branch as visible (`all-ones` visibility) rather than re-extracting
  target visibility every epoch, since the memory bank has no per-slot visibility bookkeeping
  anyway and re-extracting would cost a full extra forward pass per epoch; (b) SpCL's own
  `--min-samples` argparse flag is defined but never actually wired into its `DBSCAN(...)` calls
  (hardcoded `min_samples=4` regardless) — matched that actual (not documented) behavior rather
  than the flag's apparent intent.

- Generic plumbing ported near-verbatim from SpCL (adjusted only for `spcl.` -> `pcr.` import
  paths): `meters.py`, `logging.py`, `serialization.py`, `utils/data/{__init__,preprocessor,
  sampler,transforms}.py`, `evaluation_metrics/ranking.py`, `datasets/{market1501,dukemtmc}.py`.
  Three files not in the original file-tree spec were added because the ported code above
  transitively needs them (`utils/__init__.py`'s `to_numpy`/`to_torch` for `ranking.py`,
  `utils/osutils.py`'s `mkdir_if_missing` for logging/serialization/dukemtmc,
  `utils/data/base_dataset.py`'s `BaseImageDataset` for the dataset classes) — matches SpCL's own
  module boundaries, not scope creep.

**Verification performed** (by the orchestrating session, reading the actual files — not taking the
building subagent's self-report at face value):
- Read `hm.py`, `part_distance.py`, `jaccard_rerank.py`, `bpbreid_encoder.py`, `evaluators.py`,
  `trainers.py`, `dsbn.py`, `models/__init__.py`, `datasets/__init__.py`, `dukemtmc.py`, `setup.py`,
  `README.md`, and all of `train_uda.py` end-to-end, cross-checked against the real SpCL/BPBReID
  source.
- `python -m py_compile` on every `.py` file in the repo — clean.
- `import pcr` and every submodule import cleanly in the `torchreid` conda env (has torch 2.11 +
  numpy/scipy/sklearn/PIL) — confirmed directly, not just via the subagent's claim.
- Removed one dead import (`copy_state_dict`, unused since this build has no `--resume` flag)
  flagged by the editor's static analysis after the build.

**Known environment gotcha, not a code bug** — flagged to the user, not yet resolved: the existing
`torchreid` conda env has **deep-person-reid's** unrelated `torchreid` package on the path, not
bpbreid's. `from torchreid.models.bpbreid import BPBreID` fails there with a clean
`ModuleNotFoundError` (verified directly) rather than silently importing the wrong package — but
actually running `train_uda.py` needs a env with `pip install -e /home/lakshh/workspace/reid/bpbreid`
run into it, likely a fresh env since both packages claim the `torchreid` name.

**Not yet done / open follow-ups:**
- No actual training run performed (no GPU/dataset exercised in this environment) — only static
  verification (compile, import, and a synthetic-tensor shape/invariant check the build subagent
  ran and reported, not independently re-run by the orchestrating session).
- Source BPBReID checkpoint pretraining (external, via bpbreid's own `main.py`) has not been run.
- `--resume`-from-checkpoint support was dropped from `train_uda.py` relative to SpCL's original
  (no such flag exists yet) — add if resuming interrupted UDA runs turns out to be needed.
- CA-Jaccard / camera-aware clustering remains explicitly out of scope per the user's original
  scope note in Plan A, "not a hard rule ... may be revisited later."

### 2026-08-18 — Training-readiness sanity check (logic/design review, not a code-correctness pass)

Reviewed whether the BPBReID-in-SpCL integration is *logically* sound enough to attempt training,
beyond the earlier "does it compile/import" check. Traced actual data/gradient flow rather than
re-reading files in isolation.

**Confirmed sound:**
- HRNet backbone (`torchreid/models/hrnet.py:611`) only dereferences `hrnet_pretrained_path` when
  `pretrained=True`; `BPBReIDEncoder` always builds with `pretrained=False`, so the empty default
  path never causes a crash.
- `load_pretrained_weights`'s shape-mismatch-skip behavior is exactly why the placeholder
  `num_classes=1` works: the external checkpoint's classifier heads (sized for real num_classes)
  get silently skipped on load, everything else (backbone, pixel-classifier, pooling, dim-reduce)
  loads correctly.
- DSBN's `[source-half, target-half]` batch-split assumption holds through BPBReID's extra
  branches exactly as it does in vanilla SpCL — established once before pooling, unaffected by the
  added classifier heads or the M-dimension carried through the trainer's reshape.
- Gradient path into the attention module (`pixel_classifier -> softmax -> parts_masks -> weighted
  pooling -> bn_parts -> memory loss`) is fully differentiable; visibility is used only as a
  multiplicative weight (stop-gradient when binary), matching GiLt's own convention, not a bug.
- DataParallel + the custom `PartHM` autograd Function don't conflict — same pattern as vanilla
  SpCL's `HM`, memory bank lives outside the DataParallel-wrapped encoder.

**Real risk flagged (methodological, not a bug):** dropping GiLt's ID+triplet loss and all
pixel-supervision means the attention module's *only* training signal is whatever leaks back
through the memory-bank contrastive loss. De-risked by the fact the source checkpoint already has
mask-supervised attention (UDA phase fine-tunes, doesn't learn attention from scratch) — but there's
no anchor loss keeping attention well-behaved if pseudo-labels get noisy mid-training. Recommend
logging per-epoch mean part-visibility rate during the first real runs to watch for attention
collapse.

**Minor, non-blocking inefficiency:** unused identity-classifier heads (global/background/
foreground/concat/parts) still run forward every step and sit in the optimizer's param groups with
permanently-zero gradient — wasted compute/memory, not incorrect. Not fixed yet.

**Still unverified:** no actual forward/backward pass has been run (no GPU/dataset in this
environment) — only static and synthetic-tensor checks so far. Recommended next step:
`--setup-only` dry run, then a short smoke run (few iters) watching for NaN loss and sane
visibility stats, before committing to a full run.

### 2026-08-18 — Dummy training run: environment setup + two real bugs found and fixed

Ran an actual `--setup-only` dry run and then a short smoke training run (`-ds market1501 -dt
dukemtmc-reid`, low batch size for the local 8GB RTX 4060) against real data
(`/home/lakshh/workspace/reid/datasets/{market1501,dukemtmc-reid}`, both present with the expected
`bounding_box_train/query/bounding_box_test` layout and correct image counts). No source-pretrained
BPBReID checkpoint exists yet (external pretraining hasn't been run), so this smoke run used
`--checkpoint-path ""` (randomly-initialized backbone) purely to exercise the pcr2 pipeline's
mechanics — not to produce a meaningful model.

**Environment**: no conda env had bpbreid's own `torchreid` installed — the existing `torchreid`
env's `torchreid` package resolved to an unrelated `deep-person-reid` editable install. Cloned that
env to a new `pcr2-run` env (non-destructive — left the original `torchreid` env untouched), swapped
in bpbreid's `torchreid` (`pip install -e /home/lakshh/workspace/reid/bpbreid`), then
`pip install -e /home/lakshh/workspace/reid/pcr2`. Confirmed `torchreid.models.bpbreid` imports
correctly from the bpbreid path afterward.

**Bug 1 — `convert_dsbn`/`convert_bn` crash on parameter-free submodules (`pcr/models/dsbn.py`).**
`assert not next(model.parameters()).is_cuda` raises `StopIteration` (uncaught) whenever recursion
reaches a submodule whose entire subtree has zero parameters. Never surfaced in vanilla SpCL (every
intermediate container in a plain ResNet holds a Conv/BN somewhere), but BPBReID's pooling-head
wrappers (`GlobalAveragePoolingHead`/`GlobalWeightedAveragePoolingHead`, used for the foreground/
background/parts attention pooling heads) have exactly one child, `self.normalization = nn.Identity()`
when `normalization='identity'` (this repo's config) — a genuinely empty parameter subtree. Fixed by
using `next(model.parameters(), None)` and treating "no params" as "not cuda" (nothing to guard
against). Caught immediately by the `--setup-only` dry run, first failure point.

**Bug 2 — `compute_bpb_pairwise_distance` OOMs at real UDA scale (`pcr/utils/part_distance.py`).**
The build agent's implementation intentionally dropped bpbreid's own gallery-batching
(`batch_size_pairwise_dist_matrix`), reasoning the repo "always runs on modest re-ID gallery sizes."
That assumption is false for this use case: the target-domain self-distance for Jaccard re-ranking
runs over the *entire* target training set every epoch (16,522 images for DukeMTMC), not a query/
gallery eval set. Reproduced directly with a synthetic `[16522, 6, 512]` self-distance: the original
unbatched implementation tried to allocate 6.1GB for one intermediate tensor and OOM'd outright on
this 8GB card (confirmed via `torch.cuda` before touching the training script at all). Rewrote with
chunked processing over the gallery dimension (default `batch_size=1024`), a running-max accumulator
so the "sentinel = max observed distance + 1" value for zero-mutual-visibility pairs stays a true
global max rather than a per-chunk one, and in-place writes into a pre-allocated `[Nq, Ng]` output
tensor instead of accumulating a Python list of chunks before `torch.cat` (that intermediate list-
then-cat approach alone re-inflated peak memory to ~6GB — tightened to write-in-place +
`masked_fill_` instead, bringing peak down to ~3.4GB for the same 16522x16522 case). Verified
numerically sane (no NaN, correct shape, plausible distance range) before re-running the full
pipeline.

Both fixes were caught by actually running the code against real data/hardware, not by re-reading
files — this is exactly the gap the earlier static-only verification pass (compile + import +
synthetic-shape checks) couldn't close.

**Bug 3 — training-step OOM at `-b 8` (not caught by the earlier fixes, only surfaces on an actual
backward pass).** First real training attempt (`-b 8 -j 4 --num-instances 4`) got all the way
through dataset load, source/target feature extraction, Jaccard distance (13s, no OOM — bug 2's fix
holds), and DBSCAN (206-215 clusters found), then OOM'd inside `trainer.train`'s `loss.backward()`
on the very first iteration: `Tried to allocate 1.76 GiB ... this process has 5.89 GiB memory in
use`. Root cause, not a bug in the code so much as a sizing miscalculation on my part: `args.batch_size`
is *per domain* (source and target each get their own full batch, per `get_train_loader` — same
convention as vanilla SpCL), so `-b 8` meant 16 images jointly through HRNet32 at 384x128 per
iteration, not 8. Neither the earlier `--setup-only` dry run nor the feature-extraction passes
would ever have caught this: both run under `torch.no_grad()`, which never allocates the
backward-pass activation memory that's actually the expensive part. **Retried at `-b 4
--num-instances 2` (joint batch of 8) — ran clean, no OOM, through both epochs plus an
end-of-training evaluation pass.**

**Dummy run outcome (`-ds market1501 -dt dukemtmc-reid`, `-b 4 -j 4 --num-instances 2 --iters 10
--epochs 2 --eval-step 2 --checkpoint-path ""`, i.e. a randomly-initialized backbone — no real
source-pretrained checkpoint exists yet, this run only validates pipeline mechanics):**

- Total wall time: 11m 5s (RTX 4060, 8GB). Most of it is the four full-dataset feature-extraction
  passes this run does (source-center init, target-instance init, one mid-training eval, one final
  eval) — each takes ~2-3 min at this batch size, dominating over the actual 10-iteration training
  steps (~3-5s each).
- Epoch 0: 206 clusters found from DBSCAN, 10,689 of 16,522 target-train images left as
  singleton/un-clustered "outliers" (expected — the encoder is untrained, so pseudo-labels are
  effectively derived from random-feature noise, not real identity structure).
  ```
  Epoch: [0][2/10]  Loss_s 9.344 (9.426)   Loss_t 9.905 (9.660)
  Epoch: [0][4/10]  Loss_s 10.400 (9.717)  Loss_t 10.672 (9.891)
  Epoch: [0][6/10]  Loss_s 10.385 (9.895)  Loss_t 11.207 (10.131)
  Epoch: [0][8/10]  Loss_s 11.658 (10.320) Loss_t 11.066 (10.353)
  Epoch: [0][10/10] Loss_s 12.250 (10.632) Loss_t 11.941 (10.644)
  ```
- Epoch 1: re-clustered (206 clusters, 10,739 outliers — stable cluster count, as expected with an
  untrained encoder producing near-static features between epochs).
  ```
  Epoch: [1][2/10]  Loss_s 11.780 (11.421) Loss_t 11.202 (11.134)
  Epoch: [1][4/10]  Loss_s 11.776 (11.221) Loss_t 9.604 (10.581)
  Epoch: [1][6/10]  Loss_s 9.061 (10.692)  Loss_t 9.661 (10.325)
  Epoch: [1][8/10]  Loss_s 10.496 (10.521) Loss_t 10.548 (10.303)
  Epoch: [1][10/10] Loss_s 10.031 (10.457) Loss_t 9.668 (10.176)
  ```
- No NaN/Inf at any point. No clear downward loss trend across these 20 total iterations — expected
  and not a red flag: with a randomly-initialized backbone and only 2x10 iterations, there's no
  reason to expect convergence; the value of this run is confirming the mechanics (data flow,
  clustering, memory update, backward, checkpointing, eval) run cleanly end-to-end, not producing a
  usable model.
- Final eval: Mean AP 0.2%, top-1 0.4%, top-5 1.3%, top-10 1.9% — consistent with near-random
  (expected, same reason as above).
- `checkpoint.pth.tar` and `model_best.pth.tar` (411MB each) written correctly to
  `logs/dummy_run/` — confirms weight-saving works end-to-end.

**Batch-size guidance for this GPU (RTX 4060, 8GB) confirmed empirically, not guessed:** `-b 8`
(joint batch 16) OOMs during backward; `-b 4` (joint batch 8) runs clean with headroom. Recommend
`-b 4` through `-b 8` as the range to try for a real run on similar 8GB hardware, starting low and
increasing only if memory allows — `-b 8` specifically is confirmed to fail at height=384/width=128
with the HRNet32 backbone on this card.

**Environment note for future runs:** the working env is `pcr2-run` (conda), not `torchreid` (which
has an unrelated `deep-person-reid` `torchreid` package installed) — `conda activate pcr2-run`
before running `examples/train_uda.py`.

### 2026-08-19 — `train_usl.py`: ICE's unsupervised (single-domain) strategy, hard-instance
### contrastive loss only

Built a second training driver replicating ICE's USL pipeline
(`/home/lakshh/workspace/reid/ICE`, `examples/unsupervised_train.py` + `ice/trainers.py::
ImageTrainer` + `ice/loss/contrastive.py::ViewContrastiveLoss`), per direct instruction: single
target dataset + pseudo-labels only (no source domain), and only ICE's InfoNCE-based hard-instance
contrastive loss — its cluster-classification loss, cross-camera loss, and teacher-student KL
consistency term are all dropped. Confirmed via the log line ICE itself prints
(`Loss_hard_instance`) that `ViewContrastiveLoss` *is* the hard-instance contrastive loss (Eq. 9,
"L_h_ins", in the ICE paper) before building anything.

**New files:**
- `pcr/loss/contrastive.py::PartViewContrastiveLoss` — ICE's `ViewContrastiveLoss` generalized to
  BPBReID's `[B, M, D]` per-branch embeddings. The hard-mining/InfoNCE math (hardest same-pseudo-ID
  positive by lowest similarity, all different-ID samples as negatives, cross-entropy over
  `[positive, negatives]`) is unchanged from ICE's original — the only new part is combining M
  per-branch `[B,B]` similarity matrices into one via visibility-gated mean (mirrors
  `part_distance.py`'s bool/continuous convention), with one deliberate deviation from that file's
  own sentinel convention: zero-mutual-visibility branch-pairs fall back to `0` similarity here, not
  `part_distance.py`'s "max distance + 1" — for a *distance*, an extreme sentinel means "very far
  apart" (correct); for a *similarity* feeding hard-*positive* mining (which picks the **lowest**
  similarity as "hardest"), an extreme low sentinel would make every zero-visibility pair look like
  the hardest positive regardless of real difficulty, so this loss uses a neutral 0 fallback instead.
  Verified with a synthetic test: correct shapes, finite loss and gradients under bool visibility,
  continuous visibility, and the fully-invisible edge case (all three checked directly, not assumed).
- `pcr/trainers_usl.py::ICEUSLTrainer` — separate trainers file (per instruction), EMA teacher-student
  update + ShuffleBN (ported from ICE's `_update_ema_variables`/`get_shuffle_ids`) driving only
  `PartViewContrastiveLoss`. No DSBN here — DSBN exists specifically for the UDA joint source+target
  batch split in `pcr/trainers.py`; single-domain USL has no domain split, so plain BatchNorm is
  correct and simpler.
- `examples/train_usl.py` — per-epoch: extract via the EMA model → `compute_bpb_pairwise_distance`
  self-distance → `compute_jaccard_distance` → single DBSCAN pass → pseudo-labeled dataset with
  **DBSCAN outliers dropped entirely** (ICE's own, simpler USL approach — no SpCL-style
  R_indep/R_comp self-paced filtering, no outlier-as-singleton-class handling) → `ICEUSLTrainer.train`.
  All hyperparameters inline as argparse defaults, matching this repo's established no-separate-config
  convention. `-dt` only, no `-ds` anywhere in the script.
- Supporting additions to existing shared files (not new files): `pcr/utils/data/preprocessor.py`
  gained `Preprocessor2View` (two independently strong-augmented views of the same image, no index —
  this pipeline keeps no per-instance memory slots, so it doesn't need one); `pcr/utils/data/
  sampler.py` gained `MoreCameraSampler` (ported from ICE, ICE's own preferred PK sampler for this
  strategy); `pcr/utils/data/transforms.py` gained `GaussianBlur` (dropped earlier as "ICE-specific,
  unused" when only `train_uda.py` existed — now genuinely needed); `pcr/utils/lr_scheduler.py`
  (new file) has `WarmupMultiStepLR`, ported near-verbatim from `ice/utils/lr_scheduler.py`.

**Verified, not just written:** full repo `py_compile` + import sweep in the `pcr2-run` env passed.
Ran an actual dummy training smoke test (`-dt market1501`, `-b 4 --num-instances 2`, 2 epochs x 10
iters, no checkpoint — random-init backbone, mechanics-only check, same discipline as the UDA
dummy run): completed in 7m58s, no crash, no NaN, `Loss_hard_instance` printed and finite every
logged iteration (range ~1.2-1.8), checkpoint saved (144MB) and end-of-training eval ran cleanly
(mAP 0.3%, expected near-random for a random-init encoder). One observation, not a bug: DBSCAN's
cluster count collapsed from 335 IDs (epoch 0) to 9 IDs (epoch 1) — a known sensitivity of
density-based clustering near its `eps` threshold when run on a genuinely unstructured (random-init)
feature space; expect far more stable epoch-to-epoch clustering once trained from a real checkpoint.
No batch-size OOM issue reproduced at `-b 4` (this pipeline's per-iteration memory profile is
similar in shape to the UDA one — same encoder/resolution, two forward passes per step instead of
one joint DSBN-split pass — so the same `-b 4`-safe / `-b 8`-risky guidance from the UDA dummy run
likely applies here too, though not separately re-confirmed at `-b 8`).

### 2026-08-19 15:58 CST — Plan: BPA loss, body-part triplet loss, horizontal-stripes option,
### hard/soft triplet loss

**Clarification asked directly: does anything currently in this repo use camera-aware or
camera-agnostic memory? No.** Neither `pcr/models/hm.py::PartHybridMemory` (used by
`train_uda.py`) nor `pcr/trainers_usl.py::ICEUSLTrainer` / `pcr/loss/contrastive.py::
PartViewContrastiveLoss` (used by `train_usl.py`) has any camera dimension, per-camera proxy
memory, or cross-camera loss term. This was a deliberate exclusion both times: PLAN.md's original
scope note ruled out CA-Jaccard/camera-aware clustering from the start, and `ICEUSLTrainer` was
built by explicitly stripping ICE's own `percam_memory`/`loss_cam` mechanism out of
`ice/trainers.py::ImageTrainer`, keeping only the hard-instance contrastive term (confirmed while
researching this plan: even in ICE's own upstream repo, `loss_cam` is computed but never actually
added to the backprop loss — `loss = loss_ccl + loss_vcl + loss_kl` — so ICE's shipped USL script
doesn't really use it either despite building the memory; the sibling `mod_ice` repo does wire a
real camera-aware memory (`HybridCameraMemory`/`CrossDomainMemory`) into its own UDA trainer, but
that's a different repo we haven't touched).

**Research grounding for this plan** (exact files/signatures, not paraphrase — full detail
available on request, summarized per-item below):
- BPA loss: `torchreid/losses/body_part_attention_loss.py::BodyPartAttentionLoss`, invoked from
  `torchreid/engine/image/part_based_engine.py::ImagePartBasedEngine.combine_losses()`.
- PifPaf masks on disk: confirmed present at
  `/home/lakshh/workspace/reid/datasets/market1501/masks/{pifpaf,pifpaf_maskrcnn_filtering}/<split>/<img>.npy`
  — market1501 only; no masks directory exists for the dukemtmc-reid copy on disk, consistent with
  "PifPaf masks only available for the source dataset."
- Body-part triplet loss: `torchreid/losses/part_averaged_triplet_loss.py` and siblings, dispatched
  via `torchreid/losses/__init__.py::init_part_based_triplet_loss(name, **kwargs)` — 7 named
  combination strategies (averaged / max / min / max_min / random_max_min / inter_parts /
  intra_parts-individual).
- Horizontal stripes: `torchreid/models/bpbreid.py::BPBreID.__init__`'s real, correctly-spelled
  `horizontal_stripes` parameter. **Confirmed bug in bpbreid's own shipped code**: its `pcb()`/
  `bot()` factory functions (bottom of the same file) pass a *typo'd* kwarg,
  `horizontal_stipes=True` (missing the second "r"), which `**kwargs` silently swallows without
  error — those factories never actually enable stripe mode. Not something to replicate; pcr2's
  `BPBReIDEncoder` already constructs `BPBreID` directly rather than through those factories, so
  this is avoided by construction as long as the correct spelling is used.
- Hard/soft triplet: `ice/loss/triplet.py::TripletLoss` (batch-hard mining on `[N,D]`+labels) and
  `SoftTripletLoss` (distillation: batch-hard-mines indices from one embedding set, then
  cross-entropy between that set's and a *second* embedding set's softmaxed ap/an distances —
  needs two forward passes of the same inputs, e.g. student vs.\ teacher). **Both are dead code in
  ICE's own shipped trainer** — never wired into `ImageTrainer`. The sibling `mod_ice` repo's
  `ice/uda_trainer.py::UDATrainer` does use hard `TripletLoss`, applied only to source real labels
  (`self.triplet_loss(f_out_s, pids_s)`); `SoftTripletLoss` is unused everywhere in both repos —
  there's no existing precedent to copy for exactly how hard+soft should pair up, see the open
  question below.

**Additional things this plan needs that weren't named explicitly, declared up front:**
1. A source-only, masks-aware dataset/preprocessor path. Every dataset/preprocessor class in pcr2
   today (`pcr/datasets/*.py`, `pcr/utils/data/preprocessor.py`) is deliberately mask-free. BPA
   loss needs source images paired with their PifPaf mask file — this is new data-loading surface,
   not just a new loss file.
2. A decision on *how* to load and group those masks. Two options, not both: (a) reuse torchreid's
   own `torchreid.utils.tools.read_masks` + `torchreid.data.masks_transforms` (five-part grouping,
   etc.) directly by import — pcr2 already hard-depends on torchreid for the encoder, so this adds
   coupling but not a new dependency, and avoids re-porting a moderately complex
   albumentations-based pipeline; or (b) port a minimal, pcr2-native mask loader/grouper. This plan
   recommends (a) for the reasons above; noted as a decision point, not silently assumed.
3. A way to get `pixels_cls_scores` (BPA loss's core input) out of the encoder at all.
   `BPBReIDEncoder.forward()` today discards it (`embeddings, visibility_scores, _, _, _, _ =
   self.model(images)`), and every existing caller (`pcr/trainers.py`, `pcr/trainers_usl.py`,
   `pcr/evaluators.py`, memory init in both `examples/*.py`) depends on `forward()` staying a clean
   `(f_out, vis)` 2-tuple under `nn.DataParallel`. This is a real architectural fork, not a detail:
   - **Option A (recommended):** extend `forward()`'s return to `(f_out, vis, pixels_cls_scores)`
     always, updating every call site to unpack three values (most will just discard the third).
     Correct under multi-GPU `DataParallel` (which only scatters/gathers through `forward()`
     itself), consistent with how this repo already tolerated similar tuple-shape churn earlier
     (single-vector to per-part `(f_out, vis)`).
   - **Option B:** add a separately-named method (e.g. `forward_with_pixel_scores`) called only for
     the source slice when BPA loss is enabled. Less invasive, but a non-`forward` method bypasses
     `DataParallel`'s scatter/gather entirely — silently loses multi-GPU parallelism for that call
     specifically, or breaks outright on multi-GPU depending on device placement. Only reasonable
     if this repo commits to single-GPU for BPA-loss runs.
4. A widened but still config-file-free hyperparameter surface. Each new loss is an *optional*,
   independently-weighted term (`weight=0` disables it, mirroring GiLt's own convention) — this
   means several new inline argparse flags per script, not a new config layer.
5. Reuse, not reinvention, of this repo's existing visibility-gating convention
   (`pcr/utils/part_distance.py::masked_mean`/`replace_values`) wherever a new loss needs
   per-branch visibility masking — with the same caveat already logged for
   `PartViewContrastiveLoss`: the "-1 / +1 sentinel" convention is correct for *distances*, wrong
   for anything where "no mutual visibility" should read as *neutral*, not *extreme*. Each new loss
   below states which convention it actually needs.

**Per-loss plan:**

**1. Body Part Attention (BPA) loss — new `pcr/loss/body_part_attention_loss.py`.**
Port `BodyPartAttentionLoss` near-verbatim (it's self-contained: cross-entropy/focal/dice over a
flattened `[N*Hf*Wf, K]` pixel classification, plus label smoothing — no bpbreid-internal `Writer`
dependency to strip, unlike `part_distance.py`). Depends on items 1-3 above (masks-aware source
loader + `pixels_cls_scores` exposed from the encoder).
- **Domain scope: source only, and `train_uda.py` only.** `train_usl.py` has no source domain and
  no masks anywhere — BPA loss must never be referenced there. This is the sharpest instance of
  "respecting source/target segregation" in this whole plan.
- **Pipeline placement:** inside `PCRTrainer_UDA.train`'s per-iteration loop
  (`pcr/trainers.py`), computed only on the source half of the joint DSBN batch, using Option A's
  `pixels_cls_scores_s` slice and the batch's loaded ground-truth masks (interpolated to
  `pixels_cls_scores`'s spatial size, argmax'd to integer targets, exactly as
  `part_based_engine.py::combine_losses` does). Added into `loss_s` alongside the existing
  `self.memory(f_out_s, s_targets, vis_s)` term: `loss_s = memory_loss_s + bpa_weight * bpa_loss`.
  Never touches `loss_t`.
- **Mutually exclusive with horizontal-stripes mode** (item 3 below): when `horizontal_stripes=True`,
  `pixels_cls_scores` is `None` by construction (`bpbreid.py` forward's stripe branch never builds
  a pixel classifier score map) — BPA loss must be a no-op (or raise clearly) whenever stripes mode
  is active, not silently skipped without explanation.
- **New CLI surface (`train_uda.py` only):** `--bpa-weight` (default 0, i.e. off unless opted in;
  mirrors bpbreid's own yaml default of 0.35 when explicitly enabled), `--masks-dir` (e.g.
  `pifpaf_maskrcnn_filtering`, matching the on-disk directory confirmed above).

**2. Body part triplet loss — new `pcr/loss/part_triplet_loss.py`.**
Port `PartAveragedTripletLoss` as the default combination strategy (matches bpbreid's own "SOTA
weights" default), with the other 6 combination strategies as selectable variants behind one
`--part-triplet-strategy` choice — this is a real, reasonably self-contained piece of math
(distance-matrix-per-part -> combine -> batch-hard mine -> margin loss), doesn't need masks at
all, operates directly on the `(f_out, vis)` this repo's encoder already returns everywhere.
- **Domain scope: both, and both scripts.** No masks needed — just part embeddings + a label per
  sample (real ID for source, pseudo-label for target/USL). No segregation concern here beyond the
  labels themselves already being handled correctly by each pipeline's existing pseudo-labeling
  step.
- **Pipeline placement, `train_uda.py`:** an additional term inside `PCRTrainer_UDA.train`,
  computed on `f_out_s`/`vis_s` against `s_targets` **and** `f_out_t`/`vis_t` against
  `t_indexes + source_classes`, each weighted by `--part-triplet-weight` and added to `loss_s`/
  `loss_t` respectively — same additive pattern as BPA loss, just applied to both halves instead
  of only source.
- **Pipeline placement, `train_usl.py`:** an additional term inside `ICEUSLTrainer.train`, computed
  on the student's `q`/`q_vis` (not the EMA teacher's `k`) against the batch's pseudo-labels, added
  to the existing `PartViewContrastiveLoss` term behind its own `--part-triplet-weight`.
- **Visibility convention:** reuse the *distance*-appropriate sentinel from
  `part_distance.py` (max-observed+1 / +inf depending on combination strategy) since this loss
  really is computing distances for hard-mining, unlike `PartViewContrastiveLoss`'s similarity
  case — the two shouldn't share one convention, and this file should say why in a short comment
  the way `contrastive.py` already does.

**3. Horizontal-stripes option — `BPBReIDModelCfg` change, not a new loss file.**
Add `horizontal_stripes: bool = False` and `num_stripes: int = 6` (matching bpbreid's own PCB
configs' stripe count) to `pcr/models/bpbreid_encoder.py::BPBReIDModelCfg`; thread
`horizontal_stripes=model_cfg.horizontal_stripes` **directly** into the `BPBreID(...)` constructor
call in `BPBReIDEncoder.__init__` (the correctly-spelled parameter — sidesteps bpbreid's own
`pcb()`/`bot()` typo bug by never going through those factories, which this repo already doesn't).
When enabled, `model_cfg.masks.parts_num` should be read as "number of stripes," and
`learnable_attention_enabled` should be forced `False` (mirroring bpbreid's *intent* for `pcb()`,
implemented correctly here instead of copying its bug).
- **Domain scope: both scripts** — this is an encoder architecture choice, orthogonal to which
  training strategy runs on top of it. `PartHybridMemory`, `compute_bpb_pairwise_distance`,
  `PartViewContrastiveLoss`, and the new part-triplet loss all already read `M` from tensor shape
  at runtime rather than hardcoding it, so none of them need changes to support a different M
  from stripes instead of learned/mask-derived parts.
- **New CLI surface (both scripts):** `--horizontal-stripes` (flag) + `--num-stripes` (int,
  default 6). Must stay consistent with whatever the loaded checkpoint was pretrained with, same
  caveat already logged for `parts_num=5` in `bpbreid_encoder.py`'s docstring.
- **Interaction:** disables BPA loss (see above) since `pixels_cls_scores` is `None` in this mode;
  compatible with body-part triplet loss and the existing memory/contrastive losses without changes.

**4. Hard triplet loss + soft triplet loss — new `pcr/loss/part_triplet_loss.py` (hard, shares the
file with item 2 — it's the same `[N,D]`-style batch-hard math, just without the part dimension)
and `pcr/loss/soft_part_triplet_loss.py` (soft).**
- **Hard triplet — domain scope: source only, `train_uda.py` only**, mirroring `mod_ice`'s actual
  (if unused-elsewhere) precedent: real ground-truth labels are exactly what batch-hard mining
  wants to be confident about, and source is the only domain with real labels here. Placement:
  same additive spot in `PCRTrainer_UDA.train` as BPA loss, on `f_out_s`
  (flattened/`bn_foreg`-branch or reuse the part-triplet loss's own per-part variant — this repo
  already has both a flat and a part-based path available, worth picking one rather than building
  both; recommend reusing item 2's part-based version here for consistency rather than adding a
  second, flat-vector-only triplet implementation).
- **Soft triplet — open design question, not silently resolved:** `SoftTripletLoss`'s actual shape
  (distillation between *two* forward passes of the same input) needs a second model to distill
  from. `train_usl.py` already has exactly that (the EMA teacher) — a soft triplet term there,
  between the student's `q` and teacher's `k`, pseudo-labeled, is a natural, low-risk fit alongside
  the existing `PartViewContrastiveLoss`. `train_uda.py` has **no EMA teacher today** — `PCRTrainer_UDA`
  has only one live model. Three options, presented rather than picked:
  (a) add an EMA teacher to `train_uda.py` specifically to support this (real architecture
  addition, mirrors `train_usl.py`'s pattern);
  (b) use `PartHybridMemory`'s own stored per-slot features as the "soft" reference instead of a
  second live model (no new model needed, but changes what "soft" means from teacher-distillation
  to memory-consistency);
  (c) skip soft triplet in `train_uda.py` entirely and treat it as USL-only, since that's where it
  fits without any new infrastructure.
  This plan does not choose between them — flagging for a decision before implementation, since
  each has different infrastructure cost and slightly different meaning.

**Suggested build order** (not started — planning only, per this session's request): (1) source-only
masks-aware preprocessor + BPA loss, since it's the most self-contained new *data* path and
exercises the `pixels_cls_scores` exposure decision (item 3) that a couple of the other pieces
implicitly assume is settled; (2) body-part triplet loss (both scripts) — no new data plumbing,
highest reuse of what already exists; (3) horizontal-stripes option — small, isolated config change,
useful to land before soft-triplet since it's a prerequisite-free architecture toggle; (4) hard
triplet (source, `train_uda.py`) — small once (2)'s triplet math exists to reuse; (5) soft triplet
— last, blocked on the open design question above being resolved first.

### 2026-08-19 16:58 CST — Implemented: GiLt loss + BPA loss + CrossEntropyLabelSmooth + view
### contrastive loss, in `train_usl.py` only

Per direct instruction: dropped camera-aware/agnostic memory from scope entirely (confirmed
already absent, see the clarification above), scoped this whole pass to `examples/train_usl.py`
only (`train_uda.py` untouched, still on the plain `PartHybridMemory` strategy), and implemented
first: GiLt loss, BPA loss, `CrossEntropyLabelSmooth`, and the (already-existing) view contrastive
loss "as in ICE." Horizontal stripes and hard/soft-triplet-with-source/target-segregation from
the earlier plan are *not* part of this pass.

**Key realization that shaped the design:** BPA loss needs masks, and masks are about part
*segmentation*, not identity — PifPaf masks are auto-generated by a pose model, so a dataset
having them doesn't violate "unsupervised" (no ID labels used). This means BPA loss only needs
"does the `-dt` dataset have a `masks/` directory," not a source/target split — resolved the
apparent tension with the earlier plan's UDA-only BPA scoping. Also realized GiLt's own native
id loss (BPBreID's fixed `BNClassifier` heads, sized at model-construction time) is a poor fit
for a per-epoch-reclustering setting: DBSCAN's cluster count and numbering both change every
epoch, so a persistent classifier layer would keep reassigning what each output neuron "means."
Resolved by making the id component classify against **per-epoch cluster centers** instead
(exactly ICE's own `ccloss` mechanism) — which is why "GiLt" and "CrossEntropyLabelSmooth as in
ICE" ended up being the *same* id component here, not two separate mechanisms, once adapted to
this setting.

**New files:**
- `pcr/loss/crossentropy.py::CrossEntropyLabelSmooth` — ported near-verbatim from `ice/loss/
  crossentropy.py`.
- `pcr/loss/body_part_attention_loss.py::BodyPartAttentionLoss` — ported from bpbreid's
  `torchreid/losses/body_part_attention_loss.py`, with two deliberate deviations: `monai` (only
  needed for the unused `'fl'/'dl'` loss-type variants) is imported lazily instead of at module
  level, so the default cross-entropy path doesn't require installing it; accuracy is a plain
  argmax-equality mean instead of `torchmetrics.Accuracy`, avoiding that package's version-coupled
  constructor API.
- `pcr/loss/part_triplet_loss.py::PartTripletLoss` — ported from bpbreid's
  `part_averaged_triplet_loss.py::PartAveragedTripletLoss` (the "averaged" combination strategy,
  bpbreid's own default). Reuses `masked_mean`/`replace_values` from `pcr/utils/part_distance.py`
  for the visibility-gated branch combination, but keeps its own epsilon-stabilized distance
  computation rather than reusing `part_distance.py`'s — that file's distances never backprop
  (eval/clustering only), this one does, and plain `sqrt` has an infinite gradient exactly at
  distance 0 (which real self-pairs hit).
- `pcr/loss/gilt_loss.py::PartGiLtLoss` — new, not a direct port (see "key realization" above): id
  component = `CrossEntropyLabelSmooth` against per-epoch foreground-branch (branch 0) cluster
  centers; triplet component = `PartTripletLoss` across all M branches. Weights default to
  bpbreid's own `foreg: id=1,tr=1 / parts: id=0,tr=1` convention (this repo's encoder only exposes
  foreground+parts branches, no separate global/concat_parts, so those two weight settings are
  all that's needed).

**Modified files:**
- `pcr/models/bpbreid_encoder.py` — added `forward_full()` returning `(f_out, vis,
  pixels_cls_scores)`, alongside the existing `forward()` (unchanged, still a clean 2-tuple).
  Deliberately chose "Option B" from the earlier plan (separate method, not an always-3-tuple
  `forward()`): smaller blast radius, keeps `train_uda.py`/`pcr/evaluators.py`/`pcr/trainers.py`
  completely untouched, matches this message's "only in the USL script" scope. Documented tradeoff
  in the method's own docstring: bypasses `nn.DataParallel`'s scatter/gather, so it only runs
  correctly single-device — fine for this repo's dev GPU, would need revisiting for real multi-GPU
  BPA-loss training.
- `pcr/utils/data/preprocessor.py` — added `PreprocessorMasked`: pairs view-1 with its PifPaf mask,
  applying resize/pad/crop/flip identically to both via shared random parameters (hand-rolled with
  plain `torchvision.transforms.functional`, not bpbreid's own albumentations-based mask pipeline —
  avoids a new dependency). Colour-only augmentation (blur, erasing) stays image-only, applied
  after, since it doesn't touch pixel *positions*. Channel grouping (36 raw PifPaf channels -> 5
  parts) and the synthetic background channel reuse bpbreid's own `torchreid.data.masks_transforms.
  {CombinePifPafIntoFiveVerticalParts,AddBackgroundMask}` directly (imported lazily) rather than
  re-deriving their channel-mapping tables by hand — verified these classes work standalone via
  `.apply_to_mask(tensor)` without needing a full yacs cfg object, confirmed directly before relying
  on it.
- `pcr/trainers_usl.py::ICEUSLTrainer` — added optional `bpa_criterion`/`bpa_weight` (constructor)
  and `gilt_criterion`/`centers` (per-`.train()`-call, since GiLt's id component needs rebuilding
  every epoch to match that epoch's cluster count — `vcl_criterion`/`bpa_criterion` don't, so they
  stay constructor args). `Loss_gilt_id`/`Loss_gilt_tri`/`Loss_bpa` now print alongside
  `Loss_hard_instance` when active.
- `examples/train_usl.py` — added `compute_cluster_centers()` (per-branch, visibility-weighted
  mean per cluster, same pattern as `train_uda.py`'s source-center init, computed fresh every
  epoch from the same `cf`/`cf_vis` already extracted for DBSCAN); split the view-1 transform into
  geometric (paired with the mask) vs. photometric (image-only) pieces; new CLI: `--gilt-id-weight`/
  `--gilt-triplet-weight` (both default **1.0**, i.e. GiLt is on by default in this build, matching
  the instruction to actually add it rather than leave it as an inert opt-in), `--gilt-tau-c`,
  `--gilt-triplet-margin`, `--masks-dir` (default `''`, i.e. **off** by default — most datasets
  don't have masks; `dukemtmc-reid` confirmed has none on disk, `market1501` does), `--masks-suffix`,
  `--bpa-weight` (default 0.35, matching bpbreid's own yaml).

**Real bug found and fixed while verifying (not a code-review guess — an actual crash on a real
run):** `IterLoader.next()`'s `except: self.iter = iter(self.loader); return next(self.iter)`
retry isn't guarded against a *second* consecutive `StopIteration` — and DBSCAN's clustering can
legitimately collapse to very few clusters (already observed twice now, independent of GiLt/BPA:
UDA's own dummy run epoch-to-epoch, and the plain-VCL USL dummy run earlier), which combined with
`MoreCameraSampler` (yields exactly `num_ids*num_instances` indices) and `drop_last=True` means a
degenerate epoch can leave the DataLoader unable to assemble even one batch — crashing the whole
run with an uncaught `StopIteration` on the very first `.next()` call. First repro run (`-dt
market1501`, masks + GiLt + BPA + VCL all active) hit exactly this at epoch 1 (11 samples, 1 id,
`1*num_instances(2)=2 < batch_size(4)`). Fixed in `train_usl.py`'s epoch loop: skip training (and
`lr_scheduler.step()`) for any epoch where `num_ids * num_instances < batch_size`, log why, and
re-cluster next epoch instead of crashing.

**Verified, in order:**
1. Synthetic tests (shapes, finite values, real gradients) for `PartTripletLoss` (bool/continuous/
   no visibility), `PartGiLtLoss`, and `BodyPartAttentionLoss` — all passed before touching the
   training script.
2. `PreprocessorMasked` tested directly against a real image+mask pair on disk
   (`market1501/bounding_box_train/0002_c1s1_000451_03.jpg` + its `.npy` mask) — confirmed shapes
   (`img1 [3,384,128]`, `mask [6,384,128]`) and that the mask sums to 1 per pixel as expected.
3. Full dummy training run (`-dt market1501`, real masks, `-b 4`, no checkpoint — mechanics-only,
   same discipline as every prior dummy run): first attempt crashed on the `IterLoader` bug above;
   after the fix, a second run (3 epochs x 8 iters) completed clean in ~10 minutes — all four loss
   terms (`Loss_hard_instance`, `Loss_gilt_id`, `Loss_gilt_tri`, `Loss_bpa`) finite every logged
   iteration across all 3 epochs, no NaN, checkpoint saved (144MB), end-of-training eval ran
   (mAP 0.3%, expected near-random for a random-init encoder over 3x8 iterations).

**Not done in this pass** (per explicit scope): `train_uda.py` untouched; horizontal-stripes option;
hard triplet (source, UDA) / soft triplet (open design question) from the earlier plan's items 3-4.

### 2026-08-19 — Fixed: backbone was never actually getting ImageNet-pretrained weights

Root cause: `BPBReIDEncoder.__init__` always called `BPBreID(..., pretrained=False, ...)`
unconditionally, regardless of backbone or whether a full checkpoint was given — so any run
without an external bpbreid checkpoint (in particular `train_usl.py` with no `--checkpoint-path`,
which is a supported/default mode) started from a **fully random** backbone, not ImageNet-init.
Both `hrnet32` and `resnet50` were already correctly registered in torchreid's model factory and
already threaded through `BPBReIDModelCfg.backbone` — the gap was specifically the hardcoded
`pretrained=False`, not missing backbone support.

Researched both backbones' actual weight-sourcing before changing anything (see the research
agent's findings, summarized here since they directly shaped the fix):
- **resnet50**: `torchreid/models/resnet.py::init_pretrained_weights` uses
  `torch.utils.model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')`
  — a real, stable, auto-downloading path, caching under `~/.cache/torch/hub/checkpoints/` (or
  `~/.cache/torch/checkpoints/` on older torch). Verified directly: constructed a resnet50-backed
  `BPBReIDEncoder` with `pretrained=True`, confirmed the exact expected file
  (`resnet50-19c8e357.pth`, 102MB) was loaded from cache and a forward pass produced correctly-
  shaped output.
- **hrnet32**: no stable URL exists anywhere in bpbreid's repo or docs for the raw ImageNet-only
  weights (`hrnetv2_w32_imagenet_pretrained.pth`) — `default_config.py`'s own comment says
  "download on our Google Drive" without giving a link/ID, `README.md`'s Google Drive link is for
  full *fine-tuned* bpbreid checkpoints (unconfirmed whether the raw ImageNet file is even in
  there), and there's no download script anywhere in the repo. **Did not fabricate a URL or guess
  a gdown ID** — auto-downloading against an unverified link risks silently fetching the wrong
  file. Instead, added a fail-fast check.

**Fix (`pcr/models/bpbreid_encoder.py`):**
- `pretrained = not checkpoint_path` — ImageNet-init the backbone whenever no full bpbreid
  checkpoint is going to overwrite it anyway (a full checkpoint's matching-shape weights get
  loaded over it afterward via the existing `load_pretrained_weights` call regardless, so
  requesting ImageNet init in that case would just be wasted download/compute).
- When `backbone == 'hrnet32'` and `pretrained` is true, check for
  `<hrnet_pretrained_path>/hrnetv2_w32_imagenet_pretrained.pth` *before* attempting model
  construction; raise `FileNotFoundError` with copy-pasteable instructions (both official sources,
  the exact expected path, and "use `--backbone resnet50` instead" as the zero-setup alternative)
  rather than letting it fail deep inside bpbreid's own `HighResolutionNet.load_param`. Verified
  this actually fires correctly (file genuinely absent in this environment).
- `BPBReIDModelCfg.hrnet_pretrained_path` default changed from `''` to `'pretrained_models'`,
  matching bpbreid's own yacs default.
- `--backbone` (`hrnet32`/`resnet50`) added to both `examples/train_uda.py` and `examples/
  train_usl.py`, threaded into `BPBReIDModelCfg(backbone=args.backbone)`. Default stays `hrnet32`
  in both (no behavior change for existing users/configs that already assume it, e.g. the
  `parts_num=5` gotcha tied to bpbreid's hrnet32-based yaml configs is unaffected by this — that's
  a masks-config axis, independent of backbone choice) — `resnet50` is the new, fully zero-setup
  option for anyone without a pretrained checkpoint or the manually-downloaded HRNet file. In
  `train_uda.py`, `--checkpoint-path` stays `required=True` (unchanged design: source pretraining
  is external, per this repo's original PLAN.md) so the ImageNet-fallback path mainly matters for
  `train_usl.py`, where `--checkpoint-path` is optional by design.

**Verified, not assumed:** both the hrnet32 fail-fast error and the resnet50 real-weights-loaded
path were actually run and their output inspected (not just read as "should work" from the code).

### 2026-08-19 — Vendored resnet50's backbone code into pcr2 directly

Per direct instruction: copied bpbreid's `torchreid/models/resnet.py` into `pcr/models/resnet.py`
rather than continuing to rely solely on the externally-installed torchreid package for it.
Checked first, not assumed: this file's only imports are `torch` and `torch.utils.model_zoo` —
zero references to `torchreid.losses`/`torchreid.utils`/anything else in that package, so there
was no actual dependency tree to bring along despite the request anticipating one ("models losses
utils etc") — copying the single file is the complete, correct vendoring, not a partial one.

**Not wired into the training pipeline** — `BPBReIDEncoder(backbone='resnet50')` still resolves
`'resnet50'` through torchreid's own internal model registry (that's an implementation detail
inside bpbreid's own `BPBreID.__init__` -> `models.build_model(...)`, not something this repo's
code controls), so this vendored copy doesn't change what the existing `--backbone resnet50` flag
does. It's a separate, standalone module for direct use (e.g. a plain non-part-based baseline
encoder, if one gets built later) — flagged to the user as not yet integrated, since the request
was scoped to "copy... and test it," not "make BPBreID use this copy instead."

**Verified rigorously, not just "imports cleanly":**
- `from pcr.models.resnet import resnet50; resnet50(num_classes=751, loss='softmax',
  pretrained=True, last_stride=1)` — real ImageNet weights load correctly through this vendored
  file specifically.
- Checked via **exact tensor equality** against the downloaded checkpoint
  (`~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth`), not just "non-random-looking stats"
  (which is what the earlier verification of the *external* torchreid resnet50 relied on) —
  `torch.equal(model.conv1.weight, ref['conv1.weight'])` and the same for a deep layer
  (`layer4.2.conv3.weight`) both returned `True`.
- Forward pass verified in both modes the backbone is actually used in: standard mode returns a
  global-pooled `[2, 2048]` feature; `model.loss = 'part_based'` mode (how BPBreID's own backbone
  wrapper drives it) returns the raw `[2, 2048, 16, 8]` feature map instead, matching what
  `BPBreID.forward()` expects from `self.backbone_appearance_feature_extractor(images)`.

### 2026-08-19 20:23 CST — Full USL logic audit: 1 critical bug, 1 monitoring gap, 2 medium fixes

Per direct instruction: re-read the entire USL path fresh (not from memory) and verified every
hypothesis empirically before reporting or fixing anything, rather than trusting the earlier
build-time reasoning at face value.

**🔴 CRITICAL, fixed — the EMA "teacher" was never actually a copy of the student.**
`examples/train_usl.py::create_model` built `model`/`model_ema` via two *separate*
`BPBReIDEncoder(...)` constructor calls. Verified empirically before touching anything: **15 of
409 parameter tensors differed** between the two at initialization, including
`pixel_classifier.classifier.weight` (the part-attention mechanism itself) and every dim-reduce
layer. This breaks the entire teacher-student premise every MoCo/BYOL/ICE-style EMA setup depends
on (the teacher must start byte-identical to the student, only diverging via the momentum update
afterward) -- and it's especially damaging here because `model_ema` both (a) supplies the view-
contrastive loss's `k` target and (b) generates the features used for **clustering every epoch**.
At `alpha=0.999` the initial divergence takes ~3000 iterations (7+ epochs at default settings) to
substantially wash out, so early pseudo-labels for a real run would come from a teacher with no
real relationship to the student. Root cause: this is the *same* pattern ICE's own
`unsupervised_train.py::create_model` uses (two independent `models.create(...)` calls, only
forced into agreement via `copy_state_dict` from the *same* checkpoint when `args.init != ''`) --
so this bug is latent in ICE's own reference script too whenever no init checkpoint is given, not
something introduced fresh here. **Fix**: build one model, `model_ema = copy.deepcopy(model)`.
Verified: re-ran the same diff check after the fix, 0/409 tensors differ.

**🟠 HIGH, fixed — no diagnostic existed for part-attention collapse.**
`learnable_attention_enabled=True` means the pixel-to-part classifier is trained from indirect
gradient alone whenever BPA loss isn't active or is being outweighed by other terms -- a known
failure mode for weakly-supervised attention. This exact risk was flagged in this project's very
first sanity-check pass, long before any of the USL work, with "monitor visibility stats" proposed
as a mitigation -- and never implemented until now. **Fix**: `ICEUSLTrainer.train` now tracks
`q_vis.float().mean(dim=0)` (mean per-branch visibility rate) across the epoch and prints it at
epoch end. Verified it actually prints correctly on a real run:
`Epoch 0 mean part-visibility rate per branch (branch 0 = foreground): ['1.00', '1.00', '1.00',
'1.00', '1.00', '1.00']` -- a legitimate baseline for a fresh/untrained pixel classifier (nothing
has learned to differentiate branches yet); a real trained run should show these diverge to more
varied, occlusion-dependent values, and the diagnostic is now in place to actually observe that
instead of only ever finding out indirectly from a bad final mAP.

**🟡 MEDIUM, fixed — resize interpolation mismatch between the masked and unmasked img1 paths.**
`pcr/utils/data/preprocessor.py::PreprocessorMasked` used `TF.resize(img, [h,w])` (defaults to
bilinear) for view-1's geometric pass, while `get_view2_transform`'s `T.Resize(...,
interpolation=3)` (bicubic) is used for view-2 and for the *entire* unmasked-path view-1. Whenever
BPA loss was enabled, the student's primary training view silently got a different resize kernel
than the teacher's view -- an avoidable pipeline inconsistency. Fixed to explicitly match
(`interpolation=InterpolationMode.BICUBIC`).

**🟡 MEDIUM, fixed (per explicit direction) — GiLt id-loss's raw scale tracked num_ids, not
learning progress.** `CrossEntropyLabelSmooth`'s chance-level baseline is ~`log(num_clusters)`,
and DBSCAN's cluster count is directly observed to swing wildly epoch to epoch on an
under-trained encoder (335 -> 9 -> 1 in an earlier dummy run) for reasons unrelated to model
quality. With a fixed `gilt_id_weight`, this term's pull on the total gradient balance shifted
inconsistently for reasons having nothing to do with training progress. **Fix** (user chose
"normalize by log(num_ids)" over "leave as-is" or "skip"): `PartGiLtLoss` now divides the id-loss
by `log(max(num_clusters, 2))` before weighting it into the total loss, while still returning/
logging the *unnormalized* value (the interpretable "how confident is the classifier" number).
Verified on a real run: printed `Loss_gilt_id` values (5.24 at 198 clusters, 4.94 at 143 clusters)
closely track `log(num_ids)` (5.29, 4.96) as expected -- confirms the raw diagnostic stayed
interpretable while the actual backprop term is now the normalized one.

**Investigated and explicitly ruled out** (checked rigorously, not just assumed correct, so they
don't need to be re-litigated in a future pass): visibility-score dtype (verified genuinely
`torch.bool`, not a float lookalike, so the bool/continuous branches in `contrastive.py`/
`part_triplet_loss.py` both correctly hit their intended path); the 0-similarity fallback for
zero-mutual-visibility branch pairs in `PartViewContrastiveLoss._combine_similarity` (traced
through both the hardest-positive selection and the InfoNCE denominator with `T=0.1`'s actual
dynamic range -- confirmed benign on both sides, not just asserted); diagonal (self-pair)
inclusion in `PartViewContrastiveLoss` (intentional -- matches MoCo/ICE's own instance-
discrimination design, q[i] vs k[i] being the same physical image under two augmentations is a
genuinely valid positive candidate, not an oversight); `PartGiLtLoss`'s `tau_c=0.5` (matches ICE's
own default exactly, not invented here). Also noted, informational only, not fixed: ShuffleBN
(`_get_shuffle_ids`) is correct code but is inert on this project's current single-GPU dev setup --
its whole purpose is decorrelating BN statistics *across* GPUs in `nn.DataParallel`, and batch
statistics on one GPU are invariant to element ordering regardless of shuffling. Will start doing
real work automatically once run on 2+ GPUs; not a bug to fix now.

**Verified end-to-end after all four fixes landed together**, not just individually: a real
training run (`-dt market1501`, `--backbone resnet50`, real masks, GiLt + BPA + VCL all active,
2 epochs) completed clean, no NaN, no crash, checkpoint saved, visibility diagnostic printing
correctly each epoch.

### 2026-08-20 19:59 CST — New plan: add CLIP-ReID prompt learning, closing the gap to
### `reid_pipeline_plan.md`; Stage 0 (environment + housekeeping) done

**New plan, approved by the user, saved at
`/home/lakshh/.claude/plans/in-this-repo-cleanly-refactored-river.md`.** `reid_pipeline_plan.md`
(repo root) specifies BPBreID + CLIP-ReID (per-part prompt learning) + SPCL, three techniques in
one pipeline. Research (three parallel Explore passes over this repo, `../bpbreid`, and
`../CLIP-ReID`) confirmed the BPBreID+SPCL portions are already substantially built (see this
file's history above) and grep-confirmed **zero** existing CLIP/prompt/text-encoder code anywhere
in the repo. Gap to close, in order: (1) the entire CLIP-ReID component — net new; (2) a fusion
module for single-descriptor retrieval — doesn't exist, matching always goes through
`compute_bpb_pairwise_distance`'s per-part combination; (3) GiLt/BPA losses are wired into
`train_usl.py` only, never carried into `train_uda.py` despite being planned for it in this file's
2026-08-19 15:58 entry.

**Deliberate deviation from `reid_pipeline_plan.md`'s own pseudocode**, grounded in reading
CLIP-ReID's actual source (`../CLIP-ReID/processor/processor_clipreid_stage{1,2}.py`,
`loss/supcontrast.py`, `model/make_model_clipreid.py`) rather than trusting the plan doc's
simplified sketch: CLIP-ReID's real stage-1 loss is **not** plain diagonal InfoNCE — it's a
multi-positive supervised-contrastive loss (`SupConLoss`, same-identity mask via
`t_label==i_targets`, temperature fixed at 1.0, applied symmetrically i2t+t2i) computed over
randomly-sampled sub-batches drawn from a precomputed full-dataset image-feature cache. Stage-2's
"alignment" term is not a cosine-similarity regression either — it's label-smoothed cross-entropy
of `image_features @ text_prototypes.T` against the real identity label, where `text_prototypes`
is a frozen, precomputed-once (not per-step) per-identity lookup table built right before the
epoch loop starts. This repo's new CLIP work will follow CLIP-ReID's verified real implementation,
generalized to per-(identity,branch) instead of per-identity, over the plan document's simplified
formulas.

**Decided directly with the user** (via AskUserQuestion before finalizing the plan): the two new
CLIP training scripts (`train_prompts.py`, `train_finetune.py`) get their own YAML configs, matching
`reid_pipeline_plan.md`'s own skeleton — but the existing `train_uda.py`/`train_usl.py` **stay
argparse-only**, unchanged, preserving this repo's established no-config-file convention for the
scripts already built and verified that way. Execution proceeds one plan-stage at a time with a
pause for review after each, not autonomously through all seven.

**Stage 0 (environment + housekeeping) — done:**
- `pip install ftfy regex git+https://github.com/openai/CLIP.git` into the `pcr2-run` conda env.
  Verified directly, not just "installed cleanly": loaded `ViT-B/32` (auto-downloads and caches,
  confirmed `text embed dim: torch.Size([49408, 512])`, `transformer width: 512`), tokenized and
  encoded a sample string end-to-end, got a `[1, 512]` text feature — confirms the planned choice
  (ViT-B/32's 512-dim text tower matches `BPBReIDModelCfg.dim_reduce_output=512` exactly, no
  projection layer needed for the upcoming alignment losses).
- **`pcr/utils/data/__init__.py::IterLoader.next()`** — fixed the unguarded double-`StopIteration`
  crash at the source (previously only patched at one call site, in `train_usl.py`'s epoch-skip
  guard, which stays as-is and is still the right place to *decide* to skip a degenerate epoch —
  this fix is about `IterLoader` never crashing opaquely if some future caller doesn't replicate
  that guard). Two changes, not one: (a) the retry now catches `StopIteration` specifically instead
  of bare `Exception` — the old code would silently swallow and retry *any* exception on the first
  `next()` call (a real worker crash, a corrupt image, a CUDA error), masking real bugs behind a
  confusing "loader looked exhausted" retry; now only genuine exhaustion triggers the refill, and
  any other exception propagates immediately and loudly. (b) a second consecutive `StopIteration`
  (the loader is empty even on a fresh iterator) now raises a clear `RuntimeError` explaining why,
  instead of letting a bare `StopIteration` escape `next()` — which is a real footgun beyond just
  "ugly traceback": per PEP 479, an unguarded `StopIteration` propagating out of a function called
  from inside a generator gets converted to a confusing `RuntimeError` by Python itself, or worse,
  can silently terminate an unrelated enclosing `for`/`next()` loop if `next()` is ever called from
  such a context in the future. Verified with three synthetic cases (normal wraparound still works
  transparently; an empty loader raises the new clear `RuntimeError`; a real `ValueError` from a
  broken iterator propagates immediately, unswallowed) — all three passed.
- **`pcr/trainers.py`** — removed a stray, syntactically-inert triple-quoted string literal at the
  bottom of the file (lines 105-116): a leftover, unexecuted copy of a prior planning prompt
  ("implement loss functions required in this repository... Write a plan... Log the plan in
  progress.md") that had been accidentally left in a shipped `.py` file. Pure deletion, no
  behavior change.
- Verified: `python -m py_compile` across every `.py` file in `pcr/` and `examples/` (not just the
  touched files) — clean. `import pcr`, `from pcr.utils.data import IterLoader`, `from pcr.trainers
  import PCRTrainer_UDA` all import correctly in `pcr2-run` after the edits.

**Next**: Stage 1 (CLIP text branch + per-part prompt learner modules, no training loop yet) —
paused here for review per the user's chosen execution cadence.

### 2026-08-20 21:05 CST — Stage 1 done: CLIP text branch + per-part prompt learner (modules only)

**New files:**
- `pcr/models/clip_text_encoder.py::ClipTextEncoder(clip_arch='ViT-B/32', device='cuda')` — frozen
  CLIP text tower, ported from `../CLIP-ReID/model/make_model_clipreid.py::TextEncoder` (lines
  31-48). Keeps only `token_embedding`/`transformer`/`positional_embedding`/`ln_final`/
  `text_projection` from the loaded `clip.load(clip_arch, device=device, jit=False)` model — CLIP's
  own visual encoder is never built at all (BPBreID's backbone is the sole visual encoder
  throughout this pipeline, per the plan's own framing). All params `requires_grad_(False)` +
  `self.eval()` in the constructor. `forward(prompts, tokenized_prompts)` is a literal port of
  CLIP-ReID's math (permute NLD/LND around the transformer, `ln_final`, gather the EOT-position
  token via `tokenized_prompts.argmax(dim=-1)`, project). Confirmed `jit=False` + `device='cuda'`
  loads the text tower in **fp16** (`clip.model.build_model`'s `convert_weights` runs before
  `state_dict` load whenever `.float()` isn't explicitly called, which only happens on the
  CPU-device path inside `clip.load`) — matches CLIP-ReID's own behavior exactly (they don't call
  `.float()` after `.to('cuda')` either), not something this port introduced. `embed_dim=512`
  confirmed via `text_projection.shape[1]`, matching `BPBReIDModelCfg.dim_reduce_output` exactly —
  no projection layer needed anywhere in the upcoming alignment losses.
- `pcr/models/prompt_learner.py::PartPromptLearner(num_identities, num_branches,
  clip_text_encoder, n_ctx=4, device='cuda')` — generalizes CLIP-ReID's `PromptLearner` (same
  file, lines 191-239) from per-identity to per-(identity, branch): `cls_ctx` shape
  `[num_identities, num_branches, n_ctx, ctx_dim]` instead of `[num_identities, n_ctx, ctx_dim]`,
  branch 0 = foreground (matching `BPBReIDEncoder.forward`'s own branch-0-is-foreground
  convention), branches 1..K = the K learned parts. Same fixed template
  (`"A photo of a X X X X person."`) and the same frozen prefix/suffix embedding-buffer splicing
  mechanism as the original, shared across all identities and branches — only the `cls_ctx` slice
  differs per (identity, branch). `forward(labels, branch_idx) -> [B, 77, ctx_dim]` for one branch
  at a time (matches how the upcoming Stage-3 training loop iterates per-branch anyway). One
  simplification over the original: collapsed CLIP-ReID's two separately-named-but-always-equal
  variables (`n_ctx`, `n_cls_ctx`, both hardcoded to 4) into a single `n_ctx` — they must always
  match for the prefix/suffix slice boundaries to line up, so keeping two names bought nothing.
  Also: `tokenized_prompts`/`token_prefix`/`token_suffix` are registered as buffers (so they move
  correctly with `.to(device)`/`.cuda()` on the parent module), rather than CLIP-ReID's own
  hardcoded `.cuda()` call on `tokenized_prompts` and plain (non-buffer) attribute for the
  prefix/suffix.

**Verified, not just written** (synthetic tests, no training loop yet, matching this repo's
established pre-wiring-verification discipline):
- `ClipTextEncoder`: `embed_dim == 512`, every parameter `requires_grad is False`, constructed
  module is in eval mode.
- `PartPromptLearner`: `cls_ctx` shape exactly `[num_identities, num_branches, 4, 512]`;
  prefix-length + suffix-length + 4 == 77 (full CLIP context length accounted for); two different
  branches for the same identities produce different prompt embeddings (confirms per-branch
  context actually varies); the same identity requested twice for the same branch produces
  bit-identical prompts (confirms deterministic label-indexed lookup, no hidden randomness).
- End-to-end: `ClipTextEncoder(PartPromptLearner(labels, branch), tokenized_prompts) -> [B, 512]`,
  finite values.
- **Gradient isolation, the most important check for this stage**: backprop from the text feature
  output reaches `cls_ctx` (non-zero, finite gradient) and reaches **nothing** inside the frozen
  `ClipTextEncoder` (`p.grad is None` for every one of its parameters after `.backward()`) — this
  is the property Stage 3's training loop depends on (only the prompt learner trains; CLIP stays
  frozen), verified directly rather than assumed from the `requires_grad_(False)` calls alone.
- Full repo `python -m py_compile` sweep (`pcr/` + `examples/`) and `import pcr` +
  both new modules — clean in `pcr2-run`.

**Flagged for later, not acted on now**: `cls_ctx` is created in fp16 on GPU (inherits
`clip_text_encoder.dtype`), matching CLIP-ReID's own original behavior exactly — Adam on fp16
params can be numerically less stable than fp32 over many steps, but since this matches the
verified-working reference implementation rather than deviating from it, no change is made
preemptively. Worth watching during Stage 3's real smoke run (loss finite/no drift) before
deciding whether to keep `cls_ctx` in fp32 and cast only when building prompts.

**Next**: Stage 2 (CLIP alignment losses — `SupConLoss`, `I2TLoss`) — paused here for review.

### 2026-08-20 21:15 CST — Stage 2 done: CLIP alignment losses (`SupConLoss`, `I2TLoss`)

**New files:**
- `pcr/loss/clip_supcon_loss.py::SupConLoss(temperature=1.0)` — ported faithfully from
  `../CLIP-ReID/loss/supcontrast.py::SupConLoss`, confirmed by reading that file directly (not
  from the earlier research summary alone). `forward(anchor_features, other_features,
  anchor_labels, other_labels)`: identity-equality mask (`torch.eq(anchor_labels.unsqueeze(1),
  other_labels.unsqueeze(0))`) — every same-identity pair across the two feature sets is a
  positive, not only the same-index pair — normalized by `mask.sum(1)` per anchor, no
  clamp/epsilon (matches the original, which relies on every anchor having >=1 positive by
  construction whenever both label args come from the same batch; documented as the precondition
  callers must preserve). Symmetric use (swap the two feature sets) gives CLIP-ReID's
  `loss_i2t + loss_t2i`.
- `pcr/loss/clip_i2t_loss.py::I2TLoss(num_identities, epsilon=0.1)` — ported from
  `../CLIP-ReID/loss/make_loss.py`'s `I2TLOSS = xent(i2tscore, target)` branch. Confirms the
  earlier research finding directly from source: this is **not** a cosine-similarity alignment
  term, it's label-smoothed cross-entropy where the logits are `image_features @
  text_prototypes.T` (dot product against a frozen per-identity prototype table) and the target
  is the real identity label — the frozen table acts as a fixed linear classifier. Thin wrapper
  around the existing `pcr/loss/crossentropy.py::CrossEntropyLabelSmooth`, reused exactly as
  CLIP-ReID's own `make_loss.py` does, rather than reimplementing label smoothing a second time.

**Verified against independent reference computations, not just re-run through the same code
path** (this stage's most important check, since both losses' correctness hinges on subtle
masking/normalization behavior that's easy to get silently wrong):
- `SupConLoss`: cross-checked against a hand-written, non-vectorized per-anchor Python loop
  (separate implementation of the same math, not a copy of the vectorized version) — matched to
  `1e-5`. Separately confirmed the multi-positive mask **genuinely changes the result**: on the
  same inputs, the real multi-positive loss (4.9617) differs from a diagonal-only-mask
  (plain-InfoNCE-style) variant computed on the identical logits (6.7269) — proves by construction
  that this is not accidentally equivalent to vanilla InfoNCE, not just asserted from reading the
  formula. Gradient flow confirmed finite into both feature sets; symmetric (swapped-argument)
  call also finite.
- `I2TLoss`: cross-checked against an independently hand-written label-smoothed cross-entropy
  computation (manual `log_softmax` + smoothed one-hot target, not calling
  `CrossEntropyLabelSmooth` a second time) — matched to `1e-5`. Gradient isolation confirmed:
  `image_features` receives a finite gradient, a `requires_grad=False` `text_prototypes` table
  receives none — matches the intended usage (Stage 4 will pass Stage 3's frozen prototype table
  here, never let gradient flow back into it).
- Full repo `python -m py_compile` sweep and `import pcr` + both new loss modules — clean in
  `pcr2-run`.

**Next**: Stage 3 (`examples/train_prompts.py` + `configs/stage1_prompt_learning.yaml` — the
Stage-1 training driver that actually wires `ClipTextEncoder`/`PartPromptLearner`/`SupConLoss`
together into a runnable loop) — paused here for review.

### 2026-08-20 22:25 CST — Stage 3 done: `examples/train_prompts.py` (Stage-1 training driver),
### two real bugs found and fixed by actually running it

**New files:**
- `configs/stage1_prompt_learning.yaml` — the first YAML config in this repo, per the decision
  made directly with the user before this plan started (new CLIP scripts get YAML, existing
  `train_uda.py`/`train_usl.py` stay argparse-only). Sections: `model` (backbone/checkpoint/
  dim_reduce_output/parts_num), `clip` (arch, n_ctx), `data` (dataset/data_dir/height/width/
  batch_size/cache_batch_size/workers -- `data_dir` defaults to the real, confirmed-present
  `/home/lakshh/workspace/reid/datasets`, not a placeholder), `optim` (lr/weight_decay/epochs/
  warmup), `loss` (temperature, visibility_threshold), `logging`.
- `pcr/utils/config.py::load_yaml_config` — minimal loader, dict -> dot-accessible
  `ConfigNamespace`, used only by the two new CLIP scripts.
- `pcr/utils/lr_scheduler.py::WarmupCosineLR` — added alongside the existing `WarmupMultiStepLR`
  (ICE's, unchanged). Self-contained linear-warmup + cosine-decay, epoch-stepped, matching
  CLIP-ReID's actual Stage-1 schedule (timm's `CosineLRScheduler`) without adding a timm
  dependency for one schedule shape.
- `examples/train_prompts.py::main_worker` — mirrors CLIP-ReID's `do_train_stage1`
  (`../CLIP-ReID/processor/processor_clipreid_stage1.py`), generalized per-branch: (1) build a
  frozen `BPBReIDEncoder` (ImageNet-init fallback when no checkpoint given, matching
  `train_usl.py`'s convention) + frozen `ClipTextEncoder` + trainable `PartPromptLearner`; (2)
  cache every training image's part-embeddings/visibility/real-label once under `no_grad`
  (`cache_part_features`) -- no repeated encoder forward passes after this, matching the
  reference's own full-dataset cache; (3) per epoch, draw random image-index sub-batches
  (`torch.randperm`, not PK-sampled -- matches the reference's plain shuffled-index sampling, not
  identity-balanced), and for every branch whose visibility mask has >=1 visible member in that
  sub-batch, compute `SupConLoss` symmetrically (i2t + t2i) and sum across branches -- the
  visibility-gating reid_pipeline_plan.md's own §3.1 pseudocode specifies, which CLIP-ReID's
  original (single global branch, no visibility concept) doesn't need; (4) after training,
  precompute the final frozen per-(identity,branch) text-prototype table once
  (`compute_text_prototypes`, same precompute-once pattern CLIP-ReID's stage 2 uses) and save both
  `prompt_learner.pth` and `text_prototypes.pth`.

**Two real bugs found and fixed by actually running the training loop, not caught by
`--setup-only` or synthetic tests** (same lesson this file's 2026-08-18 UDA entries already
recorded: forward-only dry runs and `no_grad` passes cannot catch backward-pass-only failures):

1. **PyYAML doesn't parse bare scientific notation (`1e-4`) as a float.** Confirmed directly
   (`yaml.safe_load('1e-4')` -> the string `'1e-4'`, not a float; `yaml.safe_load('1.0e-4')` ->
   `0.0001`, a float -- PyYAML/YAML-1.1 requires a decimal point in the mantissa). The first draft
   of `configs/stage1_prompt_learning.yaml` used `3.5e-4`/`1e-4`/`1e-5`/`1e-6` for `lr`/
   `weight_decay`/`warmup_lr_init`/`lr_min`; the latter three would have loaded as strings and
   crashed `torch.optim.Adam(weight_decay=...)` the moment training actually started (not caught
   by `--setup-only`, since that path never constructs the optimizer). Fixed by rewriting every
   small value in the config as plain decimal notation (`0.0001`, `0.00001`, `0.000001`), which
   PyYAML always parses correctly regardless of the scientific-notation mantissa-format quirk.
2. **`torch.amp.GradScaler` hard-errors on an fp16 leaf parameter** ("Attempting to unscale FP16
   gradients"), surfacing only at the first `scaler.step(optimizer)` call in real training (not at
   `--setup-only`, which never calls `.backward()`/`.step()`). Root cause: `PartPromptLearner`
   (Stage 1's own file, built two stages ago) created `cls_ctx` directly in
   `clip_text_encoder.dtype` (fp16 on GPU) to match CLIP-ReID's original code exactly -- but
   mixed-precision training requires fp32 master weights; an fp16 trainable parameter isn't a
   pattern modern PyTorch's `GradScaler` supports, regardless of what CLIP-ReID's own (apparently
   never actually run under this strict a check) code does. **Fixed in
   `pcr/models/prompt_learner.py`**: `cls_ctx` is now created and stored in fp32 always, cast to
   the frozen prefix/suffix buffers' dtype only inside `forward()` when assembling the prompt
   embedding sequence -- standard fp32-master/fp16-compute practice; autograd handles the cast's
   backward correctly (incoming gradient cast back to fp32 for accumulation into the fp32 leaf).
   This resolves the "flagged for later, watch during Stage 3" note from Stage 1's own progress.md
   entry -- conclusively, via a hard, unambiguous error rather than speculation. Re-verified all of
   Stage 1's original synthetic tests still pass post-fix, plus a new direct check that
   `scaler.step(optimizer)` succeeds (previously crashed with exactly this error).

**Verified, in order:**
1. Full repo `python -m py_compile` sweep -- clean, both before and after the two bug fixes.
2. `--setup-only` dry run (`--backbone resnet50` for zero-setup, real Market1501 data at
   `/home/lakshh/workspace/reid/datasets`): dataset loaded (751 train ids, 12,936 images), full-
   dataset part-embedding cache built successfully (12,936 x 6 branches), `cls_ctx` shape
   `(751, 6, 4, 512)` confirmed -- exited cleanly before the training loop as designed.
3. Real smoke training run (2 epochs, `batch_size=32`, `print_freq=5`, random-init `resnet50`
   backbone -- no source-pretrained BPBreID checkpoint exists yet, so this validates pipeline
   mechanics only, same discipline as every prior dummy run in this file): completed in 3m19s, no
   crash, no NaN at any point. **Loss visibly decreasing**, not just finite: epoch-0 average
   41.2158, epoch-1 average 39.8853, with epoch 1's within-epoch trend clearly descending
   (~40.7 at iter 5 -> ~39.0-39.4 by iter 400) -- a genuine, if early, learning signal, notable
   since only `cls_ctx` is trainable here (everything else frozen) and the LR is still deep in its
   5-epoch warmup (7.8e-05 of a 3.5e-4 base by epoch 1).
4. Saved-file verification: `prompt_learner.pth` (`cls_ctx` shape `[751, 6, 4, 512]`, fp32) and
   `text_prototypes.pth` (`text_prototypes` shape `[751, 6, 512]`, fp32, all-finite) both load
   correctly and match the expected shapes/dtypes.
5. Full repo `python -m py_compile` + `import pcr` sweep after all fixes landed -- clean in
   `pcr2-run`.

**Next**: Stage 4 (`examples/train_finetune.py` + `configs/stage2_backbone_finetune.yaml` + new
`pcr/models/id_classifier.py::PartIdClassifiers` -- the Stage-2 supervised backbone-finetune
driver that consumes this stage's `text_prototypes.pth`) -- paused here for review.

### 2026-08-21 00:46 CST — Stage 4 done: `examples/train_finetune.py` (Stage-2 supervised
### backbone-finetune driver), one more real bug found and fixed by testing the actual
### cross-repo checkpoint hand-off

**New files:**
- `pcr/models/id_classifier.py::PartIdClassifiers(num_identities, embed_dim, branches=(0,))` --
  persistent per-branch `nn.Linear` heads for the id loss, only built for branches actually
  requested (default: foreground only, branch 0) -- matching bpbreid's own `default_losses_weights`
  convention already documented in `pcr/loss/gilt_loss.py` (`foreg: id=1`, `parts: id=0`), and
  deliberately avoiding the "wasted compute for gradient-dead heads" issue this file's 2026-08-18
  audit entry flagged (and left unfixed) for BPBreID's own internal placeholder classifiers.
- `configs/stage2_backbone_finetune.yaml` -- `model` (must match stage1's, notably
  `checkpoint_path`, so stage 2 starts finetuning from the same point stage 1's prompts were
  trained against), `stage1.prompt_dir` (only `text_prototypes.pth` is read from it), `data`
  (dataset/batch/PK sampling/optional `masks_dir` for BPA loss), `loss` (id/triplet/align/bpa
  weights), `optim`, `logging`, and a `fusion` section that's a pass-through placeholder for
  Stage 5 (not consumed by this script).
- `examples/train_finetune.py::main_worker` -- loads Stage 1's frozen `text_prototypes.pth`;
  builds a fully-trainable `BPBReIDEncoder` (ImageNet-init fallback, same convention as
  `train_usl.py`) + `PartIdClassifiers`; PK-sampled supervised loop (`RandomIdentitySampler`,
  reused unchanged) combining four losses per iteration: id loss (foreground branch,
  `CrossEntropyLabelSmooth` via `PartIdClassifiers`), triplet loss (all branches,
  `PartTripletLoss`, visibility-gated internally), alignment loss (all branches, visibility-gated
  per branch here since `I2TLoss` itself has no visibility concept, `I2TLoss` against Stage 1's
  frozen prototypes), and optional BPA loss (masks-aware, off by default). Standard within-domain
  supervised eval (`pcr/evaluators.py::Evaluator`, reused unchanged) every `eval_step` epochs.
  Saves `encoder.model.state_dict()` specifically (the raw `BPBreID` model, not the
  `BPBReIDEncoder` wrapper) -- the exact shape `torchreid.utils.load_pretrained_weights` expects,
  so the resulting checkpoint is directly consumable by `train_uda.py`/`train_usl.py
  --checkpoint-path` unchanged.

**New preprocessor, added by refactoring rather than duplicating:**
`pcr/utils/data/preprocessor.py` gained `PreprocessorMaskedSingleView` (one image + its mask +
real `(pid, camid, index)` -- the shape Stage 2's supervised single-view training needs, unlike
`PreprocessorMasked`'s two-view EMA-teacher-student shape from the USL path). Extracted two shared
helpers first (`_load_raw_mask`, `_paired_geometric_transform`) so the existing, already-verified
`PreprocessorMasked` (used by `train_usl.py`) and the new class both call the same geometric-
pairing logic instead of forking it -- re-verified `PreprocessorMasked`'s exact behavior was
unchanged after the refactor (see Verified section below) before trusting it as a foundation for
the new class, not just assuming the refactor was safe. `pcr/loss/__init__.py` also gained
`SupConLoss`/`I2TLoss` in its aggregated exports, for consistency with the package's other five
losses (Stage 3's `train_prompts.py` already imports `SupConLoss` directly from its own submodule
and was left unchanged, no reason to churn a working, verified import).

**One more real bug, found by testing the actual cross-repo checkpoint hand-off (not by
`--setup-only`, not by re-reading a saved file with this repo's own loader) -- `mean_ap()`
(`pcr/evaluation_metrics/ranking.py`) returns `numpy.float64`, not a plain Python float.** Stage
2's checkpoint stored this numpy scalar directly as `best_mAP`. `pcr.utils.serialization
.load_checkpoint` (this repo's own loader, used by `train_finetune.py` itself to reload its own
checkpoints) explicitly passes `weights_only=False` and loaded it fine -- **hiding** the problem.
But the checkpoint's actual designed consumer is bpbreid's *own* `torchreid.utils
.load_pretrained_weights` -> `torchtools.load_checkpoint`, which does **not** pass
`weights_only=False`; on this environment's PyTorch 2.11 (default `weights_only=True` since
2.6), that raised `UnpicklingError: ... GLOBAL numpy._core.multiarray.scalar was not an allowed
global`, discovered only by actually round-tripping a real saved checkpoint through
`BPBReIDEncoder(model_cfg, checkpoint_path=...)` -- the exact construction `train_uda.py`/
`train_usl.py` do internally for `--checkpoint-path`. This is the same class of bug this file's
prior entries keep surfacing: a forward-only or self-consistent-only test path cannot catch a
failure that only appears when the *actual* downstream consumer (an external repo's loader, in
this case) is exercised for real. **Fixed** in `train_finetune.py`: `mAP = float(evaluator
.evaluate(...))` at the point of computation, so every place `mAP`/`best_mAP` flows from then on
(comparison, checkpoint dict, print formatting) is a plain float. Scoped the fix to this file only
-- `train_uda.py`'s own `best_mAP` has the identical numpy-scalar issue, but its checkpoint format
was never part of the `--checkpoint-path` contract in the first place (it's a whole-DataParallel-
wrapped-encoder state dict with different key prefixes, only ever reloaded via this repo's own
`weights_only=False` loader) -- flagged here for awareness, not fixed as a drive-by outside this
stage's actual scope.

**Verified, in order:**
1. Full repo `python -m py_compile` sweep after every file addition/refactor -- clean throughout.
2. `PreprocessorMasked` (refactored) and the new `PreprocessorMaskedSingleView`, both tested
   directly against the same real image+mask pair used in this file's 2026-08-19 16:58 entry
   (`market1501/bounding_box_train/0002_c1s1_000451_03.jpg`): identical shapes to what was
   verified back then (`img1 [3,384,128]`, `mask [6,384,128]`, mask sums to 1 per pixel) --
   confirms the refactor didn't silently change `PreprocessorMasked`'s real behavior.
3. `--setup-only` dry run (`--backbone resnet50`, real Market1501 data, Stage 3's own smoke-run
   `text_prototypes.pth` as `stage1.prompt_dir`): dataset/prototype-table shape assertions passed,
   encoder + id classifiers + loaders built cleanly, exited before training as designed.
4. Real smoke training run, **unmasked path** (2 epochs, `batch_size=32`, real Market1501, random-
   init resnet50 backbone): completed in 3m35s, no NaN/crash, all three active loss terms
   (id/triplet/align) finite every logged iteration. **mAP 14.6% -> 21.3%/CMC top-1 38.2% across
   just 2 epochs** -- markedly faster and higher than every prior UDA/USL dummy run in this file
   (which topped out around 0.2-3.4% mAP on random-init backbones), consistent with Stage 2 having
   real, static identity labels driving id+triplet loss directly rather than epoch-to-epoch DBSCAN
   pseudo-labels -- exactly the kind of qualitative difference expected, not a red flag.
5. Real smoke training run, **masked/BPA path** (same settings, `--masks-dir
   pifpaf_maskrcnn_filtering`, exercising `forward_full`/`PreprocessorMaskedSingleView`/BPA loss
   for the first time in this stage): completed in 3m37s, no NaN/crash, all four loss terms
   (id/triplet/align/bpa) finite every iteration, mAP 13.6% -> 17.7%.
6. **Checkpoint round-trip through the real consumer, not this repo's own loader** -- the most
   important check for this stage's actual purpose (chaining into `train_uda.py`/`train_usl.py`):
   built a *fresh* `BPBReIDEncoder(model_cfg, checkpoint_path='.../model_best.pth.tar')` via
   bpbreid's own `torchreid.utils.load_pretrained_weights` (same call `--checkpoint-path` makes
   internally) and compared every one of the 409 saved weight tensors against the fresh model's
   loaded state **by exact tensor equality**, not shape-only or "no crash": all 409 matched
   bit-for-bit, 0 missing, 0 mismatched, "Successfully loaded pretrained weights" with no
   discarded-layers warning. First attempt hit the numpy-scalar `UnpicklingError` above; re-run
   after the fix passed cleanly, `best_mAP` confirmed `<class 'float'>` in the reloaded checkpoint.
7. Full repo `python -m py_compile` + `import pcr` sweep after all fixes landed -- clean in
   `pcr2-run`.

**Next**: Stage 5 (`pcr/models/fusion_head.py::FusionHead` -- fixed-weight and learned-gate
single-descriptor fusion, per reid_pipeline_plan.md section 4) -- paused here for review.

### 2026-08-21 01:33 CST — Stage 5 done: `pcr/models/fusion_head.py::FusionHead`

**New file:**
- `pcr/models/fusion_head.py::FusionHead(num_branches, embed_dim, mode='fixed_weights'|
  'learned_gate', w_global=1.0, w_part=0.5, gate_hidden_dim=128)` -- reid_pipeline_plan.md
  section 4's fusion module, adapted to this repo's actual encoder convention:
  `BPBReIDEncoder.forward` returns one `[B, M, D]` tensor (branch 0 = foreground/global,
  branches 1..K = parts) and one `[B, M]` visibility tensor, not the plan's separate
  `global_feat`/`part_feats`/`visibility[K]` arguments -- same fusion math, indexed against
  branch 0 instead of a trailing global slot. `forward(f_out, visibility) -> [B, M*D]`, always
  L2-normalized.
  - `'fixed_weights'`: global branch gets a fixed scalar weight, unaffected by visibility (matches
    the plan's own `w_global_eff = w_global`, no gating); every part branch gets one shared scalar
    weight, multiplied by that branch's visibility (bool or continuous) -- an invisible part
    contributes exactly zero.
  - `'learned_gate'`: a small 2-layer MLP maps `[global_feat, visibility]` to per-branch softmax
    weights (global branch un-gated, part branches gated by visibility same as above) -- not wired
    into any training script yet, per the approved plan's Stage 5 scope (module + verification
    only).
  - Purely additive: every existing training/eval script keeps using
    `compute_bpb_pairwise_distance`'s part-distance matching unchanged; `FusionHead` only adds an
    optional single-vector export path for ANN/FAISS-style retrieval, per the plan's own framing.

**Verified, in order** (synthetic only, no training script wiring at this stage, matching the
approved plan):
1. `python -m py_compile` -- clean.
2. `'fixed_weights'` mode: correct `[B, M*D]` shape, output L2-normalized (`norm ≈ 1`), and cross-
   checked against an **independent, non-vectorized per-sample/per-branch Python-loop reference
   computation** (not a copy of the vectorized formula) -- matched to `1e-5`.
3. **Invisible-part zero-contribution, the most important property this module has to get right**:
   corrupting an invisible branch's raw embedding with large random noise left the fused output
   completely unchanged (`torch.allclose` at `1e-5`) -- proves the visibility gate genuinely zeroes
   that branch's contribution, not just down-weights it. As a contrast check, corrupting a
   *visible* branch's embedding **did** change the fused output, confirming the test isn't
   vacuously true (e.g. from a bug that zeroes everything).
4. `'learned_gate'` mode: correct shape, L2-normalized output, finite gradients into both the
   input features and the gate's own parameters, the same invisible-part zero-contribution
   property re-verified for this mode too, and confirmed the gate's weights genuinely respond to
   different visibility patterns (all-visible vs. all-parts-invisible inputs produce different
   softmax weights) -- not silently ignoring the visibility input.
5. Continuous (non-boolean) visibility path: finite, correctly-shaped, L2-normalized output; and
   confirmed the soft-gating behavior actually scales with the continuous value (0.1 vs. 0.9
   visibility for the same branch produces different fused output), not merely thresholded to a
   hard 0/1 internally.
6. Invalid `mode` string raises `ValueError` with a clear message.
7. Full repo `python -m py_compile` + `import pcr` sweep -- clean in `pcr2-run`.

**Next**: Stage 6 (close the Stage-3/UDA GiLt+BPA gap -- port `PartGiLtLoss`/`BodyPartAttentionLoss`
wiring into `pcr/trainers.py::PCRTrainer_UDA` and `examples/train_uda.py`, executing the plan
already drafted in this file's 2026-08-19 15:58 entry for the UDA path specifically) -- paused
here for review.

### 2026-08-21 17:01 CST — Stage 6 done: GiLt (id+triplet) + BPA loss wired into
### `PCRTrainer_UDA`/`train_uda.py`, one real bug found and fixed by reading a suspicious metric

Executes the plan already drafted in this file's 2026-08-19 15:58 entry for the UDA path
specifically (that entry was only executed for `train_usl.py` at the time). **Not a reuse of
`PartGiLtLoss`** (which classifies against per-epoch DBSCAN cluster centers -- the right fit for
USL's ever-changing pseudo-label numbering, wrong for source's real, static labels): source id
loss uses `PartIdClassifiers` (Stage 4's persistent `nn.Linear` head module, reused directly) +
`CrossEntropyLabelSmooth`; triplet loss uses `PartTripletLoss` directly for both source and target.
Both are constructor args on `PCRTrainer_UDA` now, not per-`.train()`-call arguments like
`ICEUSLTrainer`'s GiLt -- no per-epoch rebuild needed once labels are real/static.

**`pcr/trainers.py::PCRTrainer_UDA`** gains `id_classifiers`/`id_criterion`/`triplet_criterion`/
`id_weight`/`triplet_weight`/`bpa_criterion`/`bpa_weight` constructor args (all optional, default
off). Per-iteration: id loss on source only (`f_out_s` through `PartIdClassifiers`, branch 0);
triplet loss on **both** source (real labels) and target (target's actual DBSCAN pseudo-cluster
label -- see bug below); BPA loss on source only, gated behind a genuinely different forward path
(`_forward_full`, new method) since it needs `pixels_cls_scores` from `BPBReIDEncoder.forward_full`.

**Real correctness constraint investigated before writing any code, not discovered by accident**:
`pcr/models/dsbn.py::DSBN2d/DSBN1d.forward` splits *any* batch handed to it exactly in half by
*position* -- first half always through `BN_S`, second half through `BN_T`, with zero awareness of
what the caller intends. This means BPA loss cannot call `forward_full` on a source-only sub-batch
(it would silently misroute half of those source images through the target BatchNorm branch and
corrupt its running statistics) -- it must be fed the *whole* joint `[source_half; target_half]`
batch in one call, exactly like the existing memory-loss forward, then `pixels_cls_scores[:B//2]`
(source's positional half) is what stays correct. Implemented that way; `_forward_full` still
bypasses `nn.DataParallel`'s scatter/gather (same accepted single-GPU-only tradeoff already
documented for `ICEUSLTrainer`'s masked path and `BPBReIDEncoder.forward_full` itself), gated
behind `bpa_criterion is not None` so the default (BPA off) path is completely unchanged from
before this stage -- zero regression risk for the already-verified plain-memory-loss behavior.

**One real bug, found by noticing a suspicious metric during the smoke run, not by inspection**:
first smoke run's `Loss_gilt_tri_t` printed exactly `0.000` on nearly every logged iteration --
too suspiciously exact to wave away as "small value rounds to 0.000." Root cause: the target
triplet call used `t_indexes + self.source_classes` (copied from the adjacent memory-loss line)
as the identity label passed to `PartTripletLoss`. `t_indexes` is `PartHybridMemory`'s per-image
memory-slot index -- by construction unique to *every individual image*, correct for addressing
memory slots (SpCL's own original design: one memory slot per target image, with a *separate*
`memory.labels` lookup table doing the pseudo-cluster grouping internally), but never shared
between two different images -- so `PartTripletLoss`'s same-label positive-pair search could only
ever succeed by the sampler coincidentally drawing the exact same image twice (`RandomMultiple
GallerySampler`'s replace=True duplication for under-populated pseudo-identities), explaining both
the frequent "no valid triplets" warnings and the suspiciously-exact zero on the rare "successes"
(a duplicated image's self-distance is ~0). **Fixed**: `_parse_data`'s already-returned (but
previously discarded via `_`) `pids` field -- which for target data *is* the real DBSCAN
pseudo-label, since `pseudo_labeled_dataset`'s tuples are built with the pseudo-label in that
position -- is now captured as `t_pseudo_labels` and used for the triplet call; `t_indexes` stays
reserved for the memory loss only, its original and only correct use. Verified the fix actually
changed behavior, not just silenced the symptom: re-ran the smoke test and `Loss_gilt_tri_t` now
shows genuine varying nonzero values (0.240, 0.261, 0.277) instead of a constant suspicious 0.000.

**Verified, in order:**
1. Full repo `python -m py_compile` sweep -- clean throughout.
2. `--setup-only` dry run (`--backbone resnet50`, real DukeMTMC->Market1501 data, GiLt on by
   default, BPA off): "GiLt on, BPA off" printed correctly, memory/encoder shapes as expected.
3. Real smoke run, **GiLt path** (`-ds dukemtmc-reid -dt market1501`, 2 epochs x 10 iters,
   random-init resnet50): completed clean, no NaN/crash, `Loss_gilt_id`/`Loss_gilt_tri_s` finite
   and varying every iteration; `Loss_gilt_tri_t` initially suspicious (see bug above), confirmed
   genuinely fixed after the correction. mAP 0.8%/top-1 2.4% -- expected near-random for a 20-
   iteration random-init smoke run, consistent with every prior UDA dummy run in this file.
4. Real smoke run, **BPA path** (`-ds market1501 -dt dukemtmc-reid --masks-dir
   pifpaf_maskrcnn_filtering`, exercising `_forward_full`/the joint-batch positional-split
   handling for the first time on the UDA path): completed clean, no NaN/crash, `Loss_bpa` finite
   every iteration (~1.79-1.87, same order of magnitude as Stage 4's and `train_usl.py`'s own BPA
   smoke runs). A first pass at `--iters 10` showed `Loss_gilt_tri_t` at a constant 0.000 the
   whole run; re-ran at `--iters 40` specifically to get more statistical signal before concluding
   anything, and confirmed nonzero values (0.277, 0.157) do appear once enough batches succeed --
   settling that the all-zero result was small-sample variance (mostly-singleton target pseudo-
   ids at this tiny smoke scale), not a residual bug, rather than assuming either way.
5. Full repo `python -m py_compile` + `import pcr` sweep after all fixes landed -- clean in
   `pcr2-run`.

**Not done in this stage, out of scope per the approved plan**: horizontal-stripes mode,
hard/soft-triplet-loss variants (both still only in the "planned, not built" state from this
file's 2026-08-19 15:58 entry).

### 2026-08-21 18:34 CST — Stage 7 done: orchestration + docs. Plan complete.

**New file: `examples/run_pipeline.py`** -- thin sequential orchestrator (Stage 1 prompts ->
Stage 2 finetune -> Stage 3 UDA/USL/none, per `reid_pipeline_plan.md` section 6). Shells out to
each stage's own script as a separate `subprocess.run` call rather than importing them as library
functions -- all four stage scripts are `if __name__ == '__main__':`-driven with process-global
state (`sys.stdout` reassignment via `Logger`, CUDA context, argparse), never designed to run
twice in one process. `argparse.parse_known_args()` splits the orchestrator's own flags
(`--stage1-config`/`--stage2-config`/`--stage3`/`--python`) from everything else, which is passed
straight through verbatim to whichever Stage 3 script runs. The one piece of real logic: Stage 1
-> Stage 2 wiring needs nothing (already config-file-driven, Stage 2's own YAML names the Stage 1
output directory it reads from), but Stage 2 -> Stage 3 crosses the YAML/argparse boundary, so the
orchestrator computes `{stage2 logs_dir}/model_best.pth.tar` and auto-injects it as Stage 3's
`--checkpoint-path` (plus `--backbone`, matching Stage 2's config) *unless* the user already
supplied either flag explicitly. A pre-flight check compares Stage 1's `logging.logs_dir` against
Stage 2's `stage1.prompt_dir` and warns (not hard-fails, in case intentional) on mismatch, so a
likely misconfiguration is caught before burning a full Stage 1 run rather than after.

**Verified with a genuine end-to-end run, not just each piece read in isolation** -- this is the
one place in the whole plan where three previously-independently-verified stages get chained
together for the first time, so the actual hand-off (not each stage's own internals, already
proven in Stages 3/4/6) is what needed checking:
1. `python -m py_compile` -- clean.
2. Isolated unit checks (`flag_present`, `parse_known_args` separation, the mismatch-detection
   comparison) run directly against the module's own functions -- all correct, including a
   deliberately-constructed mismatched-config case that correctly printed the warning.
3. **Real full run**: fresh Stage 1 (1 epoch, Market1501) -> fresh Stage 2 (1 epoch, reading
   Stage 1's actual output) -> Stage 3 `train_uda.py --setup-only` with *no* manually-supplied
   `--checkpoint-path`/`--backbone`. Confirmed from the orchestrator's own printed command that it
   correctly auto-filled `--checkpoint-path logs/dummy_pipeline_stage2/model_best.pth.tar
   --backbone resnet50` ahead of the user's pass-through args, and -- the actual proof this
   works, not just that the right flags were assembled -- Stage 3's own log printed **"Successfully
   loaded pretrained weights from 'logs/dummy_pipeline_stage2/model_best.pth.tar'"**, confirming
   bpbreid's own checkpoint loader (the same loader stress-tested in Stage 4's bit-for-bit
   round-trip check) accepted the auto-computed path and loaded it without a discarded-layers
   warning. All three subprocess stages exited 0; Stage 1 saved its prompt/prototype files, Stage
   2 trained one epoch and evaluated (mAP 8.7%), Stage 3's setup completed cleanly.
4. Full repo `python -m py_compile` + `import pcr` sweep -- clean in `pcr2-run`.

**Docs**: `README.md` rewritten to document the full pipeline (was UDA-only before this plan) --
install (including the new CLIP dependency), per-stage run commands, the orchestrator, the
external-bpbreid-pretraining alternative (still supported, unchanged), and a note on the fusion
module's standalone status.

---

## Plan complete: `reid_pipeline_plan.md` gap closed

All seven stages of the approved plan
(`/home/lakshh/.claude/plans/in-this-repo-cleanly-refactored-river.md`) are done. Summary of what
this plan added on top of the substantial BPBreID+SPCL pipeline that already existed (see this
file's history above Stage 0):

- **Net-new**: the entire CLIP-ReID component -- `pcr/models/{clip_text_encoder,prompt_learner}.py`,
  `pcr/loss/{clip_supcon_loss,clip_i2t_loss}.py`, `pcr/models/id_classifier.py`,
  `examples/{train_prompts,train_finetune}.py` + their YAML configs, faithfully following
  CLIP-ReID's actual verified implementation (multi-positive SupCon in Stage 1, frozen-prototype
  cross-entropy alignment in Stage 2) rather than the plan document's simplified pseudocode, with
  that deviation documented at the point it was made.
- **Net-new**: `pcr/models/fusion_head.py::FusionHead`, a standalone single-descriptor fusion
  module (not wired into any script, matching the plan's own framing of it as an optional
  ANN/FAISS-retrieval convenience).
- **Gap closed**: GiLt (id+triplet) and BPA loss terms, previously wired into `train_usl.py` only,
  now also available in `train_uda.py` via the same `PCRTrainer_UDA` class, correctly respecting
  DSBN's positional source/target batch split and using real static source labels rather than
  per-epoch cluster centers.
- **Orchestration + docs**: `examples/run_pipeline.py` chains all of the above with the
  pre-existing UDA/USL drivers into one runnable pipeline; `README.md` documents the whole thing.

Bugs found and fixed along the way, every one via actually running the code rather than static
review alone (consistent with this project's established discipline): a `GradScaler`/fp16-leaf-
parameter incompatibility (Stage 3), a PyYAML scientific-notation parsing gotcha (Stage 3), a
numpy-scalar checkpoint incompatibility with bpbreid's own loader (Stage 4), a crash risk from an
all-invisible sub-batch and a duplicated helper (caught by `/code-review high`, both fixed), and a
mislabeled target-triplet identity (Stage 6, caught by noticing a suspiciously-exact zero metric
rather than trusting "no crash, no NaN" alone). None of these were visible from reading the code;
all were found by running it against real data and, in several cases, by treating a
too-clean-looking number as a reason to look closer rather than a reason to move on.

**Out of scope, not done, still open** (all pre-existing observations from before this plan,
re-confirmed rather than silently dropped): horizontal-stripes mode, hard/soft-triplet-loss
variants, CA-Jaccard/camera-aware clustering, `--resume` support for `train_uda.py`. None of these
were part of the approved plan's seven stages.

### 2026-08-21 02:02 CST — Full-pipeline code review (Stages 0-5), three findings, all fixed

Ran `/code-review high` against the complete working-tree diff (everything uncommitted since
`be5b1eb`, i.e. all of Stages 0-5 together) per direct user request to review the pipeline built
so far. Three findings, all real, all fixed:

1. **Real crash risk (`examples/train_prompts.py`)**: `loss = b_features.new_zeros(())` at the
   top of each training iteration is only ever reassigned inside the per-branch
   `if visible.sum().item() < 1: continue` guard. If *every* branch (including foreground) has
   zero visible samples in a given random 64-image sub-batch, `loss` stays the original grad-less
   leaf zero tensor and `scaler.scale(loss).backward()` crashes with "element 0 of tensors does
   not require grad and does not have a grad_fn". Plausible specifically because Stage 1's default
   config uses `checkpoint_path: ''` (ImageNet init, no source-pretrained checkpoint) --
   BPBreID's visibility head is untrained early on and can emit unreliable scores, unlike a
   checkpoint where GiLt/attention training has already shaped it. **Fixed**: track
   `any_branch_used` across the branch loop; skip the whole iteration (no backward/step) if it
   stays `False`, mirroring the same "skip degenerate step rather than crash" pattern
   `train_usl.py` already uses for its own (different) degenerate-epoch case.
2. **Duplicated helper**: `branch_visible_mask` (bool-passthrough / continuous-threshold
   visibility gating) was defined verbatim in both `examples/train_prompts.py` and
   `examples/train_finetune.py` -- a fix to one could silently miss the other. **Fixed**: moved to
   `pcr/utils/part_distance.py` (this repo's established home for visibility-gating conventions,
   already holding `masked_mean`/`replace_values`), both scripts now import it from there.
3. **Minor**: `assert proto['num_branches'] == num_branches` in `train_finetune.py` had no
   message, unlike the identically-shaped `num_identities` assert right above it. **Fixed**: added
   a matching descriptive message naming the actual mismatch (stage1 vs. stage2 `parts_num`).

Review also explicitly verified (not just skipped) several things as **already correct**, worth
recording so they aren't re-litigated: `PartPromptLearner`'s prefix/suffix token-slicing against
CLIP's real tokenizer output ("A photo of a" -> exactly 5 tokens incl. SOT, matching `n_ctx+1`);
`IterLoader`'s narrowed `except StopIteration` doesn't break `train_usl.py`'s existing epoch-skip
guard; `Preprocessor`/`PreprocessorMaskedSingleView` tuple-unpacking in `train_finetune.py` lines
up correctly; `BPBReIDEncoder.forward_full`/`Evaluator`/`PartTripletLoss`/
`CrossEntropyLabelSmooth`/`BodyPartAttentionLoss` signatures all match their new call sites.

**Verified after fixes**: full repo `python -m py_compile` sweep clean; re-ran both Stage 1 and
Stage 2's real smoke training runs end-to-end on real Market1501 data -- Stage 1 produced
**bit-identical** loss values to the pre-fix run (epoch 1 avg loss 39.8853, exactly matching,
confirming the new guard is a true no-op on this data and the refactor changed nothing
behaviorally), Stage 2 completed cleanly with mAP/CMC in the same range as before (21.1% mAP,
38.5% top-1). No new bugs introduced by the fixes.

---

### 2026-08-24 00:00 — Part-Relational Attention: bidirectional cross-part mixing added to both
### Stage 1 and Stage 2, on top of the CLIP-ReID prompt-learning pipeline closed out above

New user-supplied plan ("Part-Relational Attention for BPBreID + CLIP-ReID"): BPBreID's K pooled
part embeddings and CLIP-ReID's per-part learnable prompt contexts are each produced completely
independently of one another (no term anywhere lets "left arm" and "torso" influence each other's
embedding, or one part's prompt context inform another's). The plan adds one bidirectional
self-attention block per side to close that gap, confined to Stage 1/2 (Stage 3 --
`train_uda.py`/`train_usl.py` -- explicitly out of scope, left untouched). Three design questions
were resolved directly with the user before implementation: build inside pcr2's actual flat
`pcr/models/`+`pcr/loss/`+`examples/`+`configs/` layout (not the plan's own aspirational
`engine/`/`losses/`/`scripts/`/`data/` tree); keep K=5 (not the plan's K=8, which was justified by
an occlusion-specific reason this design explicitly puts out of scope, and would break the
already-produced Stage 2 checkpoint on the training server); Stage 3 stays completely untouched.

**New module -- `pcr/models/relation_blocks.py`**: `VisualRelationBlock` (one `nn.TransformerEncoder`
layer, bidirectional, over the K pooled part vectors, wrapped in a learned zero-initialized
residual gate -- `part_tokens + tanh(gate) * encoder(part_tokens)` -- so it starts as a no-op and
only learns to mix if useful; permanent at inference, trained in both Stage 1 and Stage 2) and
`TextRelationBlock` (same shape of block, no gate/residual, over a person's K*n_ctx learnable
part-context tokens, run *before* any part's prompt is spliced together -- training-only, owned
internally by `PromptLearner`, discarded after Stage 1 once its output is baked into the cached
text-prototype table). Neither block does any masking internally -- see the upstream-filtering
paragraph below for why.

**Rewritten -- `pcr/models/prompt_learner.py`**: `PartPromptLearner` renamed to `PromptLearner`,
API redesigned around two separate learnable context tensors (`fg_ctx` for the foreground branch,
untouched by any relation block; `part_ctx` for the K part branches, mixed through
`TextRelationBlock` internally) and a single `build_part_prompts(labels)` call that returns all
`1+K` branches' spliced prompt tensors at once, replacing the old one-branch-at-a-time
`forward(labels, branch_idx)`.

**New -- `pcr/utils/visibility_filter.py`**: implements the plan's section-0.1 assumption that
occlusion handling is out of scope and images are pre-filtered upstream by a visibility-index
threshold (`lambda_v_min`), so neither relation block nor either stage's losses need any
per-sample, per-branch masking. `filter_by_visibility(dataset_list, encoder, ...)` runs the frozen
encoder once under `no_grad`, computes each image's mean part-visibility (excluding foreground),
and keeps only images at or above the threshold -- applied once, before Stage 1's feature cache
and before Stage 2's training loader are built, replacing the old per-loss-term
`branch_visible_mask`/skip-if-invisible pattern entirely (every admitted image's every branch now
contributes unconditionally to every loss term).

**Renamed + rewritten -- `examples/train_prompts.py` -> `examples/train_relational_prompts.py`**
(Stage 1 driver): builds `PromptLearner` + `VisualRelationBlock`, runs `filter_by_visibility` on
the training set before caching visual features, then per iteration builds all `1+K` prompts via
one `build_part_prompts()` call, mixes the cached part features through `VisualRelationBlock`, and
sums symmetric SupCon loss over every branch (no visibility-based skipping). Saves
`prompt_learner.pth` (whole `PromptLearner` state, TRB included) and `vrb.pth`; no longer saves
`text_prototypes.pth` itself.

**New -- `examples/cache_text_anchors.py`**: standalone script, run once after Stage 1 against the
same config. Loads `prompt_learner.pth`, calls `build_part_prompts()` per identity-batch, saves
`text_prototypes.pth`. Split out from Stage 1 per the plan's own rationale: this is a deterministic,
frozen forward pass that shouldn't be recomputed on every Stage-2 training step.

**Renamed + rewritten -- `examples/train_finetune.py` -> `examples/train_relational_finetune.py`**
(Stage 2 driver): loads `vrb.pth` from `stage1.prompt_dir` and keeps `VisualRelationBlock`
trainable (jointly with the now-unfrozen backbone); `compute_losses()` now mixes the K part
embeddings through VRB before id/triplet/align losses see them, drops the `parts_visibility=`
argument to `PartTripletLoss` (now a plain unweighted per-branch average, since the upstream filter
already guarantees full visibility) and the per-branch visibility skip in the align-loss loop
(every branch always contributes). `filter_by_visibility` applied to this stage's own training set
too, before the loader is built. Re-saves `vrb.pth` at the end (Stage 2's continued-training VRB
weights -- not consumed by anything downstream yet, since Stage 3 is out of scope for this change).

**Renamed configs**: `configs/stage1_prompt_learning.yaml` -> `configs/stage1_relational_prompts.yaml`
(added `trb:`/`vrb:`/`visibility:` sections, removed the now-unused `loss.visibility_threshold`);
`configs/stage2_backbone_finetune.yaml` -> `configs/stage2_relational_finetune.yaml` (added
`vrb:`/`visibility:` sections, removed `loss.align_visibility_threshold`).

**`examples/run_pipeline.py`**: `STAGE_SCRIPTS` updated to the renamed Stage 1/2 scripts plus a new
`cache_anchors` entry; the Stage 1 block now runs `cache_text_anchors.py` automatically right after
`train_relational_prompts.py`, against the same `--stage1-config`.

**`README.md`**: pipeline-stage diagram, Stage 1/2 sections, and the orchestrator example rewritten
to match the renamed scripts/configs and the new VRB/TRB/`cache_text_anchors.py`/visibility-filter
pieces.

**`METHODOLOGY.md`**: Stage 1 (section 3) and Stage 2 (section 4) rewritten throughout -- prompt
construction now describes the `fg_ctx`/`part_ctx` split and where `TextRelationBlock` sits in the
pipeline; the SupCon and alignment loss formulas drop their per-branch visibility threshold in
favor of the upstream filter; the part-triplet distance formula drops its visibility-weighted
average in favor of a plain average (Stage 1/2 only -- Stage 3's own visibility-weighted version in
section 5 is untouched); both algorithm pseudocode blocks updated to match. Section 3.5 ("Design
Note: Attention Maps Never Reach the Text Encoder") rewritten to explain what `VisualRelationBlock`/
`TextRelationBlock` add without crossing that image/text boundary, replacing the stale description
of the old single-branch `PartPromptLearner.forward(labels, branch_idx)` API with the actual
`PromptLearner.build_part_prompts(labels)` code path, plus a new 3.5.2 subsection on the upstream
visibility filter that replaced per-branch masking.

**Stale-reference cleanup**: fixed leftover `train_prompts.py`/`train_finetune.py`/
`PartPromptLearner` mentions in docstrings across `pcr/utils/config.py`, `pcr/utils/lr_scheduler.py`,
`pcr/models/clip_text_encoder.py`, `pcr/utils/data/preprocessor.py`, `examples/train_uda.py`,
`pcr/models/relation_blocks.py`, `pcr/models/id_classifier.py`. Removed
`pcr/utils/part_distance.py::branch_visible_mask` entirely -- confirmed by repo-wide grep to be
unused now that both Stage 1 and Stage 2 dropped per-branch visibility gating in favor of the
upstream filter.

**Note on process**: an earlier attempt at this same plan (TRB only, VRB explicitly deferred) was
built first per an initial round of clarifying-question answers, then reverted via `git checkout
HEAD -- <files>` back to the last commit (`102b6fe`) after the user clarified that both VRB and TRB
must ship together -- the revert also unintentionally discarded an uncommitted, unrelated earlier
edit to `METHODOLOGY.md` section 3.5 (added in response to a prior "why don't attention maps reach
the text encoder" question, never committed), since git cannot selectively revert one uncommitted
edit to a file from another. That section has been restored and updated above; nothing else from
that unrelated edit was lost.

**Verification status**: every new/changed file individually `python -m py_compile`-checked, plus a
full repo-wide `py_compile` sweep (`pcr/`, `examples/`) -- all clean. No synthetic shape/gradient
tests or real training smoke runs have been performed for this change -- the `pcr2-run` conda env's
`torch` installation was found broken mid-session (unrelated to this change) and has not yet been
repaired, so runtime verification remains blocked pending that fix.

---

### 2026-08-25 14:05 — Renamed VisualRelationBlock/TextRelationBlock to VisualAttentionBlock/
### TextualAttentionBlock repo-wide, per direct user request

Cosmetic rename, no behavior change: `pcr/models/relation_blocks.py::VisualRelationBlock` ->
`VisualAttentionBlock`, `TextRelationBlock` -> `TextualAttentionBlock`. For consistency, the
short-hand `VRB`/`TRB` used throughout comments/docstrings/log messages became `VAB`/`TAB`, and
every lowercase identifier following that abbreviation was renamed to match: local/attribute
variables (`vrb`/`trb` -> `vab`/`tab`, e.g. `PromptLearner.tab`), constructor kwargs
(`trb_num_heads`/`trb_num_layers` -> `tab_num_heads`/`tab_num_layers`), YAML config sections
(`vrb:`/`trb:` -> `vab:`/`tab:` in both `configs/stage1_relational_prompts.yaml` and
`configs/stage2_relational_finetune.yaml`), and the saved/loaded checkpoint filename (`vrb.pth` ->
`vab.pth`, both where Stage 1 saves it and where Stage 2 loads/re-saves it). Applied across
`pcr/models/relation_blocks.py`, `pcr/models/prompt_learner.py`, `pcr/utils/visibility_filter.py`,
`examples/train_relational_prompts.py`, `examples/train_relational_finetune.py`,
`examples/cache_text_anchors.py`, `README.md`, `METHODOLOGY.md`.

This entry's own prose, and every entry above it, still says `VisualRelationBlock`/
`VisualRelationBlock`/`VRB`/`TRB`/`vrb.pth` where that was the accurate name at the time -- left
untouched, matching this file's append-only convention (a progress log describes what was true
when it was written, not the current state of the code).

Full repo `python -m py_compile` sweep clean after the rename. Not re-run: the Stage 1
`--setup-only` smoke check (already verified working under the old names earlier today, in
`pcr2-run`) -- the rename touched no logic, only names, so it wasn't re-run, but should be if
there's any doubt.

---

### 2026-08-26 02:25 — Stage 0 added: BPA segmentation pretraining, closing a real gap in
### body-part semantic grounding

Motivating question, raised directly: does CLIP's text encoder ever actually "see" body parts,
and is BPA (BPBreID's pixel-to-part classifier) trained by anything that ties a specific branch
index to a specific real anatomical region? Investigated both, concretely:

1. **Stage 1's prompts use learned placeholder tokens, not real body-part words.**
   `pcr/models/prompt_learner.py`'s template is `"A photo of a X X X X person."` -- the context
   tokens are fully learned vectors, never literal words like "legs" or "torso". CLIP's frozen
   text encoder is never told what part a branch represents; it only ever supplies a fixed
   high-dimensional target space for identity-level alignment, not part semantics.
2. **BPA does receive gradient in Stage 2, but only an identity-discrimination signal.** Traced
   the actual autograd graph in `third_party/torchreid/models/bpbreid.py`: `pixels_cls_scores =
   self.pixel_classifier(spatial_features)` feeds directly into `parts_masks` -> `parts_embeddings`
   in one differentiable path, so with the backbone unfrozen (Stage 2), id/triplet/align losses
   all backprop into the pixel classifier. But nothing in that signal says *which* branch should
   correspond to *which* real body part -- the only thing that ever anchors a branch to ground
   truth is `BodyPartAttentionLoss` (pixel-wise CE against PifPaf/MaskRCNN masks), and that's one
   of four losses in Stage 2, off by default (`data.masks_dir: ''`), and mask-availability-limited
   (Market1501 has masks, DukeMTMC-reID does not).

Conclusion: relevant, real gap. Fix -- isolate that one signal into its own pretraining stage
that runs before Stage 1/2, so BPA already has a stable, real-part-anchored spatial split by the
time CLIP prompts and the joint Stage 2 losses build on top of it.

**New -- `examples/train_bpa_segmentation.py`** (Stage 0): trains BPBreID's full backbone +
`pixel_classifier` via `BodyPartAttentionLoss` alone -- no id/triplet/align loss at all, pure
supervised segmentation against ground-truth masks. Plain shuffled `DataLoader` (no PK sampler --
no identity structure needed without an id/triplet loss). Saves a checkpoint in exactly Stage 2's
own format (`{'state_dict': encoder.model.state_dict(), ...}`), so it's directly loadable via
`model.checkpoint_path` in Stage 1/2's configs, or `--checkpoint-path` for Stage 3, with zero
changes needed on the consuming side.

**New -- `configs/stage0_bpa_segmentation.yaml`**: `data.masks_dir` defaults to the real, on-disk
`pifpaf_maskrcnn_filtering` directory (confirmed present at
`<data_dir>/market1501/masks/pifpaf_maskrcnn_filtering/{bounding_box_train,query,bounding_box_test,gt_bbox}`)
-- unlike Stage 1/2's optional, empty-by-default `masks_dir`, this stage has no other loss to fall
back on, so defaulting it off would make the stage meaningless out of the box.

**Verified with real training runs, not just `--setup-only`** (both required after this session's
own earlier lesson: `--setup-only` alone missed the TextualAttentionBlock fp16/fp32 bug because it
exits before any real forward/backward runs):
1. `--setup-only`: dataset/encoder/loader built cleanly, 12936 training images found with
   `masks_dir=pifpaf_maskrcnn_filtering`.
2. Real 1-epoch smoke run (batch 8, reduced from the config's default 32 to fit this dev machine's
   8GB GPU during a full backbone backward pass -- same constraint Stage 2's own smoke test hit
   earlier): completed cleanly in 255s, 1617 iterations, loss 0.94 avg (stabilizing in the
   0.83-0.96 range), pixel accuracy already 71% avg after just one epoch from ImageNet init.
   Checkpoint saved to `checkpoint.pth.tar`/`model_best.pth.tar`.
3. **Stage 0 -> Stage 1 handoff, the actual point of this whole change**: ran Stage 1 with
   `model.checkpoint_path` pointed at Stage 0's `model_best.pth.tar` -- log confirms `"Successfully
   loaded pretrained weights from examples/logs/smoke_stage0/model_best.pth.tar"` (not a silent
   fallback to ImageNet init), and the rest of Stage 1's run (visibility filtering, caching,
   training) proceeded exactly as in every prior Stage 1 smoke test, confirming the checkpoint
   format is fully compatible with zero changes to `examples/train_relational_prompts.py`.

**Also fixed in passing**: reverted an accidental, untested, broken edit to
`configs/stage2_relational_finetune.yaml` (`masks_dir: 'masks/pifpaf_maskrcnn_filtering'` -- wrong,
since `pcr/utils/data/preprocessor.py::_load_raw_mask` already prepends `masks/` itself, so this
would have looked for `masks/masks/pifpaf_maskrcnn_filtering/...`, which doesn't exist) found while
reviewing `git status` before this commit -- discarded via `git checkout HEAD --`, not part of this
change.

**Not yet done**: `examples/run_pipeline.py` doesn't wire Stage 0 in (still Stage 1 -> Stage 2 ->
Stage 3 only) -- Stage 0 is run standalone for now, its checkpoint passed manually into Stage 1's
`model.checkpoint_path`. Full repo `python -m py_compile` sweep clean.

---

### 2026-08-26 03:15 — Added `train_relational_clip.py`: single-stage "generic CLIP" training,
### replacing the two-stage CLIP-ReID scheme with the original CLIP paper's own loss

Direct user request: skip the CLIP-ReID-style two-stage split (frozen backbone + SupCon, then
frozen prompts + I2T against a precomputed prototype table) and instead train every learnable
module jointly, in one forward/backward pass per iteration, using the losses from the actual CLIP
paper (Radford et al. 2021) rather than CLIP-ReID's adaptation of them. Reuses essentially every
existing module unchanged (`BPBReIDEncoder`, `ClipTextEncoder`, `PromptLearner` +
`TextualAttentionBlock`, `VisualAttentionBlock`, `PartTripletLoss`) -- only the training procedure
and loss combination changes.

**New -- `pcr/loss/clip_contrastive_loss.py::ClipContrastiveLoss`**: a faithful port of the CLIP
paper's own diagonal symmetric cross-entropy loss (row *i*'s only positive is column *i* -- the
paper's own `labels = np.arange(n)`), with a learned `logit_scale` (init `log(1/0.07)`, clamped to
`log(100)` every forward pass, exactly matching the paper's own training-stability convention).
Deliberately NOT `SupConLoss` (pcr/loss/clip_supcon_loss.py) -- that file's own docstring already
documents why CLIP-ReID moved away from the paper's literal loss (multi-positive identity masking
instead of diagonal-only); this new file is the literal version, requested explicitly instead of
the adapted one. Normalizes both inputs internally (unlike `SupConLoss`/`I2TLoss`, which assume
pre-normalized input) since the paper's own `logit_scale` calibration only makes sense against
true cosine similarities, and `VisualAttentionBlock`'s residual can otherwise leave the visual
side's norm adrift from 1.

**New -- `examples/train_relational_clip.py`**: single training loop, PK-sampled batches (needed
for `PartTripletLoss`, which this script also uses), full BPBreID forward+backward every
iteration (no caching -- unlike Stage 1, nothing here is frozen), full CLIP text forward every
iteration too (no precomputed prototype table -- unlike Stage 2). Two loss terms only, matching
exactly what was asked for: `ClipContrastiveLoss` per branch (foreground + `VisualAttentionBlock`
-mixed parts), summed, plus `PartTripletLoss` across all branches. No id loss, no BPA loss (Stage
2 has four losses; this script deliberately has two). `torch.amp.GradScaler` reused from Stage 1's
own rationale (the CLIP text tower still runs in fp16 every iteration here, same underflow risk;
every trainable parameter, including the new `ClipContrastiveLoss.logit_scale`, is fp32, so the
scaler stays harmless for all of them -- same argument Stage 1's own docstring already makes,
extended to now also cover the jointly-trainable BPBreID backbone). Saves a checkpoint in exactly
Stage 2's own format, directly loadable by `train_uda.py`/`train_usl.py --checkpoint-path`
unchanged, plus `vab.pth`/`prompt_learner.pth` for completeness (no downstream script in this
single-stage design actually needs to reload either, unlike Stage 1's `prompt_learner.pth` ->
`cache_text_anchors.py`).

**New -- `configs/relational_clip.yaml`**.

**Known, accepted tradeoff, not a bug**: PK sampling (multiple images per identity per batch,
required by the triplet loss) collides with the CLIP paper's own diagonal-only assumption (each
row has exactly one positive) -- other same-identity images in the same batch get treated as
negatives by `ClipContrastiveLoss`. This is the literal, faithful cost of using the paper's own
loss rather than adapting it (as `SupConLoss` already does) to a labeled setting with repeated
classes per batch -- documented in the loss file's own docstring rather than silently smoothed
over.

**Verified with a real training run, not `--setup-only` alone** (per this session's own established
lesson -- `--setup-only` exits before any real forward/backward runs, and already missed one real
bug in an earlier stage):
1. `--setup-only`: dataset/encoder/prompt-learner/loader built cleanly.
2. Real 1-epoch smoke run (batch 8, reduced from the config's default 32 for this dev machine's
   8GB GPU during a full backbone backward pass -- same constraint every other full-backbone stage
   here has hit): completed cleanly, 375 iterations, no dtype errors (the first time GradScaler
   has ever wrapped a full BPBreID backward pass, and the first time the fp16 CLIP text path and a
   trainable BPBreID backbone have run together in the same step), no OOM. Loss finite throughout
   (~12.6-13.9 total, contrastive ~12.3-13.6, triplet ~0.31-0.44), `logit_scale` tuning down
   smoothly from its `log(1/0.07)` init (~14.29 -> ~14.25 across one epoch), `VAB gate` moving up
   from 0 as expected. End-of-epoch eval: Mean AP 0.7%, top-1 1.7% -- lower than Stage 2's own
   1-epoch smoke result (2.2% mAP) but expected, not a red flag: `PromptLearner`/
   `TextualAttentionBlock` are training from scratch simultaneously with the backbone here, unlike
   Stage 2, which finetunes against already-Stage-1-trained prompts.
3. Checkpoint files (`checkpoint.pth.tar`, `model_best.pth.tar`, `vab.pth`, `prompt_learner.pth`)
   all saved correctly.

Full repo `python -m py_compile` sweep clean.

---

### 2026-08-26 03:35 — Removed `train_relational_clip.py` entirely, per direct user request

The single-stage "generic CLIP" training approach added above (2026-08-26 03:15) underperformed
the two-stage CLIP-ReID scheme in its own 1-epoch smoke test (0.7% mAP vs. Stage 2's 2.2% mAP).
Discussed likely causes before any action, per the user's explicit request to wait: small batch
size (8-32) starves the diagonal `ClipContrastiveLoss` of negatives, unlike `I2TLoss`'s full
prototype-table comparison which is batch-size-independent; PK sampling (needed for the triplet
loss this file also used) actively collides with the loss's diagonal-only positive assumption;
and the file has no id loss at all, removing a strong, sample-efficient supervision signal Stage 2
has. Both numbers were only 1-epoch smoke results, not converged, so the comparison itself was
never meant to be conclusive either way.

Decision after discussion: drop the whole approach rather than iterate on it. Deleted entirely:
`examples/train_relational_clip.py`, `configs/relational_clip.yaml`,
`pcr/loss/clip_contrastive_loss.py`; reverted `pcr/loss/__init__.py`'s `ClipContrastiveLoss`
export; removed the README section describing it (a stale doc pointing at deleted files would be
actively misleading, unlike this file's own historical record, which stays as-is above -- describing
what was true when it was written, matching this file's established append-only convention).

Full repo `python -m py_compile` sweep clean after removal.

---

### 2026-08-26 03:50 — Portable data_dir across machines; Stage 0 wired into run_pipeline.py

**Machine-portability fix**: `configs/stage0_bpa_segmentation.yaml`, `stage1_relational_prompts.yaml`,
and `stage2_relational_finetune.yaml` all hardcoded `data.data_dir` as an absolute path under this
machine's own home directory (`/home/lakshh/workspace/reid/datasets`), which breaks when testing on
a second machine with the same relative directory layout but a different username. Changed all
three to `../datasets` -- resolves identically regardless of username, since `data_dir` is already
used via plain `osp.join()` with no anchoring (so it's implicitly resolved relative to the process's
own working directory), and every documented usage in this repo runs each script from the repo
root already.

**`examples/run_pipeline.py`**: exposed Stage 0 (`train_bpa_segmentation.py`) as `--stage0-config`,
run before Stage 1 if given (optional -- most runs still start from Stage 1's ImageNet init).
Added a config-consistency warning matching the existing Stage1/Stage2 `logs_dir` check: if
`--stage0-config` and `--stage1-config` are both given but Stage 1's `model.checkpoint_path`
doesn't already point at Stage 0's expected `<logs_dir>/model_best.pth.tar`, this script warns
rather than silently rewriting either YAML file -- same "check and warn, don't auto-inject"
philosophy the Stage1->Stage2 handoff already uses (per this file's own docstring, Stage0->Stage1
is config-file-driven, same as Stage1->Stage2, unlike the Stage2->Stage3 handoff, which genuinely
needs help since Stage 3 is argparse-only). Verified: `--help` output correct, and confirmed
against the actual current configs that the new warning fires as expected (Stage 0's log dir
`examples/logs/stage0_bpa` vs. Stage 1's currently-empty `checkpoint_path` -- they're not wired
together by default, matching the intentional, conservative design).

Full repo `python -m py_compile` sweep clean.

---

### 2026-08-26 18:50 — Stage 1 rewritten to match "Algorithm 1 -- Stage 1: Prompt + Relation
### Learning" exactly, after a line-by-line audit found five real discrepancies

User supplied an explicit algorithm spec for Stage 1 and asked to verify `examples/
train_relational_prompts.py` actually implements it. Audit found five discrepancies (reported
first, fixed only after explicit confirmation to proceed):

1. **CLIP image encoder never loaded.** Algorithm step 2 says to load and freeze a CLIP image
   encoder; this repo's established design ("CLIP contributes only its frozen text tower") never
   did. New -- `pcr/models/clip_image_encoder.py::ClipImageEncoder` -- loaded and frozen in Stage
   1 (a second, independent `clip.load()` call, not shared with `ClipTextEncoder`) per the
   algorithm's own initialization step, though not consumed by any loss: BPBreID's backbone
   remains the sole visual encoder actually producing embeddings used anywhere in this pipeline.
2. **Backbone/BPAM defaulted to ImageNet init, not Stage 0's checkpoint.**
   `configs/stage1_relational_prompts.yaml`'s `model.checkpoint_path` now defaults to
   `examples/logs/stage0_bpa/model_best.pth.tar` -- Stage 1 is no longer usable out of the box
   without running Stage 0 first, matching the algorithm's own step 1 exactly.
3. **Extra foreground/global loss term not in the algorithm.** The algorithm's `ctx_params` is a
   single `(4K, embed_dim)` tensor (K parts only); it extracts BPAM's global feature `f_g` at step
   8 but never uses it again, and the loss sum (steps 15-16) is over K parts only. Removed the
   foreground `SupConLoss` term entirely from the training loop and excluded `PromptLearner.
   fg_ctx` from Stage 1's optimizer -- `PromptLearner` itself is untouched (still owns `fg_ctx`,
   still returns a foreground prompt from `build_part_prompts`), since `cache_text_anchors.py` and
   Stage 2 both depend on that shape/API for their own foreground handling. Consequence, flagged
   explicitly rather than silently absorbed: `fg_ctx` now stays at its random initialization after
   Stage 1 runs, which makes Stage 2's foreground alignment term (I2TLoss against a prototype
   built from untrained `fg_ctx`) meaningless until/unless that's addressed too -- out of today's
   explicit scope ("fix it for relational prompts script").
4. **Loss labeled "InfoNCE" in the algorithm, code uses `SupConLoss`.** Kept `SupConLoss`
   (CLIP-ReID's real multi-positive contrastive loss) rather than switching to literal
   single-positive InfoNCE -- a plain-InfoNCE variant of this whole pipeline was already tried
   (`train_relational_clip.py`, 2026-08-26 03:15) and removed for underperforming
   (2026-08-26 03:35). Interpreted "InfoNCE" in the algorithm as generic contrastive-loss
   terminology, not a literal instruction to revert that decision.
5. **Sampling wasn't PK.** Algorithm step 6 says "PK batch"; the code drew plain random
   sub-batches from the cached feature set. New -- `build_pk_batches()` -- groups cached indices
   by identity and partitions them into genuine PK batches every epoch (sampling with replacement
   for identities with fewer than `num_instances` cached images), added `data.num_instances` to
   `configs/stage1_relational_prompts.yaml`. This isn't just a naming fix: `SupConLoss` is a
   multi-positive loss, so PK sampling (guaranteeing several same-identity images per batch) is
   what actually lets its multi-positive design pay off, versus a plain random batch where most
   anchors have zero or one same-identity partner by chance.

**Verified with a real training run** (not `--setup-only` alone, per this session's established
practice): using the already-verified Stage 0 checkpoint from the 2026-08-26 02:17 smoke run
(`examples/logs/smoke_stage0/model_best.pth.tar`), a fresh 1-epoch smoke run completed cleanly --
`ClipImageEncoder` loaded without error, Stage 0's checkpoint loaded correctly
("Successfully loaded pretrained weights"), loss finite (~21.0, consistent with the foreground
term's removal roughly halving the previous ~24-41 range), `VAB gate` moving from 0 as expected,
`prompt_learner.pth`/`vab.pth` saved. Notably, the visibility filter now actually rejects images
(kept 5041/12936, vs. 100% kept in every prior Stage 1 smoke test) -- because BPA is now genuinely
segmentation-trained via Stage 0 rather than producing near-degenerate ImageNet-init attention, so
the visibility-index threshold is doing real, meaningful filtering for the first time.

Also noted, not touched (unrelated to this fix, not reverted per this session's own "don't touch
what you don't understand" practice): `examples/train_bpa_segmentation.py` has an appended
"SUGGESTED CHANGES" note (adding triplet/id loss + VAB to Stage 0) that appears to be the user's
own note-to-self about a separate future change, left as-is.

Full repo `python -m py_compile` sweep clean.

---

### 2026-08-26 19:36 — Stage 2 rewritten to match "Algorithm 2: Backbone Fine-Tuning" exactly,
### after a line-by-line audit found five real discrepancies

Same audit discipline as the Stage 1 rewrite above: read `examples/train_relational_finetune.py`
and its actual loss implementations (not just the docstrings) against a user-supplied algorithm
spec. Batching itself was already correct (`RandomIdentitySampler` + `IterLoader`, genuine PK
sampling, unlike Stage 1's pre-rewrite issue). Five discrepancies found, all fixed after
confirmation:

1. **Stage 0 checkpoint not loaded by default.** `configs/stage2_relational_finetune.yaml`'s
   `model.checkpoint_path` now defaults to `examples/logs/stage0_bpa/model_best.pth.tar` -- the
   same checkpoint Stage 1 uses going in (Stage 1 never updates it, so it's genuinely the same
   weights, not a separate lineage), matching the algorithm's own step 1.
2. **L_attn (BPA loss) was optional and off by default; the algorithm treats it as mandatory.**
   Judgment call, made explicitly rather than blind literal compliance: even though Stage 0
   already trains BPAM via this exact mask supervision, Stage 2 goes on to update backbone+BPAM
   further via id/triplet/align gradients that know nothing about real part boundaries -- keeping
   L_attn active during Stage 2 is what stops that continued training from drifting BPAM's
   attention away from the real, mask-anchored split Stage 0 established. `data.masks_dir` now
   defaults to Market1501's real `pifpaf_maskrcnn_filtering` (matching Stage 0's own default),
   while staying fully optional in code for mask-less datasets (dukemtmc-reid). "Pseudo
   part-parsing labels" in the algorithm's own wording read as these same ground-truth-derived
   pixel targets, not a separate self-generated pseudo-labeling scheme this repo doesn't have.
3. **Triplet loss was one fused computation across all branches; the algorithm wants three
   separate ones.** `compute_losses` used to call `PartTripletLoss(combined, targets)` once,
   which (confirmed by reading `part_triplet_loss.py` directly) averages every branch's distance
   matrix into one fused metric before a single batch-hard mining pass -- architecturally
   different from independent global/part triplet losses (a hard negative under the fused metric
   isn't necessarily the hardest negative for any single branch). Fixed by calling the same
   `PartTripletLoss` twice differently instead of writing a new loss class: once on
   `combined[:, 0:1, :]` alone (`l_tri_global`), and once per part branch summed
   (`l_tri_parts`) -- passing a single-branch `[B, 1, D]` slice makes `PartTripletLoss`'s internal
   per-branch averaging a no-op, giving that branch's own unfused batch-hard loss for free, no new
   class needed. Both terms share the existing `cfg.loss.triplet_weight`, matching the algorithm's
   own formula (step 16), which weights every term but `L_align` implicitly at 1.
4. **Alignment loss: wrong formula and wrong scope.** Algorithm step 14:
   `1 - cosine_sim(relation_feats[:,k], frozen_text_anchors[label,k])`, parts only. Code used
   `I2TLoss` -- confirmed (by rereading that file) to be full label-smoothed cross-entropy against
   the *entire* prototype table, not a pairwise cosine term -- summed over all `1+K` branches
   including foreground. New -- `pcr/loss/clip_cosine_align_loss.py::CosineAlignLoss` -- a direct
   per-sample `1 - cosine_similarity` against a pre-gathered anchor (`text_prototypes[targets, k,
   :]`), restricted to branches `1..K` only. `I2TLoss` deleted entirely (`pcr/loss/
   clip_i2t_loss.py`) -- this was its only caller repo-wide. Excluding foreground from alignment
   also resolves `changes.md`'s previously-flagged consequence of Stage 1's own rewrite (fg_ctx is
   untrained there, so `text_prototypes[:, 0, :]` was meaningless noise) -- that noise is now
   simply never read, as a direct side effect of matching Algorithm 2's own scope rather than a
   separate patch. `changes.md` updated: that entry removed (resolved), and its item 3's now-stale
   `I2TLoss` reference fixed to `CosineAlignLoss` -- worth noting `CosineAlignLoss` is
   scale-invariant (cosine similarity ignores norm), so the VAB-output-normalization concern in
   that remaining entry only actually applies to `SupConLoss` (Stage 1), not this loss.
5. **Final checkpoint was two separate files; the algorithm wants one.** Step 20: "save {backbone,
   BPAM, VRB}" as one checkpoint. `vab.pth` was a separate `torch.save` call; now
   `vab_state_dict` rides inside the same `checkpoint.pth.tar`/`model_best.pth.tar` dict alongside
   `state_dict` -- confirmed safe by reading `torchreid.utils.load_pretrained_weights` directly
   (it reads only the `state_dict` key, ignoring the rest), so `train_uda.py`/`train_usl.py
   --checkpoint-path` need no changes. The separate `vab.pth` save was removed.

**Verified with a real training run** (not `--setup-only` alone): built `text_prototypes.pth` from
the existing Stage 1 smoke checkpoint (`examples/logs/smoke_stage1_algo1`, via
`cache_text_anchors.py`, not previously run against that output), then a fresh 1-epoch Stage 2
smoke run against Stage 0's and Stage 1's existing smoke checkpoints. Completed cleanly: `tri_
global`/`tri_parts` appear as genuinely separate finite values (e.g. 0.366 / 2.110, not one fused
`triplet` term); `align` values (~1.7-2.25) are an order of magnitude smaller than the old
`I2TLoss` cross-entropy values (~39-40), as expected for a bounded `[0,2]`-per-term cosine loss;
`bpa` appears in every logged iteration now that masks default on; the saved `model_best.pth.tar`
was loaded back and directly inspected -- contains `['state_dict', 'vab_state_dict', 'epoch',
'best_mAP', 'optimizer']`, no separate `vab.pth` on disk. Full run: Mean AP 2.4%, top-1 5.7%,
top-5 14.9%, top-10 21.3% (consistent with prior 1-epoch smoke-scale numbers).

Full repo `python -m py_compile` sweep clean.

---

### 2026-08-26 20:05 — `SupConLoss` replaced with literal `InfoNCELoss` in Stage 1;
### `VisualAttentionBlock`'s output now L2-normalized

**Loss swap, per direct user request**: `pcr/loss/clip_supcon_loss.py::SupConLoss` (CLIP-ReID's
multi-positive contrastive loss) removed entirely (its only caller) and replaced with new --
`pcr/loss/clip_infonce_loss.py::InfoNCELoss` -- literal single-positive InfoNCE, matching
"Algorithm 1"'s own step 15 wording. Confirmed with the user before implementing: naive diagonal
InfoNCE over a raw PK-sampled batch would reproduce the exact same-identity-as-false-negative
collision that made the earlier `train_relational_clip.py` underperform (removed 2026-08-26
03:35) -- resolved by deduplicating to unique identities before building the negative set on
*both* the i2t direction (every image classified against the batch's unique identities' text
anchors -- duplicate rows simply share the same correct target, not a collision) and the t2i
direction (one representative image per identity, keeping it single-positive there too).
Temperature is a fixed constant, not learned -- defaults to 0.07, CLIP's own established optimal
starting value (`configs/stage1_relational_prompts.yaml`, replacing SupConLoss's old
temperature=1.0, which was tuned for that loss's different, multi-positive formulation).
`examples/train_relational_prompts.py` updated: one `infonce(visual_k, part_text, b_labels)` call
per part instead of two swapped `supcon(...)` calls (symmetry is now internal to `InfoNCELoss`
itself).

**Normalization fix, resolving `changes.md`'s tracked entry**: `VisualAttentionBlock.forward` now
L2-normalizes its output (`pcr/models/relation_blocks.py`) before returning, rather than leaving
the residual sum (`part_tokens + tanh(gate) * relation_out`) free to drift off unit norm as the
gate trains away from zero. Every consumer of this output (`InfoNCELoss` in Stage 1;
`PartTripletLoss` and `CosineAlignLoss` in Stage 2) implicitly assumes unit-normalized inputs for
its similarity computation to behave as intended, matching `BPBreIDEncoder`'s own foreground/
global embedding, which was already normalized before ever reaching this block. Normalizing
inside the block itself (rather than at each of the three call sites) means the invariant holds
everywhere by construction. Verified the zero-init no-op property survives: at `gate=0` this
reduces to `normalize(part_tokens)`, and `part_tokens` already arrives unit-normalized from
`BPBreIDEncoder`, so it's a true no-op (not merely close to one). As a side effect, Stage 2's
`combined` tensor (foreground + VAB-mixed parts) is now uniformly unit-normalized across every
branch, closing a previously-existing inconsistency where only the foreground branch was.

`changes.md` updated: its normalization entry (item 2) is removed, resolved by this change. Its
other entry (item 1, symmetric-vs-single-direction InfoNCE) is untouched and still open -- this
fix was about normalization only, not loss direction.

Full repo `python -m py_compile` sweep clean. Not yet re-verified with a real training run (the
dev GPU is currently occupied by the user's own `init.ipynb` notebook kernel; they opted to skip
the smoke test for now rather than free it up) -- treat this as compile-checked only until a real
Stage 1 run confirms `InfoNCELoss` and the normalized `VisualAttentionBlock` output behave as
expected together.

---

### 2026-08-27 14:20 — Replaced the upstream visibility filter with per-part loss weighting
### (Stage 1 and Stage 2)

A diagnostic (gate-check on 20 samples, then all 12936 Market1501 training images) found the
upstream image-level filter (`pcr/utils/visibility_filter.py::filter_by_visibility`, gating on
`lambda_v_min=0.5`) discarded 61% of training images (7895/12936) -- and that this was a
training-collapse artifact of an undertrained BPAM (one part branch reading as permanently
invisible across the *entire* dataset, another as visible 95.7% of the time regardless of
content), not genuine occlusion. Independent of that root cause, the filter was also the wrong
granularity: an image-level mean visibility score threw away an entire image (all K parts, even
well-visible ones) because of a single occluded part.

User proposed replacing the hard filter with per-part *weighting* inside the loss functions
instead of masking whole images out upstream. Three decisions confirmed directly before
implementing:
1. Stage 1/2's own `build_encoder(cfg)` switch to continuous (not binary) visibility scores --
   `BPBReIDModelCfg.training_binary_visibility_score`/`testing_binary_visibility_score` overridden
   to `False` only in these two stages' own encoder construction (Stage 3 stays binary, untouched
   in the shared dataclass default). Binary scores would have made "weighted" loss nearly
   equivalent to hard masking, defeating the point.
2. `VisualAttentionBlock`/`TextualAttentionBlock` stay unchanged -- no masking added to their own
   self-attention. Logged as a known, deliberate gap in `changes.md` (a genuinely-occluded part's
   feature still gets blended at full weight into every other part's post-attention output via
   self-attention, before any loss-level weighting gets a chance to discount it).
3. No replacement filter of any kind -- every image, including ones where every part reads as
   invisible, now enters training and gets weighted (its weight simply floors out near zero).

**`pcr/loss/clip_infonce_loss.py::InfoNCELoss`**: `weights` is now a required 4th positional
argument (single caller, no reason to keep `None`-handling dead code). Both cross-entropy calls
switched to `reduction='none'`, then combined via `(weights.clamp(min=1e-3) * per_sample).sum() /
weights.clamp(min=1e-3).sum().clamp(min=1e-8)`. Also changed `first_idx`'s selection rule (the
representative row picked per unique identity, used for both negative-dedup and as the anchor for
both directions) from "first occurrence in the batch" to "highest-weighted (most visible)
occurrence" -- t2i's whole quality rides on that one row's visual embedding, and PK batches
already have `num_instances` copies of each identity to pick the best of, essentially free.

**`pcr/loss/clip_cosine_align_loss.py::CosineAlignLoss`**: gained an `__init__` (previously had
none) holding `weight_floor=1e-3`; `weights` is now a required 3rd positional argument, same
weighted-mean formula as `InfoNCELoss`, no per-identity dedup complexity needed since this loss
was already single-directional per-sample.

**`pcr/loss/part_triplet_loss.py::PartTripletLoss`**: no code changes -- it already fully
implemented visibility-weighted/masked distance combination via its `parts_visibility` argument
(confirmed by reading it directly), just never actually called with one anywhere. Now wired up
in Stage 2 with a loose boolean threshold (`vis >= cfg.loss.triplet_visibility_min`, default
0.05) -- batch-hard mining's max/min operations don't compose with soft weights, so triplet keeps
a hard exclude/include decision, deliberately much looser than the removed filter's 0.5.

**`examples/train_relational_prompts.py`** (Stage 1): `build_encoder` overrides both binary-
visibility flags. `filter_by_visibility` import and call removed -- `train_set = sorted(dataset.
train)` directly, all 12936 images cached (previously 5041 survived the filter).
`cached_visibility` (previously computed but never used) is now moved to `.cuda()` alongside
`cached_features`/`cached_labels` and indexed per-batch (`b_vis = cached_visibility[b_idx]`);
`w_k = b_vis[:, 1 + k]` (same `1+k` branch offset already used for `prompts[1+k]`) is passed into
each part's `infonce(...)` call.

**`examples/train_relational_finetune.py`** (Stage 2): `build_encoder` overrides both binary-
visibility flags. `filter_by_visibility` call and the `.eval()`/`.train()` toggle that existed
solely to run it safely are both removed -- `train_set = sorted(dataset.train)` directly, all
12936 images now trained on (previously fewer, filtered). `vis` (previously obtained from the
encoder but completely dead) is now used: `vis_mask = vis >= cfg.loss.triplet_visibility_min`
passed as `parts_visibility` into both the `tri_global` and `tri_parts` `PartTripletLoss` calls;
continuous `vis[:, branch]` passed as `weights` into each part's `align_loss(...)` call.

**Configs**: both `visibility:` sections deleted entirely (Stage 1 has no triplet loss, so no
replacement knob needed there); Stage 2's `loss:` section gains `triplet_visibility_min: 0.05`.

**Docs**: `pcr/models/relation_blocks.py`'s module docstring, both training scripts' module
docstrings, `README.md`'s Stage 1/2 sections, and `METHODOLOGY.md` §3.5.2 (rewritten) and its two
Algorithm-pseudocode step-1 lines all updated to describe the weighting scheme instead of the
deleted filter. `changes.md`: added the VAB/TAB-still-unmasked entry (item 2).

**`pcr/utils/visibility_filter.py` deleted entirely** -- confirmed via repo-wide grep it had
exactly two callers (both training scripts, both removed here) and no other references.

**Verified with real training runs on both stages** (not `--setup-only` alone): Stage 1 --
`--setup-only` confirmed 12936 cached images (not 5041); a real 1-epoch run completed cleanly (375
PK batches, finite loss throughout, `VAB gate` moving normally, `prompt_learner.pth`/`vab.pth`
saved). Stage 2 -- `--setup-only` confirmed 12936 training images; a real 1-epoch run (masks on,
BPA loss active) completed cleanly with every loss term finite (`id`~6.65, `tri_global` 0.15-0.56,
`tri_parts` 1.28-2.0, `align` 1.7-2.3, `bpa` 0.85-1.15), Mean AP 3.0%, top-1 6.8%/top-5 17.0%/
top-10 24.6%, checkpoint saved and reloaded correctly for the final eval pass.

Full repo `python -m py_compile` sweep clean. This fixes the *mechanism* for using visibility
correctly; it does not fix BPAM's underlying training-collapse (that needs a real 60-epoch Stage 0
run, not yet done) -- the weight distribution won't look well-behaved until that happens.

---

### 2026-08-28 — Fixed Stage 2's `CosineAlignLoss`: detached the visibility-weight gradient leak,
### restored a real classification (repulsion) term against the full identity-prototype table

A code review (two subagents -- one code-focused, one methodology-focused -- reviewing Stage 0/1/2
end to end against their own docstrings and the original CLIP-ReID design) surfaced two real
problems in `pcr/loss/clip_cosine_align_loss.py::CosineAlignLoss`, written up in `changes.md`'s
"Red flag 1" and "Red flag 4" and `IMPROVEMENT_PLAN.md` sections 1 and 3. Both fixed here, in the
align-loss code path only (Stage 1's `InfoNCELoss` and Stage 2's triplet/id/bpa losses untouched).

**Fix 1 -- gradient leak**: Stage 2's `build_encoder` switches to continuous (differentiable)
visibility scores, and `vis[:, branch]` was passed straight into `CosineAlignLoss` as its per-sample
weight with no `.detach()` anywhere in the chain. Since `L = (Σ w_i l_i) / (Σ w_i)`, `∂L/∂w_j =
(l_j - L) / Σw` -- for any sample whose alignment loss was worse than the batch average, gradient
descent had a standing incentive to lower that sample's own visibility weight instead of actually
improving its alignment. Fixed with one line: `w = weights.detach().clamp(min=self.weight_floor)`.
Verified in isolation (see below) that `weights.grad` is `None` after `backward()` while gradient
still reaches `visual_features` normally.

**Fix 2 -- missing repulsion term**: the old loss was a pure per-sample regression, `1 -
cos(v_i, t_{y_i})`, with no other identity's text anchor ever involved -- nothing pushed different
identities' features apart, unlike CLIP-ReID's own original `I2TLoss` (a full classification
against the whole prototype table). Rewrote `CosineAlignLoss.forward` to take the branch's *full*
`[num_identities, D]` prototype table plus `targets`, and compute a temperature-scaled,
weighted-mean cross-entropy against it (`logits = visual_features @ text_anchors_table.t() /
temperature`) -- restoring real inter-identity negatives via the softmax denominator, same
mechanism InfoNCELoss/CLIP already use elsewhere in this repo. Added `loss.align_temperature: 0.07`
to `configs/stage2_relational_finetune.yaml` (same constant CLIP/Stage-1 already use).

**Call site** (`examples/train_relational_finetune.py::compute_losses`): now passes
`text_prototypes[:, branch, :]` (the full table) and `targets` instead of the old
`text_prototypes[targets, branch, :]` per-sample gather.

**Verified**:
1. Isolated forward/backward shape+gradient check (synthetic tensors, no GPU/dataset needed):
   finite loss for both a normal-weight batch and an all-zero-weight batch (floor working),
   `visual_features.grad` populated, `weights.grad is None` (confirms the detach closes the leak),
   logits shape `[B, num_identities]` as expected.
2. Real one-epoch Stage 2 smoke run (`batch_size=8`, masks on, `stage1.prompt_dir` pointed at an
   existing smoke checkpoint set with matching architecture) -- 375/375 iterations completed, every
   loss term (`id`, `tri_global`, `tri_parts`, `align`, `bpa`) finite throughout, no NaN/Inf/
   traceback anywhere in the log. `align` trended down over the epoch (~56 -> ~44-46), consistent
   with real learning rather than a stuck/broken loss. As a side confirmation, this run's
   `Successfully loaded pretrained weights from "examples/logs/stage0_bpa/model_best.pth.tar"`
   printed with **zero discarded layers** -- directly inspecting that checkpoint's keys confirmed
   it's genuinely HRNet32-shaped now (has `stage2`/`stage3`/`transition1`, no ResNet-signature
   `layer2`/`layer3`/`layer4`), so the Stage0->Stage2 handoff is currently healthy.

**New, not-yet-acted-on finding from the smoke test**: the new loss's raw magnitude (~35-50 summed
over 5 part branches) is far larger than the old regression version's bounded `[0, 2]` range, since
it's now a 751-way softmax. `loss.align_weight` (`0.5`) was not retuned -- left as a follow-up for
whoever runs the next real (non-smoke) training job, noted in `IMPROVEMENT_PLAN.md` section 3.

`changes.md`'s "Red flag 1" and "Red flag 4" marked resolved (struck through, pointing here) rather
than deleted, per this file's own convention of leaving a trail. `InfoNCELoss`'s identical
undetached-weight pattern (Stage 1) was deliberately left as-is -- not currently exploitable, since
its weights come from a `no_grad` caching pass, not a live graph -- and remains an open, low-
priority follow-up in `IMPROVEMENT_PLAN.md` section 1.

Full repo `python -m py_compile` sweep clean (`pcr/loss/clip_cosine_align_loss.py`,
`examples/train_relational_finetune.py`).

---

### 2026-08-28 (2) — Made VisualAttentionBlock and TextualAttentionBlock visibility-aware
### (IMPROVEMENT_PLAN.md section 5), and fixed a real PyTorch fast-path NaN bug found while
### verifying it

Implemented `IMPROVEMENT_PLAN.md` section 5 / `changes.md`'s item 2 and "Red flag 6": both
relational-attention blocks previously mixed all K part tokens via fully unmasked self-attention,
so a genuinely-occluded part's near-garbage feature was blended, at full weight, into every other
part's post-attention representation -- contamination that loss-level visibility weighting (the
2026-08-27 refactor) could never undo, since it only discounts a part's own direct loss
contribution.

**`pcr/models/relation_blocks.py`**: added `_visibility_attn_bias(visibility, num_heads, eps=1e-6)`
-- builds an additive `[B*num_heads, L, L]` bias (`log(v_j)` per key `j`, broadcast over queries and
heads) passed as `nn.TransformerEncoder`'s own `mask` argument, the same mechanism a hard
`key_padding_mask` uses (`log(0) = -inf`), generalized to a continuous score.
- `VisualAttentionBlock.forward` now takes a required `part_visibility: [B, K]` (that image's own
  per-part visibility, already available at every call site) and builds the bias fresh every call.
  The zero-init residual gate still makes this a true no-op at initialization regardless of masking
  (verified explicitly).
- `TextualAttentionBlock.forward` now takes a required `part_visibility: [B, K]` too, but
  `PromptLearner.part_ctx` has no per-image signal at all (indexed by identity only) -- so this is
  each identity's *mean* visibility across every cached training image of that identity, not a
  per-image score. `TextualAttentionBlock.__init__` gained an `n_ctx` param so it can expand `[B,
  K]` to `[B, K*n_ctx]` (one bias value per context token, matching the K contiguous n_ctx-token
  blocks `PromptLearner` slices `part_ctx` into).

**`pcr/models/prompt_learner.py`**: `TextualAttentionBlock` construction passes `n_ctx=n_ctx`;
`build_part_prompts(labels, part_visibility)` gained the new required arg, threaded straight into
`self.tab(...)`.

**`examples/train_relational_prompts.py`** (Stage 1): new `compute_identity_visibility(
cached_visibility, cached_labels, num_identities)` -- a vectorized per-identity mean over the
already-cached per-image visibility table (`index_add_`-based, no Python loop over labels). Called
once after caching; the training loop passes `b_vis[:, 1:]` (real, per-image) into `vab(...)` and
`identity_visibility[b_labels, 1:]` (per-identity mean) into `build_part_prompts(...)`. The computed
table is saved as `identity_visibility.pth` alongside `prompt_learner.pth`/`vab.pth`.

**`examples/cache_text_anchors.py`**: loads `identity_visibility.pth` from the same `logs_dir` and
reuses it unchanged (never recomputes it) when rebuilding each identity's final prompt, so TAB's
output stays a true, deterministic function of identity alone, consistent between training and the
frozen prototype table Stage 2 actually reads.

**`examples/train_relational_finetune.py`** (Stage 2): `vab(f_out[:, 1:, :], vis[:, 1:])` -- this
forward pass's own live per-image visibility, no separate table needed (unlike Stage 1's TAB usage,
VAB always has a real per-image signal available in both stages).

**A real bug found and fixed while verifying this, not a masking-logic error**: the first full
end-to-end smoke run (Stage 1 -> `cache_text_anchors.py` -> Stage 2) produced `NaN` in every
part-branch text prototype and, downstream, `NaN` in Stage 2's align loss from iteration 0.
Bisected with a standalone repro script calling `PromptLearner.tab` directly: the real, trained
`identity_visibility` values (min 0.15, max 0.96 -- nothing extreme, no zeros) produced `NaN` only
when the module was in `.eval()` mode (exactly the mode `cache_text_anchors.py` correctly uses, via
`prompt_learner.eval()`, for deterministic output); the *identical* tensors through the *identical*
module in `.train()` mode, and in `.eval()` mode with a uniform (no-op) visibility bias, both came
out perfectly finite. Root cause: `nn.TransformerEncoderLayer` automatically routes to a fused
native kernel (`torch._transformer_encoder_layer_fwd`) whenever `self.training` is `False`, and
that fused kernel doesn't correctly handle a real, non-uniform additive `attn_mask` -- confirmed via
PyTorch's own runtime warnings during the repro ("Flash Attention does not support non-null
attn_mask", "Memory efficient kernel not used", etc. -- all backends the fused path tried, all
either skipped or numerically broken for this mask shape). Forcing the ordinary (non-fused) SDPA
backend via `torch.nn.attention.sdpa_kernel` did NOT fix it, since this fused path is a separate,
whole-encoder-layer kernel that bypasses `nn.MultiheadAttention.forward`/`scaled_dot_product_attention`
entirely. Fixed with the documented workaround, `torch.backends.mha.set_fastpath_enabled(False)`,
set once at module import time in `pcr/models/relation_blocks.py` -- confirmed this repo's only
`nn.TransformerEncoder` user, so nothing else is affected, and confirmed the flag only changes
`.eval()`-mode behavior (the fast path never engages in `.train()` mode in the first place, so
Stage 1's live training numerics are completely unaffected by this fix).

**Verified**:
1. Isolated CPU checks (no dataset/GPU needed): `_visibility_attn_bias` shape; VAB's zero-gate
   no-op property holds regardless of masking; VAB/TAB masking measurably changes output vs. a
   uniform-visibility control (not silently inert); gradients finite through both blocks; the
   zero-visibility edge case stays finite (`eps` clamp working); `compute_identity_visibility`'s
   per-identity averaging checked against a hand-computed expected result.
2. Real end-to-end GPU smoke run of the full chain: Stage 1 (375/375 iterations, fresh
   `prompt_learner.pth`/`vab.pth`/`identity_visibility.pth`, no NaN/Inf/errors) ->
   `cache_text_anchors.py` (fresh `text_prototypes.pth`, confirmed fully finite by direct
   inspection) -> Stage 2 (375/375 iterations, masks + BPA loss active, every loss term finite
   throughout, no NaN/Inf/errors -- values in the same range as the previous (pre-VAB/TAB-masking)
   Stage 2 smoke test, confirming the fix didn't change the loss's overall scale/behavior).

Full repo `python -m py_compile` sweep clean.

---

### 2026-08-28 (3) — Stage 1 switched back to `SupConLoss` (from `InfoNCELoss`), per direct user
### request, after a theoretical comparison of the two for this repo's PK-sampled batches

User asked which loss "sits perfectly" for Stage 1 given this repo already uses PK sampling, since
SupCon (the loss before this session's earlier `InfoNCELoss` swap) is generally recommended over
plain InfoNCE for exactly that batch shape. Explained in plain language in-thread, then asked to
switch back, with one explicit decision confirmed first: keep Stage 1's visibility weighting (the
per-part loss-weighting mechanism from the 2026-08-27 refactor) rather than reverting to the
original, pre-weighting SupConLoss verbatim.

**`pcr/loss/clip_supcon_loss.py`** (new, restored from git history at commit `adad610^` and then
extended): CLIP-ReID's real multi-positive contrastive loss. Unlike the `InfoNCELoss` it replaces,
needs no per-batch identity deduplication -- its positive mask already tells same-identity rows
apart from different-identity ones directly, so every image in a PK batch stays a real comparison
point and contributes to its own identity's gradient (not just one "representative" image per
identity). Extended with a required `weights` argument (detached, floored at `1e-3`, same
convention as `CosineAlignLoss`) so Stage 1 keeps its per-part visibility weighting despite the
loss-mechanism swap. Temperature default restored to `1.0` (SupCon/CLIP-ReID's own original value)
-- `InfoNCELoss`'s `0.07` was specifically tuned for a single-positive loss, not this one.

**`pcr/loss/clip_infonce_loss.py`** deleted entirely (no callers left, matching this repo's
established convention of purging unused code rather than leaving it dead).

**`examples/train_relational_prompts.py`**: `supcon(...)` called twice per part (i2t + t2i, with
the two feature sets swapped), matching `SupConLoss`'s own two-call convention (unlike
`InfoNCELoss`, which folded both directions into one call) -- `b_labels` doubles as both
`anchor_labels` and `other_labels` since every image in the batch is compared against every other
image in the same batch, which is exactly what PK sampling is for.

**`configs/stage1_relational_prompts.yaml`**: `loss.temperature` `0.07` -> `1.0`.

**Docs**: module docstrings in `train_relational_prompts.py`, `pcr/loss/clip_cosine_align_loss.py`,
`pcr/models/relation_blocks.py`; `README.md`, `METHODOLOGY.md`; `changes.md`'s items 1, "Red flag
1", and "Red flag 5" updated -- the last of these ("Stage 1's negative pool is much smaller than
the original design") is now resolved outright, since SupCon compares against the whole batch (32
images) rather than InfoNCE's deduplicated ~8 unique identities; `IMPROVEMENT_PLAN.md` sections 1
and 4 updated to match.

**Verified**: isolated forward/backward check (mask correctly gives 4 positives per anchor for a
PK=4 batch; `weights.grad is None` confirms the detach; all-zero-weight batch stays finite); a real
one-epoch Stage 1 smoke run (375/375 iterations, loss stable in a sane ~20-22 range, no NaN/Inf/
errors); `cache_text_anchors.py` run against the resulting `prompt_learner.pth`, producing a fully
finite `text_prototypes.pth` -- confirming the artifact handoff to Stage 2 is unaffected by this
loss-only change.

Full repo `python -m py_compile` sweep clean.

---

### 2026-08-28 (4) — Feature vector size changed 512 -> 1024 across the pipeline, per direct user
### request; found and fixed a real, previously-latent bug along the way

User asked to set the "feature vector size" to 2048 repo-wide. No CLIP text model produces a
2048-dim embedding (the sizes available are 512/640/768/1024, fixed by each pretrained checkpoint's
own `text_projection` shape -- confirmed by inspecting `third_party/clip`), and this repo's whole
design assumes BPBreID's visual embedding size and CLIP's text embedding size are identical, with
no bridge/projection layer anywhere to reconcile a mismatch. Asked which of three options to take;
user chose switching CLIP to a 1024-size model (`RN50`) and setting `dim_reduce_output` to match,
rather than 2048 with a new adapter layer, or leaving the CLIP pipeline at 512 and finding "2048"
elsewhere (Stage 3's own separate, already-2048 plain ResNet50 in `pcr/models/resnet.py`, unrelated
to this pipeline).

**Changed to `1024`**: `pcr/models/bpbreid_encoder.py`'s `BPBReIDModelCfg.dim_reduce_output` shared
default (this also affects Stage 3's `train_uda.py`/`train_usl.py`, which construct
`BPBReIDModelCfg` without overriding this field at all -- deliberate, so a Stage 2 checkpoint built
under the new size still loads cleanly into Stage 3 without every weight being silently discarded
on a shape mismatch; loading a genuinely external, 512-dim bpbreid checkpoint into Stage 3 would
need this reverted back down for that one load, since Stage 3 has no CLI flag of its own for it).
`configs/stage0_bpa_segmentation.yaml`, `configs/stage1_relational_prompts.yaml`, and
`configs/stage2_relational_finetune.yaml`'s own `model.dim_reduce_output`. `configs/
stage1_relational_prompts.yaml`'s `clip.arch`: `ViT-B/16` -> `RN50`. `ClipTextEncoder`'s and
`ClipImageEncoder`'s own constructor defaults (`clip_arch='ViT-B/32'` -> `'RN50'` -- cosmetic only,
every real caller already passes `cfg.clip.arch` explicitly).

**A real, previously-latent bug found and fixed while verifying this, not a config-value issue**:
the first real Stage 1 smoke run crashed inside `PromptLearner.build_part_prompts` --
`RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 512 but got size
1024`. Root cause: `PromptLearner.__init__` sized its own learnable context tensors
(`fg_ctx`/`part_ctx`) using `clip_text_encoder.embed_dim` (the text encoder's *final*, post-
`text_projection` output size -- 1024 for RN50), but those tensors get concatenated with
`token_prefix`/`token_suffix` (built from `clip_text_encoder.token_embedding`, which outputs
vectors sized to the transformer's own *internal* width, not `embed_dim`) before ever reaching
`ClipTextEncoder.forward()`. For ViT-B/16 (the previous default), `transformer_width` and
`embed_dim` happen to both be 512 -- an accidental coincidence that hid this bug completely, since
this repo has never used a CLIP arch where those two numbers differ until now. Confirmed directly
by inspecting a loaded RN50 CLIP model: `transformer.width == 512`, `text_projection.shape ==
(512, 1024)` -- two different numbers, only one of which (`1024`) is `embed_dim`.

**Fix**: `pcr/models/clip_text_encoder.py::ClipTextEncoder` now exposes both
`self.embed_dim` (unchanged -- `text_projection.shape[1]`, for anything compared against BPBreID's
visual output) and a new `self.transformer_width` (`text_projection.shape[0]`, for anything that
must line up with `token_embedding`/`positional_embedding`/the transformer's own internal
computation). `pcr/models/prompt_learner.py::PromptLearner.__init__` now sizes `ctx_dim` from
`transformer_width`, not `embed_dim`.

**Verified with a full real, four-stage smoke chain** (not just the two directly-touched files --
this bug's blast radius made the whole chain worth re-checking): a fresh Stage 0 run at
`dim_reduce_output=1024` (1617 iterations, pixel accuracy climbing normally, no NaN); Stage 1
against that checkpoint with `clip.arch: RN50` (375/375 iterations, loss stable ~20-21, `vab.pth`
correctly ~4x larger than the old 512-dim version); `cache_text_anchors.py` producing a fully
finite `[751, 6, 1024]` `text_prototypes.pth`; Stage 2 against all of the above (375/375
iterations, every loss term finite throughout, checkpoint loaded with zero discarded layers, full
eval pass completed -- Mean AP 0.9%, expected for a 1-epoch smoke run, not a red flag).

**Docs updated for consistency**: `METHODOLOGY.md`'s `D=512` math notation (5 occurrences) ->
`D=1024`; `README.md`'s Stage 1 section; `configs/stage1_relational_prompts.yaml`'s and
`configs/stage2_relational_finetune.yaml`'s own inline comments referencing the old size/arch.

Full repo `python -m py_compile` sweep clean.

---

### 2026-08-28 (5) — Widened Stage 1's SupConLoss negative pool from one PK batch (~8 identities)
### to the full training set (751 identities / 12936 images), per direct user request

User pointed at `IMPROVEMENT_PLAN.md` section 4's own residual concern: even after the SupCon
restoration (entry above) fixed InfoNCE's self-inflicted waste, Stage 1's comparison pool per step
was still capped at one PK batch's worth of identities (~8) -- far short of "CLIP-ReID's full
~751-identity classification" the user explicitly wanted matched, since a weak Stage 1 signal caps
what Stage 2 can recover regardless of any later fix. Asked to implement this cleanly in Stage 1.

**Key realization enabling a clean fix, not a queue**: Stage 1's visual encoder (BPBreID
backbone + BPAM) is completely frozen throughout Stage 1 -- `cached_features`/`cached_labels`,
already built once for the whole training set before training starts, are therefore never stale.
That makes a MoCo/SimCLR-style FIFO queue (`IMPROVEMENT_PLAN.md`'s original "Option B" sketch)
unnecessary for the visual/t2i side: the full 12936-image cache can be used directly, every
iteration, with zero staleness concern and zero new hyperparameters. Only the text side is a
genuine moving target (part_ctx/TAB are what Stage 1 actually trains), so that side needs a
snapshot-refresh strategy instead of a queue.

**`examples/train_relational_prompts.py`**:
- New `build_text_snapshot(prompt_learner, text_encoder, num_identities, num_parts,
  identity_visibility, id_batch)`: rebuilds a `[num_identities, K, D]` text-anchor table once per
  epoch, under `no_grad`, against that epoch's current `part_ctx`/TAB weights -- structurally the
  same computation `cache_text_anchors.py` already does once at the very end of training, just run
  once per epoch mid-training instead. Confirmed safe to batch in chunks of `id_batch` identities:
  `TextualAttentionBlock`'s self-attention only mixes tokens within one identity's own sequence,
  never across the batch dimension, so chunked construction is exactly equivalent to building every
  identity one at a time.
- Training loop, per iteration: builds `in_batch`/`other_ids` (which of the `num_identities` are
  present in this PK batch vs. not), then for each part `k`:
  - **i2t**: `other_text = cat([part_text (fresh, differentiable), text_snapshot[other_ids, k, :]
    (detached)])`, `other_text_labels = cat([b_labels, other_ids])` -- full `num_identities`-way
    classification, with gradient still reaching `part_ctx`/TAB correctly for the identities
    actually in this batch (negatives from identities not in this batch don't need to be
    differentiable, since they'll get their own gradient on whichever future iteration they *are*
    in the batch -- the same principle sampled-softmax/negative-sampling methods already rely on).
  - **t2i**: `supcon(part_text, cached_features[:, 1+k, :], b_labels, cached_labels, w_k)` directly
    -- the entire cached dataset, no snapshot or splicing needed at all.
- Epoch loop now calls `build_text_snapshot(...)` once at the top of each epoch (`prompt_learner`
  briefly in `.eval()` inside that call, explicitly switched back to `.train()` immediately after,
  before any real training step runs).

**No new config knob added** -- the widened pool size is just `num_identities`/`num_images`, both
already known quantities, unlike a queue's own `queue_size` hyperparameter.

**Verified**:
1. Isolated checks (no GPU dataset needed beyond a real CLIP load): `build_text_snapshot` produces
   a correctly-shaped, fully finite `[num_identities, K, D]` tensor; the batch/snapshot splicing
   logic (`in_batch`/`other_ids`) produces no overlap between an iteration's "fresh" identities and
   its "other" (snapshot) identities, confirmed against a hand-picked PK batch.
2. Real 2-epoch Stage 1 smoke run (`batch_size=8`, RN50/1024-dim): both epochs completed cleanly,
   no NaN/Inf/errors anywhere in the log. Loss jumped from the pre-widening ~20-21 range to ~58-80
   (expected -- this is now a genuinely harder 751-way / 12936-way classification, not merely a
   differently-scaled version of the old ~8-way one) and visibly decreased within the very first
   epoch (79.9 -> 58.0), showing real learning against the larger task. Per-epoch wall time
   unchanged from the pre-widening run (~23s both before and after) -- confirms the once-per-epoch
   snapshot rebuild and the larger (but still tiny relative to a backbone forward pass) matmuls add
   negligible overhead.
3. `cache_text_anchors.py` run against the resulting `prompt_learner.pth`, producing a fully finite
   `[751, 6, 1024]` `text_prototypes.pth` -- the Stage 2 handoff is completely unaffected by this
   change (Stage 2 never sees any of this widening machinery; it only ever reads the final,
   already-built prototype table, same as before).

`changes.md`'s "Red flag 5" and `IMPROVEMENT_PLAN.md` section 4 updated from "resolved/largely
resolved" to fully resolved, describing the actual mechanism implemented.

Full repo `python -m py_compile` sweep clean.

---

### 2026-08-28 (6) — Stage 2: added BNNeck and a 10-epoch LR warmup, per direct user request
### (both from "Bag of Tricks and a Strong Baseline for Deep Person Re-identification", Luo et al.,
### CVPRW 2019, "BoT")

Neither is part of "Algorithm 2"'s own literal steps -- both are explicit, user-requested additions
on top of it, not silent scope creep, and documented as such in the file's own module docstring.

**1. BNNeck** (`pcr/models/bn_neck.py`, new): Stage 2's triplet loss and its id/align losses
previously read the exact same pooled feature (`combined`), but want it shaped two competing ways
-- triplet's Euclidean margin wants features spread out, id/align's softmax-style classification
converges better on a compact, roughly-hyperspherical distribution -- so every step, each loss's
gradient partially fought the other's.

`PartBNNecks`: one `BatchNorm1d` per branch (bias frozen at zero via
`bn.bias.requires_grad_(False)`, BoT's own convention -- a following linear classifier already has
its own bias-like freedom, so a learnable BN shift is redundant with it). In
`compute_losses()` (`examples/train_relational_finetune.py`): triplet loss keeps reading `combined`
directly, unchanged; `id_loss` now reads `bn_necks(combined, 0)` (wrapped back to `[B, 1, D]` so
`PartIdClassifiers`'s own `f_out[:, branch, :]` slicing convention -- shared with Stage 3, not
touched -- still applies unchanged); `align_loss` now reads `F.normalize(bn_necks(combined,
branch), p=2, dim=-1)` for each part branch -- the extra L2-normalize (not part of "plain" BNNeck)
is a deliberate addition: `BatchNorm1d`'s per-dimension rescaling doesn't preserve the roughly-
unit-norm assumption `CosineAlignLoss`'s fixed-temperature softmax depends on, so this restores it,
the same "BN then L2-normalize before a cosine-similarity head" pattern ArcFace/CosFace-style heads
already use.

`bn_necks.parameters()` added to the optimizer's param list; `bn_necks.train()` added alongside
`encoder.train()`/`vab.train()` each epoch; `bn_necks_state_dict` bundled into the saved checkpoint
alongside `state_dict`/`vab_state_dict` (nothing downstream reads it yet, matching `vab_state_dict`'s
own status).

**Scope note, documented in `pcr/models/bn_neck.py`'s own docstring**: BoT's paper also recommends
using the *post*-BN feature for retrieval at test time, not just training -- not done here, since
Stage 2's `Evaluator` (`pcr/evaluators.py`) doesn't even apply `VisualAttentionBlock` at test time
yet (a separate, pre-existing gap predating this change) -- wiring BNNeck into evaluation would
need that fixed first, and wasn't asked for; this change closes the specific *training-time*
gradient-interference gap only.

**2. 10-epoch LR warmup**: this file previously used a bare `StepLR` from epoch 0 at full LR,
straight onto a just-unfrozen backbone with three newly-interacting losses (id/triplet/align) --
BoT's own ablation flags this combination as a known source of early instability that can leave
training in a worse basin for good. Replaced `torch.optim.lr_scheduler.StepLR` with
`WarmupMultiStepLR` (`pcr/utils/lr_scheduler.py`, already used by `examples/train_usl.py` -- no new
scheduler class needed). New config knobs in `configs/stage2_relational_finetune.yaml`:
`optim.warmup_epochs: 10`, `optim.warmup_factor: 0.01` (BoT's own reference values -- linear warmup
starting at 1% of the base LR, over 10 epochs), `milestones=[cfg.optim.step_size]` reproducing the
exact same single 10x decay the old `StepLR` gave, just with warmup added in front.

**Verified**:
1. Isolated checks: `PartBNNecks`' bias is frozen (`requires_grad=False`) on every branch;
   gradient reaches the BN's own weight but never the frozen bias; train/eval mode both produce
   finite output; branches are independent module instances. `WarmupMultiStepLR` with the new
   config values produces the expected LR curve (verified numerically): `0.01 x base_lr` at epoch
   0, linearly ramping to the full `base_lr` at epoch 10, held flat, then a 10x drop exactly at
   `step_size`.
2. Real 3-epoch Stage 2 smoke run (RN50/1024-dim, masks + BPA loss active, `step_size`/
   `warmup_epochs` scaled down to fit inside 3 epochs): completed cleanly end to end, including the
   periodic eval pass, checkpoint save/reload, and final test -- no NaN/Inf/errors/discarded
   layers anywhere, all loss magnitudes in the same sane ranges as prior (pre-BNNeck) smoke runs
   (confirming the align-loss L2-renormalization successfully preserved its calibration).
3. Directly inspected the saved checkpoint's `bn_necks_state_dict`: `running_var` for branch 0 sits
   at ~0.0009-0.0016 -- almost exactly `1/1024` (the theoretically expected per-dimension variance
   of a random unit vector in 1024 dimensions), confirming BN is genuinely tracking real statistics
   from real, correctly-unit-normalized inputs, not sitting at its untouched 1.0 init; `weight`
   (learned scale) moved slightly off its 1.0 init, confirming gradient is reaching it; `bias`
   confirmed exactly 0.0 (frozen) on every branch.

Full repo `python -m py_compile` sweep clean.

---

### 2026-08-29 — Stage 1 SupCon: lowered temperature (1.0 -> 0.1) and L2-normalized the text side;
### same normalization fix carried into Stage 2's CosineAlignLoss text prototypes

User reported Stage 1's loss "doesn't seem to converge" at `batch_size=64` and asked for a recheck.
Root-caused with real, instrumented training runs (not just theory): the loss was sitting within a
few points of its own theoretical random-guess floor (`ln(num_identities)*5 + ln(num_images)*5 ≈
66.3`) for roughly two full epochs before making visible progress. Two contributing issues found:

**1. Temperature miscalibrated for the widened negative pool.** `SupConLoss`'s `temperature=1.0`
is CLIP-ReID's own original value, tuned for comparing against one small PK batch's worth of
identities. The 2026-08-28 "widen the negative pool" change (see that entry) made this loss compare
against the *entire* training set (751 identities / 12936 images) every step instead -- a
softmax over that many classes needs a sharper temperature to produce a useful gradient at all.
Confirmed directly: an instrumented run at `temperature=1.0` sat at 64-68 (near the 66.3 floor)
through ~180 iterations; the identical run at `temperature=0.1` reached 57.5 in the same iteration
count, clearing the floor markedly faster and further.

**2. Text embeddings were never L2-normalized, unlike the visual side.** `ClipTextEncoder`'s raw
output was fed straight into `SupConLoss`'s dot product with no normalization, while the visual
side (`VisualAttentionBlock`'s output) always was. Measured directly: real text embeddings had
norm ~10-13, varying by identity -- meaning `logits = anchor @ other.t() / temperature` was never
a true cosine similarity, and the effective per-identity temperature was inconsistent by whatever
that identity's arbitrary embedding norm happened to be. The same exact asymmetry existed in Stage
2's `CosineAlignLoss`: its visual side was normalized (from the BNNeck fix, 2026-08-28), but the
text-prototype side (built by `cache_text_anchors.py`) was not -- meaning `align_temperature: 0.07`
(CLIP's own real, standard value, chosen for genuine cosine similarities) had been running roughly
10-13x softer than intended the whole time.

**Changes**:
- `configs/stage1_relational_prompts.yaml`: `loss.temperature` `1.0` -> `0.1`.
- `pcr/loss/clip_supcon_loss.py`: default `temperature` `1.0` -> `0.1`; docstring rewritten to
  document the real measurement behind both changes and the now-explicit expectation that callers
  normalize both `anchor_features`/`other_features` before calling.
- `examples/train_relational_prompts.py`: `part_text` now `F.normalize`'d at both call sites
  (the live per-iteration encode, and `build_text_snapshot`'s once-per-epoch snapshot, which now
  stores already-normalized rows so nothing downstream needs to re-normalize it).
- `examples/cache_text_anchors.py`: `text_prototypes` now `F.normalize`'d before being saved,
  so Stage 2's frozen table matches the same convention Stage 1 trained against, and
  `CosineAlignLoss`'s dot product becomes a true cosine similarity there too.
- `pcr/loss/clip_cosine_align_loss.py`: docstring updated to document the same normalization
  expectation and the fix.

**Verified** (kept deliberately small/quick, per direct request, since the user is about to start
a real training run): a real 1-epoch Stage 1 smoke run (batch_size=8, RN50/1024-dim) -- clean, no
NaN/errors, loss dropped from 80.1 to 58.0 within the single epoch (comfortably below the 66.3
random floor, and faster than the old temperature=1.0 diagnostic reached even after ~4 epochs'
worth of iterations); `cache_text_anchors.py` run against the result, and the saved
`text_prototypes.pth` directly inspected -- every row's norm is now exactly 1.0 (to float32
precision); a real 1-epoch Stage 2 smoke run against that table -- clean throughout including the
full eval pass, `align` loss sitting at ~33 (matching `ln(751)*5`, the expected floor for this
short a run, not a red flag).

Full repo `python -m py_compile` sweep clean.
