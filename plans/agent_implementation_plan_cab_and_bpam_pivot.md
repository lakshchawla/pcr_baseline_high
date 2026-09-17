# Agent Implementation Plan
## Track 1 (BUILD NOW): CAB integration on top of the existing pipeline — `train_relational_prompts.py`
## Track 2 (PLAN ONLY / PLACEHOLDER): BPAM pivot — `train_relational_prompts_combined.py`

---

## 0. How to execute this document (read this first, agent)

This file contains **two independent tracks**. Do not interleave their code changes.

- **Track 1 is the active build target.** Execute it fully, in the step order given, with a verification gate at the end of every step before moving to the next one. If a verification gate fails, stop and report — do not proceed to the next step with a known-broken intermediate state.
- **Track 2 is planning-only for now.** Its deliverable is a single new file, `examples/train_relational_prompts_combined.py`, containing a structured docstring plan and stub class/function signatures with `raise NotImplementedError(...)` bodies — not a working implementation. Do not implement Track 2's internals unless explicitly told to proceed past the placeholder stage. The reason for creating the file now rather than later is so the architectural decision is recorded in-repo, version-controlled, and reviewable, without committing engineering time to it yet.
- Work in a dedicated branch per track (`feature/cab-integration` for Track 1, `plan/bpam-pivot` for Track 2) so Track 2's placeholder commit doesn't block or get entangled with Track 1's real changes.
- Every step below states: **what to touch**, **what not to touch**, **exact code/signature**, and a **verification gate**. Treat the verification gate as a hard stop condition, not a suggestion.

---

# TRACK 1 — Option 1 pathway: full CAB integration, BPAM/Stage 0 unchanged

## 1.1 Objective, restated for an executor with no prior context

The existing pipeline is: **Stage 0** (pretrain BPAM alone) → **Stage 1** (`train_relational_prompts.py`: freeze backbone+BPAM, train `ctx_params`+TRB+VRB via contrastive loss) → **cache_text_anchors.py** (freeze `ctx_params`+TRB, forward-pass once, save a lookup table) → **Stage 2** (`stage2_engine.py`: unfreeze backbone+BPAM+VRB, train against cached anchors) → **Stage 3** (optional SPCL UDA).

The diagnosed problem: the cached text anchors are calibrated against BPAM's part definition *at the moment Stage 1 finished*, but BPAM keeps changing in Stage 2 — so by late Stage 2, the anchors describe a part boundary that no longer matches what BPAM currently outputs.

This track does two things on top of the existing pipeline, without touching Stage 0 or Stage 1's core logic:
1. **Bound the drift** (Option 1b: a floor on `L_attn`'s weight so BPAM can't wander arbitrarily far from its Stage-0/Stage-1 anatomical convention).
2. **Add a genuine cross-attention module (CAB)** between the visual and text branches in Stage 2, per Section 3 of the integration plan, so the visual and text sides actively ground each other instead of only being pulled together by a static cosine-similarity loss against a fixed lookup table.

## 1.2 Full file manifest for this track

| File | Status | Role |
|---|---|---|
| `configs/stage0_bpa_segmentation.yaml` | **unchanged** | do not touch |
| `examples/train_relational_prompts.py` | **modified, minimally** | Stage 1 entry point; gains one new responsibility (see Step 6) |
| `configs/stage1_relational_prompts.yaml` | **unchanged** | do not touch |
| `scripts/cache_text_anchors.py` | **modified** | carry forward the `L_attn` floor value as metadata; no logic change to the anchor computation itself |
| `configs/stage2_backbone_finetune.yaml` | **modified** | add `attn_weight_floor`, `cab` block, `crossalign` schedule params |
| `models/relation_blocks.py` | **modified** | add `CrossAttentionBlock` class (new), alongside existing `VisualRelationBlock`/`TextRelationBlock` (untouched) |
| `models/clip_text_branch.py` | **modified** | add a forward-hook accessor for the internal self-attention matrix |
| `models/fusion_head.py` | **modified** | consume `vis_grounded` (post-CAB) instead of `relation_feats` (post-VRB only) |
| `losses/cross_attn_align_loss.py` | **new file** | implements `L_crossalign` |
| `engine/stage2_engine.py` | **modified** | instantiate CAB, wire `vis_grounded` through, add `L_attn` floor and `L_crossalign` with its warmup schedule |
| `scripts/run_pipeline.py` | **modified, trivial** | no new stages inserted; only passes through the two new config blocks |

Everything under Stage 0 and Stage 1 stays exactly as it is today. **Do not modify `models/attention_head.py` in this track.**

## 1.3 Step-by-step tasks

### Step 1 — Baseline verification (no code changes)

Before touching anything, confirm the current pipeline runs end-to-end on a small subset (e.g. 20 identities, 2 epochs per stage) and record baseline mAP/Rank-1 on Market-1501's standard test split. This is your regression baseline for every later step.

**Verification gate:** a baseline run completes stage0→stage1→cache→stage2 without error, and you have a recorded mAP/Rank-1 number written down before any further edits. Do not proceed to Step 2 without this number — every later step's verification depends on being able to say "did this go up, down, or stay flat" relative to it.

### Step 2 — Add the `L_attn` weight floor (Option 1b)

**Touch:** `configs/stage2_backbone_finetune.yaml`, `engine/stage2_engine.py`.

Add to the Stage 2 config:
```yaml
loss_weights:
  attn_floor: 0.15        # never let L_attn's weight decay below this
  attn_initial: 1.0
  attn_decay_epochs: 20   # cosine-decay from attn_initial to attn_floor over this many epochs
```

In `stage2_engine.py`, find the existing schedule that decays `L_attn`'s weight (if one already exists — check before assuming; if `L_attn` currently runs at a constant weight with no decay, you're adding a decay schedule, not just a floor). The scheduling function should be:

```python
def attn_loss_weight(epoch, initial, floor, decay_epochs):
    if epoch >= decay_epochs:
        return floor
    progress = epoch / decay_epochs
    return floor + 0.5 * (initial - floor) * (1 + math.cos(math.pi * progress))
```

Wire this into the existing training loop where `L_attn` is computed and summed into the total loss — multiply by `attn_loss_weight(current_epoch, cfg.attn_initial, cfg.attn_floor, cfg.attn_decay_epochs)` rather than a fixed constant.

**Verification gate:** run Stage 2 alone (reusing Stage 0/1/cache artifacts from Step 1's baseline) for the same number of epochs as baseline. Confirm training doesn't diverge (loss curves stay finite, no NaN) and mAP is within noise of baseline (this step alone shouldn't move the needle much — it's a safety net for what comes next, not the fix itself). If mAP drops noticeably, the floor value is probably too high (BPAM being held too close to its stale definition) — try `attn_floor: 0.05` before proceeding.

### Step 3 — Build `CrossAttentionBlock` (CAB)

**Touch:** `models/relation_blocks.py`. Add a new class; do not modify `VisualRelationBlock` or `TextRelationBlock`.

```python
class CrossAttentionBlock(nn.Module):
    """
    Cross-attends a query token set (K+1 tokens) against a context token set
    (K+1 tokens) from the other modality. Used bidirectionally but asymmetrically:
    one instance for text-queries-image, a separate instance for image-queries-text.
    Do not share weights between the two directions.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(1))   # starts at 0 -> tanh(0)=0 -> block starts as identity

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor):
        """
        query_tokens:   (B, K+1, D)
        context_tokens: (B, K+1, D)
        Returns:
            updated_query: (B, K+1, D)
            attn_weights:  (B, num_heads, K+1, K+1)  -- averaged over heads before returning to caller
        """
        B, N, D = query_tokens.shape
        Q = self.q_proj(query_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K_ = self.k_proj(context_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(Q @ K_.transpose(-1, -2) / (self.head_dim ** 0.5), dim=-1)  # (B, heads, N, N)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        updated_query = query_tokens + torch.tanh(self.gate) * out
        return updated_query, attn.mean(dim=1)   # average heads -> (B, N, N) for the alignment loss
```

**Verification gate:** write a standalone unit test (`tests/test_cross_attention_block.py`) that instantiates `CrossAttentionBlock(dim=512, num_heads=4)`, feeds random tensors of shape `(4, K+1, 512)` for both query and context, and asserts (a) output shape matches input query shape, (b) with `gate` initialized to 0, output is numerically identical to the input query (confirms the identity-start property, critical for not destabilizing Stage 2 on the first forward pass), (c) `attn_weights` rows sum to 1 (confirms softmax is applied over the correct dimension). Do not proceed to Step 4 until this test passes.

### Step 4 — Expose the frozen CLIP text encoder's internal self-attention

**Touch:** `models/clip_text_branch.py`.

Add a forward-hook accessor. Do not modify the frozen encoder's weights or forward logic — this is a read-only tap.

```python
class ClipTextBranch(nn.Module):
    # ... existing __init__, forward, etc. unchanged ...

    def register_self_attn_hook(self, layer_index: int = -1):
        """
        Registers a forward hook on the specified transformer resblock's
        self-attention module to capture its attention weights.
        layer_index=-1 means the last transformer block.
        Must be called once before any forward pass you want to capture from.
        """
        self._captured_attn = None
        target_block = self.clip_model.transformer.resblocks[layer_index].attn

        def hook(module, input, output):
            # output[1] is attn_output_weights for nn.MultiheadAttention with need_weights=True;
            # confirm this against the actual CLIP implementation's attention module signature
            # before assuming index 1 -- OpenCLIP's custom attention may return a different tuple shape.
            self._captured_attn = output[1].detach()

        self._hook_handle = target_block.register_forward_hook(hook)

    def get_captured_self_attention(self):
        if self._captured_attn is None:
            raise RuntimeError("No self-attention captured -- did you call register_self_attn_hook() "
                                "and then run a forward pass before calling this?")
        return self._captured_attn   # (B, num_tokens, num_tokens) or (B, heads, num_tokens, num_tokens) -- verify shape empirically

    def remove_self_attn_hook(self):
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
```

**Important implementation caution, flag before writing final code:** whether CLIP's text transformer's attention module exposes per-head weights via a simple forward hook depends on the exact implementation (OpenAI's original CLIP code vs. OpenCLIP vs. HuggingFace's `CLIPTextModel` all differ here — some require `output_attentions=True` passed at call time rather than a hook, since the attention weights may not be computed/retained at all in the default fast-path, e.g. if scaled-dot-product-attention kernels are used which don't materialize the full attention matrix). **Before implementing this step for real, the agent must first inspect which CLIP implementation `clip_text_branch.py` actually wraps** (`grep -n "import clip\|import open_clip\|transformers" models/clip_text_branch.py`) and confirm the correct mechanism for that specific library. Do not guess and ship a hook that silently returns `None` or garbage.

**Verification gate:** after implementing (with the correct mechanism for your actual CLIP backend confirmed), run a forward pass on a dummy batch of the combined part-prompt sequence and confirm `get_captured_self_attention()` returns a tensor of shape `(B, K+1, K+1)` (after averaging heads if needed) with rows summing to ~1. If the combined sequence includes the fixed template words (not just the K+1 learnable slots), you'll need to slice out just the K+1 slot positions from the full token-length attention matrix — confirm the slicing indices are correct by checking that the diagonal is highest for identical/self positions before moving on.

### Step 5 — Implement `L_crossalign`

**Touch:** new file `losses/cross_attn_align_loss.py`.

```python
import torch
import torch.nn.functional as F

def cross_attention_alignment_loss(A_cross: torch.Tensor, A_text_internal: torch.Tensor) -> torch.Tensor:
    """
    A_cross:         (B, K+1, K+1) -- CAB's image-queries-text cross-attention matrix
    A_text_internal: (B, K+1, K+1) -- CLIP text encoder's internal self-attention over the
                                        same K+1 part/global slots, stop-gradiented (already
                                        .detach()'d by the caller via get_captured_self_attention())
    Returns: scalar loss (mean KL divergence, row-wise, averaged over batch and rows)
    """
    A_cross_log = torch.log(A_cross.clamp_min(1e-8))
    loss = F.kl_div(A_cross_log, A_text_internal, reduction="none", log_target=False)
    return loss.sum(dim=-1).mean()   # sum over the K+1 target distribution dim, mean over batch and rows
```

Do not import anything from `engine/stage2_engine.py` here — keep this a pure function with no side effects, so it's independently unit-testable.

**Verification gate:** unit test with two identical uniform `(2, 5, 5)` distributions should return ~0 loss; two maximally different one-hot distributions per row should return a large, finite, non-NaN loss.

### Step 6 — One addition to `train_relational_prompts.py` (Stage 1)

**Touch:** `examples/train_relational_prompts.py`. This is the only change to Stage 1 in this entire track, and it is additive/non-breaking.

Stage 1 currently trains `ctx_params`+TRB+VRB and (presumably) discards the combined part-prompt token sequence after each forward pass. For CAB's `L_crossalign` in Stage 2 to work, `get_captured_self_attention()` needs to be called against the **same fixed template + `ctx_params` structure** that Stage 2 will reconstruct at inference time from the cached checkpoint. Concretely:

1. Locate where `train_relational_prompts.py` currently constructs the combined part-prompt sequence (fixed template tokens + `ctx_params` interspersed) before feeding it to the frozen text encoder.
2. Add one line saving that exact sequence-construction function (or its output shape/slot-index mapping) into the Stage 1 checkpoint metadata, e.g. `checkpoint["prompt_slot_indices"] = slot_indices`, so Stage 2 can reconstruct the identical sequence layout when it later calls `register_self_attn_hook()` + a forward pass to get `A_text_internal`. Without this, Stage 2 risks slicing the wrong token positions out of the attention matrix (see Step 4's caution) because it reconstructed the sequence slightly differently than Stage 1 did.

This is a small, additive change — do not restructure how Stage 1 trains anything.

**Verification gate:** re-run Stage 1 (small subset is fine), confirm the checkpoint now contains `prompt_slot_indices`, and confirm loading that checkpoint in a throwaway script reconstructs a sequence whose learnable-token positions match what `prompt_slot_indices` claims.

### Step 7 — Wire CAB into `stage2_engine.py`

**Touch:** `engine/stage2_engine.py`, `configs/stage2_backbone_finetune.yaml`.

Add to config:
```yaml
cab:
  num_heads: 4
  crossalign_warmup_fraction: 0.25   # fraction of Stage 2 total epochs before L_crossalign activates
  crossalign_ramp_fraction: 0.15     # fraction of Stage 2 total epochs over which lambda ramps 0 -> max
  crossalign_lambda_max: 0.3
```

In the engine:
```python
# instantiation (once, at Stage 2 setup)
cab_t2i = CrossAttentionBlock(dim=feature_dim, num_heads=cfg.cab.num_heads)
cab_i2t = CrossAttentionBlock(dim=feature_dim, num_heads=cfg.cab.num_heads)
clip_text_branch.register_self_attn_hook(layer_index=-1)

# inside the training step, after existing relation_feats / prompt_feats are computed
text_grounded, A_cross_t2i = cab_t2i(query_tokens=prompt_feats, context_tokens=relation_feats)
vis_grounded,  A_cross_i2t = cab_i2t(query_tokens=relation_feats, context_tokens=prompt_feats)

# vis_grounded replaces relation_feats for everything downstream from here on:
#   fusion_head(vis_grounded) instead of fusion_head(relation_feats)
#   L_align computed against vis_grounded instead of relation_feats
#   L_gilt computed against vis_grounded instead of relation_feats

A_text_internal = clip_text_branch.get_captured_self_attention()   # already detached inside the accessor
lambda_crossalign = crossalign_schedule(current_epoch, total_epochs, cfg.cab)
L_crossalign = lambda_crossalign * cross_attention_alignment_loss(A_cross_i2t, A_text_internal)

loss = L_attn_weighted + L_gilt + L_id_global + L_tri_global + L_align + L_crossalign
```

```python
def crossalign_schedule(epoch, total_epochs, cab_cfg):
    warmup_end = cab_cfg.crossalign_warmup_fraction * total_epochs
    ramp_end = warmup_end + cab_cfg.crossalign_ramp_fraction * total_epochs
    if epoch < warmup_end:
        return 0.0
    if epoch >= ramp_end:
        return cab_cfg.crossalign_lambda_max
    progress = (epoch - warmup_end) / (ramp_end - warmup_end)
    return cab_cfg.crossalign_lambda_max * progress
```

**Verification gate:** run Stage 2 for a full short schedule (e.g. 30 epochs on the small subset). Confirm: (a) loss curves for `L_id_global`/`L_tri_global`/`L_align` behave similarly to the Step 2 checkpoint run for the first `warmup_end` epochs (since `L_crossalign` is exactly 0 during that window, CAB's presence shouldn't change anything else yet — if it does, something is wrong with how `vis_grounded` is wired in, since CAB's gate starts at 0 and should be a no-op at initialization); (b) once `L_crossalign` activates, it decreases over subsequent epochs (confirms CAB's cross-attention is actually learning to align, not just adding noise); (c) no NaNs, no divergence.

### Step 8 — Update `fusion_head.py`

**Touch:** `models/fusion_head.py`.

Change the input this module expects from `relation_feats` to `vis_grounded`. This should be close to a one-line signature/docstring change if the module is already shape-agnostic (both are `(B, K+1, D)`); confirm no hardcoded assumptions elsewhere about which tensor "relation_feats" refers to (search for other call sites: `grep -rn "relation_feats" --include=*.py .`).

**Verification gate:** confirm every call site of `fusion_head` in `stage2_engine.py` (and anywhere else it's called — check `eval/metrics.py`'s inference path too, since inference also needs to run BPAM→VRB→CAB→fusion in the same order as training) has been updated consistently. Run the existing evaluation script end-to-end on a small gallery/query split and confirm it doesn't crash and produces a plausible (non-degenerate) mAP number.

### Step 9 — `run_pipeline.py` and `cache_text_anchors.py` — trivial pass-through updates

**Touch:** `scripts/run_pipeline.py` (pass the new config blocks through, no new stage insertion), `scripts/cache_text_anchors.py` (no logic change — just confirm it still runs correctly given Step 6's addition of `prompt_slot_indices` into the Stage 1 checkpoint; it should simply carry that field through into its own output artifact for Stage 2 to read).

**Verification gate:** full pipeline run, Stage 0 through Stage 2, on the small subset, start to finish, with no manual intervention between stages.

### Step 10 — Full ablation run (only after all of the above passes)

Run on the full Market-1501 training set (751 identities), comparing against the Step 1 baseline:

| Run | Config |
|---|---|
| Baseline (Step 1 recorded number) | no floor, no CAB |
| Floor only | Step 2's `attn_floor` active, no CAB |
| Floor + CAB, no `L_crossalign` | `crossalign_lambda_max: 0.0` |
| Floor + CAB + `L_crossalign` (full Track 1) | full config as specified above |

**This four-row table is the actual deliverable of Track 1** — it tells you how much of any improvement comes from the cheap floor fix alone versus how much CAB and `L_crossalign` add on top, which is exactly the information needed to decide whether Track 2 (a much larger architectural change) is worth pursuing at all.

---

# TRACK 2 — Option 3 pathway (placeholder plan): pivoting away from standalone BPAM

## 2.1 Objective, and why this is a placeholder, not a build order

Option 3 (from the integration plan, §2) proposes that the part localizer stop being a pure function of the image alone (`Conv1x1(F) → softmax`, no text input) and instead become a function of both the image **and** a signal derived from the current text-side part representation — closing the staleness loop structurally, at the cost of a materially more complex, jointly-trained architecture with no discrete "Stage 0 checkpoint" to fall back on.

This is a bigger change than Track 1 and should **only be built if Track 1's ablation (Step 10 above) shows that CAB + the floor fix still leave a meaningful, measurable gap** — i.e., this track exists to be picked up later with evidence, not spent on speculatively. The deliverable right now is a complete, reviewable **plan recorded as a placeholder file**, not working code.

## 2.2 Why "combined" is the right name for this file

In Track 1's pipeline, Stage 0 (vision-only BPAM pretraining) and Stage 1 (text-side prompt learning) are cleanly separable — hence two scripts, one checkpoint hand-off. In this pivoted design, the part localizer's own forward pass depends on a text-derived conditioning signal from the start, so there is no meaningful way to pretrain it "alone" the way BPAM currently is in Stage 0 — the localizer and at least a minimal text representation have to exist and interact from the very first training step. That's what "combined" refers to: Stage 0 and Stage 1, as separate scripts, cease to exist as a concept in this variant; they're combined into one training entry point.

## 2.3 High-level architecture sketch (planning-level, not final)

Replace `attention_head.py`'s `Conv1x1 → softmax` with a **FiLM-conditioned** localizer:

```python
class ConditionalPartLocalizer(nn.Module):
    """
    PLANNING STUB -- not implemented.
    Produces the same (K+1)-channel spatial attention map as the current BPAM conv,
    but the conv's output is modulated by a conditioning vector derived from the
    current state of the text-side part representation, so the localizer's part
    boundaries can react to what the text side currently expects "part k" to mean,
    rather than being fixed once at a separate Stage-0 pretraining step.
    """
    def __init__(self, in_channels: int, num_parts: int, text_cond_dim: int):
        super().__init__()
        raise NotImplementedError(
            "Track 2 placeholder -- do not implement until Track 1's ablation "
            "(see Step 10 of the main plan) shows a measurable residual gap "
            "that this architecture is expected to close."
        )
        # Planned structure, for whoever picks this up:
        #   self.conv = nn.Conv2d(in_channels, num_parts + 1, kernel_size=1)
        #   self.film_generator = nn.Linear(text_cond_dim, 2 * (num_parts + 1))  # per-channel scale+shift
        # forward(feature_map, text_cond_vector):
        #   raw = self.conv(feature_map)                          # (B, K+1, H, W)
        #   scale, shift = self.film_generator(text_cond_vector).chunk(2, dim=-1)
        #   modulated = raw * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        #   return softmax(modulated, dim=1)

    def forward(self, feature_map, text_cond_vector):
        raise NotImplementedError("See class docstring.")
```

**Open design questions to resolve before this leaves placeholder status** (do not answer these speculatively now — they need dedicated design time when this track is picked up):
1. What exactly is `text_cond_vector`? Candidates: (a) an exponential moving average of `ctx_params`' current state, updated every N steps; (b) the live output of TRB on the current batch's identities; (c) a single global, identity-agnostic vector representing "current anatomical convention," decoupled from per-identity prompts entirely. Each has different training-stability implications and this choice materially changes the rest of the design — do not default to (b) just because it seems most direct, since it reintroduces a tight per-step coupling between two initially-untrained components, exactly the instability CLIP-ReID's own ablation warns against.
2. Since there's no separate Stage 0 checkpoint anymore, what provides the initial, stable, anatomically-sane starting point that Stage 0 currently gives Track 1's pipeline for free? Likely answer: still pretrain `self.conv` alone for a short warmup with `text_cond_vector` fixed at zero (equivalent to no conditioning) before switching on the FiLM pathway — but this needs to be validated, not assumed, since it partially reintroduces the sequential-then-joint structure this track was meant to move away from.
3. How does `generate_part_pseudolabels.py` (the external pseudo-label generation step) interact with this — does `L_attn` still apply, and if so, against which output (raw conv, or FiLM-modulated)? Likely: still against the FiLM-modulated output, since that's what actually produces the part maps used downstream, but confirm this doesn't create a degenerate shortcut where the FiLM pathway learns to suppress the conv's contribution entirely.

## 2.4 Placeholder file contents (what to actually create now)

Create `examples/train_relational_prompts_combined.py` with:
- A module-level docstring containing §2.1-2.3 of this document (copy verbatim, so the plan lives with the code it describes).
- The `ConditionalPartLocalizer` stub class from §2.3, with `NotImplementedError` bodies.
- A `main()` function stub with a docstring listing the expected CLI/config surface (`--config configs/stage_combined.yaml`) but no body beyond `raise NotImplementedError(...)`.
- No config YAML yet (`configs/stage_combined.yaml` should not be created until implementation actually begins — an empty/placeholder config file invites accidental use before the code behind it exists).

**Verification gate for Track 2 (right now):** the file exists, imports cleanly (no syntax errors), and running it produces a clear `NotImplementedError` with a message pointing back to this plan document and the Track 1 ablation gate — not a silent failure or a stack trace with no context.

## 2.5 Trigger condition to promote Track 2 from placeholder to active build

Do not begin implementing §2.3's open design questions until Track 1's Step 10 ablation table shows: `Floor + CAB + L_crossalign` still trails a defined target (state the target mAP/Rank-1 threshold before running the ablation, not after — deciding the bar retroactively based on what Track 1 happened to achieve is not a valid trigger condition).
