# Implementation Plan: Part-Aware, Semantically-Regularized, Domain-Adaptive Person ReID Pipeline

## 0. Objective

Build a person ReID training pipeline that combines three techniques into one system:

1. **BPBreID** (Somers et al., WACV 2023) — attention-based body-part pooling with visibility-aware masking (GiLt).
2. **CLIP-ReID** (Li et al., AAAI 2023) — two-stage prompt learning that uses a frozen CLIP text encoder as a training-time semantic regularizer, applied per body-part rather than only globally.
3. **SPCL** (Ge et al., NeurIPS 2020, "Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID") — as the *domain adaptation strategy*, used when labeled source data and unlabeled target data both exist.

The three are not competitors — they slot into a single pipeline at different points: BPBreID defines *what features exist* (part-level + global), CLIP-ReID defines *how those features get semantically regularized during supervised training*, and SPCL defines *how the whole thing generalizes to a new unlabeled domain* after supervised pretraining.

Framing choice for the agent: **build this as a `torchreid`-style module**, not a monolithic script. Reasons: torchreid already separates `data/` (datasets, samplers, transforms), `models/` (backbones), `losses/`, `engine/` (train/test loop), `metrics/` (mAP, CMC) — this pipeline fits that separation cleanly, and it makes Stage 1 / Stage 2 / SPCL-stage swappable as different `engine.Engine` subclasses without rewriting data loading or eval code. If the agent is more comfortable with SPCL's own repo structure (which already implements hybrid-memory UDA training loops), that's an acceptable alternative base — but torchreid's engine abstraction is recommended because it already has mature ReID evaluation (mAP/CMC/re-ranking) built in, which SPCL's repo has too but is more UDA-specific and less modular for the CLIP prompt stage.

---

## 1. Repository / Module Structure

```
reid_pipeline/
├── configs/
│   ├── stage1_prompt_learning.yaml
│   ├── stage2_backbone_finetune.yaml
│   └── stage3_spcl_domain_adapt.yaml       # optional, only if target domain exists
├── data/
│   ├── datasets/                            # Market-1501, MSMT17, custom, etc. (torchreid ImageDataset subclasses)
│   ├── samplers.py                          # PK sampler (RandomIdentitySampler)
│   ├── transforms.py
│   └── part_labels/                         # offline pseudo part-segmentation masks (see §2.1)
├── models/
│   ├── backbone.py                          # ResNet50-IBN-a or ViT-B/16, torchreid-compatible
│   ├── attention_head.py                    # BPBreID part-attention module (§2)
│   ├── clip_text_branch.py                  # frozen CLIP text encoder + learnable prompts (§3)
│   └── fusion_head.py                       # visibility-weighted concat (§4)
├── losses/
│   ├── gilt_losses.py                       # visibility-masked ID + triplet loss (§2.2)
│   ├── clip_contrastive.py                  # InfoNCE alignment loss (§3)
│   └── spcl_hybrid_memory.py                # cluster-based contrastive memory (§5)
├── engine/
│   ├── stage1_engine.py                     # prompt learning loop
│   ├── stage2_engine.py                     # supervised backbone fine-tune loop
│   └── stage3_spcl_engine.py                # UDA loop (source labeled + target unlabeled)
├── scripts/
│   ├── generate_part_pseudolabels.py        # run SCHP / PifPaf offline, once
│   └── run_pipeline.py                      # orchestrates stage1 -> stage2 -> stage3
└── eval/
    └── metrics.py                           # mAP, CMC, re-ranking (reuse torchreid.metrics)
```

Each stage is a separate `Engine` object with its own optimizer, its own frozen/trainable parameter sets, and its own YAML config — this makes freezing/unfreezing explicit and auditable rather than buried in flags.

---

## 2. Component: BPBreID Attention Pooling

### 2.1 Pseudo part-label generation (offline, Stage 0)

Run once, before any training:
- Use a human parsing model (SCHP — Self-Correction Human Parsing) or a pose estimator (PifPaf/HRNet keypoints → heatmap-to-region mapping) over the full training set.
- Produce K semantic region masks per image (e.g. K=5: head, torso, arms, legs, feet) + 1 background/foreground mask.
- Store as low-res label maps aligned to the backbone's feature map resolution (e.g. 24×8 for a 384×128 input with stride 16).
- These are **only used to supervise the attention head's training loss** — never needed at inference.

### 2.2 Architecture

```python
class PartAttentionHead(nn.Module):
    def __init__(self, in_channels, num_parts=5):
        self.k = num_parts
        self.conv = nn.Conv2d(in_channels, num_parts + 1, kernel_size=1)  # +1 = background

    def forward(self, feat_map):                    # feat_map: (B, C, H, W)
        logits = self.conv(feat_map)                 # (B, K+1, H, W)
        attn = F.softmax(logits, dim=1)               # spatial-channel softmax over parts
        return attn                                   # (B, K+1, H, W)
```

Pooling:
```python
def part_pool(feat_map, attn):
    # feat_map: (B, C, H, W), attn: (B, K+1, H, W)
    B, C, H, W = feat_map.shape
    parts = []
    for k in range(attn.shape[1] - 1):                # exclude background channel
        weight = attn[:, k:k+1, :, :]                 # (B, 1, H, W)
        pooled = (feat_map * weight).sum(dim=[2, 3])   # (B, C)
        parts.append(pooled)
    visibility = attn[:, :-1, :, :].sum(dim=[2, 3])    # (B, K) — used for GiLt gating
    global_feat = feat_map.mean(dim=[2, 3])             # (B, C) simple global branch (GAP)
    return global_feat, torch.stack(parts, dim=1), visibility
```

Supervise attention with pixel-wise cross-entropy against the offline pseudo-labels:
```python
L_attn = F.cross_entropy(logits, pseudo_part_label_map)   # per-pixel, K+1 classes
```

### 2.3 GiLt Masking (Global-and-local with local masking)

Rule: a part's ID/triplet loss only contributes gradient for samples where that part is visible above a threshold. This is the mechanism that makes part-based ReID robust to occlusion — instead of forcing the network to hallucinate features for occluded regions, it simply excludes them from the loss for that sample.

```python
def gilt_masked_loss(part_features, visibility, labels, part_classifiers, visibility_thresh=0.3):
    total_loss = 0.0
    for k in range(part_features.shape[1]):
        visible_mask = visibility[:, k] > visibility_thresh    # (B,) bool
        if visible_mask.sum() == 0:
            continue
        feats_k = part_features[visible_mask, k, :]
        labels_k = labels[visible_mask]
        logits_k = part_classifiers[k](feats_k)
        total_loss += F.cross_entropy(logits_k, labels_k)
        total_loss += batch_hard_triplet_loss(feats_k, labels_k)   # only among visible samples
    return total_loss
```

Important: triplet mining for part *k* must only use the subset of the batch where part *k* is visible — do not compute triplet distances against samples where that part is occluded, or the "hardest negative" may just be noise.

**Concatenation question (answered again for the spec):** part feature vectors are *not* concatenated during loss computation — each part has its own classifier/loss, masked independently. Concatenation happens only at the final descriptor-fusion step (§4), after all training is done.

---

## 3. Component: CLIP-ReID Prompt Learning (per-part)

### 3.1 Stage 1 — Prompt learning (backbone frozen)

For each identity *i* and each part *k* (plus one "global" prompt), initialize a learnable pseudo-token sequence:

```
prompt_i_k = [SOT] "a photo of a" [V1]_{i,k} [V2]_{i,k} ... [VM]_{i,k} "person" [EOT]
```

`[Vj]_{i,k}` are `nn.Parameter` continuous embeddings (not real vocabulary words), M ≈ 4–8 tokens is typical.

```python
class PromptLearner(nn.Module):
    def __init__(self, num_identities, num_parts, ctx_len=8, embed_dim=512):
        self.ctx = nn.Parameter(torch.randn(num_identities, num_parts + 1, ctx_len, embed_dim) * 0.02)
        self.fixed_prefix = clip_tokenizer.encode("a photo of a")
        self.fixed_suffix = clip_tokenizer.encode("person")

    def build_prompt(self, identity_id, part_id):
        return torch.cat([self.fixed_prefix, self.ctx[identity_id, part_id], self.fixed_suffix])
```

Training loop:
```python
freeze(backbone); freeze(attention_head); freeze(clip_text_encoder)
optimizer = Adam(prompt_learner.parameters(), lr=...)

for batch in pk_dataloader:
    with torch.no_grad():
        feat_map = backbone(images)
        global_feat, part_feats, visibility = attention_head_pool(feat_map)  # frozen, no grad

    loss = 0.0
    for k in range(num_parts + 1):   # +1 for global
        text_tokens = prompt_learner.build_prompt(labels, k)
        text_embed = clip_text_encoder(text_tokens)             # frozen encoder, grad flows to prompt only
        visual_embed = part_feats[:, k] if k < num_parts else global_feat
        visible_mask = visibility[:, k] > thresh if k < num_parts else torch.ones_like(labels).bool()
        loss += info_nce(visual_embed[visible_mask], text_embed[visible_mask], temperature=0.07)
    loss.backward(); optimizer.step()
```

`info_nce` is the standard symmetric CLIP contrastive loss (row-wise + column-wise cross-entropy over the cosine-similarity matrix, temperature-scaled).

Result of Stage 1: one fixed text embedding per (identity, part) — a set of frozen semantic anchors, saved to disk.

### 3.2 Stage 2 — Backbone fine-tuning (prompts + text encoder frozen)

```python
freeze(prompt_learner); freeze(clip_text_encoder)
unfreeze(backbone); unfreeze(attention_head); unfreeze(id_classifiers)
optimizer = Adam([backbone.params, attention_head.params, classifiers.params], lr=...)

for batch in pk_dataloader:
    feat_map = backbone(images)
    global_feat, part_feats, visibility = attention_head_pool(feat_map)

    L_attn = pixel_ce(attention_head.last_logits, pseudo_part_labels)
    L_gilt = gilt_masked_loss(part_feats, visibility, labels, part_classifiers)
    L_id_global = ce(global_classifier(global_feat), labels)
    L_tri_global = batch_hard_triplet(global_feat, labels)

    L_align = 0.0
    for k in range(num_parts + 1):
        text_anchor_k = frozen_text_anchors[labels, k]           # precomputed in Stage 1, no grad
        visual_k = part_feats[:, k] if k < num_parts else global_feat
        visible_mask = visibility[:, k] > thresh if k < num_parts else torch.ones_like(labels).bool()
        L_align += (1 - cosine_sim(visual_k[visible_mask], text_anchor_k[visible_mask])).mean()

    loss = L_attn + L_gilt + L_id_global + L_tri_global + lambda_clip * L_align
    loss.backward(); optimizer.step()
```

**Concatenation question (CLIP text side):** the per-part text embeddings are never concatenated into a single text vector, and text embeddings never enter the final image descriptor at all — they exist purely as fixed regularization targets (`L_align`) during Stage 2, then are discarded entirely before inference.

---

## 4. Component: Fusion into a Single Descriptor

At inference (and for the eval/gallery-indexing pipeline), drop the CLIP text branch entirely. Combine global + part features into one descriptor:

```python
def fuse(global_feat, part_feats, visibility, w_global, w_parts, learnable_gate=None):
    global_n = F.normalize(global_feat, dim=-1)
    part_n = F.normalize(part_feats, dim=-1)              # (B, K, C)

    if learnable_gate is not None:
        gate_input = torch.cat([global_feat, visibility], dim=-1)
        weights = F.softmax(learnable_gate(gate_input), dim=-1)   # (B, K+1)
        w_global_eff = weights[:, 0:1]
        w_parts_eff = weights[:, 1:] * visibility
    else:
        w_global_eff = w_global
        w_parts_eff = w_parts * visibility                 # zero-out invisible parts

    parts_weighted = [w_parts_eff[:, k:k+1] * part_n[:, k] for k in range(part_n.shape[1])]
    fused = torch.cat([w_global_eff * global_n] + parts_weighted, dim=-1)
    return F.normalize(fused, dim=-1)
```

Provide both fusion modes in config: fixed scalar weights (simple, robust baseline) and a learned gating MLP (better under variable occlusion, adds a small number of extra parameters trained jointly in Stage 2).

For heavy-occlusion datasets, also implement an optional **re-ranking path**: use the fused vector for fast top-N retrieval (ANN/FAISS), then recompute a visibility-masked part-wise distance (BPBreID's native matching strategy) only on the top-N candidates for final ranking.

---

## 5. Component: SPCL-Style Domain Adaptation (Stage 3, optional)

Use this stage only if there is a labeled **source** dataset (e.g. Market-1501) and an unlabeled **target** dataset (e.g. a new deployment camera network) that the model needs to generalize to.

### 5.1 Why SPCL fits here

SPCL (Ge et al., NeurIPS 2020) builds a **hybrid memory bank** holding both source-class centroids (from labels) and target instance/cluster features (from pseudo-labels), and trains with a unified contrastive loss against this memory — self-paced, meaning it gradually trusts more target clusters as training stabilizes (measured via cluster reliability / independence scores). This slots in naturally after Stage 2: the model already has strong part+global features; SPCL's job is purely to adapt them to the target domain's distribution without needing target ID labels.

### 5.2 Pipeline

```python
# initialize source classifier centroids from labeled source data (fixed class prototypes)
source_centroids = compute_class_means(source_features, source_labels)

# initial target pseudo-labels via clustering (DBSCAN or k-means on target features from Stage-2 model)
target_pseudo_labels, target_clusters = run_clustering(target_features)

hybrid_memory = HybridMemory(source_centroids, target_clusters)  # single memory bank

for epoch in spcl_epochs:
    # re-cluster target periodically (e.g. every epoch) as features improve
    target_pseudo_labels, target_clusters = run_clustering(extract_features(target_data, model))
    hybrid_memory.update_target(target_clusters)

    for batch in mixed_source_target_pk_dataloader:
        global_feat, part_feats, visibility = model(batch.images)
        # fuse into single descriptor as in §4 for the contrastive memory lookup
        fused = fuse(global_feat, part_feats, visibility, ...)

        loss = hybrid_memory_contrastive_loss(fused, batch.pseudo_or_real_labels, hybrid_memory,
                                               temperature=0.05, self_paced_weight=reliability_score(batch))
        loss.backward(); optimizer.step()
```

Key SPCL details the agent must implement faithfully:
- **Hybrid memory** stores a running-averaged feature vector per source class *and* per target cluster, updated with momentum after each batch (not recomputed from scratch every step).
- **Self-paced weighting**: down-weight or exclude target samples belonging to low-reliability clusters (measured via independence/compactness metrics on the clustering, as defined in the SPCL paper) especially in early epochs, gradually increasing target-cluster trust as training proceeds.
- Both the BPBreID part branch and GiLt masking carry over unchanged into this stage — SPCL only changes *which loss/memory* the fused (or per-part, if extended) features are compared against, not the backbone architecture.
- Recommended extension beyond vanilla SPCL: run the hybrid memory contrastive loss **per part plus global** (K+1 separate memory banks) rather than only on the fused vector, so target-domain part-level occlusion patterns are also adapted, not just global appearance.

---

## 6. Orchestration (`scripts/run_pipeline.py`)

```
Stage 0: generate_part_pseudolabels.py     → offline part masks (SCHP/pose-based)
Stage 1: stage1_engine.py                  → learn CLIP prompts per (identity, part), backbone frozen
Stage 2: stage2_engine.py                  → fine-tune backbone + attention head, prompts frozen
Stage 3: stage3_spcl_engine.py  [optional] → domain-adapt fused features to unlabeled target via SPCL hybrid memory
Eval:    eval/metrics.py                   → mAP, CMC, re-ranking on target test split
```

Each stage should be resumable independently (save/load checkpoints per stage), and Stage 3 should be skippable entirely for single-domain (no UDA) use cases — in that case Stage 2's checkpoint is the final model.

---

## 7. Config skeleton (`stage2_backbone_finetune.yaml`)

```yaml
model:
  backbone: resnet50_ibn_a       # or vit_base_patch16
  num_parts: 5
  clip_arch: ViT-B/32

data:
  sampler: RandomIdentitySampler  # PK sampling
  P: 16
  K: 4
  part_label_dir: data/part_labels/

loss:
  id_weight: 1.0
  triplet_weight: 1.0
  attn_weight: 1.0
  clip_align_weight: 0.5
  gilt_visibility_threshold: 0.3

optim:
  lr: 3.5e-4
  epochs: 120
  scheduler: warmup_cosine

fusion:
  mode: learned_gate            # or fixed_weights
  w_global: 1.0
  w_part: 0.5
```

---

## 8. References the agent should consult while implementing

- Somers, V. et al. "BPBreID: Body Part-based Representation Learning for Occluded Person Re-Identification." WACV 2023. — attention pooling + GiLt masking.
- Li, S. et al. "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels." AAAI 2023. — two-stage prompt learning.
- Ge, Y. et al. "Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID." NeurIPS 2020. — hybrid memory + self-paced UDA.
- Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP). ICML 2021. — base image/text encoder architecture.
- Hermans, A. et al. "In Defense of the Triplet Loss for Person Re-Identification." 2017. — batch-hard triplet + PK sampling.
- Li, P. et al. "Self-Correction for Human Parsing." — for offline pseudo part-label generation.
- torchreid (Zhou & Xiang) — as the base framework for data/engine/metrics modules.

---

## 9. Summary of what to build, in order

1. similar repo skeleton with separated `data/`, `models/`, `losses/`, `engine/`, one train_clipreid.py file in examples dir.
2. Offline part-pseudo-label generation script (SCHP-based).
3. `PartAttentionHead` module + GiLt-masked ID/triplet losses.
4. `PromptLearner` module + Stage-1 engine (backbone frozen, prompts trainable).
5. Stage-2 engine (backbone/attention trainable, prompts frozen, adds `L_align`).
6. Fusion module (fixed-weight and learned-gate variants) for single-descriptor inference.
7. (Optional) SPCL hybrid-memory module + Stage-3 UDA engine for source→target adaptation.
8. Evaluation harness reusing standard mAP/CMC/re-ranking metrics.
