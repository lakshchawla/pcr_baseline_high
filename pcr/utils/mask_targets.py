"""Pixel-level part targets from PifPaf masks, shared by the stages that train PixelToPartClassifier
against real masks (Stage 1's BPAM continuation; Stage 0 / Stage 2 keep their own identical local
copies of mask_to_pixel_targets, untouched here)."""
import torch
import torch.nn.functional as F


def mask_to_pixel_targets(mask, pixels_cls_scores):
    """mask: [B, 1+parts_num, H, W] (soft, sums to 1 per pixel). Resized to pixels_cls_scores'
    spatial size and argmax'd into an integer target per pixel -- matches bpbreid's own
    part_based_engine.py::combine_losses."""
    mask = F.interpolate(mask, size=pixels_cls_scores.shape[2:], mode='bilinear', align_corners=True)
    return mask.argmax(dim=1)


def estimate_class_weights(mask_batches, num_classes, grid_size, num_batches=50):
    """Background dominates these masks (~78% of pixels at a 16x8 grid vs 1-7% per part), so
    plain CE drifts toward predicting background everywhere. Counts each class's pixel
    frequency over `num_batches` batches of masks (`mask_batches`: an iterable yielding
    [B, 1+K, H, W] mask tensors -- mask loading only, no model forward) and returns
    inverse-sqrt-frequency weights normalized to sum to num_classes -- same recipe Stage 0's
    train_bpa_segmentation_rn50.py uses, so a classifier continued in Stage 1 sees the same
    class balance it was pretrained under."""
    counts = torch.zeros(num_classes)
    for seen, mask in enumerate(mask_batches):
        fake_scores = torch.zeros(mask.size(0), num_classes, *grid_size)
        targets = mask_to_pixel_targets(mask, fake_scores)
        for k in range(num_classes):
            counts[k] += (targets == k).sum().item()
        if seen + 1 >= num_batches:
            break
    freq = counts / counts.sum()
    weight = 1.0 / freq.sqrt()
    return weight * (num_classes / weight.sum()), freq


def soft_pixel_distillation(pixels_cls_scores, target_probs):
    """Soft-target cross-entropy between the classifier's per-pixel logits and a per-patch
    probability map -- the distillation term that bakes the text-refined (blended) part
    assignment into the text-free pixel classifier. pixels_cls_scores: [B, 1+K, H, W].
    target_probs: [B, N=H*W, 1+K], detached by the caller. Returns the mean over patches of
    -sum_c target[c] * log softmax(scores)[c]. The target contains the classifier's own current
    output at weight (1 - blend), which contributes zero gradient at the optimum, so the real
    signal is the blend * (text map - classifier map) difference; the BPA loss on real masks is
    what keeps the classifier from drifting after the text."""
    B, C, H, W = pixels_cls_scores.shape
    log_probs = F.log_softmax(pixels_cls_scores, dim=1).reshape(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
    return -(target_probs * log_probs).sum(dim=-1).mean()
