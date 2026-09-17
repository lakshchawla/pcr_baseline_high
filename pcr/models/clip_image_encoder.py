"""Frozen CLIP visual tower (`clip_model.visual`), loaded and frozen per the Stage 1 algorithm's
own initialization step ("load pretrained CLIP image encoder... freeze"). Not consumed by Stage
1's forward pass -- BPBreID's backbone remains the sole visual encoder actually producing the
embeddings used in this pipeline's losses (see pcr/models/bpbreid_encoder.py and
pcr/models/clip_text_encoder.py's own docstring for why). This class exists so the algorithm's
initialization step is followed literally; `forward()` is provided for completeness/future use
(mirrors CLIP's own `encode_image`) but nothing in examples/train_relational_prompts.py currently
calls it.

Loaded independently from pcr/models/clip_text_encoder.py::ClipTextEncoder (a second `clip.load()`
call) rather than sharing one loaded model between the two -- keeps each class self-contained and
single-purpose, at the cost of loading CLIP's weights into memory twice. A one-time load cost, not
a per-iteration one, so left as the simpler design rather than plumbing a shared model instance
through two otherwise-independent classes.
"""
import clip
import torch.nn as nn


class ClipImageEncoder(nn.Module):
    def __init__(self, clip_arch='RN50', device='cuda'):
        super(ClipImageEncoder, self).__init__()
        clip_model, _ = clip.load(clip_arch, device=device, jit=False)
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, images):
        """images: [B, 3, H, W] already preprocessed to CLIP's own expected input resolution/
        normalization (not BPBreID's -- the two backbones do not share a preprocessing pipeline).
        Mirrors CLIP's own CLIP.encode_image (third_party/clip/model.py)."""
        return self.visual(images.type(self.dtype))
