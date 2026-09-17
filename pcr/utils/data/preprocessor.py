from __future__ import absolute_import
import os.path as osp
import random

import numpy as np
import torch
import torch.nn.functional as nnF
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop, InterpolationMode
from torch.utils.data import Dataset
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index


class Preprocessor2View(Dataset):
    """Two independently-augmented views of the same image, both under `transform`. Used by
    the ICE-style USL pipeline (pcr/trainers_usl.py): one view feeds the student encoder, the
    other feeds the EMA teacher -- that's the whole "instance discrimination between two views"
    setup the hard-instance contrastive loss operates on. No index is returned: unlike the UDA
    pipeline's PartHybridMemory, this training loop keeps no per-instance memory slots, so the
    pseudo-label already baked into the dataset tuple's `pid` field is all that's needed."""

    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor2View, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        img1 = self.transform(img)
        img2 = self.transform(img.copy())
        return img1, img2, pid, camid


def _load_raw_mask(masks_root, masks_dir, mask_suffix, fname):
    split_folder = osp.basename(osp.dirname(fname))
    basename = osp.splitext(osp.basename(fname))[0]
    mask_path = osp.join(masks_root, 'masks', masks_dir, split_folder, basename + mask_suffix)
    raw = np.load(mask_path)  # [C, H, W]
    return torch.from_numpy(raw).float()


def _paired_geometric_transform(img, mask, height, width, pad, flip_p):
    """Applies resize/pad/crop/flip identically to an image and its raw mask via shared random
    parameters -- deliberately hand-rolled instead of pulling in bpbreid's own albumentations-
    based mask pipeline, since only these 4 ops need pairing at all (colour-only augmentation
    doesn't touch pixel *positions*, so it's applied to the image alone by the caller,
    afterward). Shared by PreprocessorMasked and PreprocessorMaskedSingleView. Returns
    (img: PIL.Image, mask: Tensor[C,H,W]) -- photometric/tensor conversion is left to the caller.

    Bicubic resize, matching this repo's other transforms' T.Resize(interpolation=3) -- TF.resize's
    own default (bilinear) would otherwise give this view a subtly different resize kernel than a
    sibling view built through the torchvision Resize transform (e.g. PreprocessorMasked's view 2).
    """
    mask = nnF.interpolate(mask.unsqueeze(0), size=(height, width),
                            mode='bilinear', align_corners=True).squeeze(0)
    img = TF.resize(img, [height, width], interpolation=InterpolationMode.BICUBIC)
    img = TF.pad(img, pad)
    mask = nnF.pad(mask, (pad, pad, pad, pad))

    i, j, h, w = RandomCrop.get_params(img, output_size=(height, width))
    img = TF.crop(img, i, j, h, w)
    mask = mask[:, i:i + h, j:j + w]

    if random.random() < flip_p:
        img = TF.hflip(img)
        mask = mask.flip(dims=[-1])

    return img, mask


class PreprocessorMasked(Dataset):
    """Like Preprocessor2View, but view 1 is paired with its PifPaf ground-truth part mask for
    BodyPartAttentionLoss (view 2, feeding the EMA teacher, never needs a mask). Channel grouping
    (36 raw PifPaf channels -> 5 named parts) and the synthetic background channel are computed
    via bpbreid's own torchreid.data.masks_transforms classes, reused directly rather than
    re-deriving their channel-mapping tables by hand (imported lazily so plain Preprocessor/
    Preprocessor2View usage never requires torchreid's masks_transforms module at all).
    """

    def __init__(self, dataset, masks_root, masks_dir, height, width, photometric_transform,
                 view2_transform, root=None, pad=10, flip_p=0.5, mask_suffix='.npy'):
        super(PreprocessorMasked, self).__init__()
        from torchreid.data.masks_transforms import CombinePifPafIntoFiveVerticalParts, AddBackgroundMask

        self.dataset = dataset
        self.root = root
        self.masks_root = masks_root
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.height = height
        self.width = width
        self.pad = pad
        self.flip_p = flip_p
        self.photometric_transform = photometric_transform
        self.view2_transform = view2_transform
        self.group_masks = CombinePifPafIntoFiveVerticalParts()
        self.add_background = AddBackgroundMask()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = osp.join(self.root, fname) if self.root is not None else fname

        img = Image.open(fpath).convert('RGB')
        img2 = self.view2_transform(img.copy())

        mask = _load_raw_mask(self.masks_root, self.masks_dir, self.mask_suffix, fname)
        img1, mask = _paired_geometric_transform(img, mask, self.height, self.width,
                                                  self.pad, self.flip_p)

        mask = self.group_masks.apply_to_mask(mask)
        mask = self.add_background.apply_to_mask(mask)  # [1+parts_num, height, width]

        img1 = self.photometric_transform(img1)

        return img1, img2, mask, pid, camid


class PreprocessorMaskedSingleView(Dataset):
    """Like PreprocessorMasked, but for supervised single-view training (examples/
    train_relational_finetune.py): no second augmented view, no EMA-teacher pairing -- just one image + its
    mask + the real (pid, camid, index) the supervised id/triplet/alignment losses need. Shares
    the geometric-pairing helper with PreprocessorMasked rather than duplicating it."""

    def __init__(self, dataset, masks_root, masks_dir, height, width, photometric_transform,
                 root=None, pad=10, flip_p=0.5, mask_suffix='.npy'):
        super(PreprocessorMaskedSingleView, self).__init__()
        from torchreid.data.masks_transforms import CombinePifPafIntoFiveVerticalParts, AddBackgroundMask

        self.dataset = dataset
        self.root = root
        self.masks_root = masks_root
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.height = height
        self.width = width
        self.pad = pad
        self.flip_p = flip_p
        self.photometric_transform = photometric_transform
        self.group_masks = CombinePifPafIntoFiveVerticalParts()
        self.add_background = AddBackgroundMask()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = osp.join(self.root, fname) if self.root is not None else fname

        img = Image.open(fpath).convert('RGB')
        mask = _load_raw_mask(self.masks_root, self.masks_dir, self.mask_suffix, fname)
        img, mask = _paired_geometric_transform(img, mask, self.height, self.width,
                                                 self.pad, self.flip_p)

        mask = self.group_masks.apply_to_mask(mask)
        mask = self.add_background.apply_to_mask(mask)  # [1+parts_num, height, width]

        img = self.photometric_transform(img)

        return img, mask, pid, camid, index
