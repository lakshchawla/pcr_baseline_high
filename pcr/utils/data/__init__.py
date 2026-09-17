from __future__ import absolute_import

from .base_dataset import BaseDataset, BaseImageDataset
from .preprocessor import Preprocessor


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            try:
                return next(self.iter)
            except StopIteration:
                raise RuntimeError(
                    'IterLoader: underlying DataLoader yielded zero batches on a fresh '
                    'iterator -- the dataset is smaller than one batch for the current '
                    'batch_size/sampler/drop_last configuration. Callers that rebuild the '
                    'dataset per-epoch (e.g. after pseudo-label clustering) should check '
                    'the dataset size before calling next() rather than relying on this '
                    'exception, since a bare second StopIteration here would otherwise '
                    'propagate confusingly to whatever loop called next().')
