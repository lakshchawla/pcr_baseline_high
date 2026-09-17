"""Persistent per-branch nn.Linear identity classifiers for Stage 2's supervised id loss
(examples/train_relational_finetune.py). Unlike pcr/loss/gilt_loss.py::PartGiLtLoss (which classifies
against per-epoch DBSCAN cluster centers, since USL's pseudo-label count/numbering changes every
epoch), Stage 2 trains on real, static identity labels -- a persistent parametric classifier head
is the right fit here, matching bpbreid's own original GiLt design (which pcr2's DBSCAN-cluster
USL variant deliberately departed from, for the reason stated in gilt_loss.py's own docstring).

Only builds classifier heads for the branches actually requested (default: foreground only,
branch 0), matching bpbreid's own default_losses_weights convention (`foreg: id=1`, `parts: id=0`,
already documented in gilt_loss.py) -- avoids running (and holding optimizer state for) classifier
heads that would never receive a meaningful gradient, the same "wasted compute/memory" issue
flagged and left unfixed for BPBreID's own internal placeholder heads in this repo's 2026-08-18
audit entry (progress.md) -- not repeated here since this module's heads are opt-in per branch.
"""
import torch.nn as nn


class PartIdClassifiers(nn.Module):
    def __init__(self, num_identities, embed_dim, branches=(0,)):
        super(PartIdClassifiers, self).__init__()
        self.branches = list(branches)
        self.classifiers = nn.ModuleDict({
            str(b): nn.Linear(embed_dim, num_identities) for b in self.branches
        })

    def forward(self, f_out, branch):
        """f_out: [B, M, D]. branch: int, must be one of the branches this module was built
        for (self.branches)."""
        return self.classifiers[str(branch)](f_out[:, branch, :])
