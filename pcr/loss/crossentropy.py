"""Ported near-verbatim from ice/loss/crossentropy.py. Used as the "id" component of
PartGiLtLoss (pcr/loss/gilt_loss.py): classification against per-epoch cluster centers rather
than a fixed parametric classifier head, since DBSCAN's cluster numbering/count changes every
epoch -- a fixed nn.Linear classifier would silently reassign what each output neuron "means"
across epochs, which this center-matmul approach sidesteps entirely (ICE's own reason for using
it this way).
"""
import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return (-targets * log_probs).mean(0).sum()
