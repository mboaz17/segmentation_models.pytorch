import torch.nn as nn

from . import base
from . import functional as F
from ..base.modules import Activation


class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, class_intervals=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.class_intervals = class_intervals

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            class_intervals=self.class_intervals,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    def __init__(self, eps=1e-6, activation=None, ignore_channels=None, class_intervals=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.class_intervals = class_intervals

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        loss = F.CE(y_pr, y_gt, eps=self.eps, threshold=None, ignore_channels=self.ignore_channels, class_intervals=self.class_intervals)

        return loss


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    def __init__(self, eps=1e-6, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        loss = F.BCE(y_pr, y_gt, eps=self.eps, threshold=None, ignore_channels=self.ignore_channels)

        return loss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
