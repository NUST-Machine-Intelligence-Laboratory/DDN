# -*- coding: utf-8 -*-

import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super().__init__()
        self._gamma = gamma
        self._alpha = alpha
        if isinstance(alpha, (float, int)):
            self._alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self._alpha = torch.Tensor(alpha)
        self._size_average = size_average

    def forward(self, x, y):
        assert x.dim() == 2, 'x dimension has to be (batch size, 2)'
        y = y.view(-1, 1).long()
        log_pt = x.log()
        log_pt = log_pt.gather(1, y)
        log_pt = log_pt.view(-1)
        pt = log_pt.data.exp()

        if self._alpha is not None:
            if self._alpha.type() != x.data.type():
                self._alpha = self._alpha.type_as(x.data)
            alpha_t = self._alpha.gather(0, y.data.view(-1))
            log_pt = log_pt * alpha_t

        loss = -1 * (1 - pt)**self._gamma * log_pt
        if self._size_average:
            return loss.mean()
        else:
            return loss.sum()

