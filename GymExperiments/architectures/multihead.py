from typing import Optional, List

import torch
import torch.nn as nn


class Dualhead(nn.Module):
    def __init__(self, base: nn.Module, head0: nn.Module, head1: nn.Module, names: Optional[List[str]] = None):
        super().__init__()
        self.base = base
        self.head0 = head0
        self.head1 = head1

        self._return_dict = True if names is not None else False
        if self._return_dict:
            assert len(names) == 2
        self._name = names

    def forward(self, inp: torch.Tensor):
        x = self.base(inp)
        x0 = self.head0(x)
        x1 = self.head1(x)

        if self._return_dict:
            return {self._names[0]: x0, self._names[1]: x1}
        else:
            return x0, x1


class ReprDualhead(nn.Module):
    def __init__(self, encoder: nn.Module, middle: nn.Module, head0: nn.Module, head1: nn.Module, names: Optional[List[str]] = None):
        super().__init__()
        self.encoder = encoder
        self.middle = middle
        self.head0 = head0
        self.head1 = head1

        self._return_dict = True if names is not None else False
        if self._return_dict:
            assert len(names) == 4
        self._name = names

    def forward(self, inp: torch.Tensor):
        repre, repre_var = self.encoder(inp)
        x = self.middle(repre)
        x0 = self.head0(x)
        x1 = self.head1(x)

        if self._return_dict:
            return {self._names[0]: x0, self._names[1]: x1, self._names[2]: repre, self._names[3]: repre_var}
        else:
            return x0, x1, repre, repre_var
