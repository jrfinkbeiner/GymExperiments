import torch
import torch.nn as nn

class Dualhead(nn.Module):
    def __init__(self, base: nn.Module, head0: nn.Module, head1: nn.Module):
        super().__init__()
        self.base = base
        self.head0 = head0
        self.head1 = head1

    def forward(self, inp: torch.Tensor):
        x = self.base(inp)
        x0 = self.head0(x)
        x1 = self.head0(x)

        return x0, x1


class ReprDualhead(nn.Module):
    def __init__(self, encoder: nn.Module, middle: nn.Module, head0: nn.Module, head1: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.middle = middle
        self.head0 = head0
        self.head1 = head1

    def forward(self, inp: torch.Tensor):
        repre, repre_var = self.encoder(inp)
        x = self.middle(repre)
        x0 = self.head0(x)
        x1 = self.head0(x)

        return x0, x1, repre, repre_var