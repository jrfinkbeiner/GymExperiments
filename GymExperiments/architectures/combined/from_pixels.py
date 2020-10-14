import torch
import torch.nn as nn
from typing import Optional

from GymExperiments.architectures.multihead import Dualhead, ReprDualhead
from GymExperiments.architectures.blocks import MLP
from GymExperiments.architectures.cnn import SimpleCNNEncoder


def set_up_dualhead_from_pixels(
        encoder: nn.Module,
        encoder_out_dim: int,
        middle: Optional[nn.Module] = None,
        head0: Optional[nn.Module] = None,
        head1: Optional[nn.Module] = None
    ):

    # if encoder is None:
    #     encoder = ...
    if feed_forward is None:
        feed_forward = MLP([nn.ReLU(), nn.ReLU()], [encoder_out_dim, 128, 64])
    if head0 is None:
        head0 = MLP([nn.ReLU(), nn.ReLU()], [64, 32, 1])
    if head1 is None:
        head1 = MLP([nn.ReLU(), nn.ReLU()], [64, 32, 1])

    base = nn.Sequential(
        encoder,
        feed_forward,
    )

    return Dualhead(base, head0, head1)


def set_up_repr_dualhead_from_pixels(
        encoder: nn.Module,
        encoder_out_dim: int,
        middle: Optional[nn.Module] = None,
        head0: Optional[nn.Module] = None,
        head1: Optional[nn.Module] = None,
        out_dim: Optional[int] = 1,
    ):

    # if encoder is None:
    #     encoder = ...
    if middle is None:
        middle = MLP([nn.ReLU()], [encoder_out_dim, 64])
    if head0 is None:
        head0 = MLP([nn.ReLU(), nn.Sigmoid()], [64, 32, out_dim])
    if head1 is None:
        head1 = MLP([nn.ReLU(), nn.Softplus()], [64, 32, out_dim])
    # if head0 is None:
    #     head0 = MLP([nn.ReLU(), nn.Tanh(), lambda x: x], [64, 32, out_dim, out_dim])
    # if head1 is None:
    #     head1 = MLP([nn.ReLU(), nn.Tanh(), lambda x: x], [64, 32, out_dim, out_dim])

    return ReprDualhead(encoder, middle, head0, head1)