from typing import Optional, Union, List

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, activations: List, num_nodes: List[int], biasses: Optional[Union[bool, List[bool]]]=True):
        super().__init__()
        assert len(activations)+1 == len(num_nodes)
        self.num_layers = len(activations)
        
        if isinstance(biasses, bool):
            biasses = [biasses]*len(activations)

        for inode,(activation, bias) in enumerate(zip(activations, biasses)):
            setattr(self, f"layer{inode+1}", nn.Linear(in_features=num_nodes[inode], out_features=num_nodes[inode+1], bias=bias))
            setattr(self, f"activation{inode+1}", activation)

    def forward(self, x: torch.Tensor):
        for ilay in range(1,self.num_layers+1):
            x = getattr(self, f"layer{ilay}")(x)
            activation = getattr(self, f"activation{ilay}")
            if activation is not None:
                x = activation(x)
        return x