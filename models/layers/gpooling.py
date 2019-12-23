import torch
import torch.nn as nn
import numpy as np


class GUnpooling(nn.Module):
    """Graph Pooling layer, aims to add additional vertices to the graph.
    The middle point of each edges are added, and its feature is simply
    the average of the two edge vertices.
    Three middle points are connected in each triangle.
    """

    def __init__(self):
        super(GUnpooling, self).__init__()

    def forward(self, inputs, unpool_idx):
        # dim info
        in_num = torch.max(unpool_idx).item()
        out_num = in_num + len(unpool_idx)
        new_features = inputs[:, unpool_idx].clone()
        new_vertices = 0.5 * new_features.sum(2)
        output = torch.cat([inputs, new_vertices], 1)

        return output

    def __repr__(self):
        return self.__class__.__name__
