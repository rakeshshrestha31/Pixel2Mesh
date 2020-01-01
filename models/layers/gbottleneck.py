import torch.nn as nn
import torch.nn.functional as F

from models.layers.gconv import GConv


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, activation=None):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim)
        self.conv2 = GConv(in_features=hidden_dim, out_features=in_dim)
        self.activation = F.relu if activation else None

    def forward(self, inputs, adj_mat):
        x = self.conv1(inputs, adj_mat)
        if self.activation:
            x = self.activation(x)
        x = self.conv2(x, adj_mat)
        if self.activation:
            x = self.activation(x)

        return (inputs + x) * 0.5


class GBottleneck(nn.Module):

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, activation=None):
        super(GBottleneck, self).__init__()

        self.resblock_layers = [
            GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim,
                      activation=activation)
            for _ in range(block_num)
        ]

        # need to make each layer attribute
        for i in range(block_num):
            setattr(self, 'block_' + str(i), self.resblock_layers[i])

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim)
        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim)
        self.activation = F.relu if activation else None

    def forward(self, inputs, adj_mat):
        x = self.conv1(inputs, adj_mat)
        if self.activation:
            x = self.activation(x)

        for block in self.resblock_layers:
            x = block(x, adj_mat)
        x_out = self.conv2(x, adj_mat)

        return x_out, x
