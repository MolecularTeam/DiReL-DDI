import sys
import torch
from torch import nn
import torch.nn.functional as F
import math

from torch_geometric.nn import GATConv

from get_args import get_config

cfg = get_config()
layer_dataset_name = cfg.dataset_name


class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim, dp, head, edge, head_out_feats):
        super().__init__()
        self.input_dim = int(input_dim)
        self.head = int(head)
        self.head_out_feats = int(head_out_feats)
        self.edge_dim = int(edge)

        out_dim = self.head_out_feats // 2

        self.intra = GATConv(
            in_channels=self.input_dim,
            out_channels=int(out_dim),
            heads=self.head,
            edge_dim=self.edge_dim,
            dropout=dp
        )

    def forward(self, data):
        input_feature, edge_index = data.x, data.edge_index
        input_feature = F.relu(input_feature)
        intra_rep = self.intra(input_feature, edge_index, data.edge_attr)
        return intra_rep


class InterGraphAttention(nn.Module):
    def __init__(self, input_dim, dp, head, edge, head_out_feats):
        super().__init__()
        self.input_dim = int(input_dim)
        self.head = int(head)
        self.head_out_feats = int(head_out_feats)
        self.edge_dim = int(edge)

        out_dim = self.head_out_feats // 2

        self.inter = GATConv(
            in_channels=(self.input_dim, self.input_dim),
            out_channels=int(out_dim),
            heads=self.head,
            dropout=dp
        )

    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.relu(h_data.x)
        t_input = F.relu(t_data.x)

        t_rep = self.inter((h_input, t_input), edge_index)
        h_rep = self.inter((t_input, h_input), edge_index[[1, 0]])

        return h_rep, t_rep
