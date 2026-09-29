import os
import sys

import pytest
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def toy_graph():
    """Small ACM-like graph: papers with features, authors with features, feature-less terms.

    Paper 5 has no terms and no citations, so walks from it through terms must dead-end.
    """
    torch.manual_seed(0)
    g = HeteroData()
    g["paper"].x = torch.randn(6, 8)
    g["paper"].y = torch.tensor([0, 1, 2, 0, 1, 2])
    g["author"].x = torch.randn(4, 5)
    g["term"].num_nodes = 3

    pa = torch.tensor([[0, 0, 1, 2, 3, 4, 5], [0, 1, 1, 2, 3, 0, 3]])
    pt = torch.tensor([[0, 0, 1, 2, 3, 4], [0, 1, 1, 2, 0, 2]])
    pc = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    g["paper", "to", "author"].edge_index = pa
    g["author", "to", "paper"].edge_index = pa.flip(0)
    g["paper", "to", "term"].edge_index = pt
    g["term", "to", "paper"].edge_index = pt.flip(0)
    g["paper", "cite", "paper"].edge_index = pc
    return g


@pytest.fixture
def toy_templates():
    return [
        ["paper", "to", "term", "to", "paper"],
        ["paper", "to", "author", "to", "paper"],
        ["paper", "cite", "paper", "to", "term", "to", "paper"],
    ]