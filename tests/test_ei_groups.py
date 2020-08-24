import pytest

from itertools import product

import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ei import topology_of, eis_between_groups


def test_ei_groups_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layer = nn.Linear(2, 2, bias=False).to(device)
    layer.weight[0, 0] = 1.0
    layer.weight[0, 1] = 0.0
    layer.weight[1, 0] = 0.0
    layer.weight[1, 1] = 1.0
    top = topology_of(layer, input=torch.zeros((1, 2)).to(device))
    ei = eis_between_groups(layer, top,
    			groups=[((0,1),(0,1))],
    			samples=1000,
    			in_range=(0, 1),
    			out_range=(0, 1),
    			in_bins=16,
    			out_bins=16,
    			activation=lambda x: x,
    			device=device)
    assert len(ei) == 1
    assert 7.5 <= ei[0] <= 8.5


def test_ei_groups_1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layer = nn.Linear(2, 2, bias=False).to(device)
    layer.weight[0, 0] = 1.0
    layer.weight[0, 1] = 0.0
    layer.weight[1, 0] = 0.0
    layer.weight[1, 1] = 1.0
    top = topology_of(layer, input=torch.zeros((1, 2)).to(device))
    eis = eis_between_groups(layer, top,
    			groups=[((0,),(0,)), ((0,),(1,)), ((1,),(0,)), ((1,),(1,))],
    			samples=1000,
    			in_range=(0, 1),
    			out_range=(0, 1),
    			in_bins=16,
    			out_bins=16,
    			activation=lambda x: x,
    			device=device)
    assert len(eis) == 4
    assert 3.75 <= eis[0] <= 4.25
    assert -0.25 <= eis[1] <= 0.25
    assert -0.25 <= eis[2] <= 0.25
    assert 3.75 <= eis[3] <= 4.25
    assert 7.5 <= sum(eis) <= 8.5



