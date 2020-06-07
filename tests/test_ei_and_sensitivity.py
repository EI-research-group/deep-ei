import pytest

from itertools import product

import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ei import topology_of, ei_of_layer, sensitivity_of_layer,\
                        ei_of_layer_matrix, sensitivity_of_layer_matrix


def test_ei_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    for in_w, out_w in product([1, 5, 10, 20], [1, 5, 10, 20]):
        layer = nn.Linear(in_w, out_w, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, in_w)).to(device))
        ei = ei_of_layer(layer, top,
                samples=1000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=64,
                out_bins=64,
                activation=nn.Sigmoid(),
                device=device)
        assert ei > 0

def test_ei_1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layer = nn.Linear(1, 1, bias=False).to(device)
    top = topology_of(layer, input=torch.zeros((1, 1)).to(device))
    layer.weight[0, 0] = 1.0
    print(layer.weight)
    ei = ei_of_layer(layer, top,
            samples=1000,
            batch_size=1000,
            in_range=(0, 1),
            out_range=(0, 1),
            in_bins=16,
            out_bins=16,
            activation=lambda x: x,
            device=device)
    assert ei < 5 and ei > 3

def test_ei_matrix_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    for in_w, out_w in product([1, 5, 10, 20], [1, 5, 10, 20]):
        layer = nn.Linear(in_w, out_w, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, in_w)).to(device))
        eis = ei_of_layer_matrix(layer, top,
                samples=1000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=64,
                out_bins=64,
                activation=nn.Sigmoid(),
                device=device)
        rows, columns = eis.shape
        assert rows == in_w and columns == out_w
        assert eis.min() >= -1e-15


def test_ei_agreement_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for in_w, out_w in product([1, 3, 7], [1, 3, 7]):
        layer = nn.Linear(in_w, out_w, bias=False).to(device)
        nn.init.eye_(layer.weight)
        top = topology_of(layer, input=torch.zeros((1, in_w)).to(device))
        ei = ei_of_layer(layer, top,
                samples=5000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=64,
                out_bins=64,
                activation=nn.Sigmoid(),
                device=device)
        eis = ei_of_layer_matrix(layer, top,
                samples=5000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=64,
                out_bins=64,
                activation=nn.Sigmoid(),
                device=device)
        ei_summed = eis.sum()
        assert ei_summed <= ei*2.0 and ei_summed >= ei/2.0


def test_sensitivity_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    for in_w, out_w in product([1, 5, 10, 20], [1, 5, 10, 20]):
        layer = nn.Linear(in_w, out_w, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, in_w)).to(device))
        sensitivity = sensitivity_of_layer(layer, top,
                samples=100,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=64,
                out_bins=64,
                activation=nn.Sigmoid(),
                device=device)
        assert sensitivity > 1e-15

def test_sensitivity_matrix_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    for in_w, out_w in product([1, 5, 10, 20], [1, 5, 10, 20]):
        layer = nn.Linear(in_w, out_w, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, in_w)).to(device))
        sensitivities = sensitivity_of_layer_matrix(layer, top,
                samples=100,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=64,
                out_bins=64,
                activation=nn.Sigmoid(),
                device=device)
        rows, columns = sensitivities.shape
        assert rows == in_w and columns == out_w
        assert sensitivities.min() >= -1e-15


