import pytest

from itertools import product

import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ei import topology_of, vector_ei_of_layer


def test_ei_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for in_w, out_w in product([1, 5, 10, 20], [1, 5, 10, 20]):
        layer = nn.Linear(in_w, out_w, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, in_w)).to(device))
        ei = vector_ei_of_layer(layer, top,
                samples=1000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=64,
                out_bins=64,
                activation=nn.Sigmoid(),
                device=device)
        assert ei > -1e-15

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
    ei = vector_ei_of_layer(layer, top,
            samples=1000,
            batch_size=1000,
            in_range=(0, 1),
            out_range=(0, 1),
            in_bins=16,
            out_bins=16,
            activation=lambda x: x,
            device=device)
    assert ei < 5 and ei > 3

def test_ei_2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layer = nn.Linear(2, 2, bias=False).to(device)
    top = topology_of(layer, input=torch.zeros((1, 2)).to(device))
    layer.weight[0, 0] = 1.0
    layer.weight[0, 1] = 0.0
    layer.weight[1, 0] = 0.0
    layer.weight[1, 1] = 1.0
    ei = vector_ei_of_layer(layer, top,
            samples=5000,
            batch_size=5000,
            in_range=(0, 1),
            out_range=(0, 1),
            in_bins=16,
            out_bins=16,
            activation=lambda x: x,
            device=device)
    assert ei < 10 and ei > 6

def test_ei_3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layer = nn.Linear(2, 2, bias=False).to(device)
    top = topology_of(layer, input=torch.zeros((1, 2)).to(device))
    nn.init.zeros_(layer.weight)
    layer.weight[0, 0] = 1.0
    layer.weight[1, 1] = 1.0
    ei = vector_ei_of_layer(layer, top,
            threshold=0.05,
            batch_size=1000,
            in_range=(0, 1),
            out_range=(0, 1),
            in_bins=32,
            out_bins=32,
            activation=lambda x: x,
            device=device)
    assert ei < 11 and ei > 9


def test_ei_4():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layer = nn.Linear(3, 3, bias=False).to(device)
    top = topology_of(layer, input=torch.zeros((1, 3)).to(device))
    nn.init.zeros_(layer.weight)
    layer.weight[0, 0] = 1.0
    layer.weight[1, 1] = 1.0
    layer.weight[2, 2] = 1.0
    ei = vector_ei_of_layer(layer, top,
            threshold=0.05,
            batch_size=1000,
            in_range=(0, 1),
            out_range=(0, 1),
            in_bins=32,
            out_bins=32,
            activation=lambda x: x,
            device=device)
    assert ei < 16 and ei > 14


def test_ei_5():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        layer = nn.Linear(n, n, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, n)).to(device))
        nn.init.zeros_(layer.weight)
        layer.weight[0, 0] = 1.0
        layer.weight[1, 1] = 1.0
        ei = vector_ei_of_layer(layer, top,
                threshold=0.05,
                batch_size=1000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=16,
                out_bins=16,
                activation=lambda x: x,
                device=device)
        assert ei < (4*2 + 0.5) and ei > (4*2 - 0.5)

def test_ei_6():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for n in [3, 4, 5, 6, 7, 8, 9, 10]:
        layer = nn.Linear(n, n, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, n)).to(device))
        nn.init.zeros_(layer.weight)
        layer.weight[0, 0] = 1.0
        layer.weight[1, 1] = 1.0
        layer.weight[2, 2] = 1.0
        ei = vector_ei_of_layer(layer, top,
                threshold=0.05,
                batch_size=1000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=16,
                out_bins=16,
                activation=lambda x: x,
                device=device)
        assert ei < (4*3 + 0.5) and ei > (4*3 - 0.5)


def test_ei_7():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for n in [10, 20, 30, 40, 50]:
        layer = nn.Linear(n, n, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, n)).to(device))
        nn.init.zeros_(layer.weight)
        layer.weight[0, 0] = 1.0
        layer.weight[3, 3] = 1.0
        layer.weight[7, 7] = 1.0
        ei = vector_ei_of_layer(layer, top,
                threshold=0.05,
                batch_size=1000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=16,
                out_bins=16,
                activation=lambda x: x,
                device=device)
        assert ei < (4*3 + 0.5) and ei > (4*3 - 0.5)


def test_ei_8():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for n in [10, 20, 30, 40, 50]:
        layer = nn.Linear(n, n, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, n)).to(device))
        nn.init.zeros_(layer.weight)
        ei = vector_ei_of_layer(layer, top,
                threshold=0.05,
                batch_size=1000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=16,
                out_bins=16,
                activation=nn.Sigmoid(),
                device=device)
        assert ei < 0.1 and ei > -0.1


def test_ei_9():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for n in [1, 2, 3, 4]:
        layer = nn.Linear(n, n, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, n)).to(device))
        nn.init.zeros_(layer.weight)
        for k in range(n):
            layer.weight[k, k] = 1.0
        ei = vector_ei_of_layer(layer, top,
                threshold=0.05,
                batch_size=1000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=16,
                out_bins=16,
                activation=lambda x: x,
                device=device)
        assert ei < (4*n + 1) and ei > (4*n - 1)

