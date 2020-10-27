import pytest

from itertools import product

import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ei import topology_of, ei_parts, sensitivity,\
                        ei_parts_matrix, sensitivity_matrix


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
        ei = ei_parts(layer, top,
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
    ei = ei_parts(layer, top,
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
    ei = ei_parts(layer, top,
            samples=1000,
            batch_size=1000,
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
    layer.weight[0, 0] = 1.0
    layer.weight[0, 1] = 0.0
    layer.weight[1, 0] = 0.0
    layer.weight[1, 1] = 1.0
    ei = ei_parts(layer, top,
            samples=10000,
            batch_size=1000,
            in_range=(0, 1),
            out_range=(0, 1),
            in_bins=64,
            out_bins=64,
            activation=lambda x: x,
            device=device)
    assert ei < 13 and ei > 11

def test_ei_4():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for w in [2, 5, 10, 20]:
        layer = nn.Linear(w, w, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, w)).to(device))
        nn.init.eye_(layer.weight)
        ei = ei_parts(layer, top,
                samples=10000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=16,
                out_bins=16,
                activation=lambda x: x,
                device=device)
        assert ei > (4*w*0.8) and ei < (4*w*1.2)

def test_ei_matrix_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    for in_w, out_w in product([1, 5, 10, 20], [1, 5, 10, 20]):
        layer = nn.Linear(in_w, out_w, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, in_w)).to(device))
        eis = ei_parts_matrix(layer, top,
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
        ei = ei_parts(layer, top,
                samples=5000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=64,
                out_bins=64,
                activation=nn.Sigmoid(),
                device=device)
        eis = ei_parts_matrix(layer, top,
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
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for in_w, out_w in product([1, 5, 10, 20], [1, 5, 10, 20]):
        layer = nn.Linear(in_w, out_w, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, in_w)).to(device))
        sens = sensitivity(layer, top,
                samples=100,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=64,
                out_bins=64,
                activation=nn.Sigmoid(),
                device=device)
        assert sens > -1e-15

def test_sensitivity_1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for w in [2, 5, 10, 20]:
        layer = nn.Linear(w, w, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, w)).to(device))
        nn.init.ones_(layer.weight)
        sens = sensitivity(layer, top,
                samples=1000,
                in_range=(0, 1),
                out_range=(0, 1),
                in_bins=16,
                out_bins=16,
                activation=lambda x: x,
                device=device)
        assert sens > (4*w*w*0.9) and sens < (4*w*w*1.1)

def test_sensitivity_matrix_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for in_w, out_w in product([1, 5, 10, 20], [1, 5, 10, 20]):
        layer = nn.Linear(in_w, out_w, bias=False).to(device)
        top = topology_of(layer, input=torch.zeros((1, in_w)).to(device))
        sensitivities = sensitivity_matrix(layer, top,
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


