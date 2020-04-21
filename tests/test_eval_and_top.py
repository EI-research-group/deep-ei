import pytest

import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ei import _eval_model, topology_of


def test_top_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    layer = nn.Linear(10, 20).to(device)
    top = topology_of(layer, input=torch.zeros((1, 10)).to(device))
    assert layer in top
    assert top.nodes[layer]['input']['shape'] == (1, 10)
    assert top.nodes[layer]['output']['shape'] == (1, 20)
    assert top.nodes[layer]['input']['activation'] is None
    assert top.nodes[layer]['output']['activation'] is None


def test_top_1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    network = nn.Sequential(
        nn.Linear(10, 20),
        nn.Sigmoid(),
        nn.Linear(20, 5),
        nn.Tanh()
    ).to(device)
    layer1, _, layer2, _ = network
    top = topology_of(network, input=torch.zeros((1, 10)).to(device))
    assert layer1 in top
    assert layer2 in top
    assert layer2 in nx.descendants(top, layer1)
    assert top.nodes[layer1]['input']['shape'] == (1, 10)
    assert top.nodes[layer1]['output']['shape'] == (1, 20)
    assert top.nodes[layer1]['input']['activation'] is None
    assert type(top.nodes[layer1]['output']['activation']) is nn.Sigmoid
    assert top.nodes[layer2]['input']['shape'] == (1, 20)
    assert top.nodes[layer2]['output']['shape'] == (1, 5)
    assert type(top.nodes[layer2]['input']['activation']) is nn.Sigmoid
    assert type(top.nodes[layer2]['output']['activation']) is nn.Tanh  


# THESE TWO TESTS ARE COMMENTED OUT TEMPORARILY
# def test_top_2():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     torch.set_default_dtype(torch.float32)
#     network = nn.Sequential(
#         nn.Conv2d(1, 5, 3),
#         nn.ReLU(),
#         nn.Flatten(),
#         nn.Linear(64 * 5, 5),
#         nn.Tanh()
#     ).to(device)
#     layer1, _, _, layer2, _ = network
#     top = topology_of(network, input=torch.zeros((1, 1, 10, 10)).to(device))
#     assert layer1 in top
#     assert layer2 in top
#     assert layer2 in nx.descendants(top, layer1)
#     assert top.nodes[layer1]['input']['shape'] == (1, 1, 10, 10)
#     assert top.nodes[layer1]['output']['shape'] == (1, 5, 8, 8)
#     assert top.nodes[layer1]['input']['activation'] is None
#     assert type(top.nodes[layer1]['output']['activation']) is nn.ReLU
#     assert top.nodes[layer2]['input']['shape'] == (1, 320)
#     assert top.nodes[layer2]['output']['shape'] == (1, 5)
#     assert top.nodes[layer2]['input']['activation'] is None
#     assert type(top.nodes[layer2]['output']['activation']) is nn.Tanh  

# def test_top_3():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     torch.set_default_dtype(torch.float32)
#     network = nn.Sequential(
#         nn.Conv2d(1, 5, 3),
#         nn.MaxPool2d(2),
#         nn.ReLU(),
#         nn.Flatten(),
#         nn.Linear(80, 5),
#         nn.Tanh()
#     ).to(device)
#     layer1, layer2, _, _, layer3, _ = network
#     top = topology_of(network, input=torch.zeros((1, 1, 10, 10)).to(device))
#     assert layer1 in top
#     assert layer2 in top
#     assert layer2 in nx.descendants(top, layer1)
#     assert layer3 in nx.descendants(top, layer1)
#     assert layer1 not in nx.descendants(top, layer2)
#     assert top.nodes[layer1]['input']['shape'] == (1, 1, 10, 10)
#     assert top.nodes[layer1]['output']['shape'] == (1, 5, 8, 8)
#     assert top.nodes[layer1]['input']['activation'] is None
#     assert top.nodes[layer1]['output']['activation'] is None
#     assert top.nodes[layer2]['input']['shape'] == (1, 5, 8, 8)
#     assert top.nodes[layer2]['output']['shape'] == (1, 5, 4, 4)
#     assert top.nodes[layer2]['input']['activation'] is None
#     assert type(top.nodes[layer2]['output']['activation']) is nn.ReLU
#     assert top.nodes[layer3]['input']['shape'] == (1, 80)
#     assert top.nodes[layer3]['output']['shape'] == (1, 5)
#     assert top.nodes[layer3]['input']['activation'] is None
#     assert type(top.nodes[layer3]['output']['activation']) is nn.Tanh 


def test_eval_0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    layer = nn.Linear(10, 10).to(device)
    x = torch.randn((5, 10)).to(device)
    top = topology_of(layer, input=x)
    x = torch.randn((5, 10)).to(device)
    correct_output = layer(x)
    evaluated_output = _eval_model(x, layer, layer, top, None)
    assert torch.all(torch.eq(evaluated_output, correct_output))


def test_eval_1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    network = nn.Sequential(
        nn.Linear(10, 20),
        nn.Linear(20, 10)
    ).to(device)
    x = torch.randn((5, 10)).to(device)
    top = topology_of(network, input=x)
    layer1, layer2 = network
    x = torch.randn((5, 10)).to(device)
    correct_output = network(x)
    evaluated_output = _eval_model(x, layer1, layer2, top, None)
    assert torch.all(torch.eq(evaluated_output, correct_output))


def test_eval_2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    network = nn.Sequential(
        nn.Linear(10, 20),
        nn.Sigmoid(),
        nn.Linear(20, 10)
    ).to(device)
    x = torch.randn((5, 10)).to(device)
    top = topology_of(network, input=x)
    layer1, _, layer2 = network
    x = torch.randn((5, 10)).to(device)
    correct_output = network(x)
    evaluated_output = _eval_model(x, layer1, layer2, top, None)
    assert torch.all(torch.eq(evaluated_output, correct_output))


def test_eval_3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    network = nn.Sequential(
        nn.Linear(10, 20),
        nn.Sigmoid(),
        nn.Linear(20, 10),
        nn.Sigmoid()
    ).to(device)
    x = torch.randn((5, 10)).to(device)
    top = topology_of(network, input=x)
    layer1, _, layer2, _ = network
    x = torch.randn((5, 10)).to(device)
    correct_output = network(x)
    evaluated_output = _eval_model(x, layer1, layer2, top, nn.Sigmoid())
    assert torch.all(torch.eq(evaluated_output, correct_output))


def test_eval_4():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    network = nn.Sequential(
        nn.Linear(10, 20),
        nn.Sigmoid(),
        nn.Linear(20, 10),
        nn.Sigmoid(),
        nn.Linear(10, 5),
        nn.Tanh()
    ).to(device)
    x = torch.randn((5, 10)).to(device)
    top = topology_of(network, input=x)
    layer1, _, layer2, _, layer3, _ = network
    x = torch.randn((5, 10)).to(device)
    correct_output = network(x)
    evaluated_output = _eval_model(x, layer1, layer3, top, nn.Tanh())
    assert torch.all(torch.eq(evaluated_output, correct_output))
    x = torch.randn((5, 20)).to(device)
    correct_output = nn.Sigmoid()(layer2(x))
    evaluated_output = _eval_model(x, layer2, layer2, top, nn.Sigmoid())
    assert torch.all(torch.eq(evaluated_output, correct_output))
    x = torch.randn((5, 20)).to(device)
    correct_output = nn.Sigmoid()(layer3(nn.Sigmoid()(layer2(x))))
    evaluated_output = _eval_model(x, layer2, layer3, top, nn.Sigmoid())
    assert torch.all(torch.eq(evaluated_output, correct_output))


def test_eval_5():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    network = nn.Sequential(
        nn.Conv2d(1, 5, 3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(80, 5),
        nn.Tanh()
    ).to(device)
    layer1, layer2, _, _, layer3, _ = network
    x = torch.randn((1, 1, 10, 10)).to(device)
    top = topology_of(network, input=x)
    correct_output = network(x)
    evaluated_output = _eval_model(x, layer1, layer3, top, nn.Tanh())
    assert torch.all(torch.eq(evaluated_output, correct_output))
    x = torch.randn((1, 1, 10, 10)).to(device)
    correct_output = layer1(x)
    evaluated_output = _eval_model(x, layer1, layer1, top, None)
    assert torch.all(torch.eq(evaluated_output, correct_output))
    x = torch.randn((1, 1, 10, 10)).to(device)
    correct_output = nn.ReLU()(layer2(layer1(x)))
    evaluated_output = _eval_model(x, layer1, layer2, top, nn.ReLU())
    assert torch.all(torch.eq(evaluated_output, correct_output))




