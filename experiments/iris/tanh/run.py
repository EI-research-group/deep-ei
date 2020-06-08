import os
from tqdm.auto import tqdm
from pathlib import Path
from random import shuffle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import sklearn.datasets

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ei import topology_of, ei_of_layer, sensitivity_of_layer

########### PARAMS ############
BINS = 256
LAYERS = [(4, 5), (5, 5), (5, 3)]
ACTIVATION = nn.Tanh()
RUNS = 3
FREQUENCY = 5 # epochs per measurement
EPOCHS = 500
# BATCH_SIZE = 10 this is fixed
print("Total Measurements: {}".format(EPOCHS / FREQUENCY))


########### Set Device ############
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
torch.set_default_dtype(dtype)
print("Using device: {}".format(device))


data = list(zip(*sklearn.datasets.load_iris(return_X_y=True)))
shuffle(data)
inputs, labels = zip(*data)
inputs = np.array(inputs) / 8
training_inputs, training_labels = inputs[:100], labels[:100]
testing_inputs, testing_labels = inputs[100:], labels[100:]

training_data = torch.utils.data.TensorDataset(torch.tensor(training_inputs, dtype=dtype),
                                               F.one_hot(torch.tensor(training_labels, dtype=torch.long)).to(dtype))
testing_data = torch.utils.data.TensorDataset(torch.tensor(testing_inputs, dtype=dtype),
                                               F.one_hot(torch.tensor(testing_labels, dtype=torch.long)).to(dtype))

training_loader = torch.utils.data.DataLoader(training_data, 
                            batch_size=10,
                            shuffle=True)

training_metrics_loader = torch.utils.data.DataLoader(training_data, 
                            batch_size=10,
                            shuffle=True)

testing_loader = torch.utils.data.DataLoader(testing_data, 
                            batch_size=10,
                            shuffle=False)

testing_metrics_loader = torch.utils.data.DataLoader(testing_data, 
                            batch_size=10,
                            shuffle=False)


########### Weight Initializer ############
initializers = {
    'kaiming': None, # (default)
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'paper': nn.init.uniform_
}

def weight_initializer(name):
    def init_weights(m):
        if name == 'paper':
            if isinstance(m, nn.Linear):
                boundary = 1 / np.sqrt(m.in_features)
                nn.init.uniform_(m.weight, a=-boundary, b=boundary)
    return init_weights


for run in range(1, RUNS+1):
    print("STARTING RUN {}".format(run))
    output_dir = Path("run{}-frames".format(run))
    if not output_dir.exists():
        output_dir.mkdir()
    ls = []
    for (in_n, out_n) in LAYERS:
        ls.append(nn.Linear(in_n, out_n, bias=False))
        ls.append(ACTIVATION)
    network = nn.Sequential(*ls).to(device)
    network.apply(weight_initializer('paper'))
    print(network)
    top = topology_of(network, input=torch.zeros((1, LAYERS[0][0])).to(device))

    optimizer = torch.optim.SGD(network.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    def compute_metrics():
        outof = 0
        training_loss = 0
        training_accuracy = 0
        with torch.no_grad():
            for sample, target in training_metrics_loader:
                output = network(sample.to(device))
                _, pred = torch.max(output, 1)
                _, answer = torch.max(target.to(device), 1)
                training_accuracy += (pred == answer).sum().item()
                training_loss += loss_fn(output, target.to(device))
                outof += len(target)
        training_loss = float(training_loss / outof)
        training_accuracy = training_accuracy / outof

        outof = 0
        testing_loss = 0
        testing_accuracy = 0
        with torch.no_grad():
            for sample, target in testing_metrics_loader:
                output = network(sample.to(device))
                _, pred = torch.max(output, 1)
                _, answer = torch.max(target.to(device), 1)
                testing_accuracy += (pred == answer).sum().item()
                testing_loss += loss_fn(output, target.to(device))
                outof += len(target)
        testing_loss = float(testing_loss / outof)
        testing_accuracy = testing_accuracy / outof
        
        metrics = {
            'training_loss': training_loss,
            'training_accuracy': training_accuracy,
            'testing_loss': testing_loss,
            'testing_accuracy': testing_accuracy
        }
        
        i = 1
        for layer in network:
            if type(layer) is nn.Linear:
                if i == 1:
                    in_range = (0, 1)
                else:
                    in_range = (-1, 1)
                ei = ei_of_layer(layer, top,
                            threshold=0.05,
                            in_range=in_range,
                            in_bins=BINS,
                            out_range=(-1, 1),
                            out_bins=BINS,
                            activation=ACTIVATION,
                            device=device)
                sensitivity = sensitivity_of_layer(layer, top,
                            samples=1000,
                            in_range=in_range,
                            in_bins=BINS,
                            out_range=(-1, 1),
                            out_bins=BINS,
                            activation=ACTIVATION,
                            device=device)
                metrics[f"ei_layer{i}"] = ei
                metrics[f"sensitivity_layer{i}"] = sensitivity
                i += 1
        return metrics
    
    num_batches = 0
    for epoch in tqdm(range(EPOCHS)):
        if epoch % FREQUENCY == 0:
            #######################################
            #          Compute Measures           #
            #######################################
            metrics = compute_metrics()
            metrics['batches'] = num_batches
            metrics['epochs'] = epoch 
            metrics['model'] = network.state_dict()
            metrics['optimizer'] = optimizer.state_dict()
            name = output_dir / "batchnum-{}.frame".format(num_batches)
            torch.save(metrics, name)
        for sample, target in training_loader:
            optimizer.zero_grad()
            batch_loss = loss_fn(network(sample.to(device)), target.to(device))
            batch_loss.backward()
            optimizer.step()
            num_batches += 1
