import os
from tqdm.auto import tqdm
from pathlib import Path
from random import shuffle
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import sklearn.datasets

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ei import topology_of, ei_of_layer, sensitivity_of_layer, vector_ei_of_layer, vector_and_pairwise_ei

########### PARAMS ############
BINS = 8
LAYERS = [(4, 5), (5, 5), (5, 3)]
ACTIVATION = nn.Sigmoid()
RUNS = 3
FREQUENCY = 50 # epochs per measurement
EPOCHS = 4000
# BATCH_SIZE = 10 this is fixed
print("Total Measurements: {}".format(EPOCHS / FREQUENCY))


########### Set Device ############
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
torch.set_default_dtype(dtype)
print("Using device: {}".format(device))


######### Set Seeds ##########
torch.set_default_dtype(torch.float32)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###### Create Datasets ######
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
        ls.append(nn.Sigmoid())
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
        
#        for (start_i, end_i) in combinations_with_replacement(range(len(LAYERS)), 2):
        for i in range(len(LAYERS)):
            start_i = end_i = i 
            start_l = ls[start_i * 2]
            end_l = ls[end_i * 2]
            vector_ei, pairwise_ei = vector_and_pairwise_ei(end_l, top,
                                            samples=int(1e7),
                                            in_layer=start_l,
                                            in_range=(0, 1),
                                            in_bins=BINS,
                                            out_range=(0, 1),
                                            out_bins=BINS,
                                            activation=ACTIVATION,
                                            device=device)
            pairwise_sensitivity = sensitivity_of_layer(end_l, top,
                                        samples=5000,
                                        in_layer=start_l,
                                        in_range=(0, 1),
                                        in_bins=BINS,
                                        out_range=(0, 1),
                                        out_bins=BINS,
                                        activation=ACTIVATION,
                                        device=device)
            metrics[f"pairwise-ei:{start_i}-{end_i}"] = pairwise_ei
            metrics[f"pairwise-sensitivity:{start_i}-{end_i}"] = pairwise_sensitivity
            metrics[f"vector-ei:{start_i}-{end_i}"] = vector_ei
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
