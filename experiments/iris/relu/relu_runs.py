from pathlib import Path
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import sklearn
import sklearn.datasets
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from deep_ei import topology_of, ei_of_layer, sensitivity_of_layer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
torch.set_default_dtype(dtype)
print(device)

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

for run in tqdm(range(15)):
    output_dir = Path("run{}-frames".format(run))
    if not output_dir.exists():
        output_dir.mkdir()
    network = nn.Sequential(
        nn.Linear(4, 5, bias=False),
        nn.ReLU(),
        nn.Linear(5, 5, bias=False),
        nn.ReLU(),
        nn.Linear(5, 3, bias=False),
        nn.ReLU()
    ).to(device)
    network.apply(weight_initializer('paper'))
    top = topology_of(network, input=torch.zeros((1, 4)).to(device))
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    num_batches = 0
    for epoch in range(400):
        if epoch % 5 == 0:
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
                'epoch': epoch,
                'batches': num_batches,
                'training_loss': training_loss,
                'training_accuracy': training_accuracy,
                'testing_loss': testing_loss,
                'testing_accuracy': testing_accuracy,
                'model': network.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            name = output_dir / "batchnum-{}.frame".format(num_batches)
            torch.save(metrics, name)   
        #######################################
        #           Update Weights            #
        #######################################    
        for sample, target in training_loader:
            optimizer.zero_grad()
            batch_loss = loss_fn(network(sample.to(device)), target.to(device))
            batch_loss.backward()
            optimizer.step()
            num_batches += 1
