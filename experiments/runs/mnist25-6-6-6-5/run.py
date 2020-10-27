import os
from tqdm.auto import tqdm
from pathlib import Path
from random import shuffle
from math import isclose, ceil
from itertools import combinations_with_replacement
import gzip
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skimage.transform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from deep_ei import topology_of, ei_of_layer, sensitivity_of_layer, vector_ei_of_layer, vector_and_pairwise_ei

########### PARAMS ############
BINS = 8
LAYERS = [(25, 6), (6, 6), (6, 6), (6, 5)]
ACTIVATION = nn.Sigmoid()
RUNS = 3
FREQUENCY = 10 # epochs per measurement
EPOCHS = 500
BATCH_SIZE = 50
print("Total Measurements: {}".format(EPOCHS / FREQUENCY))


########### Set Device ############
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
torch.set_default_dtype(dtype)
print("Using device: {}".format(device))


dir_path = Path().absolute()
dataset_path = dir_path.parent.parent / "data/mnist.pkl.gz"
if not dataset_path.exists():
    print('Downloading dataset with curl ...')
    if not dataset_path.parent.exists():
        os.mkdir(dataset_path.parent)
    url = 'http://ericjmichaud.com/downloads/mnist.pkl.gz'
    os.system('curl -L {} -o {}'.format(url, dataset_path))
print('Download failed') if not dataset_path.exists() else print('Dataset acquired')
f = gzip.open(dataset_path, 'rb')
mnist = pickle.load(f)
f.close()
print('Loaded data to variable `mnist`')


mnist0_4 = list(filter(lambda t: t[1].argmax() < 5, mnist))
for i, t in enumerate(mnist0_4):
    idx = t[1].argmax()
    new_onehot = np.zeros(5,)
    new_onehot[idx] = 1.0
    mnist0_4[i] = (t[0], new_onehot)
print('Reduced dataset to only 0-4 examples')

# ///////////////////////////////////////////
#               DEFINE `Dataset`
# ///////////////////////////////////////////
class MNISTDataset(Dataset):
    """MNIST Digits Dataset."""
    def __init__(self, data, width=5, transform=None):
        """We save the dataset images as torch.tensor since saving 
        the dataset in memory inside a `Dataset` object as a 
        python list or a numpy array causes a multiprocessiing-related 
        memory leak."""
        self.images, self.labels = zip(*data)
        self.images = torch.from_numpy(np.array(self.images)).to(dtype)
        self.labels = torch.tensor(self.labels).to(dtype)
        self.width = width
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = skimage.transform.resize(image.reshape((28, 28)), (self.width, self.width))
        if self.transform:
            image, label = self.transform((image, label))
        return image.flatten(), label

training_data = MNISTDataset(mnist0_4[:25000], width=5)
testing_data = MNISTDataset(mnist0_4[25000:], width=5)
   
training_loader = torch.utils.data.DataLoader(training_data, 
                            batch_size=BATCH_SIZE,
                            shuffle=True)

training_metrics_loader = torch.utils.data.DataLoader(training_data, 
                                batch_size=BATCH_SIZE,
                                shuffle=False)

testing_loader = torch.utils.data.DataLoader(testing_data, 
                            batch_size=BATCH_SIZE,
                            shuffle=False)

testing_metrics_loader = torch.utils.data.DataLoader(testing_data, 
                                batch_size=BATCH_SIZE,
                                shuffle=False)


batches_per_epoch = ceil(len(training_data) / BATCH_SIZE)
print("Batches / epoch: {}".format(batches_per_epoch))
print("Given frequency: {}".format(FREQUENCY))
print("Adjusted frequency: {}".format(1 / batches_per_epoch * round(FREQUENCY * batches_per_epoch)))


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
    ######### Set Seeds ##########
    torch.set_default_dtype(torch.float32)
    np.random.seed(run)
    torch.manual_seed(run)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        
#         for (start_i, end_i) in combinations_with_replacement(range(len(LAYERS)), 2):
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
        #######################################
        #          Compute Measures           #
        #######################################
        for sample, target in training_loader:
            if isclose((num_batches / batches_per_epoch) % FREQUENCY, 0, abs_tol=1e-7):
                #######################################
                #          Compute Measures           #
                #######################################
                metrics = compute_metrics()
                metrics['batches'] = num_batches
                metrics['epochs'] = num_batches / batches_per_epoch
                metrics['model'] = network.state_dict()
                metrics['optimizer'] = optimizer.state_dict()
                name = output_dir / "batchnum-{}.frame".format(num_batches)
                torch.save(metrics, name)
            optimizer.zero_grad()
            batch_loss = loss_fn(network(sample.to(device)), target.to(device))
            batch_loss.backward()
            optimizer.step()
            num_batches += 1

