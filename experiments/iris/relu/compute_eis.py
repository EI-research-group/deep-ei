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

BINS = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
torch.set_default_dtype(dtype)
print(device)

here = Path()
runs = list(here.glob("run*-frames"))
print(runs)

for run in runs:
    frames_files = list(run.glob('*.frame'))
#     frames = [get_measures(path) for path in frames_files]
#     frames.sort(key=lambda f: f['batches'])
    network = nn.Sequential(
        nn.Linear(4, 5, bias=False),
        nn.ReLU(),
        nn.Linear(5, 5, bias=False),
        nn.ReLU(),
        nn.Linear(5, 3, bias=False),
        nn.ReLU()
    ).to(device)
    for frame_file in tqdm(frames_files):
        metrics = torch.load(frame_file)
        network.load_state_dict(metrics['model'])
        top = topology_of(network, input=torch.zeros((1, 4)).to(device))
        layer1, _, layer2, _, layer3, _ = network
        ei_layer1 = ei_of_layer(layer1, top,
                        threshold=0.05,
                        in_range=(0, 1),
                        in_bins=BINS,
                        out_range=(0, 1),
                        out_bins=BINS,
                        activation=nn.ReLU(),
                        device=device)
        sensitivity_layer1 = sensitivity_of_layer(layer1, top,
                        samples=1000,
                        in_range=(0, 1),
                        in_bins=BINS,
                        out_range=(0, 1),
                        out_bins=BINS,
                        activation=nn.ReLU(),
                        device=device)
        ei_layer2 = ei_of_layer(layer2, top,
                        threshold=0.05,
                        in_range=(0, 1),
                        in_bins=BINS,
                        out_range=(0, 1),
                        out_bins=BINS,
                        activation=nn.ReLU(),
                        device=device)
        sensitivity_layer2 = sensitivity_of_layer(layer2, top,
                        samples=1000,
                        in_range=(0, 1),
                        in_bins=BINS,
                        out_range=(0, 1),
                        out_bins=BINS,
                        activation=nn.ReLU(),
                        device=device)
        ei_layer3 = ei_of_layer(layer3, top,
                        threshold=0.05,
                        in_range=(0, 1),
                        in_bins=BINS,
                        out_range=(0, 1),
                        out_bins=BINS,
                        activation=nn.ReLU(),
                        device=device)
        sensitivity_layer3 = sensitivity_of_layer(layer3, top,
                        samples=1000,
                        in_range=(0, 1),
                        in_bins=BINS,
                        out_range=(0, 1),
                        out_bins=BINS,
                        activation=nn.ReLU(),
                        device=device)
        metrics['ei_layer1'] = ei_layer1
        metrics['sensitivity_layer1'] = sensitivity_layer1
        metrics['ei_layer2'] = ei_layer2
        metrics['sensitivity_layer2'] = sensitivity_layer2,
        metrics['ei_layer3'] = ei_layer3,
        metrics['sensitivity_layer3'] = sensitivity_layer3
        torch.save(metrics, frame_file)
    