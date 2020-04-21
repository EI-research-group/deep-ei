
<div align="center">
	<h1> deep-ei </h1>
  <img src="docs/causal-plane.png" width=40%>
</div>
<div align="center">
	<img src="https://travis-ci.com/EI-research-group/deep-ei.svg?token=XQEp1pndaPyr6Dp2sp6i&branch=master">
	<img src="https://img.shields.io/badge/-SCIENCE!-blueviolet">
	<!-- <img src="http://hits.dwyl.com/EI-research-group/deep-ei.svg"> -->
	<p>This package accompanies the paper <b>"Examining the Causal Structure of Artificial Neural Networks Using Information Theory"</b> by Simon Mattsson, <a href="ericjmichaud.com">Eric J. Michaud</a>, and <a href="https://www.erikphoel.com/">Erik Hoel</a></p>
</div>


## Installation:

The simplest way to install the deep_ei module is simply to navigate to this directory and execute:
```
pip install .
```

An Anaconda environment has also been provided, so you can create a new Anaconda environment with this module and its dependencies simply with:

```
conda env create --file environment.yml
```

Since it is recommended that PyTorch be installed with Anaconda, for many it will be easiest to install pytorch first with `conda install -c pytorch pytorch`, and then execute `pip install .`, which will then install the other dependencies (scikit-learn, fast_histogram) and install the deep_ei module itself.

Some basic tests have been included, which you can run using:

```
python setup.py test
```

## Usage:

```python
import torch
import torch.nn as nn

from deep_ei import topology_of, ei_of_layer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
torch.set_default_dtype(dtype)

network = nn.Linear(10, 10, bias=False).to(device)
top = topology_of(network, input=torch.zeros((1, 10)).to(device))

EI = ei_of_layer(network, top,
                    threshold=0.05,
                    batch_size=100, 
                    in_range=(0, 1),
                    in_bins=64,
                    out_range=(0, 1),
                    out_bins=64,
                    activation=nn.Sigmoid(), 
                    device=device)
```
Which will compute the EI of our network using a sigmoid activation. By default, this function will increase the number of samples it uses until the EI levels off (characterized by whether it will change by more than threshold of its current value even if we doubled the number of samples).

Using the data from the `topology_of` function, it can infer activation function and input and output ranges:
```python
network = nn.Sequential(
    nn.Linear(20, 10, bias=False),
    nn.Sigmoid(),
    nn.Linear(10, 5, bias=False),
    nn.Tanh()
)
top = topology_of(network, input=torch.zeros((1, 20)).to(device))

layer1, _, layer2, _ = network

EI_layer1 = ei_of_layer(layer1, top,
                    threshold=0.05,
                    batch_size=100, 
                    in_range=(0, 1),
                    in_bins=64,
                    out_bins=64, 
                    device=device)

EI_layer2 = ei_of_layer(layer2, top,
                    threshold=0.05,
                    batch_size=100, 
                    in_bins=64,
                    out_bins=64, 
                    device=device)
```
Which will use an activation of `nn.Sigmoid` and an `out_range` of `(0, 1)` for the first layer and an activation of `nn.Tanh` and an `out_range` of `(-1, 1)` for the second layer. Note that we have to specify an `in_range` for the first layer.

Instead of specifying a `threshold`, you can instead manually specify a number of samples to use when computing effective information:
```python
network = nn.Linear(10, 10, bias=False).to(device)
top = topology_of(network, input=torch.zeros((1, 10)).to(device))

EI = ei_of_layer(network, top,
                    samles=50000,
                    batch_size=100, 
                    in_range=(0, 1),
                    in_bins=64,
                    out_range=(0, 1),
                    out_bins=64,
                    activation=nn.Sigmoid(), 
                    device=device)
```

You can also measure the sensitivity of a layer like so:
```python
network = nn.Linear(10, 10, bias=False).to(device)
top = topology_of(network, input=torch.zeros((1, 10)).to(device))

sensitivity = sensitivity_of_layer(network, top,
                            samples=1000,
                            batch_size=100, 
                            in_range=(0, 1),
                            in_bins=64,
                            out_range=(0, 1),
                            out_bins=64,
                            activation=nn.Sigmoid(), 
                            device=device)
```