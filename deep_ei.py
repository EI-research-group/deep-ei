# -*- coding: UTF-8 -*-

import warnings
from math import log, log2, ceil
from functools import reduce

import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.optimize import curve_fit, differential_evolution

from fast_histogram import histogram2d
import networkx as nx

import torch
import torch.nn as nn


def hack_range(range):
        """This version of fast_histogram handles edge cases differently
        than numpy, so we have to slightly adjust the bins."""
        d = 1e-6
        return ((range[0][0]-d, range[0][1]+d), (range[1][0]-d, range[1][1]+d))


def nats_to_bits(nats):
        r"""Convert information from nats to bits.

        Args:
            nats: float

        Returns:
            float: bits of information
        """
        return nats / log(2)


def MI(x, y, bins=32, range=((0, 1), (0, 1))):
    r"""Computes mutual information between time-series x and y.

    The mutual information between two distributions is a measure of
    correlation between them. If the distributions are independent, the
    mutual information will be 0. Mathematically, it is equivalent to the
    KL-divergence between the joint distribution and the product of the marginal
    distributions:
        
    .. math::
        I(x, y) = D_{KL}\( p(x, y) || p(x)p(y) \)

    Args:
        x (torch.tensor): a 1d tensor representing a time series of x values
        y (torch.tensor): a 1d tensor representing a time series of y values
        bins (int): the number of bins to discretize x and y values into
        range (array-like; 2x2): upper and lower values which bins can take for x and y

    Returns:
        float: the mutual information of the joint and marginal distributions
        inferred from the time series.

    TODO: implement custom version in pure pytorch without relying on sklearn
    """
    assert len(x) == len(y), "time series are of unequal length"
    x = x.detach().numpy()
    y = y.detach().numpy()
    cm = histogram2d(x, y, bins=bins, range=hack_range(range))
    # return H(np.sum(cm, axis=1)) + H(np.sum(cm, axis=0)) - H(cm)
    return nats_to_bits(mutual_info_score(None, None, contingency=cm))



r"""
    The modules which are currently supported. Note that skip connections
    are currently not supported. The network is assumed to be
    feedforward.
"""
VALID_MODULES = {
    nn.Linear,
    nn.Conv2d,
    nn.AvgPool2d,
    nn.MaxPool2d,
    nn.Flatten
}

r"""
    The activations which are currently supported and their output ranges.
"""
VALID_ACTIVATIONS = {
    nn.Sigmoid: (0, 1),
    nn.Tanh: (-1, 1),
    nn.ReLU: (0, 10),
    type(None): (-10, 10)
}

r"""
    Pooling Modules that are supported. Currently only 2d pooling is supported.
"""
POOLING_MODULES = {
    nn.AvgPool2d,
    nn.MaxPool2d
}

r"""
    Convolutional Modules that are supported. Currently only 2d convolutions are supported.
"""
CONVOLUTIONAL_MODULES = {
    nn.Conv2d
}


def topology_of(model, input):
    r"""Get a graph representing the connectivity of `model`.

    Each node of the returned graph contains a dictionary:

    {
        "input": {"activation": activation module, "shape": tuple},
        "output": {"activation": activation module, "shape": tuple}
    }

    Because PyTorch uses a dynamic computation graph, the number of activations
    that a given module will return is not intrinsic to the definition of the module,
    but can depend on the shape of its input. We therefore need to pass data through
    the network to determine its connectivity. 

    This function passes `input` into `model` and gets the shapes of the tensor 
    inputs and outputs of each child module in model, provided that they are
    instances of VALID_MODULES. It also finds the modules run before and after
    each child module, provided they are in VALID_ACTIVATIONS. 

    Args:
        model (nn.Module): feedforward neural network
        input (torch.tensor): a valid input to the network

    Returns:
        networkx.DiGraph representing `model` connectivity

    Examples:
        >>> network = nn.Sequential(nn.Linear(42, 20), nn.Sigmoid(), nn.Linear(20, 10))
        >>> top = topology_of(network, input=torch.zeros((1, 42)))
        >>> layer1, _, layer2 = network
        >>> top.nodes[layer1]['output']['activation']
        nn.Sigmoid instance
        >>> top.nodes[layer1]['input']['shape']
        (1, 42)
    """
    topology_G = nx.DiGraph()
    topology = {}
    hooks = []
    
    prv_module = None
    prv = None
    def register_hook(module):
        def hook(module, input, output):
            nonlocal prv, prv_module
            if type(module) in VALID_MODULES:
                structure = {
                    "input": dict(),
                    "output": dict()
                }
                structure["input"]["activation"] = prv if type(prv) in VALID_ACTIVATIONS else None
                structure["input"]["shape"] = tuple(input[0].shape)
                structure["output"]["activation"] = None
                structure["output"]["shape"] = tuple(output.shape)
                
                '''
                To deal with convolutions, track input shape from weight vectors, not from inputs per se!
                We do not need to create a larger image, because the statistics will be identical.
                # TODO: this works for convolutions and linear layers, but conv->pooling layers require additional
                work.
                '''
                if type(module) in CONVOLUTIONAL_MODULES:
                    structure["input"]["shape"] = (1,) + module._parameters["weight"].shape[1:]
                    structure["output"]["shape"] = (1,) + module._parameters["weight"].shape[0:0]

                topology[module] = structure
                topology_G.add_node(module)
                topology_G.add_edge(prv_module, module)
                prv = module
                prv_module = module
            if type(module) in VALID_ACTIVATIONS:
                if prv in topology:
                    topology[prv]["output"]["activation"] = module
                prv = module
        if type(module) in VALID_MODULES or type(module) in VALID_ACTIVATIONS:
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    model(input)
    for hook in hooks:
        hook.remove()
    nx.set_node_attributes(topology_G, topology)
    return topology_G


MEMORY_LIMIT = 100000000 # (GPU memory limit) 100 million floats ~ 0.4 GiB

def _chunk_sizes(samples, num_inputs, num_outputs, limit):
    """Generator for noise tensor sizes. 

    Sometimes, the input and output matrices are too big to store
    on the GPU, so we have to divide up samples into smaller
    chunks and evaluate on them. If :
            samples * max(num_inputs, num_outputs) <= limit,
    then just yields samples. Otherwise breaks samples into
    chunks of size limit // max(num_inputs, num_outputs),
    and also yields the remainder.
    """
    width = max(num_inputs, num_outputs)
    size = limit // width
    for _ in range(size, samples+1, size):
        yield size
    if size > samples:
        yield samples
    remainder = samples % size
    if remainder and width * samples >= limit:
        yield remainder


def _indices_and_batch_sizes(samples, batch_size):
    """Generator for batch sizes and indices into noise input
    and output tensors.

    Divides `samples` into chunks of size batch_size. Yields a
    tuple of indices, and also a batch size. Includes the remainder.
    """
    if batch_size > samples:
        yield (0, samples), samples
    start, end = 0, batch_size
    for _ in range(batch_size, samples+1, batch_size):
        yield (start, end), batch_size
        start, end = end, end + batch_size
    last_batch = samples % batch_size
    if last_batch and batch_size <= samples:
        yield (samples-last_batch, samples), last_batch


def _eval_model(x, in_layer, layer, topology, activation):
    """Passes input x through the network starting with `in_layer`
    and ending with `layer`. `layer` is forced to use `activation`
    as its activation function, overriding whatever is in `topology`."""
    if in_layer == layer:
        with torch.no_grad():
            if activation is None:
                activation = lambda x: x
            return activation(layer(x))
    assert layer in nx.descendants(topology, in_layer), "layer does not come after in_layer in network"
    current_layer = in_layer
    with torch.no_grad():
        while current_layer != layer:
            act = topology.nodes[current_layer]['output']['activation']
            if act is None:
                act = lambda x: x
            x = act(current_layer(x))
            next_layers = list(topology.neighbors(current_layer))
            assert len(next_layers) == 1, "Module cannot output to multiple other modules"
            current_layer, = next_layers
        if activation is None:
            activation = lambda x: x
        x = activation(current_layer(x))
    return x


def _EI_of_layer_manual_samples(layer, topology, samples, batch_size, \
    in_layer, in_shape, in_range, in_bins, \
    out_shape, out_range, out_bins, activation, device):
    """Helper function for ei_of_layer that computes the EI of layer `layer`
    with a set number of samples."""
    in_l, in_u = in_range
    num_inputs = reduce(lambda x, y: x * y, in_shape)
    num_outputs = reduce(lambda x, y: x * y, out_shape)

    #################################################
    #    Create histograms for each A -> B pair     #
    #################################################
    in_bin_width = (in_u - in_l) / in_bins
    if out_bins != 'dynamic':
        CMs = np.zeros((num_inputs, num_outputs, in_bins, out_bins)) # histograms for each input/output pair
    else:
        CMs = [[None for B in range(num_outputs)] for A in range(num_inputs)]
        if out_range == 'dynamic':
            dyn_out_bins = [None for B in range(num_outputs)]
        dyn_out_bins_set = False

    if out_range == 'dynamic':
        dyn_out_ranges = np.zeros((num_outputs, 2))
        dyn_ranges_set = False

    for chunk_size in _chunk_sizes(samples, num_inputs, num_outputs, MEMORY_LIMIT):
        #################################################
        #   Create buffers for layer input and output   #
        #################################################
        inputs = torch.zeros((chunk_size, *in_shape), device=device)
        outputs = torch.zeros((chunk_size, *out_shape), device=device)
        #################################################
        #           Evaluate module on noise            #
        #################################################
        for (i0, i1), bsize in _indices_and_batch_sizes(chunk_size, batch_size):
            sample = (in_u - in_l) * torch.rand((bsize, *in_shape), device=device) + in_l
            inputs[i0:i1] = sample
            try:
                result = _eval_model(sample, in_layer, layer, topology, activation)
            except:
                print(i0, i1, bsize, in_layer, layer, in_shape, out_shape)
                raise
            outputs[i0:i1] = result
        inputs = torch.flatten(inputs, start_dim=1)
        outputs = torch.flatten(outputs, start_dim=1)
        #################################################
        #    If specified to be dynamic,                #
        #    and first time in the loop,                #
        #    determine out_range for output neurons     #
        #################################################
        if out_range == 'dynamic' and not dyn_ranges_set:
            for B in range(num_outputs):
                out_l = torch.min(outputs[:, B]).item()
                out_u = torch.max(outputs[:, B]).item()
                dyn_out_ranges[B][0] = out_l
                dyn_out_ranges[B][1] = out_u
            dyn_ranges_set = True
        #################################################
        #    If specified to be dynamic,                #
        #    and first time in the loop,                #
        #    determine out_bins for output neurons      #
        #################################################        
        if out_bins == 'dynamic' and not dyn_out_bins_set:
            if out_range == 'dynamic':
                for B in range(num_outputs):
                    out_l, out_u = dyn_out_ranges[B]
                    bins = int((out_u - out_l) / in_bin_width) + 1
                    out_u = out_l + (bins * in_bin_width)
                    dyn_out_bins[B] = bins
                    dyn_out_ranges[B][1] = out_u
            else:
                out_l, out_u = out_range
                bins = int((out_u - out_l) / in_bin_width) + 1
                out_u = out_l + (bins * in_bin_width)
                dyn_out_bins = bins
                out_range = (out_l, out_u)
            for A in range(num_inputs):
                for B in range(num_outputs):
                    if out_range == 'dynamic':
                        out_b = dyn_out_bins[B]
                    else:
                        out_b = dyn_out_bins
                    CMs[A][B] = np.zeros((in_bins, out_b))
            dyn_out_bins_set = True
        #################################################
        #     Update Histograms for each A -> B pair    #
        #################################################
        for A in range(num_inputs):
            for B in range(num_outputs):
                if out_range == 'dynamic':
                    out_r = tuple(dyn_out_ranges[B])
                else:
                    out_r = out_range
                if out_bins == 'dynamic':
                    if out_range == 'dynamic':
                        out_b = dyn_out_bins[B]
                    else:
                        out_b = dyn_out_bins
                else:
                    out_b = out_bins
                # print("in_range: {}".format(in_range))
                # print("in_bins: {}".format(in_bins))
                # print("out_range: {}".format(out_r))
                # print("out_bins: {}".format(out_b))
                CMs[A][B] += histogram2d(inputs[:, A].to('cpu').detach().numpy(),
                                            outputs[:, B].to('cpu').detach().numpy(),
                                            bins=(in_bins, out_b),
                                            range=hack_range((in_range, out_r)))
    #################################################
    #           Compute mutual information          #
    #################################################
    EI = 0.0
    for A in range(num_inputs):
        for B in range(num_outputs):
            A_B_EI = nats_to_bits(mutual_info_score(None, None, contingency=CMs[A][B]))
            EI += A_B_EI
    if EI < 0.01:
        return 0.0
    else:
        return EI


def _EI_of_layer_extrapolate(layer, topology, batch_size, in_layer, in_shape, in_range, in_bins,\
    out_shape, out_range, out_bins, activation, device):
    """Helper function of ei_of_layer that computes the EI of layer `layer` by computing EI
    with several different sample sizes and fitting a curve."""
    INTERVAL = 100000
    POINTS = 20
    sample_sizes = []
    EIs = []

    in_l, in_u = in_range
    num_inputs = reduce(lambda x, y: x * y, in_shape)
    num_outputs = reduce(lambda x, y: x * y, out_shape)

    #################################################
    #    Create histograms for each A -> B pair     #
    #################################################
    in_bin_width = (in_u - in_l) / in_bins
    if out_bins != 'dynamic':
        CMs = np.zeros((num_inputs, num_outputs, in_bins, out_bins)) # histograms for each input/output pair
    else:
        CMs = [[None for B in range(num_outputs)] for A in range(num_inputs)]
        if out_range == 'dynamic':
            dyn_out_bins = [None for B in range(num_outputs)]
        dyn_out_bins_set = False

    if out_range == 'dynamic':
        dyn_out_ranges = np.zeros((num_outputs, 2))
        dyn_ranges_set = False

    for n in range(POINTS):
        for chunk_size in _chunk_sizes(INTERVAL, num_inputs, num_outputs, MEMORY_LIMIT):
            #################################################
            #   Create buffers for layer input and output   #
            #################################################
            inputs = torch.zeros((chunk_size, *in_shape), device=device)
            outputs = torch.zeros((chunk_size, *out_shape), device=device)
            #################################################
            #           Evaluate module on noise            #
            #################################################
            for (i0, i1), bsize in _indices_and_batch_sizes(chunk_size, batch_size):
                sample = (in_u - in_l) * torch.rand((bsize, *in_shape), device=device) + in_l
                inputs[i0:i1] = sample
                try:
                    result = _eval_model(sample, in_layer, layer, topology, activation)
                except:
                    print(i0, i1, bsize, in_layer, layer, in_shape, out_shape)
                    raise
                outputs[i0:i1] = result
            inputs = torch.flatten(inputs, start_dim=1)
            outputs = torch.flatten(outputs, start_dim=1)
            #################################################
            #    If specified to be dynamic,                #
            #    and first time in the loop,                #
            #    determine out_range for output neurons     #
            #################################################
            if out_range == 'dynamic' and not dyn_ranges_set:
                for B in range(num_outputs):
                    out_l = torch.min(outputs[:, B]).item()
                    out_u = torch.max(outputs[:, B]).item()
                    dyn_out_ranges[B][0] = out_l
                    dyn_out_ranges[B][1] = out_u
                dyn_ranges_set = True
            #################################################
            #    If specified to be dynamic,                #
            #    and first time in the loop,                #
            #    determine out_bins for output neurons      #
            #################################################
            if out_bins == 'dynamic' and not dyn_out_bins_set:
                if out_range == 'dynamic':
                    for B in range(num_outputs):
                        out_l, out_u = dyn_out_ranges[B]
                        bins = int((out_u - out_l) / in_bin_width) + 1
                        out_u = out_l + (bins * in_bin_width)
                        dyn_out_bins[B] = bins
                        dyn_out_ranges[B][1] = out_u
                else:
                    out_l, out_u = out_range
                    bins = int((out_u - out_l) / in_bin_width) + 1
                    out_u = out_l + (bins * in_bin_width)
                    dyn_out_bins = bins
                    out_range = (out_l, out_u)
                for A in range(num_inputs):
                    for B in range(num_outputs):
                        if out_range == 'dynamic':
                            out_b = dyn_out_bins[B]
                        else:
                            out_b = dyn_out_bins
                        CMs[A][B] = np.zeros((in_bins, out_b))
                dyn_out_bins_set = True
            #################################################
            #     Update Histograms for each A -> B pair    #
            #################################################
            for A in range(num_inputs):
                for B in range(num_outputs):
                    if out_range == 'dynamic':
                        out_r = tuple(dyn_out_ranges[B])
                    else:
                        out_r = out_range
                    if out_bins == 'dynamic':
                        if out_range == 'dynamic':
                            out_b = dyn_out_bins[B]
                        else:
                            out_b = dyn_out_bins
                    else:
                        out_b = out_bins
                    # print("in_range: {}".format(in_range))
                    # print("in_bins: {}".format(in_bins))
                    # print("out_range: {}".format(out_r))
                    # print("out_bins: {}".format(out_b))
                    CMs[A][B] += histogram2d(inputs[:, A].to('cpu').detach().numpy(),
                                                outputs[:, B].to('cpu').detach().numpy(),
                                                bins=(in_bins, out_b),
                                                range=hack_range((in_range, out_r)))
        #################################################
        #           Compute mutual information          #
        #################################################
        EI = 0.0
        for A in range(num_inputs):
            for B in range(num_outputs):
                A_B_EI = nats_to_bits(mutual_info_score(None, None, contingency=CMs[A][B]))
                EI += A_B_EI
        EIs.append(EI)
        sample_sizes.append((n + 1) * INTERVAL)
    #################################################
    #       Fit curve and determine asymptote       #
    #################################################
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EIs = np.array(EIs[4:])
        sample_sizes = np.array(sample_sizes[4:])
        def curve(x, a, e, C):
            return a / (x**e) + C
        def loss(func, params):
            return np.sum((EIs - func(sample_sizes, *params))**2)
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        a_inits = [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
        e_inits = [0, 1]
        params = []
        for a in a_inits:
            for e in e_inits:
                try:
                    ps, _ = curve_fit(curve, sample_sizes, EIs, p0=[a, e, 0], bounds=bounds, maxfev=10000)
                    params.append(ps)
                except RuntimeError:
                    pass
        best_params = min(params, key=lambda ps: loss(curve, ps))
    EI = float(curve(1e15, *best_params))
    if EI < 0.01:
        return 0.0
    else:
        return EI


def _EI_of_layer_auto_samples(layer, topology, batch_size, in_layer, in_shape, in_range, in_bins, \
    out_shape, out_range, out_bins, activation, device, threshold):
    """Helper function of ei_of_layer that computes the EI of layer `layer`
    using enough samples to be within `threshold`% of the true value. 
    """
    MULTIPLIER = 2
    INTERVAL = 10000
    SAMPLES_SO_FAR = INTERVAL
    EIs = []

    def has_converged(EIs):
        if len(EIs) < 2:
            return False
        slope = (EIs[-2] - EIs[-1]) / INTERVAL
        error = slope * SAMPLES_SO_FAR * (MULTIPLIER - 1)
        if error / EIs[-1] > threshold:
            return False
        return True
    
    in_l, in_u = in_range
    num_inputs = reduce(lambda x, y: x * y, in_shape)
    num_outputs = reduce(lambda x, y: x * y, out_shape)

    #################################################
    #    Create histograms for each A -> B pair     #
    #################################################
    in_bin_width = (in_u - in_l) / in_bins
    if out_bins != 'dynamic':
        CMs = np.zeros((num_inputs, num_outputs, in_bins, out_bins)) # histograms for each input/output pair
    else:
        CMs = [[None for B in range(num_outputs)] for A in range(num_inputs)]
        if out_range == 'dynamic':
            dyn_out_bins = [None for B in range(num_outputs)]
        dyn_out_bins_set = False

    if out_range == 'dynamic':
        dyn_out_ranges = np.zeros((num_outputs, 2))
        dyn_ranges_set = False

    while True:
        for chunk_size in _chunk_sizes(INTERVAL, num_inputs, num_outputs, MEMORY_LIMIT):
            #################################################
            #   Create buffers for layer input and output   #
            #################################################
            inputs = torch.zeros((chunk_size, *in_shape), device=device)
            outputs = torch.zeros((chunk_size, *out_shape), device=device)
            #################################################
            #           Evaluate module on noise            #
            #################################################
            for (i0, i1), bsize in _indices_and_batch_sizes(chunk_size, batch_size):
                sample = (in_u - in_l) * torch.rand((bsize, *in_shape), device=device) + in_l
                inputs[i0:i1] = sample
                try:
                    result = _eval_model(sample, in_layer, layer, topology, activation)
                except:
                    print(i0, i1, bsize, in_layer, layer, in_shape, out_shape)
                    raise
                outputs[i0:i1] = result
            inputs = torch.flatten(inputs, start_dim=1)
            outputs = torch.flatten(outputs, start_dim=1)
            #################################################
            #    If specified to be dynamic,                #
            #    and first time in the loop,                #
            #    determine out_range for output neurons     #
            #################################################
            if out_range == 'dynamic' and not dyn_ranges_set:
                for B in range(num_outputs):
                    out_l = torch.min(outputs[:, B]).item()
                    out_u = torch.max(outputs[:, B]).item()
                    dyn_out_ranges[B][0] = out_l
                    dyn_out_ranges[B][1] = out_u
                dyn_ranges_set = True
            #################################################
            #    If specified to be dynamic,                #
            #    and first time in the loop,                #
            #    determine out_bins for output neurons      #
            #################################################
            if out_bins == 'dynamic' and not dyn_out_bins_set:
                if out_range == 'dynamic':
                    for B in range(num_outputs):
                        out_l, out_u = dyn_out_ranges[B]
                        bins = int((out_u - out_l) / in_bin_width) + 1
                        out_u = out_l + (bins * in_bin_width)
                        dyn_out_bins[B] = bins
                        dyn_out_ranges[B][1] = out_u
                else:
                    out_l, out_u = out_range
                    bins = int((out_u - out_l) / in_bin_width) + 1
                    out_u = out_l + (bins * in_bin_width)
                    dyn_out_bins = bins
                    out_range = (out_l, out_u)
                for A in range(num_inputs):
                    for B in range(num_outputs):
                        if out_range == 'dynamic':
                            out_b = dyn_out_bins[B]
                        else:
                            out_b = dyn_out_bins
                        CMs[A][B] = np.zeros((in_bins, out_b))
                dyn_out_bins_set = True
            #################################################
            #     Update Histograms for each A -> B pair    #
            #################################################
            for A in range(num_inputs):
                for B in range(num_outputs):
                    if out_range == 'dynamic':
                        out_r = tuple(dyn_out_ranges[B])
                    else:
                        out_r = out_range
                    if out_bins == 'dynamic':
                        if out_range == 'dynamic':
                            out_b = dyn_out_bins[B]
                        else:
                            out_b = dyn_out_bins
                    else:
                        out_b = out_bins
                    CMs[A][B] += histogram2d(inputs[:, A].to('cpu').detach().numpy(),
                                                outputs[:, B].to('cpu').detach().numpy(),
                                                bins=(in_bins, out_b),
                                                range=hack_range((in_range, out_r)))
        #################################################
        #           Compute mutual information          #
        #################################################
        EI = 0.0
        for A in range(num_inputs):
            for B in range(num_outputs):
                A_B_EI = nats_to_bits(mutual_info_score(None, None, contingency=CMs[A][B]))
                EI += A_B_EI        
        EIs.append(EI)
        #################################################
        #        Determine whether more samples         #
        #        are needed and update how many         #
        #################################################
        if has_converged(EIs):
            EI = EIs[-1]
            if EI < 0.01:
                return 0.0
            else:
                return EI
        INTERVAL = int(SAMPLES_SO_FAR * (MULTIPLIER - 1))
        SAMPLES_SO_FAR += INTERVAL
    
    
def ei_of_layer(layer, topology, threshold=0.05, samples=None, extrapolate=False, batch_size=20, 
    in_layer=None, in_range=None, in_bins=64, \
    out_range=None, out_bins=64, 
    activation=None, device='cpu'):
    """Computes the effective information of neural network layer `layer`.

    Args:
        layer (nn.Module): a module in `topology`
        topology (dict): topology object (nested dictionary) returned from topology_of function
        threshold (float): used to dynamically determine how many samples to use.
        samples (int): if specified (defaults to None), function will manually use this many samples, which may or may not give good convergence.
        extrapolate (bool): if True, then evaluate EI at several points and then fit a curve to determine asymptotic value.
        batch_size (int): the number of samples to run `layer` on simultaneously
        in_layer (nn.Module): the module in `topology` which begins our 'layer'. By default is the same as `layer`.
        in_range (tuple): (lower_bound, upper_bound), inclusive, by default determined from `topology`
        in_bins (int): the number of bins to discretize in_range into for MI calculation
        out_range (tuple): (lower_bound, upper_bound), inclusive, by default determined from `topology`
        out_bins (int): the number of bins to discretize out_range into for MI calculation
        activation (function): the output activation of `layer`, by defualt determined from `topology`
        device: 'cpu' or 'cuda' or `torch.device` instance

    Returns:
        float: an estimate of the EI of layer `layer`
    """
    
    #################################################
    #   Determine shapes, ranges, and activations   #
    #################################################
    if in_layer is None:
        in_layer = layer
    in_shape = topology.nodes[in_layer]["input"]["shape"][1:]
    out_shape = topology.nodes[layer]["output"]["shape"][1:]
    ##############################################
    #   Conv -> Pooling layer is a special case  #
    #   TODO: this is a hack that needs work.    #
    ##############################################
    if type(layer) in POOLING_MODULES and type(in_layer) in CONVOLUTIONAL_MODULES:
        # print(layer, in_layer)
        out_shape = (in_layer.out_channels,1,1)
        in_shape = in_shape[:-2] + tuple([x + layer.stride * y for x,y in zip(in_shape[-2:], in_layer.stride)])

    # print(type(in_layer), type(layer), in_shape, out_shape)
        
    if in_range == 'dynamic':
        raise ValueError("Input range cannot be dynamic, only output range can be.")
    if in_range is None:
        activation_type = type(topology.nodes[in_layer]["input"]["activation"])
        in_range = VALID_ACTIVATIONS[activation_type]
    if out_range is None:
        activation_type = type(topology.nodes[layer]["output"]["activation"])
        out_range = VALID_ACTIVATIONS[activation_type]

    if activation is None:
        activation = topology.nodes[layer]["output"]["activation"]
        if activation is None:
            activation = lambda x: x

    #################################################
    #             Call helper functions             #
    #################################################
    if extrapolate:
        return _EI_of_layer_extrapolate(layer=layer, topology=topology,
            batch_size=batch_size,
            in_layer=in_layer,
            in_shape=in_shape,
            in_range=in_range,
            in_bins=in_bins,
            out_shape=out_shape,
            out_range=out_range,
            out_bins=out_bins,
            activation=activation,
            device=device)
    if samples is not None:
        return _EI_of_layer_manual_samples(layer=layer, topology=topology,
            samples=samples, 
            batch_size=batch_size,
            in_layer=in_layer,
            in_shape=in_shape,
            in_range=in_range,
            in_bins=in_bins,
            out_shape=out_shape,
            out_range=out_range,
            out_bins=out_bins,
            activation=activation,
            device=device)
    return _EI_of_layer_auto_samples(layer=layer, topology=topology,
                batch_size=batch_size,
                in_shape=in_shape,
                in_layer=in_layer,
                in_range=in_range,
                in_bins=in_bins,
                out_shape=out_shape,
                out_range=out_range,
                out_bins=out_bins,
                activation=activation,
                device=device,
                threshold=threshold)


def ei_of_layer_matrix(layer, topology, samples=None, batch_size=20, 
    in_layer=None, in_range=None, in_bins=64, \
    out_range=None, out_bins=64, 
    activation=None, device='cpu'):
    """Computes the effective information of all A -> B connections of 
    neural network layer `layer`.

    Args:
        layer (nn.Module): a module in `topology`
        topology (dict): topology object (nested dictionary) returned from topology_of function
        threshold (float): used to dynamically determine how many samples to use.
        samples (int): if specified (defaults to None), function will manually use this many samples, which may or may not give good convergence.
        extrapolate (bool): if True, then evaluate EI at several points and then fit a curve to determine asymptotic value.
        batch_size (int): the number of samples to run `layer` on simultaneously
        in_layer (nn.Module): the module in `topology` which begins our 'layer'. By default is the same as `layer`.
        in_range (tuple): (lower_bound, upper_bound), inclusive, by default determined from `topology`
        in_bins (int): the number of bins to discretize in_range into for MI calculation
        out_range (tuple): (lower_bound, upper_bound), inclusive, by default determined from `topology`
        out_bins (int): the number of bins to discretize out_range into for MI calculation
        activation (function): the output activation of `layer`, by defualt determined from `topology`
        device: 'cpu' or 'cuda' or `torch.device` instance

    Returns:
        np.array: The [A][B]th element is the EI from A -> B
    """
    
    #################################################
    #   Determine shapes, ranges, and activations   #
    #################################################
    if in_layer is None:
        in_layer = layer
    in_shape = topology.nodes[in_layer]["input"]["shape"][1:]
    out_shape = topology.nodes[layer]["output"]["shape"][1:]
    ##############################################
    #   Conv -> Pooling layer is a special case  #
    #   TODO: this is a hack that needs work.    #
    ##############################################
    if type(layer) in POOLING_MODULES and type(in_layer) in CONVOLUTIONAL_MODULES:
        # print(layer, in_layer)
        out_shape = (in_layer.out_channels,1,1)
        in_shape = in_shape[:-2] + tuple([x + layer.stride * y for x,y in zip(in_shape[-2:], in_layer.stride)])
        
    if in_range == 'dynamic':
        raise ValueError("Input range cannot be dynamic, only output range can be.")
    if in_range is None:
        activation_type = type(topology.nodes[in_layer]["input"]["activation"])
        in_range = VALID_ACTIVATIONS[activation_type]
    if out_range is None:
        activation_type = type(topology.nodes[layer]["output"]["activation"])
        out_range = VALID_ACTIVATIONS[activation_type]

    if activation is None:
        activation = topology.nodes[layer]["output"]["activation"]
        if activation is None:
            activation = lambda x: x

    in_l, in_u = in_range
    num_inputs = reduce(lambda x, y: x * y, in_shape)
    num_outputs = reduce(lambda x, y: x * y, out_shape)

    #################################################
    #    Create histograms for each A -> B pair     #
    #################################################
    in_bin_width = (in_u - in_l) / in_bins
    if out_bins != 'dynamic':
        CMs = np.zeros((num_inputs, num_outputs, in_bins, out_bins)) # histograms for each input/output pair
    else:
        CMs = [[None for B in range(num_outputs)] for A in range(num_inputs)]
        if out_range == 'dynamic':
            dyn_out_bins = [None for B in range(num_outputs)]
        dyn_out_bins_set = False

    if out_range == 'dynamic':
        dyn_out_ranges = np.zeros((num_outputs, 2))
        dyn_ranges_set = False

    for chunk_size in _chunk_sizes(samples, num_inputs, num_outputs, MEMORY_LIMIT):
        #################################################
        #   Create buffers for layer input and output   #
        #################################################
        inputs = torch.zeros((chunk_size, *in_shape), device=device)
        outputs = torch.zeros((chunk_size, *out_shape), device=device)
        #################################################
        #           Evaluate module on noise            #
        #################################################
        for (i0, i1), bsize in _indices_and_batch_sizes(chunk_size, batch_size):
            sample = (in_u - in_l) * torch.rand((bsize, *in_shape), device=device) + in_l
            inputs[i0:i1] = sample
            try:
                result = _eval_model(sample, in_layer, layer, topology, activation)
            except:
                print(i0, i1, bsize, in_layer, layer, in_shape, out_shape)
                raise
            outputs[i0:i1] = result
        inputs = torch.flatten(inputs, start_dim=1)
        outputs = torch.flatten(outputs, start_dim=1)
        #################################################
        #    If specified to be dynamic,                #
        #    and first time in the loop,                #
        #    determine out_range for output neurons     #
        #################################################
        if out_range == 'dynamic' and not dyn_ranges_set:
            for B in range(num_outputs):
                out_l = torch.min(outputs[:, B]).item()
                out_u = torch.max(outputs[:, B]).item()
                dyn_out_ranges[B][0] = out_l
                dyn_out_ranges[B][1] = out_u
            dyn_ranges_set = True
        #################################################
        #    If specified to be dynamic,                #
        #    and first time in the loop,                #
        #    determine out_bins for output neurons      #
        #################################################        
        if out_bins == 'dynamic' and not dyn_out_bins_set:
            if out_range == 'dynamic':
                for B in range(num_outputs):
                    out_l, out_u = dyn_out_ranges[B]
                    bins = int((out_u - out_l) / in_bin_width) + 1
                    out_u = out_l + (bins * in_bin_width)
                    dyn_out_bins[B] = bins
                    dyn_out_ranges[B][1] = out_u
            else:
                out_l, out_u = out_range
                bins = int((out_u - out_l) / in_bin_width) + 1
                out_u = out_l + (bins * in_bin_width)
                dyn_out_bins = bins
                out_range = (out_l, out_u)
            for A in range(num_inputs):
                for B in range(num_outputs):
                    if out_range == 'dynamic':
                        out_b = dyn_out_bins[B]
                    else:
                        out_b = dyn_out_bins
                    CMs[A][B] = np.zeros((in_bins, out_b))
            dyn_out_bins_set = True
        #################################################
        #     Update Histograms for each A -> B pair    #
        #################################################
        for A in range(num_inputs):
            for B in range(num_outputs):
                if out_range == 'dynamic':
                    out_r = tuple(dyn_out_ranges[B])
                else:
                    out_r = out_range
                if out_bins == 'dynamic':
                    if out_range == 'dynamic':
                        out_b = dyn_out_bins[B]
                    else:
                        out_b = dyn_out_bins
                else:
                    out_b = out_bins
                CMs[A][B] += histogram2d(inputs[:, A].to('cpu').detach().numpy(),
                                            outputs[:, B].to('cpu').detach().numpy(),
                                            bins=(in_bins, out_b),
                                            range=hack_range((in_range, out_r)))
    eis = np.zeros((num_inputs, num_outputs))
    for A in range(num_inputs):
        for B in range(num_outputs):
            A_B_EI = nats_to_bits(mutual_info_score(None, None, contingency=CMs[A][B]))
            eis[A][B] = A_B_EI
    return eis


def sensitivity_of_layer(layer, topology, samples=500, batch_size=20,
        in_layer=None, in_range=None, in_bins=64, out_range=None, out_bins=64, activation=None, device='cpu'):
    """Computes the sensitivity of neural network layer `layer`.

    Note that this does not currently support dynamic ranging or binning. There is a
    good reason for this: because the inputs we run through the network in the
    sensitivity calculation are very different from the noise run though in the EI
    calculation, each output neuron's range may be different, and we would be
    evaluating the sensitivity an EI using a different binning. The dynamic
    ranging and binning supported by the EI function should be used with
    great caution.

    Args:
        layer (nn.Module): a module in `topology`
        topology (dict): topology object (nested dictionary) returned from topology_of function
        samples (int): the number of noise samples run through `layer`
        batch_size (int): the number of samples to run `layer` on simultaneously
        in_layer (nn.Module): the module in `topology` which begins our 'layer'. By default is the same as `layer`.
        in_range (tuple): (lower_bound, upper_bound), inclusive, by default determined from `topology`
        in_bins (int): the number of bins to discretize in_range into for MI calculation
        out_range (tuple): (lower_bound, upper_bound), inclusive, by default determined from `topology`
        out_bins (int): the number of bins to discretize out_range into for MI calculation
        activation (function): the output activation of `layer`, by defualt determined from `topology`
        device: 'cpu' or 'cuda' or `torch.device` instance

    Returns:
        float: an estimate of the sensitivity of layer `layer`
    """
    
    #################################################
    #   Determine shapes, ranges, and activations   #
    #################################################
    if in_layer is None:
        in_layer = layer
    in_shape = topology.nodes[in_layer]["input"]["shape"][1:]
    out_shape = topology.nodes[layer]["output"]["shape"][1:]

    ##############################################
    #   Conv -> Pooling layer is a special case  #
    #   TODO: this is a hack that needs work.    #
    ##############################################
    if type(layer) in POOLING_MODULES and type(in_layer) in CONVOLUTIONAL_MODULES:
        #print(layer, in_layer)
        out_shape = (in_layer.out_channels,1,1)
        in_shape = in_shape[:-2] + tuple([x + layer.stride * y for x,y in zip(in_shape[-2:], in_layer.stride)])

    #print(type(in_layer), type(layer), in_shape, out_shape)
    if in_range is None:
        activation_type = type(topology.nodes[in_layer]["input"]["activation"])
        in_range = VALID_ACTIVATIONS[activation_type]
    if out_range is None:
        activation_type = type(topology.nodes[layer]["output"]["activation"])
        out_range = VALID_ACTIVATIONS[activation_type]
    in_l, in_u = in_range
    if activation is None:
        activation = topology.nodes[layer]["output"]["activation"]
        if activation is None:
            activation = lambda x: x
    #################################################
    #   Create buffers for layer input and output   #
    #################################################
    num_inputs = reduce(lambda x, y: x * y, in_shape)
    num_outputs = reduce(lambda x, y: x * y, out_shape)
    inputs = torch.zeros((samples, num_inputs), device=device)
    outputs = torch.zeros((samples, num_outputs), device=device)

    ###########################################
    #   Determine out bin ranges if dynamic   #
    ###########################################
    if out_range == 'dynamic':
        # Make two tensors, one with high and one with low ranges.
        # This makes the assumption that the activation function is convex.
        # All of the current ones are, but I wouldn't put it past someone to change that in the future.
        # Yes I am talking to you, distant future maintainer.
        mins = []
        maxs = []
        for A in range(num_inputs):
            low_tensor = torch.zeros((1,num_inputs), device = device)
            low_tensor[:,A] = torch.full((1,), in_range[0], device = device)
            high_tensor = torch.zeros((1,num_inputs), device = device)
            high_tensor[:,A] = torch.full((1,), in_range[1], device = device)
            low_result = _eval_model(low_tensor.reshape((1,*in_shape)), in_layer, layer, topology, activation)
            high_result = _eval_model(high_tensor.reshape((1,*in_shape)), in_layer, layer, topology, activation)
            min_low = torch.min(low_result).item()
            max_low = torch.max(low_result).item()
            min_high = torch.min(high_result).item()
            max_high = torch.max(high_result).item()
            mins.append(min([min_low,min_high]))
            maxs.append(max([max_low,max_high]))
        out_range = (min(mins), max(maxs))
    if out_bins == 'dynamic':
        bin_size = (in_range[1] - in_range[0]) / in_bins
        out_bins = ceil((out_range[1] - out_range[0]) / bin_size)

    sensitivity = 0.0
    for A in range(num_inputs):
        #################################################
        #           Evaluate module on noise            #
        #################################################
        for (i0, i1), size in _indices_and_batch_sizes(samples, batch_size):
            sample = torch.zeros((size, num_inputs)).to(device)
            sample[:, A] = (in_u - in_l) * torch.rand((size,), device=device) + in_l
            inputs[i0:i1] = sample
            try:
                result = _eval_model(sample.reshape((size, *in_shape)), in_layer, layer, topology, activation)
            except:
                print(i0, i1, size, in_layer, layer, in_shape, out_shape)
                continue
            outputs[i0:i1] = result.flatten(start_dim=1)
        for B in range(num_outputs):
            #################################################
            #           Compute mutual information          #
            #################################################
            sensitivity += MI(inputs[:, A].to('cpu'), 
                outputs[:, B].to('cpu'),
                bins=(in_bins, out_bins), 
                range=(in_range, out_range))
        inputs.fill_(0)
        outputs.fill_(0)
    return sensitivity




def sensitivity_of_layer_matrix(layer, topology, samples=500, batch_size=20,
        in_layer=None, in_range=None, in_bins=64, out_range=None, out_bins=64, activation=None, device='cpu'):
    """Computes the sensitivitites of each A -> B connection 
    of neural network layer `layer`.

    Note that this does not currently support dynamic ranging or binning. There is a
    good reason for this: because the inputs we run through the network in the
    sensitivity calculation are very different from the noise run though in the EI
    calculation, each output neuron's range may be different, and we would be
    evaluating the sensitivity and EI using a different binning. The dynamic
    ranging and binning supported by the EI function should be used with
    great caution.

    Args:
        layer (nn.Module): a module in `topology`
        topology (dict): topology object (nested dictionary) returned from topology_of function
        samples (int): the number of noise samples run through `layer`
        batch_size (int): the number of samples to run `layer` on simultaneously
        in_layer (nn.Module): the module in `topology` which begins our 'layer'. By default is the same as `layer`.
        in_range (tuple): (lower_bound, upper_bound), inclusive, by default determined from `topology`
        in_bins (int): the number of bins to discretize in_range into for MI calculation
        out_range (tuple): (lower_bound, upper_bound), inclusive, by default determined from `topology`
        out_bins (int): the number of bins to discretize out_range into for MI calculation
        activation (function): the output activation of `layer`, by defualt determined from `topology`
        device: 'cpu' or 'cuda' or `torch.device` instance

    Returns:
        float: an estimate of the sensitivity of layer `layer`
    """
    
    #################################################
    #   Determine shapes, ranges, and activations   #
    #################################################
    if in_layer is None:
        in_layer = layer
    in_shape = topology.nodes[in_layer]["input"]["shape"][1:]
    out_shape = topology.nodes[layer]["output"]["shape"][1:]

    ##############################################
    #   Conv -> Pooling layer is a special case  #
    #   TODO: this is a hack that needs work.    #
    ##############################################
    if type(layer) in POOLING_MODULES and type(in_layer) in CONVOLUTIONAL_MODULES:
        #print(layer, in_layer)
        out_shape = (in_layer.out_channels,1,1)
        in_shape = in_shape[:-2] + tuple([x + layer.stride * y for x,y in zip(in_shape[-2:], in_layer.stride)])

    #print(type(in_layer), type(layer), in_shape, out_shape)
    if in_range is None:
        activation_type = type(topology.nodes[in_layer]["input"]["activation"])
        in_range = VALID_ACTIVATIONS[activation_type]
    if out_range is None:
        activation_type = type(topology.nodes[layer]["output"]["activation"])
        out_range = VALID_ACTIVATIONS[activation_type]
    in_l, in_u = in_range
    if activation is None:
        activation = topology.nodes[layer]["output"]["activation"]
        if activation is None:
            activation = lambda x: x
    #################################################
    #   Create buffers for layer input and output   #
    #################################################
    num_inputs = reduce(lambda x, y: x * y, in_shape)
    num_outputs = reduce(lambda x, y: x * y, out_shape)
    inputs = torch.zeros((samples, num_inputs), device=device)
    outputs = torch.zeros((samples, num_outputs), device=device)

    ###########################################
    #   Determine out bin ranges if dynamic   #
    ###########################################
    if out_range == 'dynamic':
        # Make two tensors, one with high and one with low ranges.
        # This makes the assumption that the activation function is convex.
        # All of the current ones are, but I wouldn't put it past someone to change that in the future.
        # Yes I am talking to you, distant future maintainer.
        mins = []
        maxs = []
        for A in range(num_inputs):
            low_tensor = torch.zeros((1,num_inputs), device = device)
            low_tensor[:,A] = torch.full((1,), in_range[0], device = device)
            high_tensor = torch.zeros((1,num_inputs), device = device)
            high_tensor[:,A] = torch.full((1,), in_range[1], device = device)
            low_result = _eval_model(low_tensor.reshape((1,*in_shape)), in_layer, layer, topology, activation)
            high_result = _eval_model(high_tensor.reshape((1,*in_shape)), in_layer, layer, topology, activation)
            min_A = torch.min(low_result).item()
            max_A = torch.max(high_result).item()
            mins.append(min_A)
            maxs.append(max_A)
        out_range = (min(mins), max(maxs))
    if out_bins == 'dynamic':
        bin_size = (in_range[1] - in_range[0]) / in_bins
        out_bins = ceil((out_range[1] - out_range[0]) / bin_size)

    sensitivities = np.zeros((num_inputs, num_outputs))
    for A in range(num_inputs):
        #################################################
        #           Evaluate module on noise            #
        #################################################
        for (i0, i1), size in _indices_and_batch_sizes(samples, batch_size):
            sample = torch.zeros((size, num_inputs)).to(device)
            sample[:, A] = (in_u - in_l) * torch.rand((size,), device=device) + in_l
            inputs[i0:i1] = sample
            try:
                result = _eval_model(sample.reshape((size, *in_shape)), in_layer, layer, topology, activation)
            except:
                print(i0, i1, size, in_layer, layer, in_shape, out_shape)
                continue
            outputs[i0:i1] = result.flatten(start_dim=1)
        for B in range(num_outputs):
            #################################################
            #           Compute mutual information          #
            #################################################
            sensitivities[A][B] = MI(inputs[:, A].to('cpu'), 
                outputs[:, B].to('cpu'),
                bins=(in_bins, out_bins), 
                range=(in_range, out_range))
        inputs.fill_(0)
        outputs.fill_(0)
    return sensitivities

