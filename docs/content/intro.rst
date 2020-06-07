
Introduction
============

Installation
^^^^^^^^^^^^

The simplest way to install the ``deep_ei`` module is with:
    
    .. code-block:: bash

        pip install deep-ei

Becaues ``pytorch`` can be fragile, it is recommended that you install and test ``pytorch`` before installing ``deep-ei`` (such as with ``conda install pytorch -c pytorch``). To install ``deep-ei`` directly from the GitHub repository:

    .. code-block:: bash

        git clone https://github.com/EI-research-group/deep-ei.git
        cd deep-ei
        pip install .

Basic tests can be executed with:

    .. code-block:: bash

        python setup.py test

Note that we have also provided an anaconda environment file. You can use it to create a new environment with ``deep-ei`` and all its dependencies:

    .. code-block:: bash

        conda env create --file environment.yml


Examples
^^^^^^^^
Here are some basic examles:

    .. code-block:: python

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


This will compute the EI of a ``10 -> 10`` dense layer using a sigmoid activation. By default, this function will increase the number of samples it uses until the EI levels off (characterized by whether it will change by more than threshold of its current value even if we doubled the number of samples).

The function ``topology_of`` creates a ``networkx`` graph representing the connectivity of the network. ``ei_of_layer`` can infer argument values using this graph, such as the ranges of the inputs and outputs of the layer and its activation function:

    .. code-block:: python

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



Which will use an activation of ``nn.Sigmoid`` and an ``out_range`` of ``(0, 1)`` for the first layer and an activation of ``nn.Tanh`` and an ``out_range`` of ``(-1, 1)`` for the second layer. Note that we have to specify an ``in_range`` for the first layer.

Instead of specifying a ``threshold``, you can instead manually specify a number of samples to use when computing effective information:

    .. code-block:: python

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

Sometimes, EI takes millions of samples to converge, so be careful of specifying a value too low. 

You can also measure the sensitivity of a layer like so:

    .. code-block:: python

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

If you want to compute the EI of each edge in a layer, use the ``ei_of_layer_matrix`` function:
    
    .. code-block:: python

        network = nn.Linear(20, 10, bias=False).to(device)
        top = topology_of(network, input=torch.zeros((1, 20)).to(device))

        EI = ei_of_layer_matrix(network, top,
                            samles=50000,
                            batch_size=100, 
                            in_range=(0, 1),
                            in_bins=64,
                            out_range=(0, 1),
                            out_bins=64,
                            activation=nn.Sigmoid(), 
                            device=device)

Which will return a ``20 x 10`` matrix where the rows correspond with in-neurons and the columns correspond with out-neurons. 


Ideas for future experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We'd love for people to use and expand on this code to make new discoveries. Here are some questions we haven't looked into yet:

- How does dropout effect the EI of a layer? In otherwise identical networks, does dropout increase or decrease the EI of the network layers?
- What can EI tell us about generalization? Does EI evolve in the causal plane in different ways when a network is memorizing a dataset vs generalizing? To test this, train networks on some dataset as you would normally, but then randomize the labels in the training dataset and train new networks. This label randomization will force the network to memorize the dataset.
- On harder tasks, where deep networks are required (in MNIST and Iris, which we studied, it is unnecessary that networks be deep for them to achieve good acuracy), do the hidden layers differentiate in the causal plane?
- Can EI be measured in recurrent networks? How would this work?

Contributing & Questions
^^^^^^^^^^^^^^^^^^^^^^^^
We'd welcome feedback and contributions! Feel free to email me at ``eric.michaud99@gmail.com`` if you have questions about the code. 