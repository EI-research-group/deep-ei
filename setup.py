# -*- coding: UTF-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='deep-ei',
    version='0.5.0',
    py_modules=['deep_ei'],
    author='Simon Mattsson, Eric J. Michaud, Erik Hoel',
    author_email='eric.michaud99@gmail.com',
    license='MIT',
    url='https://github.com/EI-research-group/deep-ei',
    description='Tools for examining the causal structure of artificial neural networks built with PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch",
        "scikit-learn",
        "fast_histogram",
        "networkx"
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)