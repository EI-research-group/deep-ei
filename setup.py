# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages

setup(
    name='deep-ei',
    version='0.9.0',
    py_modules=['deep_ei'],
    author='Simon Mattsson, Eric J. Michaud, Erik Hoel',
    author_email='eric.michaud99@gmail.com',
    license='MIT',
    # url='https://github.com/ejmichaud/torch-foresight',
    description='Tools for measuring the effective information in artificial neural networks built with PyTorch',
    install_requires=[
        "torch",
        "scikit-learn",
        "fast_histogram",
        "networkx"
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)