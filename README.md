[![Build Status](https://travis-ci.org/NREL/nfp.svg?branch=master)](https://travis-ci.org/NREL/nfp)
[![PyPI version](https://badge.fury.io/py/nfp.svg)](https://badge.fury.io/py/nfp)

# Neural fingerprint (nfp)

Keras layers for end-to-end learning on molecular structure. Based on Keras, Tensorflow, and RDKit. Source code used in the study [Message-passing neural networks for high-throughput polymer screening](https://arxiv.org/abs/1807.10363)

## Related Work

1. [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)
2. [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf)
3. [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
4. [Neural Message Passing with Edge Updates for Predicting Properties of Molecules and Materials](https://arxiv.org/abs/1806.03146)

## (Main) Requirements

- [rdkit](http://www.rdkit.org/docs/Install.html)
- keras (github master, until [#11548](https://github.com/keras-team/keras/pull/11548) is included in a release)
- tensorflow

## Getting started

This library extends Keras with additional layers for handling molecular structures (i.e., graph-based inputs). There a strong familiarity with Keras is recommended.

An overview of how to build a model is shown in `examples/solubility_test_graph_output.ipynb`. Models can optionally include 3D molecular geometry; a simple example of a network using 3D geometry is found in `examples/model_3d_coordinates.ipynb`.

The current state-of-the-art architecture on QM9 (published in [4]) is included in `examples/schnet_edgeupdate.py`. This script requires qm9 preprocessing to be run before the model is evaluated with `examples/preprocess_qm9.py`.
