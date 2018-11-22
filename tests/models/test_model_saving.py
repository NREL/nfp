import warnings
import pytest
import tempfile
from numpy.testing import assert_allclose

import keras
from keras.layers import Input, Embedding, Dense, BatchNormalization
from keras.models import load_model, save_model

from nfp import custom_layers
from nfp.layers import (GRUStep, ReduceAtomToMol, Embedding2D, Squeeze,
                        MessageLayer)
from nfp.models import GraphModel


def test_save_and_load_model(get_2d_sequence, tmpdir):

    preprocessor, sequence = get_2d_sequence

    node_graph_indices = Input(shape=(1,), name='node_graph_indices', dtype='int32')
    atom_types = Input(shape=(1,), name='atom', dtype='int32')
    bond_types = Input(shape=(1,), name='bond', dtype='int32')
    connectivity = Input(shape=(2,), name='connectivity', dtype='int32')

    squeeze = Squeeze()

    snode_graph_indices = squeeze(node_graph_indices)
    satom_types = squeeze(atom_types)
    sbond_types = squeeze(bond_types)

    atom_features = 5

    atom_state = Embedding(
    preprocessor.atom_classes,
    atom_features, name='atom_embedding')(satom_types)

    bond_matrix = Embedding2D(
        preprocessor.bond_classes,
        atom_features, name='bond_embedding')(sbond_types)

    atom_rnn_layer = GRUStep(atom_features)
    message_layer = MessageLayer(reducer='sum', dropout=0.1)
    
    # Perform the message passing
    for _ in range(2):

        # Get the message updates to each atom
        message = message_layer([atom_state, bond_matrix, connectivity])

        # Update memory and atom states
        atom_state = atom_rnn_layer([message, atom_state])
    
    atom_fingerprint = Dense(64, activation='sigmoid')(atom_state)
    mol_fingerprint = ReduceAtomToMol(reducer='sum')([atom_fingerprint,
                                                      snode_graph_indices])

    out = Dense(1)(mol_fingerprint)
    model = GraphModel([node_graph_indices, atom_types, bond_types,
                        connectivity], [out])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 

        model.compile(optimizer=keras.optimizers.Adam(lr=1E-4), loss='mse')
        hist = model.fit_generator(sequence, epochs=1)
    
    loss = model.evaluate_generator(sequence)

    _, fname = tempfile.mkstemp('.h5')
    model.save(fname)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 

        model = load_model(fname, custom_objects=custom_layers)
        loss2 = model.evaluate_generator(sequence)

    assert_allclose(loss, loss2)
