import pytest
import numpy as np
from numpy.testing import assert_allclose
from keras import layers
from keras import models

from nfp.layers import (MessageLayer, Squeeze, GatherAtomToBond,
                        ReduceAtomToMol, ReduceBondToAtom, Embedding2D,
                        EdgeNetwork, GatherMolToAtomOrBond)
from nfp.models import GraphModel

def test_message():
    atom = layers.Input(name='atom', shape=(5,), dtype='float32')
    bond = layers.Input(name='bond', shape=(5,5), dtype='float32')
    connectivity = layers.Input(name='connectivity', shape=(2,), dtype='int32')

    message_layer = MessageLayer()
    o = message_layer([atom, bond, connectivity])
    assert o._keras_shape == (None, 5)

    model = GraphModel([atom, bond, connectivity], o)
    
    x1 = np.random.rand(2, 5)
    x2 = np.random.rand(2, 5, 5)
    x3 = np.array([[0, 1], [1, 0]])

    out = model.predict_on_batch({
        'atom': x1,
        'bond': x2,
        'connectivity': x3})

    assert_allclose(np.vstack([x2[0].dot(x1[1]), x2[1].dot(x1[0])]),
                    out, rtol=1E-5, atol=1E-5)

def test_GatherAtomToBond():
    atom = layers.Input(name='atom', shape=(5,), dtype='float32')
    connectivity = layers.Input(name='connectivity', shape=(2,), dtype='int32')

    gather_layer = GatherAtomToBond(index=1)
    o = gather_layer([atom, connectivity])
    assert o._keras_shape == (None, 5)

    x1 = np.random.rand(2, 5)
    x3 = np.array([[0, 1], [1, 0]])

    model = GraphModel([atom, connectivity], o)
    out = model.predict_on_batch({
        'atom': x1,
        'connectivity': x3})

    assert_allclose(out[0], x1[1])
    assert_allclose(out[1], x1[0])


def test_GatherMolToAtomOrBond():
    global_state = layers.Input(name='global_state', shape=(5,), dtype='float32')
    node_graph_indices = layers.Input(name='node_graph_indices', shape=(1,), dtype='int32')

    snode = Squeeze()(node_graph_indices)
 
    layer = GatherMolToAtomOrBond()
    o = layer([global_state, snode])
    assert o._keras_shape == (None, 5)

    model = GraphModel([global_state, node_graph_indices], o)

    x1 = np.random.rand(2, 5)
    x2 = np.array([0, 0, 0, 1, 1])

    out = model.predict_on_batch([x1, x2])
    assert_allclose(out, x1[x2])


def test_ReduceAtomToMol():
    atom = layers.Input(name='atom', shape=(5,), dtype='float32')
    node_graph_indices = layers.Input(name='node_graph_indices', shape=(1,), dtype='int32')

    snode = Squeeze()(node_graph_indices)
 
    reduce_layer = ReduceAtomToMol()
    o = reduce_layer([atom, snode])
    assert o._keras_shape == (None, 5)

    model = GraphModel([atom, node_graph_indices], o)

    x1 = np.random.rand(5, 5)
    x2 = np.array([0, 0, 0, 1, 1])

    out = model.predict_on_batch([x1, x2])

    assert_allclose(x1[:3].sum(0), out[0])
    assert_allclose(x1[3:].sum(0), out[1])


def test_ReduceBondToAtom():
    bond = layers.Input(name='bond', shape=(5,), dtype='float32')
    connectivity = layers.Input(name='connectivity', shape=(2,), dtype='int32')

    reduce_layer = ReduceBondToAtom(reducer='max')
    o = reduce_layer([bond, connectivity])
    assert o._keras_shape == (None, 5)

    model = GraphModel([bond, connectivity], o)

    x1 = np.random.rand(5, 5)
    x2 = np.array([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]]).T

    out = model.predict_on_batch([x1, x2])

    assert_allclose(x1[:3].max(0), out[0])
    assert_allclose(x1[3:].max(0), out[1])


def test_Embedding2D():

    bond = layers.Input(name='bond', shape=(1,), dtype='int32')
    sbond = Squeeze()(bond)

    embedding = Embedding2D(3, 5)
    o = embedding(sbond)
    assert o._keras_shape == (None, 5, 5)

    model = GraphModel([bond], o)

    x1 = np.array([1, 1, 2, 2, 0])
    out = model.predict_on_batch([x1])

    assert_allclose(out[0], out[1])
    assert_allclose(out[2], out[3])

    assert not (out[0] == out[-1]).all()


def test_EdgeNetwork():

    bond = layers.Input(name='bond', shape=(1,), dtype='int32')
    distance = layers.Input(name='distance', shape=(1,), dtype='float32')

    en = EdgeNetwork(5, 3)
    o = en([bond, distance])
    assert o._keras_shape == (None, 5, 5)

    model = GraphModel([bond, distance], o)

    x1 = np.array([1, 1, 2, 2, 0])
    x2 = np.array([1., 1., 2., 3., .5])
    out = model.predict_on_batch([x1, x2])

    assert_allclose(out[0], out[1])
    assert (~np.isclose(out[2], out[3])).any()
    assert (~np.isclose(out[0], out[-1])).any()

