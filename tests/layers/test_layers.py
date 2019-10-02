import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

from nfp.layers import (MessageLayer, GatherAtomToBond, Set2Set,
                        ReduceAtomToMol, ReduceBondToAtom, Embedding2D,
                        EdgeNetwork, GatherMolToAtomOrBond)

def test_message():
    atom = tf.keras.Input(name='atom', shape=(5,), dtype='float32')
    bond = tf.keras.Input(name='bond', shape=(5,5), dtype='float32')
    connectivity = tf.keras.Input(name='connectivity', shape=(2,), dtype='int32')

    message_layer = MessageLayer()
    o = message_layer([atom, bond, connectivity])
    assert o.shape.as_list() == [None, 5]

    model = tf.keras.Model(inputs=[atom, bond, connectivity], outputs=o)
    
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
    atom = tf.keras.Input(name='atom', shape=(5,), dtype='float32')
    connectivity = tf.keras.Input(name='connectivity', shape=(2,), dtype='int32')

    gather_layer = GatherAtomToBond(index=1)
    o = gather_layer([atom, connectivity])
    assert o.shape.as_list() == [None, 5]

    x1 = np.random.rand(2, 5)
    x3 = np.array([[0, 1], [1, 0]])

    model = tf.keras.Model([atom, connectivity], o)
    out = model.predict_on_batch({
        'atom': x1,
        'connectivity': x3})

    assert_allclose(out[0], x1[1])
    assert_allclose(out[1], x1[0])


def test_GatherMolToAtomOrBond():
    global_state = tf.keras.Input(name='global_state', shape=(5,), dtype='float32')
    node_graph_indices = tf.keras.Input(name='node_graph_indices', shape=(1,), dtype='int32')
    snode = tf.squeeze(node_graph_indices, 1)

    layer = GatherMolToAtomOrBond()
    o = layer([global_state, snode])
    assert o.shape.as_list() == [None, 5]

    model = tf.keras.Model(inputs=[global_state, node_graph_indices], outputs=o)

    x1 = np.random.rand(2, 5)
    x2 = np.array([0, 0, 0, 1, 1])

    out = model.predict_on_batch([x1, x2])
    assert_allclose(out, x1[x2])


def test_ReduceAtomToMol():
    atom = tf.keras.Input(name='atom', shape=(5,), dtype='float32')
    node_graph_indices = tf.keras.Input(name='node_graph_indices', shape=(1,), dtype='int32')

    snode = tf.squeeze(node_graph_indices, 1)
 
    reduce_layer = ReduceAtomToMol()
    o = reduce_layer([atom, snode])
    assert o.shape.as_list() == [None, 5]

    model = tf.keras.Model([atom, node_graph_indices], o)

    x1 = np.random.rand(5, 5)
    x2 = np.array([0, 0, 0, 1, 1])

    out = model.predict_on_batch([x1, x2])

    assert_allclose(x1[:3].sum(0), out[0])
    assert_allclose(x1[3:].sum(0), out[1])


def test_ReduceBondToAtom():
    bond = tf.keras.Input(name='bond', shape=(5,), dtype='float32')
    connectivity = tf.keras.Input(name='connectivity', shape=(2,), dtype='int32')

    reduce_layer = ReduceBondToAtom(reducer='max')
    o = reduce_layer([bond, connectivity])
    assert o.shape.as_list() == [None, 5]

    model = tf.keras.Model([bond, connectivity], o)

    x1 = np.random.rand(5, 5)
    x2 = np.array([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]]).T

    out = model.predict_on_batch([x1, x2])

    assert_allclose(x1[:3].max(0), out[0])
    assert_allclose(x1[3:].max(0), out[1])


def test_Embedding2D():

    bond = tf.keras.Input(name='bond', shape=(1,), dtype='int32')
    sbond = tf.squeeze(bond, 1)
    
    embedding = Embedding2D(3, 5)
    o = embedding(sbond)
    assert o.shape.as_list() == [None, 5, 5]

    model = tf.keras.Model([bond], o)

    x1 = np.array([1, 1, 2, 2, 0])
    out = model.predict_on_batch([x1])

    assert_allclose(out[0], out[1])
    assert_allclose(out[2], out[3])

    assert not (out[0] == out[-1]).all()


def test_EdgeNetwork():

    bond = tf.keras.Input(name='bond', shape=(1,), dtype='int32')
    distance = tf.keras.Input(name='distance', shape=(1,), dtype='float32')

    en = EdgeNetwork(5, 3)
    o = en([bond, distance])
    assert o.shape.as_list() == [None, 5, 5]

    model = tf.keras.Model([bond, distance], o)

    x1 = np.array([1, 1, 2, 2, 0])
    x2 = np.array([1., 1., 2., 3., .5])
    out = model.predict_on_batch([x1, x2])

    assert_allclose(out[0], out[1])
    assert (~np.isclose(out[2], out[3])).any()
    assert (~np.isclose(out[0], out[-1])).any()


def test_set2set():
    atom = tf.keras.Input(name='atom', shape=(5,), dtype='float32')
    node_graph_indices = tf.keras.Input(name='node_graph_indices', shape=(1,), dtype='int32')

    snode = tf.squeeze(node_graph_indices, 1)

    reduce_layer = Set2Set()
    o = reduce_layer([atom, snode])
    assert o.shape.as_list() == [None, 10]

    model = tf.keras.Model([atom, node_graph_indices], o)

    x1 = np.random.rand(5, 5)
    x2 = np.array([0, 0, 0, 1, 1])

    out = model.predict_on_batch([x1, x2])

    assert out.shape == (2, 10)
