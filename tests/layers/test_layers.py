import nfp
import numpy as np
import pytest
from nfp.frameworks import tf
from numpy.testing import assert_allclose

layers = tf.keras.layers

# def test_slice():
#     connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')
#
#     out0 = nfp.Slice(np.s_[:, :, 0])(connectivity)
#     out1 = nfp.Slice(np.s_[:, :, 1])(connectivity)
#
#     model = tf.keras.Model([connectivity], [out0, out1])
#     inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]).T
#     inputs = inputs[np.newaxis, :, :]
#
#     out = model(inputs)
#
#     assert_allclose(out[0], inputs[:, :, 0])
#     assert_allclose(out[1], inputs[:, :, 1])


# def test_gather():
#     in1 = layers.Input(shape=[None], dtype="float", name="data")
#     in2 = layers.Input(shape=[None], dtype=tf.int64, name="indices")

#     gather = nfp.Gather()([in1, in2])

#     model = tf.keras.Model([in1, in2], [gather])

#     data = np.random.rand(2, 10).astype(np.float32)
#     indices = np.array([[2, 6, 3], [5, 1, 0]])
#     out = model([data, indices])

#     assert_allclose(out, np.vstack([data[0, indices[0]], data[1, indices[1]]]))


@pytest.mark.parametrize("method", ["sum", "mean", "max", "min", "prod"])
def test_reduce(smiles_inputs, method):
    preprocessor, inputs = smiles_inputs
    func = getattr(np, method)

    atom_class = layers.Input(shape=[None], dtype=tf.int64, name="atom")
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name="bond")
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")

    atom_embed = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(
        atom_class
    )
    bond_embed = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(
        bond_class
    )

    reduced = nfp.Reduce(method)([bond_embed, connectivity[:, :, 0], atom_embed])

    model = tf.keras.Model(
        [atom_class, bond_class, connectivity], [atom_embed, bond_embed, reduced]
    )

    atom_state, bond_state, atom_reduced = model(
        [inputs["atom"], inputs["bond"], inputs["connectivity"]]
    )

    assert_allclose(atom_reduced[0, 0, :], func(bond_state[0, :4, :], 0))
    assert_allclose(atom_reduced[0, 1, :], func(bond_state[0, 4:8, :], 0))
    assert_allclose(atom_reduced[0, 2, :], bond_state[0, 9, :], 0)
    assert_allclose(atom_reduced[0, 3, :], bond_state[0, 10, :], 0)
    assert_allclose(atom_reduced[0, 4, :], bond_state[0, 11, :], 0)
    assert_allclose(atom_reduced[0, 5, :], bond_state[0, 12, :], 0)
    # assert_allclose(atom_reduced[0, 8:, :], 0.)


def test_tile():
    state = layers.Input(shape=[None], dtype="float", name="data")
    target = layers.Input(shape=[None, 3], dtype=tf.int64, name="indices")

    tile = nfp.Tile()([state, target])

    model = tf.keras.Model([state, target], [tile])

    state_input = np.random.rand(10, 16).astype(np.float32)
    target_input = np.random.rand(10, 24, 3).astype(np.float32)

    out = model([state_input, target_input])

    shape = list(state_input.shape)
    shape.insert(1, target_input.shape[1])

    assert list(out.shape) == shape
    assert np.all(out[:, 0, :] == out[:, 1, :])


def test_RBFExpansion(crystals_and_preprocessor):
    preprocessor, inputs = crystals_and_preprocessor
    input_distance = inputs["distance"].numpy()
    assert np.nanmin(input_distance) > 1  # we shouldn't be padding with zeros

    input_distance[0, 0] = 2
    input_distance[0, 1] = 3
    input_distance[0, 2] = 4

    distance = layers.Input(shape=[None], dtype=tf.float32, name="distance")
    rbf_distance = nfp.RBFExpansion(
        dimension=10,
        init_gap=10,
        init_max_distance=10,
    )(distance)

    model = tf.keras.Model([distance], [rbf_distance])
    embedded_distance = model(input_distance)

    # The first column of the valid distances should be essentially zero
    assert embedded_distance[embedded_distance._keras_mask][:, 0].numpy().max() < 1e-8

    assert embedded_distance[0, 0].numpy().argmax() == 2
    assert embedded_distance[0, 1].numpy().argmax() == 3
    assert embedded_distance[0, 2].numpy().argmax() == 4
