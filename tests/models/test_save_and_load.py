import pytest
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

import nfp

# @pytest.mark.skip(reason="slow, test whole message block instead")
# @pytest.mark.parametrize('layer', [nfp.EdgeUpdate, nfp.NodeUpdate, nfp.GlobalUpdate])
# @pytest.mark.parametrize('use_global', [True, False])
# def test_save_and_load(smiles_inputs, layer, use_global, tmpdir: 'py.path.local'):
#     preprocessor, inputs = smiles_inputs
#
#     def get_inputs(max_atoms=-1, max_bonds=-1):
#         dataset = tf.data.Dataset.from_generator(
#             lambda: (preprocessor.construct_feature_matrices(smiles, train=True)
#                      for smiles in ['CC', 'CCC', 'C(C)C', 'C']),
#             output_types=preprocessor.output_types,
#             output_shapes=preprocessor.output_shapes) \
#             .padded_batch(batch_size=4,
#                           padded_shapes=preprocessor.padded_shapes(max_atoms, max_bonds),
#                           padding_values=preprocessor.padding_values)
#
#         return list(dataset.take(1))[0]
#
#     if layer == nfp.GlobalUpdate:
#         get_layer = lambda: layer(8, 2)
#     else:
#         get_layer = layer
#
#     atom_class = layers.Input(shape=[11], dtype=tf.int64, name='atom')
#     bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')
#     connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')
#
#     atom_state = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(atom_class)
#     bond_state = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(bond_class)
#     global_state = layers.GlobalAveragePooling1D()(atom_state)
#
#     if use_global:
#         output_layer = get_layer()([atom_state, bond_state, connectivity, global_state])
#     else:
#         output_layer = get_layer()([atom_state, bond_state, connectivity])
#
#     model = tf.keras.Model([atom_class, bond_class, connectivity], [output_layer])
#     outputs = model(inputs)
#     output_pad = model(get_inputs(max_atoms=20, max_bonds=40))
#
#     model.save(tmpdir, include_optimizer=False)
#     loaded_model = tf.keras.models.load_model(tmpdir, compile=False)
#     loutputs = loaded_model(inputs)
#     loutputs_pad = model(get_inputs(max_atoms=20, max_bonds=40))
#
#     assert np.all(np.isclose(outputs, loutputs, atol=1E-4, rtol=1E-3))
#     assert np.all(np.isclose(output_pad, loutputs_pad, atol=1E-4, rtol=1E-3))


def test_save_and_load_message(smiles_inputs, tmpdir: 'py.path.local'):
    preprocessor, inputs = smiles_inputs

    def get_inputs(max_atoms=-1, max_bonds=-1):
        dataset = tf.data.Dataset.from_generator(
            lambda: (preprocessor.construct_feature_matrices(smiles, train=True)
                     for smiles in ['CC', 'CCC', 'C(C)C', 'C']),
            output_types=preprocessor.output_types,
            output_shapes=preprocessor.output_shapes) \
            .padded_batch(batch_size=4,
                          padded_shapes=preprocessor.padded_shapes(max_atoms, max_bonds),
                          padding_values=preprocessor.padding_values)

        return list(dataset.take(1))[0]

    atom_class = layers.Input(shape=[None], dtype=tf.int64, name='atom')
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

    atom_state = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(atom_class)
    bond_state = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(bond_class)
    global_state = nfp.GlobalUpdate(8, 2)([atom_state, bond_state, connectivity])

    for _ in range(3):
        new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
        bond_state = layers.Add()([bond_state, new_bond_state])

        new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])
        atom_state = layers.Add()([atom_state, new_atom_state])

        new_global_state = nfp.GlobalUpdate(8, 2)([atom_state, bond_state, connectivity])
        global_state = layers.Add()([new_global_state, global_state])

    model = tf.keras.Model([atom_class, bond_class, connectivity], [global_state])
    outputs = model(get_inputs())
    output_pad = model(get_inputs(max_atoms=20, max_bonds=40))
    assert np.all(np.isclose(outputs, output_pad, atol=1E-4, rtol=1E-4))

    model.save(tmpdir, include_optimizer=False)
    loaded_model = tf.keras.models.load_model(tmpdir, compile=False)
    loutputs = loaded_model(get_inputs())
    loutputs_pad = model(get_inputs(max_atoms=20, max_bonds=40))

    assert np.all(np.isclose(outputs, loutputs, atol=1E-4, rtol=1E-3))
    assert np.all(np.isclose(output_pad, loutputs_pad, atol=1E-4, rtol=1E-3))