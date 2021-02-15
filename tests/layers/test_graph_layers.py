import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import layers

import nfp


@pytest.mark.parametrize('layer', [nfp.EdgeUpdate, nfp.NodeUpdate])
@pytest.mark.parametrize("dropout", [0., 0.5])
def test_layer(smiles_inputs, layer, dropout):
    preprocessor, inputs = smiles_inputs

    atom_class = layers.Input(shape=[11], dtype=tf.int64, name='atom')
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

    atom_state = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(atom_class)
    bond_state = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(bond_class)
    global_state = layers.GlobalAveragePooling1D()(atom_state)

    update = layer(dropout=dropout)([atom_state, bond_state, connectivity])
    update_global = layer(dropout=dropout)([atom_state, bond_state,
                                            connectivity, global_state])

    model = tf.keras.Model([atom_class, bond_class, connectivity],
                           [update, update_global])

    update_state, update_state_global = model(
        [inputs['atom'], inputs['bond'], inputs['connectivity']])

    assert update_state.shape == update_state_global.shape
    assert not np.all(update_state == update_state_global)


@pytest.mark.parametrize("dropout", [0., 0.5])
def test_global(smiles_inputs, dropout):
    preprocessor, inputs = smiles_inputs

    atom_class = layers.Input(shape=[11], dtype=tf.int64, name='atom')
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

    atom_state = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(atom_class)
    bond_state = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(bond_class)
    global_state = layers.GlobalAveragePooling1D()(atom_state)

    update = nfp.GlobalUpdate(8, 2, dropout=dropout)(
        [atom_state, bond_state, connectivity])
    update_global = nfp.GlobalUpdate(8, 2, dropout=dropout)(
        [atom_state, bond_state, connectivity, global_state])

    model = tf.keras.Model([atom_class, bond_class, connectivity],
                           [update, update_global])

    update_state, update_state_global = model(inputs)

    assert not hasattr(update, '_keras_mask')
    assert not hasattr(update_global, '_keras_mask')
    assert update_state.shape == update_state_global.shape
    assert not np.all(update_state == update_state_global)

@pytest.mark.parametrize('layer', [nfp.EdgeUpdate, nfp.NodeUpdate, nfp.GlobalUpdate])
def test_masking(smiles_inputs, layer):
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

    if layer == nfp.GlobalUpdate:
        get_layer = lambda: layer(8, 2)
    else:
        get_layer = layer

    atom_class = layers.Input(shape=[11], dtype=tf.int64, name='atom')
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

    atom_state = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(atom_class)
    bond_state = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(bond_class)
    global_state = layers.GlobalAveragePooling1D()(atom_state)

    update = get_layer()([atom_state, bond_state, connectivity])
    update_global = get_layer()([atom_state, bond_state, connectivity, global_state])
    model = tf.keras.Model([atom_class, bond_class, connectivity], [update, update_global])

    update_state, update_state_global = model(get_inputs())
    update_state_pad, update_state_global_pad = model(get_inputs(max_atoms=20, max_bonds=40))

    if update_state.ndim > 2:  # bond or atom, need to remove the padding to compare
        update_state_pad = update_state_pad[:, :update_state.shape[1], :]

    if update_state_global_pad.ndim > 2:
        update_state_global_pad = update_state_global_pad[:, :update_state_global.shape[1], :]

    assert np.all(np.isclose(update_state, update_state_pad, atol=1E-4))
    assert np.all(np.isclose(update_state_global, update_state_global_pad, atol=1E-4))