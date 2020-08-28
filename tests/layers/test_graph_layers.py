import pytest

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

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

    update_state, update_state_global = model(
        [inputs['atom'], inputs['bond'], inputs['connectivity']])

    assert update_state.shape == update_state_global.shape
    assert not np.all(update_state == update_state_global)
