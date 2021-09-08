import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import layers

import nfp



def test_save_and_load_message(inputs_no_padding, inputs_with_padding, smiles_inputs, tmpdir: 'py.path.local'):
    """ mainly to do with https://github.com/tensorflow/tensorflow/issues/38620 """

    preprocessor, inputs = smiles_inputs

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
    outputs = model(inputs_no_padding)
    output_pad = model(inputs_with_padding)
    assert np.all(np.isclose(outputs, output_pad, atol=1E-4, rtol=1E-4))

    model.save(tmpdir, include_optimizer=False)
    loaded_model = tf.keras.models.load_model(tmpdir, compile=False)
    loutputs = model(inputs_no_padding)
    loutput_pad = model(inputs_with_padding)

    assert np.all(np.isclose(outputs, loutputs, atol=1E-4, rtol=1E-3))
    assert np.all(np.isclose(output_pad, loutput_pad, atol=1E-4, rtol=1E-3))
