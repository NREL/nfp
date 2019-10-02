import warnings
import tempfile
from numpy.testing import assert_allclose

import tensorflow as tf
from tensorflow.keras import Input, layers

from nfp import custom_layers
from nfp.layers import ReduceAtomToMol, ReduceBondToAtom, GatherAtomToBond


def test_save_and_load_model(get_2d_sequence, tmpdir):

    preprocessor, sequence = get_2d_sequence

    node_graph_indices = Input(shape=(1,), name='node_graph_indices', dtype='int32')
    atom_types = Input(shape=(1,), name='atom', dtype='int32')
    bond_types = Input(shape=(1,), name='bond', dtype='int32')
    connectivity = Input(shape=(2,), name='connectivity', dtype='int32')

    snode_graph_indices = tf.squeeze(node_graph_indices, 1)
    satom_types = tf.squeeze(atom_types, 1)
    sbond_types = tf.squeeze(bond_types, 1)

    atom_features = 5

    atom_state = layers.Embedding(
        preprocessor.atom_classes,
        atom_features, name='atom_embedding')(satom_types)

    bond_state = layers.Embedding(
        preprocessor.bond_classes,
        atom_features, name='bond_embedding')(sbond_types)

    def message_block(original_atom_state, original_bond_state, connectivity):

        atom_state = layers.BatchNormalization()(original_atom_state)
        bond_state = layers.BatchNormalization()(original_bond_state)

        source_atom_gather = GatherAtomToBond(1)
        target_atom_gather = GatherAtomToBond(0)

        source_atom = source_atom_gather([atom_state, connectivity])
        target_atom = target_atom_gather([atom_state, connectivity])
        new_bond_state = layers.Concatenate()([
            source_atom, target_atom, bond_state])
        new_bond_state = layers.Dense(
            atom_features, activation='relu')(new_bond_state)

        source_atom = layers.Dense(atom_features)(source_atom)    
        messages = layers.Multiply()([source_atom, new_bond_state])
        messages = ReduceBondToAtom(reducer='sum')([messages, connectivity])

        bond_state = layers.Add()([original_bond_state, new_bond_state])
        atom_state = layers.Add()([original_atom_state, messages])

        return atom_state, bond_state


    # Perform the message passing
    for _ in range(2):
        atom_state, bond_state = message_block(atom_state, bond_state,
                                               connectivity)

    mol_fingerprint = ReduceAtomToMol(reducer='sum')(
        [atom_state, snode_graph_indices])

    out = layers.Dense(1)(mol_fingerprint)
    model = tf.keras.Model(
        inputs=[node_graph_indices, atom_types, bond_types, connectivity],
        outputs=[out])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1E-4), loss='mse')
        model.fit_generator(sequence, epochs=1)
    
    loss = model.evaluate_generator(sequence)

    _, fname = tempfile.mkstemp('.h5')
    model.save(fname)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 

        model = tf.keras.load_model(fname, custom_objects=custom_layers)
        loss2 = model.evaluate_generator(sequence)

    assert_allclose(loss, loss2)
