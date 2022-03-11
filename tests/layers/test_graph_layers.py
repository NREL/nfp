import nfp
import numpy as np
import pytest
from nfp.frameworks import tf

layers = tf.keras.layers


@pytest.mark.parametrize("layer", [nfp.EdgeUpdate, nfp.NodeUpdate])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
@pytest.mark.parametrize("output_dtype", ["int32", "int64"])
def test_layer(smiles_inputs, layer, dropout, output_dtype):
    preprocessor, inputs = smiles_inputs
    preprocessor.output_dtype = output_dtype

    atom_class = layers.Input(shape=[None], dtype=output_dtype, name="atom")
    bond_class = layers.Input(shape=[None], dtype=output_dtype, name="bond")
    connectivity = layers.Input(
        shape=[None, 2], dtype=output_dtype, name="connectivity"
    )

    atom_state = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(
        atom_class
    )
    bond_state = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(
        bond_class
    )
    global_state = layers.GlobalAveragePooling1D()(atom_state)

    update = layer(dropout=dropout)([atom_state, bond_state, connectivity])
    update_global = layer(dropout=dropout)(
        [atom_state, bond_state, connectivity, global_state]
    )

    model = tf.keras.Model(
        [atom_class, bond_class, connectivity], [update, update_global]
    )

    update_state, update_state_global = model(
        [inputs["atom"], inputs["bond"], inputs["connectivity"]]
    )

    assert update_state.shape == update_state_global.shape
    assert not np.all(update_state == update_state_global)


@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_global(smiles_inputs, dropout):
    preprocessor, inputs = smiles_inputs

    atom_class = layers.Input(shape=[None], dtype=tf.int64, name="atom")
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name="bond")
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")

    atom_state = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(
        atom_class
    )
    bond_state = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(
        bond_class
    )
    global_state = layers.GlobalAveragePooling1D()(atom_state)

    update = nfp.GlobalUpdate(8, 2, dropout=dropout)(
        [atom_state, bond_state, connectivity]
    )
    update_global = nfp.GlobalUpdate(8, 2, dropout=dropout)(
        [atom_state, bond_state, connectivity, global_state]
    )

    model = tf.keras.Model(
        [atom_class, bond_class, connectivity], [update, update_global]
    )

    update_state, update_state_global = model(inputs)

    assert not hasattr(update, "_keras_mask")
    assert not hasattr(update_global, "_keras_mask")
    assert update_state.shape == update_state_global.shape
    assert not np.all(update_state == update_state_global)


@pytest.mark.parametrize("layer", [nfp.EdgeUpdate, nfp.NodeUpdate, nfp.GlobalUpdate])
def test_masking(inputs_no_padding, inputs_with_padding, smiles_inputs, layer):
    preprocessor, inputs = smiles_inputs

    atom_class = layers.Input(shape=[None], dtype=tf.int64, name="atom")
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name="bond")
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")

    atom_state = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(
        atom_class
    )
    bond_state = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(
        bond_class
    )
    global_state = layers.GlobalAveragePooling1D()(atom_state)

    if layer == nfp.GlobalUpdate:
        get_layer = lambda: layer(8, 2)
    else:
        get_layer = layer

    update = get_layer()([atom_state, bond_state, connectivity])
    update_global = get_layer()([atom_state, bond_state, connectivity, global_state])
    model = tf.keras.Model(
        [atom_class, bond_class, connectivity], [update, update_global]
    )

    update_state, update_state_global = model(inputs_no_padding)
    update_state_pad, update_state_global_pad = model(inputs_with_padding)

    if update_state.ndim > 2:
        # bond or atom, need to remove the padding to compare
        update_state_pad = update_state_pad[:, : update_state.shape[1], :]

    if update_state_global_pad.ndim > 2:
        update_state_global_pad = update_state_global_pad[
            :, : update_state_global.shape[1], :
        ]

    assert np.all(np.isclose(update_state, update_state_pad, atol=1e-4))
    assert np.all(np.isclose(update_state_global, update_state_global_pad, atol=1e-4))


def test_masking_message(inputs_no_padding, inputs_with_padding, smiles_inputs):
    preprocessor, inputs = smiles_inputs

    atom_class = layers.Input(shape=[None], dtype=tf.int64, name="atom")
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name="bond")
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")

    atom_state = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(
        atom_class
    )
    bond_state = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(
        bond_class
    )
    global_state = nfp.GlobalUpdate(8, 2)([atom_state, bond_state, connectivity])

    for _ in range(3):
        new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
        bond_state = layers.Add()([bond_state, new_bond_state])

        new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])
        atom_state = layers.Add()([atom_state, new_atom_state])

        new_global_state = nfp.GlobalUpdate(8, 2)(
            [atom_state, bond_state, connectivity]
        )
        global_state = layers.Add()([new_global_state, global_state])

    model = tf.keras.Model([atom_class, bond_class, connectivity], [global_state])

    output = model(inputs_no_padding)
    output_pad = model(inputs_with_padding)

    assert np.all(np.isclose(output, output_pad, atol=1e-4))


def test_no_residual(inputs_no_padding, inputs_with_padding, smiles_inputs):
    """This model might not work when saved and loaded, see
    https://github.com/tensorflow/tensorflow/issues/38620"""
    preprocessor, inputs = smiles_inputs

    atom_class = layers.Input(shape=[None], dtype=tf.int64, name="atom")
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name="bond")
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")

    atom_state = layers.Embedding(preprocessor.atom_classes, 16, mask_zero=True)(
        atom_class
    )
    bond_state = layers.Embedding(preprocessor.bond_classes, 16, mask_zero=True)(
        bond_class
    )
    global_state = nfp.GlobalUpdate(8, 2)([atom_state, bond_state, connectivity])

    for _ in range(3):
        bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
        atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])
        global_state = nfp.GlobalUpdate(8, 2)([atom_state, bond_state, connectivity])

    model = tf.keras.Model([atom_class, bond_class, connectivity], [global_state])

    output = model(inputs_no_padding)
    output_pad = model(inputs_with_padding)

    assert np.all(np.isclose(output, output_pad, atol=1e-4))
