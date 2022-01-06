import os

import pytest
import tensorflow as tf
from nfp.preprocessing.xtb_preprocessor import xTBSmilesPreprocessor

dir_path = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture()
def get_2d_smiles_json():
    train = ["C", "CO", "CC"]
    train_json = [
        os.path.join(dir_path, "xtb_inputs", filename)
        for filename in ["train_1.json", "train_2.json", "train_3.json"]
    ]

    test = ["C=O", "C=CC"]

    test_json = [
        os.path.join(dir_path, "xtb_inputs", filename)
        for filename in ["test_1.json", "test_2.json"]
    ]

    return train, train_json, test, test_json


def test_xtb_smiles_preprocessor(get_2d_smiles_json):
    train, train_json, test, test_json = get_2d_smiles_json

    preprocessor = xTBSmilesPreprocessor(explicit_hs=True)
    inputs = [
        preprocessor(smiles, jsonfile, train=True)
        for smiles, jsonfile in zip(train, train_json)
    ]

    # Make sure all bonds and atoms get a valid class
    for input_ in inputs:
        assert input_["atom_xtb"].shape == (
            len(input_["atom"]),
            len(preprocessor.xtb_atom_features),
        )
        assert input_["bond_xtb"].shape == (
            len(input_["bond"]),
            len(preprocessor.xtb_bond_features),
        )
        assert input_["mol_xtb"].shape == (len(preprocessor.xtb_mol_features),)

    test_inputs = [
        preprocessor(smiles, jsonfile, train=False)
        for smiles, jsonfile in zip(test, test_json)
    ]

    for input_ in test_inputs:
        assert input_["atom_xtb"].shape == (
            len(input_["atom"]),
            len(preprocessor.xtb_atom_features),
        )
        assert input_["bond_xtb"].shape == (
            len(input_["bond"]),
            len(preprocessor.xtb_bond_features),
        )
        assert len(input_["mol_xtb"]) == len(preprocessor.xtb_mol_features)


def test_xtb_batching(get_2d_smiles_json):

    train, train_json, test, test_json = get_2d_smiles_json

    preprocessor = xTBSmilesPreprocessor(explicit_hs=True)

    inputs = (
        preprocessor(smiles, jsonfile, train=True)
        for smiles, jsonfile in zip(train, train_json)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: inputs,
        output_signature=preprocessor.output_signature,
    ).padded_batch(batch_size=3, padding_values=preprocessor.padding_values)

    batched_inputs = next(dataset.as_numpy_iterator())
    assert "atom_xtb" in batched_inputs
