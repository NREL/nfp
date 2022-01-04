import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from nfp.preprocessing.xtb_preprocessor import xTBSmilesPreprocessor


@pytest.fixture()
def get_2d_smiles_json():
    train = ['C','CO','CC']
    train_json = ['train_1.json','train_2.json','train_3.json']
    test = ['C=O', 'C=CC']
    test_json = ['test_1.json','test_2.json']

    return train, train_json, test, test_json

@pytest.mark.parametrize('explicit_hs', [True, False])
def test_smiles_preprocessor(explicit_hs, get_2d_smiles_json):
    train, train_json, test, test_json = get_2d_smiles_json

    preprocessor = xTBSmilesPreprocessor(explicit_hs=explicit_hs)
    inputs = [preprocessor(smiles, json=jsonfile, train=True) for smiles, jsonfile in zip(train,train_json)]

    # Make sure all bonds and atoms get a valid class
    for input_ in inputs:
        assert (input_['bond'] != 0).all()
        assert (input_['atom'] != 0).all()
        assert (input_['bond'] != 1).all()
        assert (input_['atom'] != 1).all()

        assert (len(input_['atomxtbfeatures']) != 0)
        assert (len(input_['bondxtbfeatures']) != 0)

    test_inputs = [preprocessor(smiles, json=jsonfile, train=False) for smiles, jsonfile in zip(test,test_json)]

    for input_ in test_inputs:
        assert (input_['bond'] == 1).any()
        assert (input_['atom'] == 1).any()

        assert (len(input_['atomxtbfeatures']) != 0)
        assert (len(input_['bondxtbfeatures']) != 0)
