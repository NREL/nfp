import tempfile

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from nfp.preprocessing import SmilesPreprocessor


@pytest.fixture()
def get_2d_smiles():
    train = ['CC', 'CCC', 'C(C)C', 'C']
    test = ['CO', 'CCO']

    return train, test

    # data = pd.read_csv('../data/delaney.csv')
    # train = data.sample(frac=0.2, random_state=0)
    # test = data[~data.index.isin(test.index)]
    #
    # return list(train.head(100).smiles), list(test.head(25).smiles)


@pytest.fixture()
def get_3d_smiles(get_2d_smiles):
    def embed_3d(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        return mol

    train, test = get_2d_smiles

    return ([embed_3d(smile) for smile in train],
            [embed_3d(smile) for smile in test])


@pytest.mark.parametrize('explicit_hs', [True, False])
def test_smiles_preprocessor(explicit_hs, get_2d_smiles):
    train, test = get_2d_smiles

    preprocessor = SmilesPreprocessor(explicit_hs=explicit_hs)
    inputs = [preprocessor.construct_feature_matrices(smiles, train=True) for
              smiles in train]

    # Make sure all bonds and atoms get a valid class
    for input_ in inputs:
        if input_['n_atom'] > 1:
            assert (input_['bond'] != 0).all()
        assert (input_['atom'] != 0).all()
        assert (input_['bond'] != 1).all()
        assert (input_['atom'] != 1).all()

    if not explicit_hs:
        assert inputs[0]['n_atom'] == 2
        assert inputs[0]['n_bond'] == 1

    else:
        assert inputs[0]['n_atom'] == 8
        assert inputs[0]['n_bond'] == 7

    test_inputs = [preprocessor.construct_feature_matrices(smiles, train=False)
                   for smiles in test]

    for input_ in test_inputs:
        assert (input_['bond'] == 1).any()
        assert (input_['atom'] == 1).any()


@pytest.mark.parametrize('explicit_hs', [True, False])
def test_smiles_preprocessor_serialization(explicit_hs, get_2d_smiles):
    train, test = get_2d_smiles

    preprocessor = SmilesPreprocessor(explicit_hs=explicit_hs)
    input_train = [preprocessor.construct_feature_matrices(smiles, train=True)
                   for smiles in train]
    input_test = [preprocessor.construct_feature_matrices(smiles, train=False)
                  for smiles in test]

    with tempfile.NamedTemporaryFile(suffix='.json') as file:
        preprocessor.to_json(file.name)
        del preprocessor
        preprocessor = SmilesPreprocessor()
        preprocessor.from_json(file.name)

    input_train_new = [preprocessor.construct_feature_matrices(
        smiles, train=True) for smiles in train]
    input_test_new = [preprocessor.construct_feature_matrices(
        smiles, train=False) for smiles in test]

    assert str(input_train) == str(input_train_new)
    assert str(input_test) == str(input_test_new)
