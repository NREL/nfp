from nfp.preprocessing import features, Tokenizer
from rdkit import Chem
import pytest

@pytest.fixture()
def mol():
    return Chem.MolFromSmiles('CO')

def test_atom_features(mol):
    assert features.atom_features_v1(mol.GetAtomWithIdx(0)) == "('C', 1, 3, 3, False)"
    assert features.atom_features_v2(mol.GetAtomWithIdx(0)) == "(rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, 1, 1, 0, rdkit.Chem.rdchem.HybridizationType.SP3, 3, False, False, 0, 3, 0, 'C', 4, 3, 4, 0)"

def test_bond_features(mol):
    assert features.bond_features_v1(mol.GetBondWithIdx(0)) == "(rdkit.Chem.rdchem.BondType.SINGLE, False, False, ['C', 'O'])"
    assert features.bond_features_v2(mol.GetBondWithIdx(0)) == "(rdkit.Chem.rdchem.BondType.SINGLE, False, rdkit.Chem.rdchem.BondStereo.STEREONONE, 0, ['C', 'O'])"
    assert features.bond_features_v3(mol.GetBondWithIdx(0)) == '(rdkit.Chem.rdchem.BondType.SINGLE, False, rdkit.Chem.rdchem.BondStereo.STEREONONE, 0, \'O\', "(\'C\', 1, 3, 3, False)", "(\'O\', 1, 1, 1, False)")'
    
def test_tokenizer():
    token = Tokenizer()
    assert token.train

    assert token('test') == 2
    assert token('test') == 2
    assert token('test2') == 3

    assert token.num_classes == 3
    token.train = False

    assert token('test') == 2
    assert token('test2') == 3
    assert token('test3') == 1
    assert token.num_classes == 3


