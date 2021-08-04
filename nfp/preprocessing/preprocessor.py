import json
import logging
from typing import Callable, Dict, Hashable, List, Optional

import numpy as np
import rdkit
import tensorflow as tf
from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles
from tqdm import tqdm

from nfp.preprocessing import features
from nfp.preprocessing.features import Tokenizer

zero = tf.constant(0, dtype=tf.int64)


class MolPreprocessor(object):
    """A preprocessor to turn a set of SMILES strings into atom, bond, and connectivity inputs suitable for nfp's
    graph layers.

    Args:
        explicit_hs: whether to tell RDkit to add H's to a molecule.
        atom_features: A function applied to an rdkit.Atom that returns some
            representation (i.e., string, integer) for the Tokenizer class.
        bond_features: A function applied to an rdkit Bond to return some description.

    :Example:
    >>> preprocessor = MolPreprocessor(explicit_hs=False)
    >>> preprocessor.construct_feature_matrices(rdkit.Chem.MolFromSmiles('CCC'), train=True)
    {'atom': array([2, 3, 2]),
     'bond': array([2, 2, 2, 2]),
     'connectivity': array([[0, 1],
            [1, 0],
            [1, 2],
            [2, 1]])}
    """

    def __init__(self,
                 atom_features: Optional[Callable[[rdkit.Chem.Atom], Hashable]] = None,
                 bond_features: Optional[Callable[[rdkit.Chem.Bond], Hashable]] = None) -> None:

        self.atom_tokenizer = Tokenizer()
        self.bond_tokenizer = Tokenizer()

        if atom_features is None:
            atom_features = features.atom_features_v1

        if bond_features is None:
            bond_features = features.bond_features_v1

        self.atom_features = atom_features
        self.bond_features = bond_features

        # Keep track of biggest molecules seen in training
        self.max_atoms = 0
        self.max_bonds = 0

    def to_json(self, filename):
        with open(filename, 'w') as f:
            return json.dump(self, f, default=lambda x: x.__dict__)

    def from_json(self, filename):
        with open(filename, 'r') as f:
            json_data = json.load(f)

        load_from_json(self, json_data)

    @property
    def atom_classes(self):
        """ The number of atom types found (includes the 0 null-atom type) """
        return self.atom_tokenizer.num_classes + 1

    @property
    def bond_classes(self):
        """ The number of bond types found (includes the 0 null-bond type) """
        return self.bond_tokenizer.num_classes + 1

    def construct_feature_matrices(self, mol: rdkit.Chem.Mol, train: bool = False) -> {}:
        """ Convert an rdkit Mol to a list of tensors
        'atom' : (n_atom,) length list of atom classes
        'bond' : (n_bond,) list of bond classes
        'connectivity' : (n_bond, 2) array of source atom, target atom pairs.
        """

        self.atom_tokenizer.train = train
        self.bond_tokenizer.train = train

        n_atom = mol.GetNumAtoms()
        n_bond = 2 * mol.GetNumBonds()

        # If its an isolated atom, add a self-link
        if n_bond == 0:
            n_bond = 1

        atom_feature_matrix = np.zeros(n_atom, dtype='int32')
        bond_feature_matrix = np.zeros(n_bond, dtype='int32')
        connectivity = np.zeros((n_bond, 2), dtype='int32')

        if n_bond == 1:
            bond_feature_matrix[0] = self.bond_tokenizer('self-link')

        bond_index = 0
        for n, atom in enumerate(mol.GetAtoms()):

            # Atom Classes
            atom_feature_matrix[n] = self.atom_tokenizer(self.atom_features(atom))

            start_index = atom.GetIdx()

            for bond in atom.GetBonds():
                # Is the bond pointing at the target atom
                rev = bond.GetBeginAtomIdx() != start_index

                # Bond Classes
                bond_feature_matrix[bond_index] = self.bond_tokenizer(self.bond_features(bond, flipped=rev))

                # Connectivity
                if not rev:  # Original direction
                    connectivity[bond_index, 0] = bond.GetBeginAtomIdx()
                    connectivity[bond_index, 1] = bond.GetEndAtomIdx()

                else:  # Reversed
                    connectivity[bond_index, 0] = bond.GetEndAtomIdx()
                    connectivity[bond_index, 1] = bond.GetBeginAtomIdx()

                bond_index += 1

        return {
            'atom': atom_feature_matrix,
            'bond': bond_feature_matrix,
            'connectivity': connectivity,
        }

    output_signature = {'atom': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity': tf.TensorSpec(shape=(None, 2), dtype=tf.int32)}


def load_from_json(obj, data):
    for key, val in obj.__dict__.items():
        if type(val) == type(data[key]):
            obj.__dict__[key] = data[key]
        elif hasattr(val, '__dict__'):
            load_from_json(val, data[key])


class SmilesPreprocessor(MolPreprocessor):

    def __init__(self, *args, explicit_hs: bool = True, **kwargs):
        super(SmilesPreprocessor, self).__init__(*args, **kwargs)
        self.explicit_hs = explicit_hs

    def construct_feature_matrices(self, smiles: str, train: bool = False) -> {}:
        mol = rdkit.Chem.MolFromSmiles(smiles)
        if self.explicit_hs:
            mol = rdkit.Chem.AddHs(mol)
        return super(SmilesPreprocessor, self).construct_feature_matrices(mol, train=train)



def get_max_atom_bond_size(smiles_iterator, explicit_hs=True):
    """ Convienence function to get max_atoms, max_bonds for a set of input
    SMILES """

    max_atoms = 0
    max_bonds = 0
    for smiles in tqdm(smiles_iterator):
        mol = MolFromSmiles(smiles)
        if explicit_hs:
            mol = AddHs(mol)
        max_atoms = max([max_atoms, len(mol.GetAtoms())])
        max_bonds = max([max_bonds, len(mol.GetBonds())])

    return dict(max_atoms=max_atoms, max_bonds=max_bonds * 2)


def canonicalize_smiles(smiles, isomeric=True, sanitize=True):
    try:
        mol = MolFromSmiles(smiles, sanitize=sanitize)
        return MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:
        pass


def filter_keys(input_dict: Dict, keys: Optional[List[str]] = None) -> Dict:
    """Remove unnecessary model inputs from nfp.SmilesPreprocessor outputs

    Args:
        keys: desired keys
        input_dict: A dictionary containing unnecessary keys

    Returns:
        attribute_dict: The input dictionary filtered to the desired keys
    """
    if keys is None:
        keys = ['atom', 'bond', 'connectivity']

    return {key: value for key, value in input_dict.items() if key in keys}
