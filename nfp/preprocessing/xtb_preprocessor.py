from typing import Dict, Optional, Callable, Hashable

import networkx as nx
import numpy as np
import rdkit.Chem
import json

try:
    import tensorflow as tf
except ImportError:
    tf = None

from nfp.preprocessing import features
from nfp.preprocessing.preprocessor import Preprocessor
from nfp.preprocessing.tokenizer import Tokenizer

class xTBPreprocessor(Preprocessor):
    def __init__(
            self,
            atom_features: Optional[Callable[[rdkit.Chem.Atom], Hashable]] = None,
            bond_features: Optional[Callable[[rdkit.Chem.Bond], Hashable]] = None,
            **kwargs,
    ) -> None:
        super(xTBPreprocessor, self).__init__(**kwargs)

        #first set of RDKit features for keeping count of type of atom/bond features.
        if atom_features is None:
            self.atom_features_xtb = features.atom_features_xtb
        if bond_features is None:
            self.bond_features_xtb = features.bond_features_xtb

        self.atom_tokenizer = Tokenizer()
        self.bond_tokenizer = Tokenizer()


    def create_nx_graph(self, mol: rdkit.Chem.Mol, **kwargs) -> nx.DiGraph:
        #add hydrogens as wbo contains Hydrogens
        g = nx.Graph(mol=mol)
        g.add_nodes_from(((atom.GetIdx(), {
            'atom': atom
        }) for atom in mol.GetAtoms()))

        #add edges based on wilberg bond orders.
        wbo = np.array(self.json_data['Wiberg matrix'])
        g.add_edges_from(((i, j, {
            'wbo': wbo[i][j],'mol': mol
        }) for i in range(len(wbo)) for j in range(len(wbo)) if wbo[i][j] > self.cutoff ))

        return nx.DiGraph(g)

    def get_edge_features(self, edge_data: list,
                          max_num_edges) -> Dict[str, np.ndarray]:
        bond_feature_matrix = np.zeros(max_num_edges, dtype=self.output_dtype)
        for n, (start_atom, end_atom, bond_dict) in enumerate(edge_data):

            # flipped = start_atom == bond_dict['bond'].GetEndAtomIdx()
            bond_feature_matrix[n] = self.bond_tokenizer(
                self.bond_features_xtb(start_atom, end_atom, bond_dict['wbo'],bond_dict['mol']))
        return {'n_bond': max_num_edges,'bond': bond_feature_matrix}

    def get_node_features(self, node_data: list,
                          max_num_nodes) -> Dict[str, np.ndarray]:
        atom_feature_matrix = np.zeros(max_num_nodes, dtype=self.output_dtype)
        for n, atom_dict in node_data:
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features_xtb(atom_dict['atom'], n, self.json_data))
        return {'n_atom': max_num_nodes,'atom': atom_feature_matrix}

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        return {}

    @property
    def atom_classes(self) -> int:
        """ The number of atom types found (includes the 0 null-atom type) """
        return self.atom_tokenizer.num_classes + 1

    @property
    def bond_classes(self) -> int:
        """ The number of bond types found (includes the 0 null-bond type) """
        return self.bond_tokenizer.num_classes + 1

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        if tf is None:
            raise ImportError('Tensorflow was not found')
        return {
            'n_atom': tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
            'n_bond': tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
            'atom': tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
            'bond': tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
            'connectivity': tf.TensorSpec(shape=(None, 2),
                                          dtype=self.output_dtype)
        }

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        """Defaults to zero for each output"""
        if tf is None:
            raise ImportError('Tensorflow was not found')
        return {
            key: tf.constant(0, dtype=self.output_dtype)
            for key in self.output_signature.keys()
        }

    @property
    def tfrecord_features(self) -> Dict[str, tf.io.FixedLenFeature]:
        """For loading preprocessed inputs from a tf records file"""
        if tf is None:
            raise ImportError('Tensorflow was not found')
        return {
            key: tf.io.FixedLenFeature(
                [],
                dtype=self.output_dtype if len(val.shape) == 0 else tf.string)
            for key, val in self.output_signature.items()
        }


class xTBSmilesPreprocessor(xTBPreprocessor):
    def __init__(self, *args, explicit_hs: bool = True,
        cutoff: float = 0.3, **kwargs):
        super(xTBSmilesPreprocessor, self).__init__(*args, **kwargs)
        self.explicit_hs = explicit_hs
        self.cutoff = cutoff

    def create_nx_graph(self, smiles: str, jsonfile: str, **kwargs) -> nx.DiGraph:
        mol = rdkit.Chem.MolFromSmiles(smiles)
        with open(jsonfile, 'r') as f:
            self.json_data = json.load(f)
        if self.explicit_hs:
            mol = rdkit.Chem.AddHs(mol)
        return super(xTBSmilesPreprocessor, self).create_nx_graph(mol, **kwargs)
