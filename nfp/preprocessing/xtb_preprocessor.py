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
from nfp.preprocessing.mol_preprocessor import MolPreprocessor
from nfp.preprocessing.tokenizer import Tokenizer


class xTBSmilesPreprocessor(MolPreprocessor):
    def __init__(self, *args, explicit_hs: bool = True,
        xtb_atom_features: Optional[Callable[[rdkit.Chem.Atom], Hashable]] = None,
        xtb_bond_features: Optional[Callable[[rdkit.Chem.Bond], Hashable]] = None,
        cutoff: float = 0.3, **kwargs):
        super(xTBSmilesPreprocessor, self).__init__(*args, **kwargs)
        self.explicit_hs = explicit_hs
        self.cutoff = cutoff

        #update only bond features as we dont use rdkit
        self.bond_features = features.bond_features_wbo

        if xtb_atom_features is None:
            self.xtb_atom_features = [
                'mulliken charges', 'cm5 charges', 's proportion', 'p proportion',
                'd proportion', 'FOD','FOD s proportion','FOD p proportion','FOD d proportion'
            ]

        if xtb_bond_features is None:
            self.xtb_bond_features = [
                'Wiberg bond order'
            ]

    def create_nx_graph(self, smiles: str, jsonfile: str, **kwargs) -> nx.DiGraph:
        mol = rdkit.Chem.MolFromSmiles(smiles)
        with open(jsonfile, 'r') as f:
            json_data = json.load(f)
        if self.explicit_hs:
            mol = rdkit.Chem.AddHs(mol)

        #add hydrogens as wbo contains hydrogens and add xtb features to the graphs
        g = nx.Graph(mol=mol)
        g.add_nodes_from(((atom.GetIdx(), {
            'atom': atom, 'atomxtbfeatures': [json_data[prop][atom.GetIdx()] for prop in self.xtb_atom_features]
        }) for atom in mol.GetAtoms()))

        #add edges based on wilberg bond orders.
        wbo = np.array(json_data['Wiberg matrix'])
        g.add_edges_from(((i, j, {
            'wbo': wbo[i][j], 'mol': mol
        }) for i in range(len(wbo)) for j in range(len(wbo)) if wbo[i][j] > self.cutoff ))

        return nx.DiGraph(g)

    def get_edge_features(self, edge_data: list,
                          max_num_edges) -> Dict[str, np.ndarray]:

        bond_feature_matrix = np.zeros(max_num_edges, dtype=self.output_dtype)
        bond_feature_matrix_xtb = np.zeros(max_num_edges, dtype='float32')

        for n, (start_atom, end_atom, bond_dict) in enumerate(edge_data):
            bond_feature_matrix[n] = self.bond_tokenizer(
                self.bond_features(start_atom, end_atom, bond_dict['mol']))
            bond_feature_matrix_xtb[n] = bond_dict['wbo']
        return {'n_bond': max_num_edges, 'bond': bond_feature_matrix,'bondxtbfeatures': bond_feature_matrix_xtb}

    def get_node_features(self, node_data: list,
                          max_num_nodes) -> Dict[str, np.ndarray]:

        atom_feature_matrix = np.zeros(max_num_nodes, dtype=self.output_dtype)
        atom_feature_matrix_xtb = np.zeros([max_num_nodes,len(self.xtb_atom_features)], dtype='float32')

        for n, atom_dict in node_data:
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom_dict['atom']))
            atom_feature_matrix_xtb[n] = atom_dict['atomxtbfeatures']
        return {'n_atom': max_num_nodes,'atom': atom_feature_matrix,'atomxtbfeatures': atom_feature_matrix_xtb}

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
                                          dtype=self.output_dtype),
            'atomxtbfeatures': tf.TensorSpec(shape=(None, None),
                                          dtype='float32'),
            'bondxtbfeatures': tf.TensorSpec(shape=(None, None),
                                          dtype='float32')
        }
