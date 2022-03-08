import json
from typing import Callable, Dict, Hashable, Optional, List
from sklearn.preprocessing import StandardScaler
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem

import networkx as nx
import numpy as np
import rdkit.Chem

try:
    import tensorflow as tf
except ImportError:
    tf = None

from nfp.preprocessing import features
from nfp.preprocessing.mol_preprocessor import (
    MolPreprocessor,
    SmilesPreprocessor,
)


class xTBPreprocessor(MolPreprocessor):
    def __init__(
        self,
        *args,
        explicit_hs: bool = True,
        xtb_atom_features: Optional[List[str]] = None,
        xtb_bond_features: Optional[List[str]] = None,
        xtb_mol_features: Optional[List[str]] = None,
        scaler: bool = True,
        **kwargs
    ):
        super(xTBPreprocessor, self).__init__(*args, **kwargs)
        self.explicit_hs = explicit_hs
        self.scaler = scaler

        # update only bond features as we dont use rdkit
        self.bond_features = features.bond_features_v3

        if xtb_atom_features is None:
            self.xtb_atom_features = [
                "mulliken charges",
                "cm5 charges",
                "FUKUI+",
                "FUKUI-",
                "FUKUIrad",
                "s proportion",
                "p proportion",
                "d proportion",
                "FOD",
                "FOD s proportion",
                "FOD p proportion",
                "FOD d proportion",
                "Dispersion coefficient C6",
                "Polarizability alpha",
            ]
        else:
            self.xtb_atom_features = xtb_atom_features

        if xtb_bond_features is None:
            self.xtb_bond_features = ["Wiberg matrix", "bond_dist"]
        else:
            self.xtb_bond_features = xtb_bond_features

        if xtb_mol_features is None:
            self.xtb_mol_features = [
                "total energy",
                "electronic energy",
                "HOMO",
                "LUMO",
            ]
        else:
            self.xtb_mol_features = xtb_mol_features

    def create_nx_graph(
        self, mol: rdkit.Chem.Mol, jsonfile: str, **kwargs
    ) -> nx.DiGraph:

        with open(jsonfile, "r") as f:
            json_data = json.load(f)

        # add hydrogens as wbo contains hydrogens and add xtb features to the graphs
        mol_data = {"mol_xtb": [json_data[prop] for prop in self.xtb_mol_features]}

        g = nx.Graph(mol=mol, **mol_data)
        for atom in mol.GetAtoms():
            atom_data = {
                "atom": atom,
                "atom_index": atom.GetIdx(),
                "atom_xtb": [
                    json_data[prop][atom.GetIdx()] for prop in self.xtb_atom_features
                ],
            }
            g.add_node(atom.GetIdx(), **atom_data)

        for bond in mol.GetBonds():
            edge_data = {
                "bondatoms": (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                "bond": bond,
                "bond_index": bond.GetIdx(),
                "bond_xtb": [
                    json_data[prop][bond.GetBeginAtomIdx()][bond.GetEndAtomIdx()]
                    for prop in self.xtb_bond_features
                ],
            }
            g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), **edge_data)

        return nx.DiGraph(g)

    def get_edge_features(
        self, edge_data: list, max_num_edges
    ) -> Dict[str, np.ndarray]:
        bond_indices = np.zeros(max_num_edges, dtype=self.output_dtype)
        bond_atom_indices = np.zeros((max_num_edges, 2), dtype=self.output_dtype)
        bond_feature_matrix = np.zeros(max_num_edges, dtype=self.output_dtype)
        bond_feature_matrix_xtb = np.zeros(
            (max_num_edges, len(self.xtb_bond_features)), dtype="float32"
        )

        for n, (start_atom, end_atom, bond_dict) in enumerate(edge_data):
            bond_indices[n] = bond_dict["bond_index"]
            bond_atom_indices[n] = (start_atom, end_atom)
            if bond_dict["bond"] is not None:
                bond_feature_matrix[n] = self.bond_tokenizer(
                    self.bond_features(bond_dict["bond"])
                )
            else:
                bond_feature_matrix[n] = 0
            bond_feature_matrix_xtb[n] = bond_dict["bond_xtb"]

        if self.scaler:
            scaler = StandardScaler()
            bond_feature_matrix_xtb = scaler.fit_transform(bond_feature_matrix_xtb)
        return {
            "bond": bond_feature_matrix,
            "bond_xtb": bond_feature_matrix_xtb,
            "bond_indices": bond_indices,
            "bond_atom_indices": bond_atom_indices,
        }

    def get_node_features(
        self, node_data: list, max_num_nodes: int
    ) -> Dict[str, np.ndarray]:
        atom_indices = np.zeros(max_num_nodes, dtype=self.output_dtype)
        node_features = super().get_node_features(node_data, max_num_nodes)
        node_features["atom_xtb"] = np.zeros(
            [max_num_nodes, len(self.xtb_atom_features)], dtype="float32"
        )

        for n, atom_dict in node_data:
            atom_indices[n] = atom_dict["atom_index"]
            node_features["atom_xtb"][n] = atom_dict["atom_xtb"]

        if self.scaler:
            scaler = StandardScaler()
            node_features["atom_xtb"] = scaler.fit_transform(node_features["atom_xtb"])

        return {"atom_indices": atom_indices, **node_features}

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        return {"mol_xtb": np.asarray(graph_data["mol_xtb"])}

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        output_signature = super().output_signature
        output_signature["atom_xtb"] = tf.TensorSpec(
            shape=(None, None), dtype="float32"
        )
        output_signature["bond_xtb"] = tf.TensorSpec(
            shape=(None, None), dtype="float32"
        )
        output_signature["bond_indices"] = tf.TensorSpec(
            shape=(None,), dtype=self.output_dtype
        )
        output_signature["bond_atom_indices"] = tf.TensorSpec(
            shape=(None, None), dtype=self.output_dtype
        )
        output_signature["atom_indices"] = tf.TensorSpec(
            shape=(None,), dtype=self.output_dtype
        )
        output_signature["mol_xtb"] = tf.TensorSpec(shape=(None,), dtype="float32")
        return output_signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["atom_xtb"] = tf.constant(0, dtype="float32")
        padding_values["bond_xtb"] = tf.constant(0, dtype="float32")
        padding_values["bond_indices"] = tf.constant(0, dtype=self.output_dtype)
        padding_values["bond_atom_indices"] = tf.constant(0, dtype=self.output_dtype)
        padding_values["atom_indices"] = tf.constant(0, dtype=self.output_dtype)
        padding_values["mol_xtb"] = tf.constant(0, dtype="float32")
        return padding_values

    @property
    def tfrecord_features(self) -> Dict[str, tf.io.FixedLenFeature]:
        tfrecord_features = super().tfrecord_features
        tfrecord_features["atom_xtb"] = tf.io.FixedLenFeature(
            [],
            dtype="float32"
            if len(self.output_signature["atom_xtb"].shape) == 0
            else tf.string,
        )
        tfrecord_features["bond_xtb"] = tf.io.FixedLenFeature(
            [],
            dtype="float32"
            if len(self.output_signature["bond_xtb"].shape) == 0
            else tf.string,
        )
        tfrecord_features["bond_indices"] = tf.io.FixedLenFeature(
            [],
            dtype=self.output_dtype
            if len(self.output_signature["bond_indices"].shape) == 0
            else tf.string,
        )
        tfrecord_features["bond_atom_indices"] = tf.io.FixedLenFeature(
            [],
            dtype=self.output_dtype
            if len(self.output_signature["bond_atom_indices"].shape) == 0
            else tf.string,
        )
        tfrecord_features["atom_indices"] = tf.io.FixedLenFeature(
            [],
            dtype=self.output_dtype
            if len(self.output_signature["atom_indices"].shape) == 0
            else tf.string,
        )
        tfrecord_features["mol_xtb"] = tf.io.FixedLenFeature(
            [],
            dtype="float32"
            if len(self.output_signature["mol_xtb"].shape) == 0
            else tf.string,
        )
        return tfrecord_features


class xTBSmilesPreprocessor(SmilesPreprocessor, xTBPreprocessor):
    pass


class xTBWBOPreprocessor(xTBPreprocessor):
    def __init__(self, cutoff: int = 0.3, **kwargs):
        self.cutoff = cutoff
        super(xTBWBOPreprocessor, self).__init__(**kwargs)

    def create_nx_graph(
        self, mol: rdkit.Chem.Mol, jsonfile: str, **kwargs
    ) -> nx.DiGraph:

        with open(jsonfile, "r") as f:
            json_data = json.load(f)

        # add hydrogens as wbo contains hydrogens and add xtb features to the graphs
        mol_data = {"mol_xtb": [json_data[prop] for prop in self.xtb_mol_features]}

        g = nx.Graph(mol=mol, **mol_data)
        for atom in mol.GetAtoms():
            atom_data = {
                "atom": atom,
                "atom_index": atom.GetIdx(),
                "atom_xtb": [
                    json_data[prop][atom.GetIdx()] for prop in self.xtb_atom_features
                ],
            }
            g.add_node(atom.GetIdx(), **atom_data)

        # add edges based on wilberg bond orders.
        wbo = np.array(json_data["Wiberg matrix"])
        edges_to_add = (
            (i, j)
            for i in range(len(wbo))
            for j in range(len(wbo))
            if wbo[i][j] > self.cutoff and j > i
        )

        max_bonds = len(mol.GetBonds()) - 1
        for i, j in edges_to_add:
            if mol.GetBondBetweenAtoms(i, j) is not None:
                idx = mol.GetBondBetweenAtoms(i, j).GetIdx()
            else:
                max_bonds += 1
                idx = max_bonds
            edge_data = {
                "bondatoms": (mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)),
                "bond": mol.GetBondBetweenAtoms(i, j),
                "bond_index": idx,
                "bond_xtb": [json_data[prop][i][j] for prop in self.xtb_bond_features],
            }

            g.add_edge(i, j, **edge_data)

        return nx.DiGraph(g)


class xTBSmilesWBOPreprocessor(SmilesPreprocessor, xTBWBOPreprocessor):
    pass


class xTB3DPreprocessor(xTBPreprocessor):
    def __init__(self, n_neighbors: int = 100, distcutoff: int = 5, **kwargs):
        self.n_neighbors = n_neighbors
        self.distcutoff = distcutoff
        super(xTB3DPreprocessor, self).__init__(**kwargs)

    def create_nx_graph(
        self, mol: rdkit.Chem.Mol, jsonfile: str, **kwargs
    ) -> nx.DiGraph:

        with open(jsonfile, "r") as f:
            json_data = json.load(f)

        # add hydrogens as wbo contains hydrogens and add xtb features to the graphs
        mol_data = {"mol_xtb": [json_data[prop] for prop in self.xtb_mol_features]}

        g = nx.Graph(mol=mol, **mol_data)
        for atom in mol.GetAtoms():
            atom_data = {
                "atom": atom,
                "atom_index": atom.GetIdx(),
                "atom_xtb": [
                    json_data[prop][atom.GetIdx()] for prop in self.xtb_atom_features
                ],
            }
            g.add_node(atom.GetIdx(), **atom_data)

        # add edges based on neighbours from coordinates.
        distance_matrix = np.array(json_data["bond_dist"])

        edges_to_add = []
        for n, atom in enumerate(mol.GetAtoms()):
            # if n_neighbors is greater than total atoms, then each atom is a
            # neighbor.
            if (self.n_neighbors + 1) > len(mol.GetAtoms()):
                neighbor_end_index = len(mol.GetAtoms())
            else:
                neighbor_end_index = self.n_neighbors + 1

            distance_atom = distance_matrix[n, :]
            cutoff_end_index = distance_atom[distance_atom < self.distcutoff].size

            end_index = min(neighbor_end_index, cutoff_end_index)
            neighbor_inds = distance_matrix[n, :].argsort()[1:end_index]

            if len(neighbor_inds) == 0:
                neighbor_inds = [n]
            for neighbor in neighbor_inds:
                edges_to_add.append((int(n), int(neighbor)))

        max_bonds = len(mol.GetBonds()) - 1
        for i, j in edges_to_add:
            if mol.GetBondBetweenAtoms(i, j) is not None:
                idx = mol.GetBondBetweenAtoms(i, j).GetIdx()
            else:
                max_bonds += 1
                idx = max_bonds
            edge_data = {
                "bondatoms": (mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)),
                "bond": mol.GetBondBetweenAtoms(i, j),
                "bond_index": idx,
                "bond_xtb": [json_data[prop][i][j] for prop in self.xtb_bond_features],
            }

            g.add_edge(i, j, **edge_data)

        return nx.DiGraph(g)


class xTBSmiles3DPreprocessor(SmilesPreprocessor, xTB3DPreprocessor):
    pass
