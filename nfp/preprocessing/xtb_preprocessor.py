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
from nfp.preprocessing.mol_preprocessor import MolPreprocessor, SmilesPreprocessor
from nfp.preprocessing.tokenizer import Tokenizer


class xTBPreprocessor(MolPreprocessor):
    def __init__(
        self,
        *args,
        explicit_hs: bool = True,
        xtb_atom_features: Optional[Callable[[rdkit.Chem.Atom], Hashable]] = None,
        xtb_bond_features: Optional[Callable[[rdkit.Chem.Bond], Hashable]] = None,
        cutoff: float = 0.3,
        **kwargs
    ):
        super(xTBPreprocessor, self).__init__(*args, **kwargs)
        self.explicit_hs = explicit_hs
        self.cutoff = cutoff

        # update only bond features as we dont use rdkit
        self.bond_features = features.bond_features_wbo

        if xtb_atom_features is None:
            self.xtb_atom_features = [
                "mulliken charges",
                "cm5 charges",
                "s proportion",
                "p proportion",
                "d proportion",
                "FOD",
                "FOD s proportion",
                "FOD p proportion",
                "FOD d proportion",
            ]

        if xtb_bond_features is None:
            self.xtb_bond_features = ["Wiberg bond order"]

    def create_nx_graph(
        self, mol: rdkit.Chem.Mol, jsonfile: str, **kwargs
    ) -> nx.DiGraph:

        with open(jsonfile, "r") as f:
            json_data = json.load(f)

        # add hydrogens as wbo contains hydrogens and add xtb features to the graphs
        g = nx.Graph(mol=mol)
        g.add_nodes_from(
            (
                (
                    atom.GetIdx(),
                    {
                        "atom": atom,
                        "atomxtbfeatures": [
                            json_data[prop][atom.GetIdx()]
                            for prop in self.xtb_atom_features
                        ],
                    },
                )
                for atom in mol.GetAtoms()
            )
        )

        # add edges based on wilberg bond orders.
        wbo = np.array(json_data["Wiberg matrix"])
        g.add_edges_from(
            (
                (
                    i,
                    j,
                    {
                        "wbo": wbo[i][j],
                        "bondatoms": (mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)),
                    },
                )
                for i in range(len(wbo))
                for j in range(len(wbo))
                if wbo[i][j] > self.cutoff
            )
        )

        return nx.DiGraph(g)

    def get_edge_features(
        self, edge_data: list, max_num_edges
    ) -> Dict[str, np.ndarray]:

        bond_feature_matrix = np.zeros(max_num_edges, dtype=self.output_dtype)
        bond_feature_matrix_xtb = np.zeros(max_num_edges, dtype="float32")

        for n, (start_atom, end_atom, bond_dict) in enumerate(edge_data):
            bond_feature_matrix[n] = self.bond_tokenizer(
                self.bond_features(start_atom, end_atom, bond_dict["bondatoms"])
            )
            bond_feature_matrix_xtb[n] = bond_dict["wbo"]
        return {"bond": bond_feature_matrix, "bond_xtb": bond_feature_matrix_xtb}

    def get_node_features(
        self, node_data: list, max_num_nodes: int
    ) -> Dict[str, np.ndarray]:
        node_features = super().get_node_features(node_data, max_num_nodes)
        node_features["atom_xtb"] = np.zeros(
            [max_num_nodes, len(self.xtb_atom_features)], dtype="float32"
        )
        for n, atom_dict in node_data:
            node_features["atom_xtb"][n] = atom_dict["atomxtbfeatures"]
        return node_features

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        output_signature = super().output_signature
        output_signature["atom_xtb"] = tf.TensorSpec(
            shape=(None, None), dtype="float32"
        )
        output_signature["bond_xtb"] = tf.TensorSpec(
            shape=(None, None), dtype="float32"
        )
        return output_signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["atom_xtb"] = tf.constant(np.nan, dtype="float32")
        padding_values["bond_xtb"] = tf.constant(np.nan, dtype="float32")
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
            dtype=self.output_dtype
            if len(self.output_signature["bond_xtb"].shape) == 0
            else tf.string,
        )
        return tfrecord_features


class xTBSmilesPreprocessor(SmilesPreprocessor, xTBPreprocessor):
    pass
