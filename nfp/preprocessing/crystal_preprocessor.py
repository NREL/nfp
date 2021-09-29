from typing import Any, Dict

import networkx as nx
import numpy as np
import tensorflow as tf

from nfp import Preprocessor, Tokenizer


class CifPreprocessor(Preprocessor):
    def __init__(self, radius=None, num_neighbors=12):
        self.site_tokenizer = Tokenizer()
        self.radius = radius
        self.num_neighbors = num_neighbors

    def create_nx_graph(self, structure: Any, **kwargs) -> nx.DiGraph:
        pass

    def get_edge_features(self, edge_data: list,
                          max_num_edges) -> Dict[str, np.ndarray]:
        pass

    def get_node_features(self, node_data: list,
                          max_num_nodes) -> Dict[str, np.ndarray]:
        pass

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        pass

    @property
    def site_classes(self):
        return self.site_tokenizer.num_classes + 1

    @staticmethod
    def site_features(site):
        species = site.as_dict()['species']
        assert len(species) == 1
        return species[0]['element']

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        pass

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        pass

    @property
    def tfrecord_features(self) -> Dict[str, tf.io.FixedLenFeature]:
        pass
