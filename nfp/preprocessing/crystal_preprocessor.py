from typing import Dict

import networkx as nx
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None

from nfp.preprocessing.preprocessor import PreprocessorMultiGraph
from nfp.preprocessing.tokenizer import Tokenizer


class PymatgenPreprocessor(PreprocessorMultiGraph):
    def __init__(self, radius=None, num_neighbors=12, **kwargs):
        super(PymatgenPreprocessor, self).__init__(**kwargs)
        self.site_tokenizer = Tokenizer()
        self.radius = radius
        self.num_neighbors = num_neighbors

    def create_nx_graph(self, crystal, **kwargs) -> nx.MultiDiGraph:
        """ crystal should be a pymatgen.core.Structure object.
        """
        g = nx.MultiDiGraph(crystal=crystal)
        g.add_nodes_from(((i, {
            'site': site
        }) for i, site in enumerate(crystal.sites)))

        if self.radius is None:
            # Get the expected number of sites / volume, then find a radius
            # expected to yield 2x the desired number of neighbors
            desired_vol = (crystal.volume /
                           crystal.num_sites) * self.num_neighbors
            radius = 2 * (desired_vol / (4 * np.pi / 3)) ** (1 / 3)
        else:
            radius = self.radius

        for i, neighbors in enumerate(crystal.get_all_neighbors(radius)):
            if len(neighbors) < self.num_neighbors:
                raise RuntimeError(
                    f"Only {len(neighbors)} neighbors for site {i}")

            sorted_neighbors = sorted(neighbors,
                                      key=lambda x: x[1])[:self.num_neighbors]

            for _, distance, j, _ in sorted_neighbors:
                g.add_edge(i, j, distance=distance)

        return g

    def get_edge_features(self, edge_data: list,
                          max_num_edges) -> Dict[str, np.ndarray]:

        edge_feature_matrix = np.empty(max_num_edges, dtype='float32')
        edge_feature_matrix[:] = np.nan  # Initialize distances with nans

        for n, (_, _, edge_dict) in enumerate(edge_data):
            edge_feature_matrix[n] = edge_dict['distance']
        return {'distance': edge_feature_matrix}

    def get_node_features(self, node_data: list,
                          max_num_nodes) -> Dict[str, np.ndarray]:
        site_feature_matrix = np.zeros(max_num_nodes, dtype=self.output_dtype)
        for n, site_dict in node_data:
            site_feature_matrix[n] = self.site_tokenizer(
                self.site_features(site_dict['site']))
        return {'site': site_feature_matrix}

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        return {}

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
        if tf is None:
            raise ImportError('Tensorflow was not found')
        return {
            'site': tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
            'distance': tf.TensorSpec(shape=(None,), dtype='float32'),
            'connectivity': tf.TensorSpec(shape=(None, 2),
                                          dtype=self.output_dtype)
        }

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        if tf is None:
            raise ImportError('Tensorflow was not found')
        return {
            'site': tf.constant(0, dtype=self.output_dtype),
            'distance': tf.constant(np.nan, dtype='float32'),
            'connectivity': tf.constant(0, dtype=self.output_dtype)
        }

    @property
    def tfrecord_features(self) -> Dict[str, tf.io.FixedLenFeature]:
        if tf is None:
            raise ImportError('Tensorflow was not found')
        return {
            key: tf.io.FixedLenFeature([], dtype=tf.string)
            for key in self.output_signature.keys()
        }
