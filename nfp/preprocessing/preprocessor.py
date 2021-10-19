import json
import logging
import warnings
from abc import ABC, abstractmethod
from inspect import getmembers
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import tensorflow as tf

from nfp.preprocessing.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class Preprocessor(ABC):
    def __init__(self, output_dtype: str = 'int32'):
        self.output_dtype = output_dtype

    @abstractmethod
    def create_nx_graph(self, structure: Any, **kwargs) -> nx.DiGraph:
        pass

    @abstractmethod
    def get_edge_features(self, edge_data: list,
                          max_num_edges) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def get_node_features(self, node_data: list,
                          max_num_nodes) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        pass

    @property
    @abstractmethod
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        pass

    @property
    @abstractmethod
    def padding_values(self) -> Dict[str, tf.constant]:
        pass

    @property
    @abstractmethod
    def tfrecord_features(self) -> Dict[str, tf.io.FixedLenFeature]:
        pass

    @staticmethod
    def get_connectivity(graph: nx.DiGraph) -> Dict[str, np.ndarray]:
        return {'connectivity': np.asarray(graph.edges, dtype='int64')}

    def __call__(self,
                 structure: Any,
                 train: bool = False,
                 max_num_nodes: Optional[int] = None,
                 max_num_edges: Optional[int] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        nx_graph = self.create_nx_graph(structure, **kwargs)

        max_num_edges = len(nx_graph.edges) if max_num_edges is None else max_num_edges
        assert len(nx_graph.edges) <= max_num_edges, "max_num_edges too small for given input"

        max_num_nodes = len(nx_graph.nodes) if max_num_nodes is None else max_num_nodes
        assert len(nx_graph.nodes) <= max_num_nodes, "max_num_nodes too small for given input"

        # Make sure that Tokenizer classes are correctly initialized
        for _, tokenizer in getmembers(self, lambda x: type(x) == Tokenizer):
            tokenizer.train = train

        node_features = self.get_node_features(nx_graph.nodes(data=True),
                                               max_num_nodes)
        edge_features = self.get_edge_features(nx_graph.edges(data=True),
                                               max_num_edges)
        graph_features = self.get_graph_features(nx_graph.graph)
        connectivity = self.get_connectivity(nx_graph)

        return {
            **node_features,
            **edge_features,
            **graph_features,
            **connectivity
        }

    def construct_feature_matrices(self,
                                   *args,
                                   train=False,
                                   **kwargs) -> Dict[str, np.ndarray]:
        warnings.warn(
            "construct_feature_matrices is deprecated, use `call` instead as "
            "of nfp 0.4.0", DeprecationWarning)
        return self(*args, train=train, **kwargs)

    def to_json(self, filename: str) -> None:
        """Serialize the classes's data to a json file"""
        with open(filename, 'w') as f:
            json.dump(self, f, default=lambda x: x.__dict__)

    def from_json(self, filename: str) -> None:
        """Set's the class's data with attributes taken from the save file"""
        with open(filename, 'r') as f:
            json_data = json.load(f)
        load_from_json(self, json_data)


class PreprocessorMultiGraph(Preprocessor, ABC):
    """Class to handle graphs with parallel edges and self-loops"""

    @abstractmethod
    def create_nx_graph(self, structure: Any, **kwargs) -> nx.MultiDiGraph:
        pass

    @staticmethod
    def get_connectivity(graph: nx.DiGraph) -> Dict[str, np.ndarray]:
        # Don't include keys in the connectivity matrix
        return {'connectivity': np.asarray(graph.edges)[:, :2]}


def load_from_json(obj, data):
    """Function to set member attributes from json data recursively.

    Parameters
    ----------
    obj: the class to initialize
    data: a dictionary of potentially nested attribute: value pairs

    Returns
    -------
    The object, with attributes set to those from the data file.

    """

    for key, val in obj.__dict__.items():
        try:
            if isinstance(val, type(data[key])):
                obj.__dict__[key] = data[key]
            elif hasattr(val, '__dict__'):
                load_from_json(val, data[key])

        except KeyError:
            logger.warning(
                f"{key} not found in JSON file, it may have been created with"
                " an older nfp version")
