import json
import logging
import warnings
from abc import ABC, abstractmethod
from inspect import getmembers
from typing import Any, Optional

import networkx as nx
import numpy as np
import tensorflow as tf

from nfp.preprocessing.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class Preprocessor(ABC):
    @abstractmethod
    def create_nx_graph(self, structure: Any, **kwargs) -> nx.DiGraph:
        pass

    @abstractmethod
    def get_edge_features(self, edge_data: list, max_num_edges) -> {str: np.ndarray}:
        pass

    @abstractmethod
    def get_node_features(self, node_data: list, max_num_nodes) -> {str: np.ndarray}:
        pass

    @abstractmethod
    def get_graph_features(self, graph_data: dict) -> {str: np.ndarray}:
        pass

    @property
    @abstractmethod
    def output_signature(self) -> {str: tf.TensorSpec}:
        pass

    @property
    @abstractmethod
    def padding_values(self) -> {str: tf.constant}:
        pass

    @property
    @abstractmethod
    def tfrecord_features(self) -> {str: tf.io.FixedLenFeature}:
        pass

    @staticmethod
    def get_connectivity(graph: nx.DiGraph) -> {str: np.ndarray}:
        return {'connectivity': np.asarray(graph.edges)}

    def __call__(self,
                 structure: Any,
                 train: bool = False,
                 max_num_nodes: Optional[int] = None,
                 max_num_edges: Optional[int] = None,
                 **kwargs) -> {str: np.ndarray}:

        nx_graph = self.create_nx_graph(structure, **kwargs)

        max_num_edges = max(1, len(nx_graph.edges)) if max_num_edges is None else max_num_edges
        assert len(nx_graph.edges) <= max_num_edges, "max_num_edges too small for given input"

        max_num_nodes = len(nx_graph.nodes) if max_num_nodes is None else max_num_nodes
        assert len(nx_graph.nodes) <= max_num_nodes, "max_num_nodes too small for given input"

        # Make sure that Tokenizer classes are correctly initialized
        for _, tokenizer in getmembers(self, lambda x: type(x) == Tokenizer):
            tokenizer.train = train

        node_features = self.get_node_features(nx_graph.nodes(data=True), max_num_nodes)
        edge_features = self.get_edge_features(nx_graph.edges(data=True), max_num_edges)
        graph_features = self.get_graph_features(nx_graph.graph)
        connectivity = self.get_connectivity(nx_graph)

        return {**node_features, **edge_features, **graph_features, **connectivity}

    def construct_feature_matrices(self, *args, train=False, **kwargs) -> {str: np.ndarray}:
        warnings.warn("construct_feature_matrices is deprecated, use `call` instead as of nfp 0.4.0",
                      DeprecationWarning)
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


def load_from_json(obj, data):
    for key, val in obj.__dict__.items():
        try:
            if isinstance(val, type(data[key])):
                obj.__dict__[key] = data[key]
            elif hasattr(val, '__dict__'):
                load_from_json(val, data[key])

        except KeyError:
            logger.warning(f"{key} not found in JSON file, it may have been created with an older nfp version")
