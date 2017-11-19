import os
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse as sp

from app.utils.constant import GCN, NETWORK, LABEL, FEATURE
from app.utils.util import invert_dict, map_set_to_khot_vector, map_list_to_floats


class Base_Graph(ABC):
    '''Base class for the graph data structure'''

    def __init__(self, model_name=GCN, sparse_features=True):
        '''Method to initialise the graph'''
        self.preprocessed = False
        self.features = None
        # nodes X features
        self.adj = None
        self.labels = None
        # nodes X labels

        # For optimisation
        self.sparse_features = sparse_features

        # We are saving the model_name as different models would need different kind of preprocessing.
        self.model_name = model_name

        # We are saving the mappings so that we do not have to make any assumptions about the data types any more.
        # `label` and `node` are assumed to be strings as read from the input file.
        self.label_to_id_map = {}
        self.id_to_label_map = {}
        self.node_to_id_map = {}
        self.id_to_node_map = {}

        # Mapping of node to labels and lables to nodes. We would use the latter for computing scores.
        # Note that these ds are using the ids and not the raw strings read from the file.
        self.label_to_node_map = {}
        self.node_to_label_map = {}

        self.edge_count = -1

    def read_labels(self, label_data_path):
        '''
        Method to read the lables from `data_path`
        '''
        print("Reading labels from", str(label_data_path))
        data = np.genfromtxt(label_data_path, dtype=np.dtype(str))

        label_to_id_map = {label: id for id, label in enumerate(
            set(
                list(
                    map(lambda _data: _data[1], data)
                )
            )
        )}

        node_to_id_map = {node: id for id, node in enumerate(
            set(
                list(
                    map(lambda _data: _data[0], data)
                )
            )
        )}

        node_to_label_map = {}
        label_to_node_map = {}

        for node, label in map(lambda _data: (node_to_id_map[_data[0]], label_to_id_map[_data[1]]), data):
            if node not in node_to_label_map:
                node_to_label_map[node] = set()
            node_to_label_map[node].add(label)
            if label not in label_to_node_map:
                label_to_node_map[label] = set()
            label_to_node_map[label].add(node)

        label_count = len(label_to_id_map.keys())

        labels = np.asarray(list(
            map(lambda index_set: map_set_to_khot_vector(index_set=index_set, num_classes=label_count)
                , node_to_label_map.values())
        ))

        assert (len(self.id_to_node_map.keys()) == len(self.node_to_label_map.keys())), \
            "Some nodes are missing labels or ids"

        assert (len(self.id_to_label_map.keys()) == len(self.label_to_node_map.keys())), \
            "Some labels are missing nodes or ids"

        node_count = labels.shape[0]
        label_count = labels.shape[1]

        # Updating all the class variables in one place
        self.label_to_id_map = label_to_id_map
        self.node_to_id_map = node_to_id_map
        self.node_to_label_map = node_to_label_map
        self.label_to_node_map = label_to_node_map

        self.id_to_label_map, self.id_to_node_map = list(
            map(lambda _dict: invert_dict(_dict), [
                label_to_id_map, node_to_id_map
            ])
        )

        self.labels = labels

        print("{} nodes read.".format(node_count))
        print("{} labels read.".format(label_count))

    def read_features(self, feature_data_path, one_hot=False, dim=100):
        '''
        Method to read the features from `feature_data_path`
        '''
        # Check if the `feature_data_path` is set else generate default feature vectors

        node_count = len(self.id_to_node_map.keys())

        if (feature_data_path):
            features = np.genfromtxt(feature_data_path, dtype=np.dtype(str))
            features = np.asarray(
                list(map(map_list_to_floats, features[:, 1:])), dtype=np.int32)
            if self.sparse_features:
                features = sp.csr_matrix(features)

        else:
            if (one_hot):
                # In case of one_hot features, we ignore the set value of `dim` and use dim = node_count.
                dim = node_count
                assert (dim > 0), "node count  = ".format(dim)
                if self.sparse_features:
                    features = sp.identity(dim).tocsr()
                else:
                    features = sp.identity(dim).todense()
            else:
                features = np.random.uniform(low=0, high=0.5, size=(node_count, dim))

        assert (features.shape[0] == node_count), "Missing features for some nodes"
        self.features = features
        print("{} features read for each node.".format(self.features.shape[1]))

    def read_data(self, data_dir=None, dataset_name=None):
        '''
        Method to read the data corresponding to `dataset_name` from `data_dir`

        :return:
        Populates self.features, self.adjacency_matrix, self.labels
        '''
        data_path = os.path.join(data_dir, dataset_name)
        print("Reading data from", str(data_path))

        data_path_map = {}
        data_path_map[NETWORK] = os.path.join(data_path, "network.txt")
        data_path_map[LABEL] = os.path.join(data_path, "label.txt")
        data_path_map[FEATURE] = os.path.join(data_path, "feature.txt")

        self.read_labels(label_data_path=data_path_map[LABEL])
        self.read_features(feature_data_path=data_path_map[FEATURE])
        self.read_network(network_data_path=data_path_map[NETWORK])

    @abstractmethod
    def read_network(self, network_data_path):
        '''
        Method to read the network from `network_data_path`
        '''
        pass

    @abstractmethod
    def compute_supports(self, model_params):
        '''
        Method to compute the supports for the graph before feeding to the model
        '''
        pass
