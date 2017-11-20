import pickle as pkl
import sys

import networkx as nx
import numpy as np
from scipy import sparse as sp

from app.ds.graph import base_graph
from app.utils.constant import GCN

class Graph(base_graph.Base_Graph):
    '''This is the class to access the preprocessed graphs'''

    def __init__(self, model_name=GCN, sparse_features=True):
        '''Method to initialise the graph'''
        super(Graph, self).__init__(model_name=model_name, sparse_features=sparse_features)
        self.preprocessed = True

    def read_data(self, data_dir=None, dataset_name=None):
        '''
        Method to read the data corresponding to `dataset_name` from `data_dir`

        Logic borrowed from https://github.com/tkipf/gcn/blob/master/gcn/utils.py
        '''
        print("Reading data from", str(data_dir))
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("{}/{}/ind.{}.{}".format(data_dir, dataset_name, dataset_name, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("{}/{}/ind.{}.test.index".format(data_dir, dataset_name, dataset_name))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tocsr()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        self.adj = adj
        self.features = features
        self.labels = labels

        return idx_train, idx_val, idx_test

    def read_network(self, network_data_path):
        '''
        Method to read the network from `network_data_path`
        '''
        pass


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
