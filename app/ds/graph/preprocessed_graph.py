import pickle as pkl
import sys

import networkx as nx
import numpy as np
from scipy import sparse as sp

from app.ds.graph import base_graph
from app.utils.constant import GCN, SYMMETRIC


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

        node_count = self.features.shape[0]
        edges = np.genfromtxt(network_data_path, dtype=np.dtype(str))
        # I guess the code would break for the case when we have just 1 edge. I should be fixing that later

        if (edges.shape[1] == 2):
            #     unwieghted graph
            edges = np.array(list(map(self.node_to_id_map.get, edges.flatten())),
                             dtype=np.int32).reshape(edges.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(node_count, node_count), dtype=np.float32)
        else:

            # weighted graph
            edges = np.array(list(map(
                lambda _edge_data: (self.node_to_id_map[_edge_data[0]],
                                    self.node_to_id_map[_edge_data[1]],
                                    float(_edge_data[2])),
                edges)), dtype=np.int32).reshape(edges.shape)
            adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),
                                shape=(node_count, node_count), dtype=np.float32)

        self.adj = symmetic_adj(adj)
        self.edge_count = edges.shape[0]
        print("{} edges read.".format(self.edge_count))

        return adj

    def compute_supports(self, model_params):
        adj = self.adj
        is_symmetric = (model_params.norm_mode == SYMMETRIC)

        supports = [transform_adj(adj=adj, is_symmetric=is_symmetric)]

        return supports


def symmetic_adj(adj):
    '''
    Method to preprocess the adjacency matrix `adj`
    :return: symmetric adjacency matrix

    Let us say the input matrix was [[1, 2]
                                    [1, 1]]
    To make it symmetric, we compute the max of the values at  index pairs (i, j) and (j, i) and set both (i, j) and
    (j, i) to that value.
    '''

    adj_t = adj.T
    return adj + adj_t.multiply(adj_t > adj) - adj.multiply(adj_t > adj)


def transform_adj(adj, is_symmetric=True):
    '''
    Method to transform the  adjacency matrix as described in section 2 of https://arxiv.org/abs/1609.02907
    '''
    adj = adj + sp.eye(adj.shape[0])
    # Adding self connections

    adj = renormalization_trick(adj, is_symmetric)

    return adj


def renormalization_trick(adj, symmetric=True):
    if symmetric:
        # dii = sum_j(aij)
        # dii = dii ** -o.5
        d = sp.diags(
            np.power(np.asarray(adj.sum(1)), -0.5).flatten(),
            offsets=0)
        # dii . adj . dii
        return adj.dot(d).transpose().dot(d).tocsr()
        # else:
        # d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        # a_norm = d.dot(adj).tocsr()
        # return a_norm


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


    # a = Graph(model_name="test")
    # a.read_data(dataset_name="cora", data_dir="/Users/shagun/projects/pregel/data")
