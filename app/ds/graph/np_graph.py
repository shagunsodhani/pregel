import numpy as np
from scipy import sparse as sp

from app.ds.graph import base_graph
from app.utils.constant import GCN


class Graph(base_graph.Base_Graph):
    '''Base class for the graph data structure'''

    def __init__(self, model_name=GCN):
        '''Method to initialise the graph'''
        super(Graph, self).__init__(model_name=model_name)

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
