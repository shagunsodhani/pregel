import numpy as np
import tensorflow as tf

from app.ds.graph.preprocessed_graph import Graph
from app.model.params import SparseModelParams
from app.utils.constant import TRAIN, LABELS, FEATURES, SUPPORTS, MASK, VALIDATION, TEST, DROPOUT, GCN, \
    FF, GCN_POLY


class DataPipeline():
    '''Class for managing the data pipeline'''

    def __init__(self, model_params, data_dir, dataset_name):
        self.graph = None
        self._populate_graph(model_params, data_dir, dataset_name)
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.model_params = model_params
        self.num_elements = -1
        self.feature_size = self.graph.features.shape[1]
        self.node_size = self.graph.features.shape[0]
        self.label_size = self.graph.labels.shape[1]
        self.support_size = self.model_params.support_size
        self.supports = []
        self.placeholder_dict = {}
        self.train_feed_dict = {}
        self.validation_feed_dict = {}
        self.test_feed_dict = {}
        self._populate_feed_dicts()

    def _populate_graph(self, model_params, data_dir, dataset_name):
        self.graph = Graph(model_name=model_params.model_name, sparse_features=model_params.sparse_features)
        self.graph.read_data(data_dir=data_dir, dataset_name=dataset_name)

    def _set_placeholder_dict(self):
        '''
        Logic borrowed from
        https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/fully_connected_feed.py'''

        labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.label_size), name=LABELS)
        features_placeholder = tf.placeholder(tf.float32, shape=(None, self.feature_size), name=FEATURES)
        if (self.model_params.sparse_features):
            features_placeholder = tf.sparse_placeholder(tf.float32, shape=(None, self.feature_size), name=FEATURES)
        mask_placeholder = tf.placeholder(tf.float32, name=MASK)

        # For disabling dropout during testing - based on https://stackoverflow.com/questions/44971349/how-to-turn-off-dropout-for-testing-in-tensorflow
        dropout_placeholder = tf.placeholder_with_default(0.0, shape=(), name=DROPOUT)

        support_placeholder = []

        for i in range(self.support_size):
            support_placeholder.append(tf.sparse_placeholder(tf.float32, name=SUPPORTS + str(i)))

        self.placeholder_dict = {
            FEATURES: features_placeholder,
            LABELS: labels_placeholder,
            SUPPORTS: support_placeholder,
            MASK: mask_placeholder,
            DROPOUT: dropout_placeholder
        }

    def _prepare_feed_dict(self, labels, features, mask_indices, dropout):

        y = np.zeros(labels.shape)
        y[mask_indices,:] = labels[mask_indices,:]

        placeholder_dict = self.placeholder_dict
        feed_dict = {
            placeholder_dict[LABELS]: y,
            placeholder_dict[FEATURES]: features,
            placeholder_dict[MASK]: map_indices_to_mask(indices=mask_indices, mask_size=self.node_size),
            placeholder_dict[DROPOUT]: dropout
        }
        for i in range(self.support_size):
            feed_dict[placeholder_dict[SUPPORTS][i]] = self.supports[i]

        return feed_dict

    def _prepare_data_node_classifier(self, dataset_splits, shuffle_data=False):

        self._set_placeholder_dict()

        features = self.graph.features
        labels = self.graph.labels
        supports = self.graph.compute_supports(model_params=self.model_params)

        if (self.model_params.sparse_features):
            self.num_elements = features.nnz

        if(self.graph.preprocessed):
            train_index, val_index, test_index = self.graph.read_data(dataset_name=self.dataset_name, data_dir=self.data_dir)

        else:
            if(shuffle_data):
                shuffle = np.arange(self.node_size)
                np.random.shuffle(shuffle)
                features = features[shuffle]
                labels = labels[shuffle]
            train_index, val_index, test_index = self.graph.get_node_mask(dataset_splits=dataset_splits)

        features = convert_sparse_matrix_to_sparse_tensor(features)

        self.supports = list(
            map(
                lambda support: convert_sparse_matrix_to_sparse_tensor(support), supports
            )
        )

        return [[labels, features],
         [train_index, val_index, test_index]]

    def _prepare_data(self, dataset_splits, shuffle_data=False):

        if(self.model_params.model_name in set([GCN, GCN_POLY, FF])):
            return self._prepare_data_node_classifier(dataset_splits=dataset_splits,
                                                      shuffle_data=shuffle_data)
        else:
            return None

    def _populate_feed_dicts(self, dataset_splits=[140, 500, 1000]):
        '''Method to populate the feed dicts'''

        [[labels, features],
         [train_index, val_index, test_index]] = self._prepare_data(dataset_splits=dataset_splits)

        self.train_feed_dict = self._prepare_feed_dict(labels=labels,
                                                       features=features,
                                                       mask_indices=train_index,
                                                       dropout=self.model_params.dropout)

        self.validation_feed_dict = self._prepare_feed_dict(labels=labels,
                                                       features=features,
                                                       mask_indices=val_index,
                                                       dropout=0)

        self.test_feed_dict = self._prepare_feed_dict(labels=labels,
                                                       features=features,
                                                       mask_indices=test_index,
                                                       dropout=0)

    def get_feed_dict(self, mode=TRAIN):
        if mode == TRAIN:
            return self.train_feed_dict
        elif mode == VALIDATION:
            return self.validation_feed_dict
        elif mode == TEST:
            return self.test_feed_dict
        else:
            return None

    def get_placeholder_dict(self):
        '''Method to populate the feed dicts'''
        return self.placeholder_dict

    def get_sparse_model_params(self):
        return SparseModelParams(
                num_elements=self.num_elements,
                feature_size=self.feature_size
            )

def map_indices_to_mask(indices, mask_size):
    '''Method to map the indices to a mask'''
    mask = np.zeros(mask_size, dtype=np.float32)
    mask[indices] = 1.0
    return mask


def convert_sparse_matrix_to_sparse_tensor(X):
    '''
    code borrowed from https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent
    '''
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data, coo.shape)