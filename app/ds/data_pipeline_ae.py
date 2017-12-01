import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from app.ds.data_pipeline import DataPipeline, convert_sparse_matrix_to_sparse_tensor
from app.model.params import AutoEncoderModelParams
from app.utils.constant import TRAIN, LABELS, FEATURES, SUPPORTS, MASK, VALIDATION, \
    TEST, DROPOUT, GCN_AE, MODE, NORMALISATION_CONSTANT, GCN_VAE


class DataPipelineAE(DataPipeline):
    '''Class for managing the data pipeline'''

    def __init__(self, model_params, data_dir, dataset_name):

        self.autoencoder_model_params = None
        super(DataPipelineAE, self).__init__(model_params=model_params, data_dir=data_dir,
                                             dataset_name=dataset_name)

    def _set_placeholder_dict(self):
        '''
        Logic borrowed from
        https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/fully_connected_feed.py'''

        labels_placeholder = tf.sparse_placeholder(tf.float32, name=LABELS)
        # Since this is the auto-encoder model, we are basically passing along the original adjacency matrix

        features_placeholder = tf.placeholder(tf.float32, shape=(None, self.feature_size), name=FEATURES)
        if (self.model_params.sparse_features):
            features_placeholder = tf.sparse_placeholder(tf.float32, shape=(None, self.feature_size), name=FEATURES)

        # For disabling dropout during testing - based on https://stackoverflow.com/questions/44971349/how-to-turn-off-dropout-for-testing-in-tensorflow
        dropout_placeholder = tf.placeholder_with_default(0.0, shape=(), name=DROPOUT)

        mask_placeholder = tf.sparse_placeholder(tf.float32, name=MASK)

        mode_placeholder = tf.placeholder(tf.string, name=MODE)

        normalisation_constant_placeholder = tf.placeholder_with_default(0.5, shape=(), name=NORMALISATION_CONSTANT)

        support_placeholder = []

        for i in range(self.support_size):
            support_placeholder.append(tf.sparse_placeholder(tf.float32, name=SUPPORTS + str(i)))

        self.placeholder_dict = {
            LABELS: labels_placeholder,
            FEATURES: features_placeholder,
            SUPPORTS: support_placeholder,
            MASK: mask_placeholder,
            DROPOUT: dropout_placeholder,
            MODE: mode_placeholder,
            NORMALISATION_CONSTANT: normalisation_constant_placeholder
        }

    def _prepare_feed_dict(self, labels, features, mask_indices, dropout, mode):

        mask = convert_sparse_matrix_to_sparse_tensor(
            sp.csr_matrix((np.ones(len(mask_indices)), (mask_indices[:, 0], mask_indices[:, 1])),
                          shape=self.graph.adj.shape))

        placeholder_dict = self.placeholder_dict
        feed_dict = {
            placeholder_dict[LABELS]: labels,
            placeholder_dict[FEATURES]: features,
            placeholder_dict[MASK]: mask,
            placeholder_dict[DROPOUT]: dropout,
            placeholder_dict[MODE]: mode
        }
        for i in range(self.support_size):
            feed_dict[placeholder_dict[SUPPORTS][i]] = self.supports[i]

        return feed_dict

    def _prepare_data_auto_encoder(self, dataset_splits, shuffle_data=False):

        self._set_placeholder_dict()

        adj, train_index, val_index, test_index = self.graph.get_edge_mask(dataset_splits, shuffle_data=shuffle_data)

        features = self.graph.features
        supports = self.graph.compute_supports(model_params=self.model_params, adj=adj)

        if (self.model_params.sparse_features):
            self.num_elements = features.nnz

        features = convert_sparse_matrix_to_sparse_tensor(features)
        labels = convert_sparse_matrix_to_sparse_tensor(self.graph.adj)
        labels_train = convert_sparse_matrix_to_sparse_tensor(adj)

        self.supports = list(
            map(
                lambda support: convert_sparse_matrix_to_sparse_tensor(support), supports
            )
        )

        total_sample_count = float(adj.shape[0]) ** 2
        positive_sample_count = adj.sum()
        negative_sample_count = total_sample_count - positive_sample_count

        positive_sample_weight = negative_sample_count / positive_sample_count

        self.autoencoder_model_params = AutoEncoderModelParams(
            positive_sample_weight=positive_sample_weight,
            node_count=self.graph.adj.shape[0]
        )

        return [[labels, labels_train, features],
                [train_index, val_index, test_index]]

    def _prepare_data(self, dataset_splits, shuffle_data=False):

        if (self.model_params.model_name in set([GCN_AE, GCN_VAE])):
            return self._prepare_data_auto_encoder(dataset_splits=dataset_splits,
                                                   shuffle_data=shuffle_data)
        else:
            return None

    def _populate_feed_dicts(self, dataset_splits=[85, 5, 10]):
        '''Method to populate the feed dicts'''

        [[labels, labels_train, features],
         [train_index, val_index, test_index]] = self._prepare_data(dataset_splits=dataset_splits)

        self.train_feed_dict = self._prepare_feed_dict(labels=labels_train,
                                                       features=features,
                                                       mask_indices=val_index,
                                                       dropout=self.model_params.dropout,
                                                       mode=TRAIN)
        # we are actually passing mask_indices for training data as val_index as the mask is ignored for the train data

        self.validation_feed_dict = self._prepare_feed_dict(labels=labels,
                                                            features=features,
                                                            mask_indices=val_index,
                                                            dropout=0,
                                                            mode=VALIDATION)

        self.test_feed_dict = self._prepare_feed_dict(labels=labels,
                                                      features=features,
                                                      mask_indices=test_index,
                                                      dropout=0,
                                                      mode=TEST)

    def get_autoencoder_model_params(self):
        return self.autoencoder_model_params
