from app.utils.constant import GCN_AE, SUPPORTS, MODE, TRAIN, NORMALISATION_CONSTANT, LOSS, ACCURACY
from app.model.aemodel import base_model

from app.layer.GC import SparseGC
from app.layer.IPD import InnerProductDecoder

import tensorflow as tf
import numpy as np

class Model(base_model.Base_Model):
    '''Class for GCN Model'''

    def __init__(self, model_params, sparse_model_params, placeholder_dict, autoencoder_model_params):
        super(Model, self).__init__(model_params=model_params,
                                    sparse_model_params=sparse_model_params,
                                    placeholder_dict=placeholder_dict,
                                    autoencoder_model_params=autoencoder_model_params)
        self.name = GCN_AE
        self.model_op()

    def _layers_op(self):
        '''Operator to build the layers for the model.
        This function should not be called by the variables outside the class and
        is to be implemented by all the subclasses'''
        self.layers.append(SparseGC(input_dim=self.input_dim,
                                    output_dim=self.model_params.hidden_layer1_size,
                                    supports=self.supports,
                                    dropout_rate=self.dropout_rate,
                                    activation=tf.nn.relu,
                                    sparse_features=self.model_params.sparse_features,
                                    num_elements=self.num_elements))

        self.layers.append(SparseGC(input_dim=self.model_params.hidden_layer1_size,
                                    # output_dim=int(self.output_shape[1]),
                                    output_dim=self.model_params.hidden_layer2_size,
                                    supports=self.supports,
                                    dropout_rate=self.dropout_rate,
                                    activation=lambda x: x,
                                    sparse_features=False,
                                    num_elements=self.num_elements))

        # So far, we have just added the GCN model.

        self.layers.append(InnerProductDecoder(input_dim=self.input_dim,
                                    output_dim=self.input_dim,
                                    dropout_rate=self.dropout_rate,
                                    activation=lambda x: x,
                                    sparse_features=False))

        # The output of the GCN-AE model is always an adjacency matrix

    def _compute_metrics(self):
        '''Method to compute the metrics of interest'''
        self.predictions = self._prediction_op()
        self.loss = self._loss_op()
        self.accuracy = self._accuracy_op()
        self.embeddings = self.activations[2]
        tf.summary.scalar(LOSS, self.loss)
        tf.summary.scalar(ACCURACY, self.accuracy)
        self.summary_op = tf.summary.merge_all()