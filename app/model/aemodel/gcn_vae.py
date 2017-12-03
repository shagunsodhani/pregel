from app.utils.constant import GCN_VAE, SUPPORTS, MODE, TRAIN, NORMALISATION_CONSTANT, LOSS, ACCURACY
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
                                    autoencoder_model_params = autoencoder_model_params)
        self.name = GCN_VAE
        # We feed in the adjacency matrix in the sparse format and then make it dense

        # We need the mode variable to know if we need to use the mask values.
        # Since we use almost the entire adjacency matrix at training time, it is inefficient to use a mask
        # parameter at training time.

        self.mean_encoding = None
        self.log_sigma_encoding = None
        self.mean_encoder = None
        self.log_sigma_encoder = None
        self.z = None
        self.decoder = None
        self.node_count = autoencoder_model_params.node_count

        self.model_op()

    def _loss_op(self):
        '''Operator to compute the loss for the model.
        This method should not be directly called the variables outside the class.
        Note we do not need to initialise the loss as zero for each batch as process the entire data in just one batch.'''

        # Computing the KL loss analytically boils down to KL divergence between two gaussians as
        # computed here: https://arxiv.org/pdf/1312.6114.pdf (page 11)

        # Why do we have a normalisation constant in the kl_loss?
        # The reconstruction loss was normalized with respect to both the input size (node_count)
        # and input dimensionality (again node_count as it is basically an adjacency matrix).
        # So we normalize one more time with respect to node_count

        kl_loss = 0.5 * tf.reduce_mean(input_tensor=(tf.reduce_sum(input_tensor=(-2*self.log_sigma_encoding
                                                                           + tf.square(tf.exp(self.log_sigma_encoding))
                                                                           + tf.square(self.mean_encoding)
                                                                           - 1),
                                                             axis=1)))/self.node_count

        liklihood_loss = super(Model, self)._loss_op()

        return liklihood_loss + kl_loss

    def _mean_encoder_op(self):
        '''Component of the encoder op which learns the mean'''
        return SparseGC(input_dim=self.model_params.hidden_layer1_size,
                                    output_dim=self.model_params.hidden_layer2_size,
                                    supports=self.supports,
                                    dropout_rate=self.dropout_rate,
                                    activation=lambda x: x,
                                    sparse_features=False,
                                    num_elements=self.num_elements)

    def _log_sigma_encoder_op(self):
        '''Component of the encoder op which learns the log of sigma'''
        return SparseGC(input_dim=self.model_params.hidden_layer1_size,
                                    output_dim=self.model_params.hidden_layer2_size,
                                    supports=self.supports,
                                    dropout_rate=self.dropout_rate,
                                    activation=lambda x: x,
                                    sparse_features=False,
                                    num_elements=self.num_elements)

    def _layers_op(self):
        '''Operator to build the layers for the model.
        This function should not be called by the variables outside the class and
        is to be implemented by all the subclasses.

        This function implemenetation is different from the other _layer_ops as now our model is not sequential.'''

        self.layers.append(SparseGC(input_dim=self.input_dim,
                                    output_dim=self.model_params.hidden_layer1_size,
                                    supports=self.supports,
                                    dropout_rate=self.dropout_rate,
                                    activation=tf.nn.relu,
                                    sparse_features=self.model_params.sparse_features,
                                    num_elements=self.num_elements))

        self.mean_encoder = self._mean_encoder_op()

        self.log_sigma_encoder = self._log_sigma_encoder_op()

        self.decoder = InnerProductDecoder(input_dim=self.input_dim,
                                    output_dim=self.input_dim,
                                    dropout_rate=self.dropout_rate,
                                    activation=lambda x: x,
                                    sparse_features=False)

        # The output of the GCN-AE model is always an adjacency matrix

    def model_op(self):
        '''Operator to build the network.
        This function should be called by the variables outside the class.
        We can not use the model_op from the base class as now it is not just a sequential model.'''
        scope_name = self.name + "_var_to_save"
        with tf.variable_scope(name_or_scope=scope_name):
            self._layers_op()

        self.vars = {var.name: var for var in
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)}
        # self._save_op()

        self.activations = [self.inputs]

        for layer in self.layers:
            self.activations.append(
                layer(self.activations[-1])
            )

        # Activations is a list of the form input::first_common_hidden_layer::..::last_common_hidden_layer

        self.mean_encoding = self.mean_encoder(self.activations[-1])

        self.log_sigma_encoding = self.log_sigma_encoder(self.activations[-1])

        self.z = tf.random_normal(shape=[self.node_count, self.model_params.hidden_layer2_size],
                                  mean=self.mean_encoding,
                                  stddev=tf.exp(self.log_sigma_encoding))

        self.outputs = self.decoder(self.z)

        self._compute_metrics()
        self.optimizer_op = self._optimizer_op()

    def _compute_metrics(self):
        '''Method to compute the metrics of interest'''
        self.predictions = self._prediction_op()
        self.loss = self._loss_op()
        self.accuracy = self._accuracy_op()
        self.embeddings = self.z
        tf.summary.scalar(LOSS, self.loss)
        tf.summary.scalar(ACCURACY, self.accuracy)
        self.summary_op = tf.summary.merge_all()