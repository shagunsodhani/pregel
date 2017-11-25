from app.utils.constant import GCN_AE, SUPPORTS, MODE, TRAIN, NORMALISATION_CONSTANT
from app.model import base_model

from app.layer.GC import SparseGC
from app.layer.IPD import InnerProductDecoder

import tensorflow as tf
import numpy as np

class Model(base_model.Base_Model):
    '''Class for GCN Model'''

    def __init__(self, model_params, sparse_model_params, placeholder_dict, autoencoder_model_params):
        super(Model, self).__init__(model_params=model_params,
                                    sparse_model_params=sparse_model_params,
                                    placeholder_dict=placeholder_dict)
        self.name = GCN_AE
        self.labels = tf.sparse_tensor_to_dense(self.labels)
        # We feed in the adjacency matrix in the sparse format and then make it dense

        # We need the mode variable to know if we need to use the mask values.
        # Since we use almost the entire adjacency matrix at training time, it is inefficient to use a mask
        # parameter at training time.
        self.mode = placeholder_dict[MODE]

        self.supports = placeholder_dict[SUPPORTS]


        # For GCN AE model, the support is just the adj matrix. For clarity, we would save it another param, self.adj
        # We could have set this as one of the params in the AutoEncoderModelParams but are sending it via the
        # placeholderdict to keep it consistent with the base models.
        self.adj = self.supports[0]

        self.normalisation_constant = placeholder_dict[NORMALISATION_CONSTANT]
        self.positive_sample_weight = autoencoder_model_params.positive_sample_weight

        self.model_op()

    def _loss_op(self):
        '''Operator to compute the loss for the model.
        This method should not be directly called the variables outside the class.
        Not we do not need to initialise the loss as zero for each batch as process the entire data in just one batch.'''

        complete_loss = tf.nn.weighted_cross_entropy_with_logits(
                            targets = self.labels,
                            logits = self.outputs,
                            pos_weight=self.positive_sample_weight
                        )

        def _compute_masked_loss(complete_loss):
            '''Method to compute the masked loss'''
            normalized_mask = self.mask / tf.sparse_reduce_sum(self.mask)
            # normalized_mask = self.mask
            complete_loss = tf.multiply(complete_loss, tf.sparse_tensor_to_dense(normalized_mask))
            return tf.reduce_sum(complete_loss)
            # the sparse_tensor_to_dense would be the bottleneck step and shoudld be replaced by something more efficient

        complete_loss = tf.cond(tf.equal(self.mode, TRAIN),
                                true_fn=lambda : tf.reduce_mean(complete_loss),
                                false_fn=lambda : _compute_masked_loss(complete_loss))


        return complete_loss * self.normalisation_constant


    def _accuracy_op(self):
        '''Operator to compute the accuracy for the model.
        This method should not be directly called the variables outside the class.'''


        logits = tf.sigmoid(x = self.outputs, name="output_to_logits")
        predictions = tf.cast(tf.greater_equal(logits, 0.5, name="logits_to_prediction"),
                              dtype=tf.float32)

        correct_predictions = tf.cast(tf.equal(predictions,
                                       self.labels), dtype=tf.float32)

        def _compute_masked_accuracy(correct_predictions):
            '''Method to compute the masked loss'''
            normalized_mask = self.mask / tf.sparse_reduce_sum(self.mask)
            # normalized_mask = self.mask
            correct_predictions = tf.multiply(correct_predictions, tf.sparse_tensor_to_dense(normalized_mask))
            return tf.reduce_sum(correct_predictions, name="accuracy_op")

        accuracy = tf.cond(tf.equal(self.mode, TRAIN),
                                true_fn=lambda: tf.reduce_mean(correct_predictions, name="accuracy_op"),
                                false_fn=lambda: _compute_masked_accuracy(correct_predictions))

        return accuracy

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
