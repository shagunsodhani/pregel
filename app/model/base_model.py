from abc import ABC, abstractmethod

import tensorflow as tf

from app.model.util import masked_softmax_loss, masked_accuracy
from app.utils.constant import BASE_MODEL, LABELS, MASK, FEATURES, DROPOUT


class Base_Model(ABC):
    '''Base class for all the models'''

    def __init__(self, model_params, sparse_model_params, placeholder_dict):

        self.name = BASE_MODEL
        self.inputs = placeholder_dict[FEATURES]
        self.outputs = None
        self.input_dim = sparse_model_params.feature_size
        self.output_shape = placeholder_dict[LABELS].get_shape()
        self.mask = placeholder_dict[MASK]
        self.labels = placeholder_dict[LABELS]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=model_params.learning_rate)
        self.dropout_rate = placeholder_dict[DROPOUT]
        self.num_elements = sparse_model_params.num_elements
        self.loss = -1
        self.accuracy = -1
        self.vars = {}
        self.layers = []
        self.activations = []
        self.saver = None
        self.model_params = model_params
        self.optimizer_op = None

    @abstractmethod
    def _layers_op(self):
        '''Operator to build the layers for the model.
        This function should not be called by the variables outside the class and
        is to be implemented by all the subclasses'''
        pass

    def predict_op(self):
        '''Operator to make predictions using the network.'''
        return tf.nn.softmax(self.outputs)

    def _compute_softmax_loss(self):
        '''Method to compute the softmax loss'''
        return masked_softmax_loss(labels=self.labels,
                                   logits=self.outputs,
                                   mask=self.mask)

    def _l2_loss(self):
        '''Method to compute the L2 loss'''
        loss = 0
        for layer in self.layers[:-1]:
            for W in layer.weights:
                loss += tf.nn.l2_loss(W) * self.model_params.l2_weight
        return loss

    def _loss_op(self):
        '''Operator to compute the loss for the model.
        This method should not be directly called the variables outside the class.
        Not we do not need to initialise the loss as zero for each batch as process the entire data in just one batch.'''

        # Cross entropy loss
        loss = self._compute_softmax_loss()

        # L2-Regularization loss
        loss+=self._l2_loss()

        return loss

    def _accuracy_op(self):
        '''Operator to compute the accuracy for the model.
        This method should not be directly called the variables outside the class.'''
        return masked_accuracy(labels=self.labels,
                                        logits=self.outputs,
                                        mask=self.mask)

    def _optimizer_op(self):
        '''Operator to run the optimiser'''
        if (not self.optimizer):
            raise AttributeError("Optimizer (self.optimizer) not set.")
        return self.optimizer.minimize(self.loss)

    def _save_op(self):
        '''Operator to save the model to the disk.
        This method should not be directly called the variables outside the class.'''

        self.saver = tf.train.Saver(self.vars)

    def save(self, sess, save_path, global_step):
        '''Method to save the model to the disk'''
        if (not self.saver):
            raise AttributeError("Save operator not set. Call self._save_op first")
        path = self.saver.save(sess, save_path, global_step=global_step)
        print("Saving {} model for global_step {} at {}".format(self.name, global_step, path))

    def load(self, sess, save_path):
        if (not self.saver):
            raise AttributeError("Save operator not set. Call self._save_op first")
        # It is assumed that save_path contains information about the global_step, if that is needed
        self.saver.restore(sess, save_path)
        print("Restoring {} model from {}".format(self.name, save_path))

    def model_op(self):
        '''Operator to build the network.
        This function should be called by the variables outside the class'''
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
        # Activations is a list of the form input::first_hidden_layer::..::last_hidden_layer::outputs

        self.outputs = self.activations[-1]
        self.loss = self._loss_op()
        self.accuracy = self._accuracy_op()
        self.optimizer_op = self._optimizer_op()