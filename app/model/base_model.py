from abc import ABC, abstractmethod
from app.utils.constant import BASE_MODEL

import tensorflow as tf

class Base_Model(ABC):
    '''Base class for all the models'''

    def __init__(self, model_params):

        self.name = BASE_MODEL
        self.inputs = None
        self.outputs = None
        self.optimizer = tf.train.AdamOptimizer(learning_rate=model_params.learning_rate)
        self.loss = -1
        self.accuracy = -1
        self.vars = {}
        self.layers = []
        self.activations = []
        self.saver = None
        self.optimizer_op = None

    @abstractmethod
    def _layers_op(self):
        '''Operator to build the layers for the model.
        This function should not be called by the variables outside the class and
        is to be implemented by all the subclasses'''
        pass

    @abstractmethod
    def predict_op(self):
        '''Operator to make predictions using the network.'''

        pass

    @abstractmethod
    def _loss_op(self):
        '''Operator to compute the loss for the model.
        This method should not be directly called the variables outside the class.'''
        pass

    @abstractmethod
    def _accuracy_op(self):
        '''Operator to compute the accuracy for the model.
        This method should not be directly called the variables outside the class.'''
        pass

    def _optimizer_op(self):
        '''Operator to run the optimiser'''
        if (not self.optimizer):
            raise AttributeError("Optimizer (self.optimizer) not set.")
        self.optimizer.minimize(self.loss)

    def _save_op(self):
        '''Operator to save the model to the disk.
        This method should not be directly called the variables outside the class.'''

        self.saver = tf.train.Saver(self.vars)

    def save(self, sess, save_path, global_step):
        '''Method to save the model to the disk'''
        if(not self.saver):
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
        scope_name = self.name+"_var_to_save"
        with tf.variable_scope(scope=scope_name):
            self._layers_op()

        self.vars = {var.name: var for var in
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)}
        self._save_op()

        self.activations = [self.inputs]

        for layer in self.layers:
            self.activations.append(
                layer(self.activations[-1])
            )
        # Activations is a list of the form input::first_hidden_layer::..::last_hidden_layer::outputs

        self.outputs = self.activations[-1]

        self._loss_op()
        self._accuracy_op()
        self.optimizer_op = self._optimizer_op()

