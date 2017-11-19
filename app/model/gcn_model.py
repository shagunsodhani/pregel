from app.utils.constant import GCN_MODEL, SUPPORTS
from app.model import base_model

from app.layer.GC import SparseGC

import tensorflow as tf

class Model(base_model.Base_Model):
    '''Class for GCN Model'''

    def __init__(self, model_params, sparse_model_params, placeholder_dict):
        super(Model, self).__init__(model_params=model_params,
                                    sparse_model_params=sparse_model_params,
                                    placeholder_dict=placeholder_dict)
        self.name = GCN_MODEL
        self.supports = placeholder_dict[SUPPORTS]
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
                                    # sparse_features=False,
                                    num_elements=self.num_elements))

        self.layers.append(SparseGC(input_dim=self.model_params.hidden_layer1_size,
                                    output_dim=int(self.output_shape[1]),
                                    supports=self.supports,
                                    dropout_rate=self.dropout_rate,
                                    activation=lambda x: x,
                                    sparse_features=False,
                                    num_elements=self.num_elements))
