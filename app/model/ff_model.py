import tensorflow as tf

from app.layer.dense import SparseFC
# from app.layer.dense import Dense
from app.model import base_model
from app.utils.constant import FF_MODEL


class Model(base_model.Base_Model):
    '''Class for feedforwad model'''

    def __init__(self, model_params, sparse_model_params, placeholder_dict):
        super(Model, self).__init__(model_params=model_params,
                                    sparse_model_params=sparse_model_params,
                                    placeholder_dict=placeholder_dict)
        self.name = FF_MODEL
        self.model_op()

    def _layers_op(self):
        '''Operator to build the layers for the model.
        This function should not be called by the variables outside the class and
        is to be implemented by all the subclasses'''

        self.layers.append(SparseFC(input_dim=self.input_dim,
                                    output_dim=self.model_params.hidden_layer1_size,
                                    dropout_rate=self.dropout_rate,
                                    activation=tf.nn.relu,
                                    sparse_features=self.model_params.sparse_features,
                                    num_elements=self.num_elements))

        self.layers.append(SparseFC(input_dim=self.model_params.hidden_layer1_size,
                                    output_dim=self.output_shape[1],
                                    dropout_rate=self.dropout_rate,
                                    activation=lambda x: x,
                                    sparse_features=False,
                                    num_elements=self.num_elements))
