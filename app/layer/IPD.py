import tensorflow as tf
from tensorflow.contrib.keras import layers
from app.layer.util import sparse_dropout, get_dotproduct_op, get_transpose_op

# Code borrowed from
# * https://keras.io/layers/writing-your-own-keras-layers/
# * https://github.com/fchollet/keras/blob/master/keras/layers/core.py

class InnerProductDecoder(layers.Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate=0.0,
                 activation=tf.nn.sigmoid,
                 sparse_features=False,
                 num_elements=-1,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.sparse_features = sparse_features
        self.num_elements = num_elements

        super(InnerProductDecoder, self).__init__(**kwargs)

    def build(self, input_shape):

        super(InnerProductDecoder, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        '''Logic borrowed from: https://github.com/fchollet/keras/blob/master/keras/layers/core.py
        '''
        dotproduct_op = get_dotproduct_op(sparse_features=self.sparse_features)
        transpose_op = get_transpose_op(sparse_features=self.sparse_features)
        if (self.sparse_features):
            inputs = sparse_dropout(inputs, keep_prob=1 - self.dropout_rate, noise_shape=(self.num_elements,))
        else:
            inputs = tf.nn.dropout(inputs, keep_prob=1 - self.dropout_rate)
        inputs_transpose =transpose_op(inputs)

        output = dotproduct_op(inputs, inputs_transpose)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)