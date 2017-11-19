import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import initializers
from app.layer.util import sparse_dropout, get_dotproduct
from app.utils.constant import KERNEL, BIAS

# Code borrowed from
# * https://keras.io/layers/writing-your-own-keras-layers/
# * https://github.com/fchollet/keras/blob/master/keras/layers/core.py

class SparseFC(layers.Layer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate=0.0,
                 activation=tf.nn.relu,
                 sparse_features=True,
                 num_elements=-1,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.sparse_features = sparse_features
        self.num_elements = num_elements

        super(SparseFC, self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernel = self.add_weight(name=KERNEL,
                                      shape=(self.input_dim, self.output_dim),
                                      initializer=initializers.glorot_uniform(),
                                      trainable=True)

        self.bias = self.add_weight(name=BIAS,
                                    shape=(self.output_dim,),
                                    initializer=initializers.Zeros,
                                    trainable=True)

        super(SparseFC, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        '''Logic borrowed from: https://github.com/fchollet/keras/blob/master/keras/layers/core.py
        '''
        dotproduct = get_dotproduct(sparse_features=self.sparse_features)
        if (self.sparse_features):
            inputs = sparse_dropout(inputs, keep_prob=1 - self.dropout_rate, noise_shape=(self.num_elements,))
        else:
            inputs = tf.nn.dropout(inputs, keep_prob=1 - self.dropout_rate)
        output = dotproduct(inputs, self.kernel)

        output = tf.add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


