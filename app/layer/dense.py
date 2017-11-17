import tensorflow as tf
from tensorflow.contrib.keras import layers


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
        self.weight = None
        self.bias = None

        super(SparseFC, self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      # initializer='uniform',
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    # initializer='uniform',
                                    trainable=True)

        super(SparseFC, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        '''Logic borrowed from: https://github.com/fchollet/keras/blob/master/keras/layers/core.py
        '''
        if (self.sparse_features):
            inputs = sparse_dropout(inputs, keep_prob=1 - self.dropout_rate, noise_shape=(self.num_elements,))
            output = tf.sparse_tensor_dense_matmul(inputs, self.kernel)
        else:
            inputs = tf.nn.dropout(inputs, keep_prob=1 - self.dropout_rate)
            output = tf.matmul(inputs, self.kernel)

        output = tf.add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def sparse_dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
    '''borrowed logic and implementation from https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/nn_ops.py'''

    # Skipping all the assertions

    if (keep_prob == 1):
        return x

    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape,
                                       seed=seed,
                                       dtype=x.dtype)

    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = tf.floor(random_tensor)
    # Typecase necessary as mentioned on https://www.tensorflow.org/api_docs/python/tf/sparse_retain
    binary_tensor = tf.cast(binary_tensor, dtype=tf.bool)
    ret = tf.sparse_retain(x, binary_tensor)
    ret = ret * (1 / keep_prob)
    return ret
