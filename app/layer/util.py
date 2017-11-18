import tensorflow as tf

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

def get_dotproduct(sparse_features=True):
    if (sparse_features):
        return tf.sparse_tensor_dense_matmul
    else:
        return tf.matmul