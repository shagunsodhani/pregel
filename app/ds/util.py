import tensorflow as tf

from app.utils.constant import *


def get_feed_dict(g, mode=TRAIN):
    '''Method to obtain the feed dict using graph g'''
    supports = g.compue_supports
    placeholder_dict = placeholder_inputs(
        feature_size=g.features.shape[1],
        label_size=g.labels.shape[1],
        support_size=len(supports)
    )
    if(mode == TRAIN):
        mask = g.get_train_mask
    feed_dict = {
        placeholder_dict[LABELS]: g.labels,
        placeholder_dict[FEATURES]: g.features,
        placeholder_dict[SUPPORTS]: supports,
        placeholder_dict[MASK]: mask
    }
    return feed_dict


def placeholder_inputs(feature_size, label_size, support_size):
    '''
    Logic borrowed from
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/fully_connected_feed.py'''
    labels_placeholder = tf.placeholder(tf.int32, shape=(None, label_size))
    features_placeholder = tf.placeholder(tf.float32, shape=(None, feature_size))
    support_placeholder = [tf.sparse_placeholder(tf.float32)] * support_size
    mask_placeholder = tf.placeholder(tf.bool, shape=(None, 1))
    return {
        FEATURES: features_placeholder,
        LABELS: labels_placeholder,
        SUPPORTS: support_placeholder,
        MASK: mask_placeholder
    }